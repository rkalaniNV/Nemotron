# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety family stages wired into the shared BYOB dispatcher."""

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.base import ConfigPath
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json as _write_json
from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_yaml as _write_yaml
from nemotron.steps.byob.runtime.benchmark_families.safety.config import (
    ConfigError,
    ResolvedConfig,
    load_resolved,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.coverage import (
    COVERAGE_OUTPUTS,
    GAPS_FILE,
    MAX_BACKFILL_ROUNDS,
    PLAN_FILE,
    apply_backfill,
    build_coverage_plan,
    case_cap_for,
    load_eligible_rules,
    load_routed_statuses,
    quality_gate_shortfall,
    rewrite_coverage_plan,
    rule_level_rejected_policies,
    select_backfill_cells,
    span_text,
    write_coverage_plan,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import (
    DATASET_OUTPUTS,
    load_jsonl_objects,
    select_dataset,
    write_dataset,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.eval import (
    EvalInputError,
    clear_evaluation,
    evaluate_published,
    write_evaluation,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.extract import PROMPT_VERSION
from nemotron.steps.byob.runtime.benchmark_families.safety.ingest import (
    INGEST_VERSION,
    QUARANTINED,
    REJECTED,
    SAFE,
    IngestResult,
    ingest_sources,
    rule_input_digests,
    write_artifacts,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.publish import (
    MANIFEST_NAME,
    QUALITY_FILE,
    assert_published_intact,
    clear_published,
    compile_publication,
    publication_claim,
    status_for,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.review import (
    content_matches,
    install_review,
    load_review,
    pending_detail,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.rules import POLICY_RULES_FILE, run_rule_stages
from nemotron.steps.byob.runtime.benchmark_families.safety.scenarios import (
    CASES_FILE,
    CONTRAST_FILE,
    SCENARIO_OUTPUTS,
    build_scenarios,
    load_contrast_families,
    load_plan_rows,
    load_unscheduled_slots,
    write_scenarios,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.sources import (
    SourceChangedError,
    SourceError,
    canonical_source_path,
    scan_sources,
    source_fingerprint,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.status import (
    RunStatus,
    SafetyStatus,
    abort,
)

__version__ = "0.1.0"
CACHE_INDEX_VERSION = 2
REVIEW_PACKET_VERSION = 1
SIGNOFF_VERSION = 1
QUARANTINE_STATUSES = frozenset({SAFE, QUARANTINED, REJECTED})
RULE_STAGES = ("extract", "normalize", "conflicts")
RECOVERED_DIR = "recovered"
PACKET_NAME = "publication_review_packet.yaml"
SIGNOFF_NAME = "publication_signoff.json"
SIGNER_ROLES = ("benchmark_owner", "compliance_owner", "policy_author", "adjudicator", "release_owner")
# A packet may name its own required_roles, for example when no regulated rule needs a compliance owner.
PROFILE_ROLES = {
    "standard": ("benchmark_owner",),
    "regulated": ("benchmark_owner", "compliance_owner"),
    "high_assurance": ("policy_author", "adjudicator", "release_owner"),
}
DISTINCT_SIGNERS_PROFILES = frozenset({"high_assurance"})
APPROVE_COMMAND = "python -m nemotron.steps.byob.scripts.approve_safety_review --config <config>"


def prepare_safety(config_path: ConfigPath) -> Path:
    """Initialize or refresh one safety run directory. This stage does not publish a benchmark."""
    resolved = _load(config_path)
    run_dir = resolved.run_dir
    digest = _digest(resolved)
    with _run_lock(run_dir):
        fingerprint, sources = _sources(resolved)
        recovered: list[str] = []
        previous = _recover_json(run_dir / "run_record.json", run_dir, _valid_record, recovered)
        index_path = run_dir / "stage_cache" / "cache_index.json"
        index = _recover_json(index_path, run_dir, _valid_cache_index, recovered)
        index = _refresh_cache_index(index, digest, fingerprint)
        try:
            privacy_config = resolved.extras.get("privacy", {})
            privacy_labels = (
                privacy_config.get("additional_labels", {}) if isinstance(privacy_config, dict) else {}
            )
            ingested = ingest_sources(
                resolved.input_dir,
                sources,
                run_dir / "stage_cache",
                privacy_labels=privacy_labels,
            )
        except SourceChangedError as exc:
            raise abort(RunStatus.PREPARE_REQUIRED, run_dir, str(exc)) from exc
        except SourceError as exc:
            raise SafetyStatus(RunStatus.INVALID_CONFIG, f"Invalid configuration:\n- {exc}") from exc
        stop = _ingest_stop(ingested)
        now = _now()
        record = {
            "record_type": "safety_run_record",
            "publication_commit": False,
            "family": "safety",
            "expt_name": resolved.expt_name,
            "pipeline_version": __version__,
            "config_sha256": digest,
            "source_fingerprint": fingerprint,
            "source_count": len(sources),
            "status": "prepared",
            "input_dir": str(resolved.input_dir),
            "output_dir": str(resolved.output_dir),
            "run_dir": str(run_dir),
            "created_at": (previous or {}).get("created_at", now),
            "updated_at": now,
            "note": "run_manifest.json is written only when a safety benchmark is published.",
        }
        record["ingest"] = ingested.counts()
        if stop is not None:
            record["status"] = stop.value
            record["last_outcome"] = {"stage": "prepare", "status": stop.value, "at": now}
        elif _same_inputs(previous, digest, fingerprint) and "last_outcome" in previous:
            record["last_outcome"] = previous["last_outcome"]
        if recovered:
            record["recovered_artifacts"] = recovered
        index["stages"]["ingest"] = {
            "status": "completed",
            "outcome": stop.value if stop is not None else "ready",
            "ingest_version": INGEST_VERSION,
            "config_sha256": digest,
            "source_fingerprint": fingerprint,
            "counts": ingested.counts(),
            "completed_at": now,
        }
        (run_dir / "stage_cache").mkdir(exist_ok=True)
        (run_dir / "quarantine").mkdir(exist_ok=True)
        clear_published(run_dir)
        clear_evaluation(run_dir)
        _write_yaml(run_dir / "config_resolved.yaml", resolved.to_dict())
        write_artifacts(run_dir, ingested)
        record["ingest_artifacts"] = rule_input_digests(run_dir / "stage_cache")
        _write_json(index_path, index)
        # run_record.json is replaced last: it marks the prepare as committed.
        _write_json(run_dir / "run_record.json", record)
    if stop is not None:
        raise abort(stop, run_dir, _ingest_detail(ingested))
    return run_dir


def _ingest_stop(ingested: IngestResult) -> RunStatus | None:
    """The status prepare must stop with when no policy document can move on."""
    if ingested.privacy_blocked:
        return RunStatus.PRIVACY_BLOCKED
    if ingested.count(SAFE) == 0:
        return RunStatus.NO_USABLE_SOURCES
    return None


def _ingest_detail(ingested: IngestResult) -> str:
    reasons = Counter(row["reason"] for row in ingested.files if row["status"] != SAFE)
    return "Files held back: " + ", ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items())) + "."


def generate_safety(config_path: ConfigPath, /, *, skip_until: str | None = None) -> Path:
    """Build, review, resume, and publish one signed safety benchmark."""
    resolved = _load(config_path)
    run_dir = resolved.run_dir
    digest = _digest(resolved)
    if not run_dir.is_dir():
        raise abort(RunStatus.PREPARE_REQUIRED, run_dir)
    with _run_lock(run_dir):
        fingerprint, sources = _sources(resolved)
        record = _read_json(run_dir / "run_record.json", run_dir)
        if record is not None and not _valid_record(record):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT, run_dir, "run_record.json is damaged; run stage=prepare to recover it."
            )
        if record is None or not _same_inputs(record, digest, fingerprint):
            raise abort(RunStatus.PREPARE_REQUIRED, run_dir, _changed_inputs(record, digest, fingerprint))
        manifest: dict[str, Any] | None = None
        resumed = _resume_published(run_dir, record)
        if resumed is not None:
            status, detail, rule_counts = resumed
        elif skip_until is not None:
            status, detail = (
                RunStatus.PIPELINE_INCOMPLETE,
                "skip_until is not available until safety checkpoints exist.",
            )
            rule_counts = None
        else:
            try:
                status, detail, rule_counts, manifest = _publication_blocker(
                    resolved, digest, fingerprint, sources, record
                )
            except SafetyStatus as caught:
                # The exception still stops the command. The record keeps the same status.
                now = _now()
                record["status"] = caught.status.value
                record["last_outcome"] = {"stage": "generate", "status": caught.status.value, "at": now}
                record["updated_at"] = now
                _write_json(run_dir / "run_record.json", record)
                raise
            if rule_counts is not None:
                record["rules"] = rule_counts
        record["status"] = status.value
        record["last_outcome"] = {"stage": "generate", "status": status.value, "at": _now()}
        record["updated_at"] = record["last_outcome"]["at"]
        if manifest is not None:
            _commit_publication(run_dir, record, manifest)
        else:
            _write_json(run_dir / "run_record.json", record)
    raise abort(status, run_dir, detail)


def _resume_published(
    run_dir: Path, record: dict[str, Any]
) -> tuple[RunStatus, str, dict[str, int] | None] | None:
    """Return the published outcome when the commit is intact, including a torn last write."""
    if record.get("publication_commit") is True:
        assert_published_intact(run_dir, record)
        return _published_outcome(run_dir, record)
    claim = record.get("publication")
    if not isinstance(claim, dict) or not claim.get("certification"):
        return None
    if not (run_dir / MANIFEST_NAME).is_file():
        return None
    try:
        assert_published_intact(run_dir, {**record, "publication_commit": True})
    except SafetyStatus:
        return None
    record["publication_commit"] = True
    record["updated_at"] = _now()
    _write_json(run_dir / "run_record.json", record)
    return _published_outcome(run_dir, record)


def _published_outcome(
    run_dir: Path, record: dict[str, Any]
) -> tuple[RunStatus, str, dict[str, int] | None]:
    published = _read_json(run_dir / MANIFEST_NAME, run_dir)
    if published is None:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is missing after publication.")
    try:
        status = status_for(str(published.get("certification") or ""))
    except KeyError as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{MANIFEST_NAME} has an unknown certification.",
        ) from exc
    rule_counts = record.get("rules") if isinstance(record.get("rules"), dict) else None
    return status, "", rule_counts


def _commit_publication(run_dir: Path, record: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Write the manifest as the publication commit, then record that the commit exists.

    Compiled files already exist. The run record first stores the claim without
    ``publication_commit``, then the manifest is written, then the flag is set.
    A crash after the manifest is recovered by verifying the claim.
    """
    record["publication"] = publication_claim(manifest)
    record["publication_commit"] = False
    _write_json(run_dir / "run_record.json", record)
    _write_json(run_dir / MANIFEST_NAME, manifest)
    record["publication_commit"] = True
    _write_json(run_dir / "run_record.json", record)


def translate_safety(config_path: ConfigPath, /, *, skip_until: str | None = None) -> Path:
    """Keep translation off the shared dispatcher's Silver exit code."""
    del skip_until
    raise _unavailable(config_path)


def evaluate_safety(config_path: ConfigPath) -> Path:
    """Score a published benchmark with the path-1 adapter and the path-2 Guardrails log."""
    resolved = _load(config_path)
    run_dir = resolved.run_dir
    if not run_dir.is_dir():
        raise abort(RunStatus.PREPARE_REQUIRED, run_dir)
    with _run_lock(run_dir):
        record = _read_json(run_dir / "run_record.json", run_dir)
        if record is not None and not _valid_record(record):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT, run_dir, "run_record.json is damaged; run stage=prepare to recover it."
            )
        if record is None or record.get("publication_commit") is not True:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "Evaluation needs a published benchmark.")
        assert_published_intact(run_dir, record)
        evaluation = resolved.extras.get("evaluation")
        try:
            path1, path2 = evaluate_published(run_dir, evaluation, config_path=Path(config_path))
        except EvalInputError as exc:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, str(exc)) from exc
        write_evaluation(run_dir, path1, path2)
        now = _now()
        record["evaluation"] = {"path1": path1.get("status"), "path2": path2.get("status")}
        record["last_outcome"] = {"stage": "eval", "status": "evaluated", "at": now}
        record["updated_at"] = now
        _write_json(run_dir / "run_record.json", record)
    return run_dir


def _publication_blocker(
    config: ResolvedConfig, digest: str, fingerprint: str, sources: dict[str, str], record: dict[str, Any]
) -> tuple[RunStatus, str, dict[str, int] | None, dict[str, Any] | None]:
    """Return the status generate must surface before it stops, with any extra detail and rule counts.

    The privacy check runs first: rule extraction is the first stage that may call a model.
    """
    source_stop = _source_blocker(config, digest, fingerprint, sources)
    if source_stop is not None:
        return source_stop, "", None, None
    run_dir = config.run_dir
    _check_rule_inputs(record, run_dir)
    packet_path = run_dir / PACKET_NAME
    previous_packet = _read_packet(packet_path, run_dir)
    signoff = _read_json(run_dir / SIGNOFF_NAME, run_dir)
    if previous_packet is not None and signoff is not None and _bound_to(previous_packet, digest, fingerprint):
        packet_sha256 = _file_sha256(packet_path, run_dir)
        if not _signs_packet(signoff, digest, fingerprint, packet_sha256):
            clear_published(run_dir)
            return (
                RunStatus.REVIEW_REQUIRED,
                f"{SIGNOFF_NAME} does not match the current review packet, so it was not accepted. "
                "Sign the current packet again.",
                record.get("rules") if isinstance(record.get("rules"), dict) else None,
                None,
            )
        if isinstance(previous_packet.get("content"), dict):
            mismatched = content_matches(run_dir, previous_packet)
            if mismatched:
                raise abort(
                    RunStatus.INVALID_RUN_ARTIFACT,
                    run_dir,
                    f"{mismatched} changed after the review packet was signed.",
                )
    counts = _rule_stages(config, digest, fingerprint)
    coverage = _coverage_stage(config, digest, fingerprint)
    scenarios = _scenario_stage(config, digest, fingerprint)
    dataset = _dataset_stage(config, digest, fingerprint)
    coverage, scenarios, dataset = _backfill_quality_gates(
        config, digest, fingerprint, coverage, scenarios, dataset
    )
    record["coverage"] = coverage
    record["scenarios"] = scenarios
    record["dataset"] = dataset
    coverage_note = (
        f"Coverage plan: {coverage['certification']} for eligible rules, "
        f"{coverage['scheduled_cases']} scheduled, {coverage['required_cases']} required, "
        f"{len(coverage['missing_required'])} required omitted, "
        f"{len(coverage['excluded_rules'])} rules excluded."
        f"{_backfill_note(coverage)}"
    )
    scenario_note = (
        f"Scenarios: {scenarios['accepted']} accepted, {scenarios['rejected']} rejected, "
        f"{scenarios['reused_missing_leg']} missing-leg cases reused."
    )
    dataset_note = (
        f"Dataset: {dataset['validated']} validated, {dataset['rejected']} rejected, "
        f"{dataset['duplicates_removed']} duplicates removed, "
        f"{dataset['development']} development, {dataset['private_test']} private-test "
        f"({dataset['split_reason']}), {dataset['quality_gate_failures']} minimum quality gates unmet."
    )
    if counts["eligible_for_generation"] + counts["needs_decision"] == 0:
        return (
            RunStatus.NO_USABLE_RULES,
            f"{_rule_summary(counts)} {coverage_note} {scenario_note} {dataset_note}",
            counts,
            None,
        )
    rules_note = (
        f"{_rule_summary(counts)} {coverage_note} {scenario_note} {dataset_note} "
        f"Rule file: {run_dir / 'stage_cache' / POLICY_RULES_FILE}."
    )
    build = load_review(
        run_dir,
        config_sha256=digest,
        source_fingerprint=fingerprint,
        mode=config.mode,
        profile=config.assurance["profile"],
        example_cap=config.budget["max_review_items"],
    )
    current_digest = _file_sha256(packet_path, run_dir) if packet_path.is_file() else ""
    packet_changed = current_digest != build.digest
    if packet_changed:
        older = previous_packet is not None and not _bound_to(previous_packet, digest, fingerprint)
        install_review(run_dir, build)
        clear_published(run_dir)
        record["review"] = _review_record(build)
        if config.mode != "preview" or build.pending:
            if older:
                detail = f"{rules_note} {_OLDER_PACKET}"
            elif signoff is not None and _signed_roles(signoff):
                detail = (
                    f"{SIGNOFF_NAME} does not match the current review packet, so it was not accepted. "
                    "Sign the current packet again."
                )
            elif signoff is not None:
                detail = _signoff_gaps(signoff, config, build.packet, run_dir)
            else:
                detail = rules_note
            return (
                RunStatus.REVIEW_REQUIRED,
                detail,
                counts,
                None,
            )
    else:
        # Normalize the decision sheet and refresh the page without changing the signed packet.
        install_review(run_dir, build)
        clear_published(run_dir)
        record["review"] = _review_record(build)
    if signoff is None and config.mode != "preview":
        return RunStatus.REVIEW_REQUIRED, "", counts, None
    if signoff is not None and not _signs_packet(signoff, digest, fingerprint, build.digest):
        return (
            RunStatus.REVIEW_REQUIRED,
            f"{SIGNOFF_NAME} does not match the current review packet, so it was not accepted. "
            "Sign the current packet again.",
            counts,
            None,
        )
    if signoff is not None:
        gaps = _signoff_gaps(signoff, config, build.packet, run_dir)
        if gaps:
            return RunStatus.REVIEW_REQUIRED, gaps, counts, None
    if build.pending:
        return RunStatus.REVIEW_REQUIRED, pending_detail(build.pending), counts, None
    stage_cache = run_dir / "stage_cache"
    try:
        cases = load_jsonl_objects(
            (stage_cache / "deduplicated_cases.jsonl").read_text(encoding="utf-8"),
            "deduplicated_cases.jsonl",
        )
        families = load_jsonl_objects(
            (stage_cache / CONTRAST_FILE).read_text(encoding="utf-8"),
            CONTRAST_FILE,
        )
        split_manifest = json.loads((stage_cache / "split_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(split_manifest, dict):
            raise ValueError("split_manifest.json must contain an object")
        manifest = compile_publication(
            run_dir,
            build.packet,
            cases=cases,
            families=families,
            split_manifest=split_manifest,
            withheld_policies=build.withheld_policies,
            mode=config.mode,
            published_at=_now(),
            config_sha256=digest,
            source_fingerprint=fingerprint,
            packet_sha256=build.digest,
        )
    except SafetyStatus:
        clear_published(run_dir)
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        clear_published(run_dir)
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"benchmark cannot be compiled: {exc}") from exc
    if _quality_gates_block_publish(config, run_dir):
        clear_published(run_dir)
        return RunStatus.QUALITY_GATES_FAILED, "", counts, None
    return status_for(manifest["certification"]), "", counts, manifest


def _review_record(build: Any) -> dict[str, Any]:
    """Small resumable review summary for run_record.json."""
    return {
        "packet_sha256": build.digest,
        "cards": len(build.packet["cards"]),
        "pending": len(build.pending),
        "withheld_policies": len(build.withheld_policies),
        "unresolved_rules": len(build.packet["unresolved_rules"]),
        "unresolved_conflicts": len(build.packet["unresolved_conflicts"]),
    }


def _rule_stages(config: ResolvedConfig, digest: str, fingerprint: str) -> dict[str, int]:
    """Extract, normalize, and check rules, then record the three stages in the cache index."""
    run_dir = config.run_dir
    index_path = run_dir / "stage_cache" / "cache_index.json"
    index = _read_json(index_path, run_dir)
    if index is None or not _valid_cache_index(index):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "stage_cache/cache_index.json is missing or damaged; run stage=prepare.",
        )
    bound = {"config_sha256": digest, "source_fingerprint": fingerprint}
    stage_cache = run_dir / "stage_cache"
    # The plan and its cases name rules from the previous result. Drop both before the index changes.
    for name in (*COVERAGE_OUTPUTS, *SCENARIO_OUTPUTS, *DATASET_OUTPUTS):
        (stage_cache / name).unlink(missing_ok=True)
    index["stages"].pop("coverage_plan", None)
    index["stages"].pop("scenarios", None)
    index["stages"].pop("dataset", None)
    # Rule outputs are replaced from here on. Until they are written, the index must not say completed.
    for stage in RULE_STAGES:
        index["stages"][stage] = {"status": "running", **bound, "started_at": _now()}
    _write_json(index_path, index)
    try:
        result = run_rule_stages(config, run_dir)
    except Exception as exc:
        outcome = exc.status.value if isinstance(exc, SafetyStatus) else type(exc).__name__
        for stage in RULE_STAGES:
            index["stages"][stage] = {"status": "failed", "outcome": outcome, **bound, "failed_at": _now()}
        _write_json(index_path, index)
        raise
    now = _now()
    common = {"status": "completed", **bound, "completed_at": now}
    index["stages"]["extract"] = {
        **common,
        "prompt_version": PROMPT_VERSION,
        "model": result.extraction.get("model"),
        "passes": result.extraction.get("passes", []),
    }
    index["stages"]["normalize"] = {**common, "counts": result.counts}
    index["stages"]["conflicts"] = {**common, "unresolved_conflicts": result.counts["unresolved_conflicts"]}
    _write_json(index_path, index)
    return result.counts


def _coverage_stage(config: ResolvedConfig, digest: str, fingerprint: str) -> dict[str, Any]:
    """Write the coverage plan from the eligible rules. This stage does not call a model."""
    run_dir = config.run_dir
    stage_cache = run_dir / "stage_cache"
    index_path = stage_cache / "cache_index.json"
    index = _read_json(index_path, run_dir)
    if index is None or not _valid_cache_index(index):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "stage_cache/cache_index.json is missing or damaged; run stage=prepare.",
        )
    bound = {"config_sha256": digest, "source_fingerprint": fingerprint}
    index["stages"]["coverage_plan"] = {"status": "running", **bound, "started_at": _now()}
    _write_json(index_path, index)
    try:
        rules = load_eligible_rules(yaml.safe_load((stage_cache / "policy_rules_eligible.yaml").read_text("utf-8")))
        routed = load_routed_statuses((stage_cache / POLICY_RULES_FILE).read_text(encoding="utf-8"))
        spans = span_text(json.loads((stage_cache / "source_index.json").read_text(encoding="utf-8")))
        plan = build_coverage_plan(
            rules,
            spans,
            case_cap=case_cap_for(config.mode, config.budget),
            budget=config.budget,
            modules=config.generation["modules"],
            generation=config.generation,
            routed=routed,
            target_languages=config.proposals.get("target_languages", []),
        )
        summary = write_coverage_plan(stage_cache, plan)
    except Exception as exc:
        for name in COVERAGE_OUTPUTS:
            (stage_cache / name).unlink(missing_ok=True)
        outcome = exc.status.value if isinstance(exc, SafetyStatus) else type(exc).__name__
        index["stages"]["coverage_plan"] = {"status": "failed", "outcome": outcome, **bound, "failed_at": _now()}
        _write_json(index_path, index)
        if isinstance(exc, SafetyStatus):
            raise
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"coverage plan cannot be built: {exc}") from exc
    index["stages"]["coverage_plan"] = {
        "status": "completed",
        **bound,
        "completed_at": _now(),
        "certification": summary["certification"],
        "scheduled_cases": summary["scheduled_cases"],
        "required_cases": summary["required_cases"],
    }
    _write_json(index_path, index)
    return summary


def _scenario_stage(config: ResolvedConfig, digest: str, fingerprint: str) -> dict[str, Any]:
    """Write one seed per scheduled plan cell. This stage does not call a model."""
    run_dir = config.run_dir
    stage_cache = run_dir / "stage_cache"
    index_path = stage_cache / "cache_index.json"
    index = _read_json(index_path, run_dir)
    if index is None or not _valid_cache_index(index):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "stage_cache/cache_index.json is missing or damaged; run stage=prepare.",
        )
    bound = {"config_sha256": digest, "source_fingerprint": fingerprint}
    index["stages"]["scenarios"] = {"status": "running", **bound, "started_at": _now()}
    _write_json(index_path, index)
    try:
        rules = load_eligible_rules(yaml.safe_load((stage_cache / "policy_rules_eligible.yaml").read_text("utf-8")))
        rows = load_plan_rows((stage_cache / PLAN_FILE).read_text(encoding="utf-8"))
        gaps = json.loads((stage_cache / GAPS_FILE).read_text(encoding="utf-8"))
        summary = write_scenarios(
            stage_cache,
            build_scenarios(
                rows,
                load_contrast_families(gaps),
                rules,
                unscheduled=load_unscheduled_slots(gaps),
            ),
        )
    except Exception as exc:
        for name in SCENARIO_OUTPUTS:
            (stage_cache / name).unlink(missing_ok=True)
        outcome = exc.status.value if isinstance(exc, SafetyStatus) else type(exc).__name__
        index["stages"]["scenarios"] = {"status": "failed", "outcome": outcome, **bound, "failed_at": _now()}
        _write_json(index_path, index)
        if isinstance(exc, SafetyStatus):
            raise
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"scenarios cannot be built: {exc}") from exc
    index["stages"]["scenarios"] = {
        "status": "completed",
        **bound,
        "completed_at": _now(),
        "accepted": summary["accepted"],
        "rejected": summary["rejected"],
    }
    _write_json(index_path, index)
    return summary


def _dataset_stage(config: ResolvedConfig, digest: str, fingerprint: str) -> dict[str, Any]:
    """Check labels, drop exact duplicates, and split the cases. This stage does not call a model."""
    run_dir = config.run_dir
    stage_cache = run_dir / "stage_cache"
    index_path = stage_cache / "cache_index.json"
    index = _read_json(index_path, run_dir)
    if index is None or not _valid_cache_index(index):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "stage_cache/cache_index.json is missing or damaged; run stage=prepare.",
        )
    bound = {"config_sha256": digest, "source_fingerprint": fingerprint}
    index["stages"]["dataset"] = {"status": "running", **bound, "started_at": _now()}
    _write_json(index_path, index)
    try:
        rules = load_eligible_rules(yaml.safe_load((stage_cache / "policy_rules_eligible.yaml").read_text("utf-8")))
        spans = span_text(json.loads((stage_cache / "source_index.json").read_text(encoding="utf-8")))
        cases = load_jsonl_objects((stage_cache / CASES_FILE).read_text(encoding="utf-8"), CASES_FILE)
        families = load_jsonl_objects((stage_cache / CONTRAST_FILE).read_text(encoding="utf-8"), CONTRAST_FILE)
        summary = write_dataset(
            stage_cache,
            select_dataset(
                cases,
                families,
                rules,
                spans,
                quality_gates=config.extras.get("quality_gates"),
            ),
        )
    except Exception as exc:
        for name in DATASET_OUTPUTS:
            (stage_cache / name).unlink(missing_ok=True)
        outcome = exc.status.value if isinstance(exc, SafetyStatus) else type(exc).__name__
        index["stages"]["dataset"] = {"status": "failed", "outcome": outcome, **bound, "failed_at": _now()}
        _write_json(index_path, index)
        if isinstance(exc, SafetyStatus):
            raise
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"dataset cannot be built: {exc}") from exc
    index["stages"]["dataset"] = {
        "status": "completed",
        **bound,
        "completed_at": _now(),
        "validated": summary["validated"],
        "rejected": summary["rejected"],
        "duplicates_removed": summary["duplicates_removed"],
        "split_complete": summary["split_complete"],
        "quality_gates_passed": summary["quality_gates_passed"],
    }
    _write_json(index_path, index)
    return summary


def _backfill_quality_gates(
    config: ResolvedConfig,
    digest: str,
    fingerprint: str,
    coverage: dict[str, Any],
    scenarios: dict[str, Any],
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Promote unscheduled distinct cells into room left by rejected or duplicate rows.

    Quote-copies are skipped. Held translation, missing-span, and max-turn cells stay
    unscheduled, and so do cells of a policy whose cases fail because of the rule itself.
    Required cells already on the plan are not dropped. A round that does not lower the
    total gate shortfall is rolled back, because a completed contrast family or a new
    split can remove rows that already passed.
    """
    if dataset.get("quality_gates_passed") is True:
        return coverage, scenarios, dataset
    if not isinstance(config.extras.get("quality_gates"), Mapping):
        return coverage, scenarios, dataset
    run_dir = config.run_dir
    stage_cache = run_dir / "stage_cache"
    case_cap = case_cap_for(config.mode, config.budget)
    for _ in range(MAX_BACKFILL_ROUNDS):
        if dataset.get("quality_gates_passed") is True:
            break
        try:
            plan_text = (stage_cache / PLAN_FILE).read_text(encoding="utf-8")
            gaps_text = (stage_cache / GAPS_FILE).read_text(encoding="utf-8")
            rows = load_plan_rows(plan_text)
            gaps_doc = json.loads(gaps_text)
            surviving = load_jsonl_objects(
                (stage_cache / "deduplicated_cases.jsonl").read_text(encoding="utf-8"),
                "deduplicated_cases.jsonl",
            )
            rejected = load_jsonl_objects(
                (stage_cache / "rejected_cases.jsonl").read_text(encoding="utf-8"),
                "rejected_cases.jsonl",
            )
            split_manifest = json.loads((stage_cache / "split_manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"quality-gate backfill cannot read artifacts: {exc}",
            ) from exc
        if not isinstance(gaps_doc, dict) or not isinstance(split_manifest, dict):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                "quality-gate backfill needs coverage_gaps.json and split_manifest.json objects.",
            )
        gate_result = split_manifest.get("quality_gates") or {}
        added = select_backfill_cells(
            gaps_doc.get("gaps") or [],
            rows,
            surviving,
            case_cap=case_cap,
            budget=config.budget,
            gate_result=gate_result,
            blocked_policies=rule_level_rejected_policies(rejected),
        )
        if not added:
            break
        before = quality_gate_shortfall(gate_result)
        previous_coverage = coverage
        new_rows, updated = apply_backfill(rows, gaps_doc, added)
        coverage = rewrite_coverage_plan(stage_cache, new_rows, updated)
        _record_coverage_backfill(stage_cache, run_dir, digest, fingerprint, coverage)
        scenarios = _scenario_stage(config, digest, fingerprint)
        dataset = _dataset_stage(config, digest, fingerprint)
        if _current_shortfall(stage_cache, run_dir) < before:
            continue
        (stage_cache / PLAN_FILE).write_text(plan_text, encoding="utf-8")
        (stage_cache / GAPS_FILE).write_text(gaps_text, encoding="utf-8")
        coverage = previous_coverage
        _record_coverage_backfill(stage_cache, run_dir, digest, fingerprint, coverage)
        scenarios = _scenario_stage(config, digest, fingerprint)
        dataset = _dataset_stage(config, digest, fingerprint)
        break
    return coverage, scenarios, dataset


def _current_shortfall(stage_cache: Path, run_dir: Path) -> int:
    try:
        split_manifest = json.loads((stage_cache / "split_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json cannot be read.") from exc
    gates = split_manifest.get("quality_gates") if isinstance(split_manifest, dict) else None
    if not isinstance(gates, Mapping):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json has damaged quality_gates.")
    return quality_gate_shortfall(gates)


def _record_coverage_backfill(
    stage_cache: Path,
    run_dir: Path,
    digest: str,
    fingerprint: str,
    coverage: Mapping[str, Any],
) -> None:
    index_path = stage_cache / "cache_index.json"
    index = _read_json(index_path, run_dir)
    if index is None or not _valid_cache_index(index):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "stage_cache/cache_index.json is missing or damaged; run stage=prepare.",
        )
    entry = index["stages"].get("coverage_plan")
    if not isinstance(entry, dict):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "coverage_plan is missing from the cache index after backfill.",
        )
    entry["certification"] = coverage.get("certification")
    entry["scheduled_cases"] = coverage.get("scheduled_cases")
    entry["required_cases"] = coverage.get("required_cases")
    entry["config_sha256"] = digest
    entry["source_fingerprint"] = fingerprint
    backfill = coverage.get("backfill")
    if isinstance(backfill, Mapping):
        entry["backfill_rounds"] = backfill.get("rounds")
        entry["backfill_added"] = len(backfill.get("added_slot_ids") or [])
    else:
        entry.pop("backfill_rounds", None)
        entry.pop("backfill_added", None)
    index["stages"]["coverage_plan"] = entry
    _write_json(index_path, index)


def _backfill_note(coverage: Mapping[str, Any]) -> str:
    backfill = coverage.get("backfill")
    if not isinstance(backfill, Mapping):
        return ""
    added = [slot for slot in backfill.get("added_slot_ids") or [] if isinstance(slot, str)]
    if not added:
        return ""
    rounds = backfill.get("rounds")
    round_label = rounds if isinstance(rounds, int) and not isinstance(rounds, bool) else 1
    return f" Backfill added {len(added)} distinct cells in {round_label} round(s)."


def _quality_gates_block_publish(config: ResolvedConfig, run_dir: Path) -> bool:
    if config.mode == "preview":
        return False
    path = run_dir / QUALITY_FILE
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{QUALITY_FILE} cannot be read after compile.") from exc
    if not isinstance(report, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{QUALITY_FILE} is damaged after compile.")
    gates = report.get("quality_gates")
    if not isinstance(gates, Mapping):
        return False
    return gates.get("passed") is False and gates.get("on_failure") == "fail"


def _check_rule_inputs(record: dict[str, Any], run_dir: Path) -> None:
    """Refuse ingest outputs that differ from the ones prepare committed, before any model call."""
    recorded = record.get("ingest_artifacts")
    if not isinstance(recorded, dict):
        raise abort(
            RunStatus.PREPARE_REQUIRED,
            run_dir,
            "run_record.json does not record the ingest outputs; it predates this check.",
        )
    try:
        current = rule_input_digests(run_dir / "stage_cache")
    except OSError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"stage_cache cannot be read: {exc}.") from exc
    changed = sorted(name for name, digest in current.items() if recorded.get(name) != digest)
    if changed:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"stage_cache/{', stage_cache/'.join(changed)} changed after prepare; run stage=prepare to rebuild it.",
        )


def _rule_summary(counts: dict[str, int]) -> str:
    return (
        f"Rules: {counts['eligible_for_generation']} eligible, {counts['needs_decision']} need a decision, "
        f"{counts['quarantined']} quarantined, {counts['rejected']} rejected; "
        f"{counts['unresolved_conflicts']} unresolved conflicts."
    )


_OLDER_PACKET = f"{PACKET_NAME} belongs to an older config or older policy files and was ignored."


def sign_safety_packet(
    config_path: ConfigPath, *, approved_by: str, role: str | None = None, note: str | None = None
) -> dict[str, Any]:
    """Record one approver's signature on the current review packet and report the roles still missing.

    The signature binds the config, the policy files, and the exact packet bytes. Each role holds
    one signature, so signing a role again replaces it, and signing a newer packet drops the
    signatures given to an older one. Without ``role``, the approver keeps the role they already
    hold, or takes the only role still missing.
    """
    name = approved_by.strip() if isinstance(approved_by, str) else ""
    problems = []
    if not name:
        problems.append("approved_by must name the person who reviewed the packet.")
    if role is not None and role not in SIGNER_ROLES:
        problems.append(f"role must be one of {', '.join(SIGNER_ROLES)}; received {role!r}.")
    if note is not None and not isinstance(note, str):
        problems.append("note must be text.")
    if problems:
        raise _sign_error(*problems)
    resolved = _load(config_path)
    run_dir = resolved.run_dir
    digest = _digest(resolved)
    if not run_dir.is_dir():
        raise abort(RunStatus.PREPARE_REQUIRED, run_dir)
    with _run_lock(run_dir):
        fingerprint, _ = _sources(resolved)
        record = _read_json(run_dir / "run_record.json", run_dir)
        if record is None or not _same_inputs(record, digest, fingerprint):
            raise abort(RunStatus.PREPARE_REQUIRED, run_dir, _changed_inputs(record, digest, fingerprint))
        if record.get("publication_commit") is True:
            assert_published_intact(run_dir, record)
            raise _sign_error("This benchmark is already published. Run stage=prepare before starting a new review.")
        packet_path = run_dir / PACKET_NAME
        packet = _read_packet(packet_path, run_dir)
        if packet is None:
            raise abort(RunStatus.PIPELINE_INCOMPLETE, run_dir, f"There is no {PACKET_NAME} to sign yet.")
        if not _bound_to(packet, digest, fingerprint):
            raise abort(
                RunStatus.PIPELINE_INCOMPLETE, run_dir, f"{_OLDER_PACKET} Run stage=generate to rebuild it, then sign."
            )
        packet_sha256 = _file_sha256(packet_path, run_dir)
        signoff_path = run_dir / SIGNOFF_NAME
        try:
            previous = _read_json(signoff_path, run_dir)
        except SafetyStatus:
            previous = {}  # a damaged signoff is replaced, not trusted
        same_packet = bool(previous) and _signs_packet(previous, digest, fingerprint, packet_sha256)
        signed = _signed_roles(previous) if same_packet else {}
        required = _required_roles(resolved, packet, run_dir)
        held = [held_role for held_role, item in signed.items() if _same_person(item["approved_by"], name)]
        if role is None:
            role = _inferred_role(held, required, signed)
        profile = resolved.assurance["profile"]
        if profile in DISTINCT_SIGNERS_PROFILES and any(held_role != role for held_role in held):
            raise _sign_error(f"{name} already signed as {held[0]}; {profile} needs a different person for each role.")
        signature: dict[str, Any] = {"approved_by": name, "role": role, "approved_at": _now()}
        if note:
            signature["note"] = note
        signed[role] = signature
        signoff = {
            "signoff_version": SIGNOFF_VERSION,
            "config_sha256": digest,
            "source_fingerprint": fingerprint,
            "packet_sha256": packet_sha256,
            "signatures": list(signed.values()),
        }
        _write_json(signoff_path, signoff)
    return {
        "approved_by": name,
        "role": role,
        "missing_roles": [missing for missing in required if missing not in signed],
        "replaced_older_signoff": bool(previous) and not same_packet,
        "signoff": str(signoff_path),
    }


def _sign_error(*problems: str) -> SafetyStatus:
    return SafetyStatus(RunStatus.INVALID_CONFIG, "Invalid sign-off:\n- " + "\n- ".join(problems))


def _same_person(first: str, second: str) -> bool:
    return first.strip().casefold() == second.strip().casefold()


def _inferred_role(held: list[str], required: tuple[str, ...], signed: dict[str, Any]) -> str:
    if held:
        return held[0]
    missing = [role for role in required if role not in signed]
    if len(missing) == 1:
        return missing[0]
    if not missing:
        raise _sign_error("Every required role has already signed this packet. Pass --role to replace one.")
    raise _sign_error(f"Several roles still need to sign ({', '.join(missing)}). Pass --role with yours.")


def _read_packet(path: Path, run_dir: Path) -> dict | None:
    packet = _read_yaml(path, run_dir)
    if packet is not None and packet.get("packet_version") != REVIEW_PACKET_VERSION:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{PACKET_NAME} must have packet_version {REVIEW_PACKET_VERSION}."
        )
    return packet


def _signs_packet(signoff: dict, digest: str, fingerprint: str, packet_sha256: str) -> bool:
    return (
        signoff.get("signoff_version") == SIGNOFF_VERSION
        and _bound_to(signoff, digest, fingerprint)
        and signoff.get("packet_sha256") == packet_sha256
    )


def _signed_roles(signoff: dict) -> dict[str, dict[str, Any]]:
    """The valid signature held by each role. A later entry for the same role wins."""
    items = signoff.get("signatures")
    if not isinstance(items, list):
        return {}
    return {
        item["role"]: item
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("approved_by"), str)
        and item["approved_by"].strip()
        and item.get("role") in SIGNER_ROLES
    }


def _required_roles(config: ResolvedConfig, packet: dict, run_dir: Path) -> tuple[str, ...]:
    roles = packet.get("required_roles")
    if roles is None:
        return PROFILE_ROLES[config.assurance["profile"]]
    if (
        not isinstance(roles, list)
        or not roles
        or len(set(map(str, roles))) != len(roles)
        or any(role not in SIGNER_ROLES for role in roles)
    ):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{PACKET_NAME} required_roles must list distinct roles from: {', '.join(SIGNER_ROLES)}.",
        )
    return tuple(roles)


def _signoff_gaps(signoff: dict, config: ResolvedConfig, packet: dict, run_dir: Path) -> str:
    """What still blocks publication on a signoff that matches the packet; empty when nothing does."""
    required = _required_roles(config, packet, run_dir)
    signed = _signed_roles(signoff)
    missing = [role for role in required if role not in signed]
    if missing:
        return f"Still waiting for: {', '.join(missing)}. That approver runs: {APPROVE_COMMAND}"
    profile = config.assurance["profile"]
    people = {signed[role]["approved_by"].strip().casefold() for role in required}
    if profile in DISTINCT_SIGNERS_PROFILES and len(people) < len(required):
        return f"The {profile} profile needs a different person for each of: {', '.join(required)}."
    return ""


def _source_blocker(
    config: ResolvedConfig, digest: str, fingerprint: str, sources: dict[str, str]
) -> RunStatus | None:
    """The status to stop with when the quarantine report leaves no usable policy document.

    The report must describe every current source with its sha256 and a status of safe,
    quarantined, or rejected. A missing report, or one that misses a source, names an
    unknown one, or carries an older hash, is stale: generate refuses to continue rather
    than treat the gap as safe. With no safe document left, any quarantined file makes
    this a privacy stop; otherwise every file was rejected.
    """
    run_dir = config.run_dir
    report = run_dir / "quarantine" / "files.jsonl"
    if not report.is_file():
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "quarantine/files.jsonl is missing, so the privacy check cannot run. Run stage=prepare.",
        )
    file_rows: dict[str, dict] = {}
    blocked = False
    for line_number, row in _jsonl_rows(report, run_dir):
        if "run_blocked" in row:
            if row.get("run_blocked") is True and _bound_to(row, digest, fingerprint):
                blocked = True
            continue
        path = canonical_source_path(row.get("path"))
        where = f"quarantine/files.jsonl line {line_number}"
        if path is None:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{where} must name a relative POSIX path with no '.', '..', or backslash segments.",
            )
        if row.get("status") not in QUARANTINE_STATUSES or not isinstance(row.get("sha256"), str):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{where} needs status safe, quarantined, or rejected and a sha256.",
            )
        if path in file_rows:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{where} repeats {path}.")
        file_rows[path] = row
    if blocked:
        return RunStatus.PRIVACY_BLOCKED
    current = {path: row["sha256"] for path, row in file_rows.items()}
    if current != sources:
        missing = sorted(set(sources) - set(current))
        unknown = sorted(set(current) - set(sources))
        changed = sorted(path for path in set(sources) & set(current) if sources[path] != current[path])
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "quarantine/files.jsonl does not match the current policy files, so the privacy check cannot "
            f"trust it. Missing: {missing or 'none'}. Unknown: {unknown or 'none'}. Changed: {changed or 'none'}. "
            "Run stage=prepare.",
        )
    statuses = {row["status"] for row in file_rows.values()}
    if SAFE in statuses:
        return None
    return RunStatus.PRIVACY_BLOCKED if QUARANTINED in statuses else RunStatus.NO_USABLE_SOURCES


def _refresh_cache_index(index: dict | None, digest: str, fingerprint: str) -> dict[str, Any]:
    """Keep earlier stage entries. Changed inputs mark them stale instead of deleting them."""
    if index is None:
        return {
            "stage_cache_version": CACHE_INDEX_VERSION,
            "config_sha256": digest,
            "source_fingerprint": fingerprint,
            "stages": {},
        }
    reasons = []
    if index.get("config_sha256") != digest:
        reasons.append("config_changed")
    if index.get("source_fingerprint") != fingerprint:
        reasons.append("sources_changed")
    for entry in index["stages"].values():
        if reasons:
            entry["stale"] = True
            entry["stale_reasons"] = sorted(set(entry.get("stale_reasons", [])) | set(reasons))
    index["config_sha256"] = digest
    index["source_fingerprint"] = fingerprint
    return index


def _valid_cache_index(index: dict) -> bool:
    stages = index.get("stages")
    return (
        index.get("stage_cache_version") == CACHE_INDEX_VERSION
        and isinstance(stages, dict)
        and all(isinstance(entry, dict) for entry in stages.values())
    )


def _valid_record(record: dict) -> bool:
    return record.get("record_type") == "safety_run_record" and isinstance(record.get("config_sha256"), str)


def _same_inputs(record: dict | None, digest: str, fingerprint: str) -> bool:
    return bool(record) and _bound_to(record, digest, fingerprint)


def _bound_to(document: dict, digest: str, fingerprint: str) -> bool:
    return document.get("config_sha256") == digest and document.get("source_fingerprint") == fingerprint


def _changed_inputs(record: dict | None, digest: str, fingerprint: str) -> str:
    if record is None:
        return ""
    changed = []
    if record.get("config_sha256") != digest:
        changed.append("the config")
    if record.get("source_fingerprint") != fingerprint:
        changed.append("the policy files")
    return f"Changed since prepare: {' and '.join(changed)}." if changed else ""


def _sources(config: ResolvedConfig) -> tuple[str, dict[str, str]]:
    try:
        sources = scan_sources(config.input_dir)
    except SourceError as exc:
        raise SafetyStatus(RunStatus.INVALID_CONFIG, f"Invalid configuration:\n- {exc}") from exc
    return source_fingerprint(sources), sources


def _unavailable(config_path: ConfigPath) -> SafetyStatus:
    resolved = _load(config_path)
    return abort(RunStatus.STAGE_UNAVAILABLE, resolved.run_dir)


def _load(config_path: ConfigPath) -> ResolvedConfig:
    try:
        return load_resolved(Path(config_path))
    except ConfigError as exc:
        raise SafetyStatus(
            RunStatus.INVALID_CONFIG,
            "Invalid configuration:\n" + "\n".join(f"- {item}" for item in exc.messages),
        ) from exc


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """Prevent two writers from interleaving one safety run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".safety.lock"
    handle = lock_path.open("a+b")
    unlock: Callable[[], None] | None = None
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if lock_path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = partial(msvcrt.locking, handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                unlock = partial(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise abort(RunStatus.RUN_LOCKED, run_dir) from exc
        yield
    finally:
        try:
            if unlock is not None:
                unlock()
        finally:
            handle.close()


def _digest(config: ResolvedConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _recover_json(path: Path, run_dir: Path, valid: Callable[[dict], bool], recovered: list[str]) -> dict | None:
    """Read a prepare-owned file. A damaged copy is moved aside so prepare can rebuild it."""
    try:
        data = _read_json(path, run_dir)
    except SafetyStatus:
        data = None
        damaged = True
    else:
        damaged = data is not None and not valid(data)
    if not damaged:
        return data
    target = run_dir / RECOVERED_DIR / f"{path.name}.{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.damaged"
    target.parent.mkdir(exist_ok=True)
    os.replace(path, target)
    recovered.append(target.relative_to(run_dir).as_posix())
    return None


def _read_text(path: Path, run_dir: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read as UTF-8 text.") from exc


def _read_json(path: Path, run_dir: Path) -> dict | None:
    text = _read_text(path, run_dir)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} must contain a JSON object.")
    return data


def _read_yaml(path: Path, run_dir: Path) -> dict | None:
    text = _read_text(path, run_dir)
    if text is None:
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is not valid YAML.") from exc
    if not isinstance(data, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} must contain a YAML mapping.")
    return data


def _jsonl_rows(path: Path, run_dir: Path) -> Iterator[tuple[int, dict]]:
    text = _read_text(path, run_dir) or ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{path.parent.name}/{path.name} line {line_number} is not valid JSON.",
            ) from exc
        if not isinstance(row, dict):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{path.parent.name}/{path.name} line {line_number} must be an object.",
            )
        yield line_number, row


def _file_sha256(path: Path, run_dir: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read.") from exc
