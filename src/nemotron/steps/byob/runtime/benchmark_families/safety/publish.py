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

"""Compile a signed safety benchmark and refuse a file that changed after signing.

The benchmark is JSONL. ``run_manifest.json`` is the publication commit and is
written by the caller after every other publication file, including the run record.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_json, write_jsonl
from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import (
    DatasetInputError,
    evaluate_quality_gates,
    load_jsonl_objects,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.review import (
    PACKET_NAME,
    REVIEW_PAGE,
    card_is_pending,
    content_matches,
    withheld_audit_case_ids,
    withheld_case_ids,
    withheld_policies,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, abort

MANIFEST_NAME = "run_manifest.json"
BENCHMARK_FILE = "benchmark.jsonl"
DEVELOPMENT_FILE = "benchmark_development.jsonl"
QUALITY_FILE = "quality_report.json"
PUBLISHED_FILES = (BENCHMARK_FILE, DEVELOPMENT_FILE, QUALITY_FILE)
CLAIM_FIELDS = ("certification", "reason", "cases", "development", "published_at")
_REMOVED_PUBLISHED_FILES = ("benchmark_private_test.jsonl",)
_MANIFEST_VERSION = 2
_CASES_FILE = "stage_cache/deduplicated_cases.jsonl"
_FAMILIES_FILE = "stage_cache/contrast_families.jsonl"
_SPLIT_FILE = "stage_cache/split_manifest.json"
_TIERS = {
    "gold": RunStatus.GOLD_COMPLETED,
    "silver": RunStatus.SILVER_COMPLETED,
    "preview": RunStatus.PREVIEW_COMPLETED,
}


def publication_tier(
    *,
    mode: str,
    coverage_certification: str,
    coverage_reason: str,
    split_reason: str,
    development: int,
    withheld: bool,
    blocking: bool,
    review_limitations: bool = False,
    quality_gates_passed: bool = True,
    unrealized_surfaces: bool = False,
) -> tuple[str, str]:
    """Gold needs a signed benchmark with development cases and no blocking gaps.

    A required allow cell whose surface is still the source quote is not an
    over-refusal probe, so the benchmark cannot claim Gold without it.

    Preview stays Preview even when coverage is complete. It is not a comparable score.
    """
    if mode == "preview":
        return "preview", "preview"
    if withheld:
        return "silver", "review_removed_cases"
    if review_limitations:
        return "silver", "review_unresolved_items"
    if coverage_certification != "gold" or blocking:
        return "silver", coverage_reason or "required_omitted"
    if unrealized_surfaces:
        return "silver", "allow_surface_unrealized"
    if development == 0:
        return "silver", split_reason or "split_incomplete"
    if not quality_gates_passed:
        return "silver", "minimum_dataset_quality_not_met"
    return "gold", "complete"


def status_for(tier: str) -> RunStatus:
    return _TIERS[tier]


def publication_claim(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The fields ``run_record.json`` stores so a later generate can detect a rewritten manifest."""
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    return {
        **{key: manifest.get(key) for key in CLAIM_FIELDS},
        "packet_sha256": manifest.get("packet_sha256"),
        "files": {str(name): digest for name, digest in files.items() if isinstance(digest, str)},
    }


def compile_publication(
    run_dir: Path,
    packet: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    withheld_policies: frozenset[str],
    mode: str,
    published_at: str,
    config_sha256: str,
    source_fingerprint: str,
    packet_sha256: str,
) -> dict[str, Any]:
    """Write the benchmark files and the quality report. The caller writes the manifest last."""
    mismatched = content_matches(run_dir, packet)
    if mismatched:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{mismatched} changed while the review packet was written.",
        )
    _, _, grouped, quality, tier, _reason = _publication_payload(
        run_dir,
        packet,
        cases=cases,
        families=families,
        split_manifest=split_manifest,
        dropped_policies=withheld_policies,
        mode=mode,
        published_at=published_at,
        config_sha256=config_sha256,
        source_fingerprint=source_fingerprint,
        packet_sha256=packet_sha256,
    )
    for name, group in grouped.items():
        write_jsonl(run_dir / name, group)
    write_json(run_dir / QUALITY_FILE, quality)
    files = {name: _sha256(run_dir / name) for name in PUBLISHED_FILES}
    return {
        "manifest_version": _MANIFEST_VERSION,
        "publication_commit": True,
        "certification": tier,
        "reason": quality["reason"],
        "cases": quality["cases"],
        "development": quality["development"],
        "published_at": published_at,
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "packet_sha256": packet_sha256,
        "decisions_sha256": _sha256(run_dir / "review_decisions.csv"),
        "review_html_sha256": _sha256(run_dir / REVIEW_PAGE),
        "signoff_sha256": (
            _sha256(run_dir / "publication_signoff.json")
            if (run_dir / "publication_signoff.json").is_file()
            else None
        ),
        "files": files,
    }


def assert_published_intact(run_dir: Path, record: Mapping[str, Any]) -> None:
    """Refuse generate when a published file no longer matches the signed packet."""
    if record.get("publication_commit") is not True:
        return
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is missing after publication.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is damaged after publication.") from exc
    if not isinstance(manifest, dict) or manifest.get("publication_commit") is not True:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is damaged after publication.")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PUBLISHED_FILES):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{MANIFEST_NAME} files must list {', '.join(PUBLISHED_FILES)}.",
        )
    recorded = record.get("publication")
    if isinstance(recorded, Mapping) and recorded and publication_claim(manifest) != recorded:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} does not match run_record.json.")
    for name, key in (
        (PACKET_NAME, "packet_sha256"),
        ("review_decisions.csv", "decisions_sha256"),
        (REVIEW_PAGE, "review_html_sha256"),
        ("publication_signoff.json", "signoff_sha256"),
    ):
        expected = manifest.get(key)
        if name == "publication_signoff.json" and expected is None:
            if (run_dir / name).exists():
                raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{name} changed after publication.")
            continue
        if not isinstance(expected, str):
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is damaged after publication.")
        _match(run_dir, name, expected)
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str):
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{MANIFEST_NAME} is damaged after publication.")
        _match(run_dir, name, digest)
    _assert_tier_matches_packet(run_dir, manifest)


def clear_published(run_dir: Path) -> None:
    """Remove a benchmark that this run is not publishing."""
    for name in (*PUBLISHED_FILES, *_REMOVED_PUBLISHED_FILES, MANIFEST_NAME):
        (run_dir / name).unlink(missing_ok=True)


def _assert_tier_matches_packet(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Recompute Gold/Silver from the signed packet so a rewritten manifest cannot upgrade it."""
    packet = _load_packet(run_dir)
    mismatched = content_matches(run_dir, packet)
    if mismatched:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{mismatched} changed after it was signed.")
    try:
        cases = load_jsonl_objects((run_dir / _CASES_FILE).read_text(encoding="utf-8"), Path(_CASES_FILE).name)
        families = load_jsonl_objects(
            (run_dir / _FAMILIES_FILE).read_text(encoding="utf-8"), Path(_FAMILIES_FILE).name
        )
        split_manifest = json.loads((run_dir / _SPLIT_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DatasetInputError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "signed publication inputs cannot be read.") from exc
    if not isinstance(split_manifest, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json must contain an object.")
    cards = packet.get("cards") if isinstance(packet.get("cards"), list) else []
    _rows, _dropped, grouped, quality, tier, reason = _publication_payload(
        run_dir,
        packet,
        cases=cases,
        families=families,
        split_manifest=split_manifest,
        dropped_policies=withheld_policies(cards),
        mode=str(packet.get("mode") or ""),
        published_at=str(manifest.get("published_at") or ""),
        config_sha256=str(manifest.get("config_sha256") or ""),
        source_fingerprint=str(manifest.get("source_fingerprint") or ""),
        packet_sha256=str(manifest.get("packet_sha256") or ""),
    )
    if (tier, reason) != (manifest.get("certification"), manifest.get("reason")):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{MANIFEST_NAME} certification does not match the signed review packet.",
        )
    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    expected_files = {
        BENCHMARK_FILE: hashlib.sha256(_jsonl_text(grouped[BENCHMARK_FILE]).encode("utf-8")).hexdigest(),
        DEVELOPMENT_FILE: hashlib.sha256(_jsonl_text(grouped[DEVELOPMENT_FILE]).encode("utf-8")).hexdigest(),
    }
    for name, digest in expected_files.items():
        if files.get(name) != digest:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{name} does not match the signed review packet.",
            )
    try:
        on_disk = json.loads((run_dir / QUALITY_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{QUALITY_FILE} is damaged after publication.") from exc
    for key in ("certification", "reason", "cases", "development", "published_at"):
        if not isinstance(on_disk, dict) or on_disk.get(key) != quality.get(key):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{QUALITY_FILE} does not match the signed review packet.",
            )
    counts = {key: quality[key] for key in ("cases", "development")}
    if any(manifest.get(key) != value for key, value in counts.items()):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{MANIFEST_NAME} case counts do not match the signed review packet.",
        )


def _publication_payload(
    run_dir: Path,
    packet: Mapping[str, Any],
    *,
    cases: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    dropped_policies: frozenset[str],
    mode: str,
    published_at: str,
    config_sha256: str,
    source_fingerprint: str,
    packet_sha256: str,
) -> tuple[list[dict[str, Any]], set[str], dict[str, list[dict[str, Any]]], dict[str, Any], str, str]:
    dropped = withheld_case_ids(cases, families, dropped_policies) | withheld_audit_case_ids(packet)
    development = _split_ids(split_manifest, run_dir)
    indexed: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "a deduplicated case has no case_id.")
        if case_id in indexed:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"deduplicated cases repeat {case_id}.")
        indexed[case_id] = case
    unknown_split_ids = sorted(development - set(indexed))
    if unknown_split_ids:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"split_manifest.json names unknown cases: {', '.join(unknown_split_ids)}.",
        )
    unassigned = sorted(set(indexed) - development - dropped)
    if unassigned:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"deduplicated cases are missing from the split manifest: {', '.join(unassigned)}.",
        )
    rows = []
    for case in sorted(cases, key=lambda item: str(item.get("case_id"))):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or case_id in dropped:
            continue
        if case_id not in development:
            continue
        rows.append({**case, "split": "development"})
    _tag_contrast_families(rows, families)
    stored_gates = split_manifest.get("quality_gates")
    if isinstance(stored_gates, Mapping):
        gate_requirements = stored_gates.get("requirements")
        evaluated_policies = stored_gates.get("evaluated_policy_ids")
        if not isinstance(gate_requirements, Mapping) or not isinstance(evaluated_policies, list):
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                "split_manifest.json has damaged quality_gates.",
            )
        retained_policies = [
            policy_id
            for policy_id in evaluated_policies
            if isinstance(policy_id, str) and policy_id not in dropped_policies
        ]
        quality_gates = evaluate_quality_gates(rows, gate_requirements, policy_ids=retained_policies)
    else:
        quality_gates = evaluate_quality_gates(rows, {})
    coverage = packet.get("coverage") if isinstance(packet.get("coverage"), Mapping) else {}
    surfaces = split_manifest.get("unrealized_surfaces", [])
    if not isinstance(surfaces, list) or any(not isinstance(row, Mapping) for row in surfaces):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json has damaged unrealized_surfaces.")
    unrealized = [dict(row) for row in surfaces if row.get("policy_id") not in dropped_policies]
    tier, reason = publication_tier(
        mode=mode,
        coverage_certification=str(coverage.get("certification") or ""),
        coverage_reason=str(coverage.get("reason") or ""),
        split_reason=str(split_manifest.get("reason") or ""),
        development=sum(row["split"] == "development" for row in rows),
        withheld=bool(dropped),
        blocking=bool(packet.get("blocking_gaps")),
        review_limitations=bool(packet.get("unresolved_rules") or packet.get("unresolved_conflicts")),
        quality_gates_passed=quality_gates["passed"],
        unrealized_surfaces=bool(unrealized),
    )
    grouped = {
        BENCHMARK_FILE: rows,
        DEVELOPMENT_FILE: rows,
    }
    quality = {
        "certification": tier,
        "reason": reason,
        "coverage_certification": coverage.get("certification"),
        "coverage_reason": coverage.get("reason"),
        "mode": packet.get("mode"),
        "assurance_profile": packet.get("assurance_profile"),
        "cases": len(rows),
        "development": len(grouped[DEVELOPMENT_FILE]),
        "quality_gates": quality_gates,
        "blocking_gaps": list(packet.get("blocking_gaps") or []),
        "unrealized_surfaces": unrealized,
        "withheld_policies": sorted(dropped_policies),
        "withheld_cases": sorted(dropped),
        "unresolved_rules": [row.get("policy_id") for row in packet.get("unresolved_rules") or []],
        "unresolved_conflicts": [row.get("card_id") for row in packet.get("unresolved_conflicts") or []],
        "pending_cards": [card.get("card_id") for card in packet.get("cards") or [] if card_is_pending(card)],
        "split_reason": split_manifest.get("reason"),
        "packet_sha256": packet_sha256,
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "published_at": published_at,
    }
    return rows, dropped, grouped, quality, tier, reason


def _tag_contrast_families(rows: list[dict[str, Any]], families: Sequence[Mapping[str, Any]]) -> None:
    """Name each published family on its members, so eval can score both sides of a pair.

    A family counts only when every member is published. A pair with a dropped side
    cannot measure consistency.
    """
    by_id = {str(row["case_id"]): row for row in rows}
    tags: dict[str, list[str]] = {}
    for family in families:
        family_id = family.get("family_id")
        members = family.get("members")
        if not isinstance(family_id, str) or not isinstance(members, list):
            continue
        ids = [member.get("case_id") for member in members if isinstance(member, Mapping)]
        if len(ids) < 2 or not all(isinstance(case_id, str) and case_id in by_id for case_id in ids):
            continue
        for case_id in ids:
            tags.setdefault(case_id, []).append(family_id)
    for case_id, family_ids in tags.items():
        by_id[case_id]["contrast_family_ids"] = sorted(set(family_ids))


def _load_packet(run_dir: Path) -> dict[str, Any]:
    path = run_dir / PACKET_NAME
    try:
        packet = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{PACKET_NAME} cannot be read after publication."
        ) from exc
    if not isinstance(packet, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{PACKET_NAME} is damaged after publication.")
    return packet


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def _split_ids(manifest: Mapping[str, Any], run_dir: Path) -> set[str]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json must name its splits.")
    development = _ids(splits.get("development"), "development", run_dir)
    if set(splits) != {"development"}:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "split_manifest.json may only name development.")
    return development


def _ids(value: Any, name: str, run_dir: Path) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"split_manifest.json {name} must list case ids.")
    return set(value)


def _match(run_dir: Path, name: str, expected: str) -> None:
    path = run_dir / name
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        actual = ""
    if actual != expected:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{name} changed after it was signed.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
