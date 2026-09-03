# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for ``steps/curate/flow`` — one config, six steps, one command.

Running the curate category by hand means six configs plus a hand-written
approved policy, and the paths between them have to agree. Two of those
agreements are silent when broken: ``curate/nemo_curator`` ships
``emit_manifest: null`` while ``curate/audit`` ships ``declared_manifest: null``,
so an audit run against a producer that emitted no manifest reports counts as
*informational* and claims nothing — which reads exactly like a clean result.
This step derives both from one path, so they cannot disagree.

The shape is deliberately thin. Every step already exposes ``run(cfg) -> report``
(see ``tests/steps/curate/test_step_seam.py``), so this module owns three things
and no algorithms:

``derive``     one authored config becomes six per-step configs
``preflight``  everything that can be refused is refused before any work starts
``run``        the enabled steps, in dependency order, with their reports collected

**The approval gate is not collapsed.** ``curate/profile`` measures and emits
``candidate_policies.yaml`` with ``approved: false``; a threshold only becomes
executable when a person writes an ``approve:`` block naming who decided and on
what evidence. That block cannot appear on the first run, because the candidates
it selects from do not exist yet. So the flow is one config run twice, not one
command that decides for you — and the second run additionally recomputes the
corpus fingerprint and refuses an approval that was granted against different
data, which is what stops a config being copied between corpora with its
approval signature attached.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from nemotron.steps.curate.runtime import integrity
from nemotron.steps.curate.runtime import manifest as run_manifest
from nemotron.steps.curate.runtime import policy as policy_module
from nemotron.steps.curate.runtime import registry as signal_registry

logger = logging.getLogger("curate.flow")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "flow" / "config" / "default.yaml"

#: Bumped when a change would alter which steps a given config runs.
SCHEMA_VERSION = 1


class FlowConfigError(ValueError):
    """The flow cannot start as configured.

    One class rather than one per cause: preflight collects *every* problem it
    finds and raises once, because a run that is wrong in three ways should say
    so in one pass instead of three edit-run cycles.
    """


# -- the plan -----------------------------------------------------------------


@dataclass(frozen=True)
class StepPlan:
    """One step's place in the flow.

    ``produces`` and ``needs`` are artifact *names*, not paths: the flow resolves
    them to paths in one place so a downstream step cannot be pointed somewhere
    the upstream step never wrote.
    """

    key: str
    step_id: str
    subdir: str
    produces: tuple[str, ...] = ()
    #: Artifacts without which the step cannot run at all.
    needs: tuple[str, ...] = ()
    #: Artifacts the step runs without, but reports less for. Absent, they are a
    #: warning naming the producer — never a refusal, because the step's own
    #: degraded report is a legitimate answer, and never silence, because that
    #: degraded report reads exactly like a clean one.
    optional: tuple[str, ...] = ()
    gpus: int = 0


#: Dependency order, and the reason for it. ``profile`` reads the corpus *before*
#: filtering — profiling the filtered output would measure the gates you are
#: trying to understand after they have already run.
STEP_ORDER: tuple[StepPlan, ...] = (
    # First, because everything downstream reads JSONL with a document id and a
    # raw corpus has neither guaranteed. Disabled by default: a corpus already in
    # that shape should not be rewritten just to be read.
    StepPlan("ingest", "curate/ingest", "ingested", produces=("prepared",)),
    StepPlan("profile", "curate/profile", "profile", produces=("candidates", "profile_report")),
    StepPlan("filter", "curate/nemo_curator", "filtered_jsonl", produces=("corpus", "manifest", "ledger")),
    # manifest and ledger are optional on purpose: without a manifest the audit
    # reports completeness as informational, and without a ledger it reports
    # attribution as unavailable. Both are real answers, and both look like a
    # clean result unless something says why.
    StepPlan("audit", "curate/audit", "audit", needs=("corpus",), optional=("manifest", "ledger")),
    StepPlan("subset", "curate/subset", "subset", needs=("corpus",)),
    StepPlan("decontamination", "curate/decontamination", "decontaminated", needs=("corpus",), gpus=1),
)

STEPS_BY_KEY = {plan.key: plan for plan in STEP_ORDER}


@dataclass
class Resolved:
    """A step's derived config plus whether it will actually run."""

    plan: StepPlan
    enabled: bool
    config: dict[str, Any]
    notes: list[str] = field(default_factory=list)


# -- derivation ---------------------------------------------------------------


def _corpus(cfg: dict) -> dict:
    block = dict(cfg.get("corpus") or {})
    if not block.get("input"):
        raise FlowConfigError("corpus.input is required: the flow has nothing to read")
    return block


def _artifact_paths(root: Path) -> dict[str, str]:
    """Where each named artifact lives. One table, so two steps cannot disagree."""
    return {
        "corpus": str(root / "filtered_jsonl"),
        "manifest": str(root / "filtered_jsonl" / "run_manifest.json"),
        "ledger": str(root / "filtered_jsonl" / "curation_ledger.json"),
        "prepared": str(root / "ingested"),
        "candidates": str(root / "profile" / "candidate_policies.yaml"),
        "profile_report": str(root / "profile" / "profile_report.json"),
        "approved_policy": str(root / "policy" / "approved_policy.yaml"),
    }


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Commit a JSON artifact without exposing an interrupted partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def derive(cfg: dict) -> tuple[list[Resolved], dict[str, str]]:
    """Turn one authored config into one config per step.

    Anything a step needs that the flow can compute is computed here rather than
    asked for again: output paths, the manifest and ledger paths that must match
    between producer and auditor, and the reference corpus the audit compares
    against.
    """
    corpus = _corpus(cfg)
    root = Path(cfg.get("output_root") or "./output/curate")
    paths = _artifact_paths(root)
    steps_cfg = cfg.get("steps") or {}
    ingest_on = bool((steps_cfg.get("ingest") or {}).get("enabled"))

    # Ingest reads the author's raw field names but deliberately emits the
    # category's canonical schema. Every consumer must therefore read the
    # canonical names when ingest is enabled; carrying the raw names forward
    # makes a perfectly valid `content`/`doc_id` corpus disappear at profile or
    # subset time.
    shared = {
        "text_field": "text" if ingest_on else corpus.get("text_field", "text"),
        "id_field": "id" if ingest_on else corpus.get("id_field"),
        "source_field": (
            "source"
            if ingest_on and (corpus.get("source_field_in_source") or corpus.get("source_value"))
            else (None if ingest_on else corpus.get("source_field"))
        ),
    }

    # What profile and filter actually read. When ingest runs it is the corpus
    # ingest wrote, not the raw files: reading the raw corpus after normalising
    # it would measure a different set of documents from the one being filtered.
    read_from = f"{paths['prepared']}/*.jsonl" if ingest_on else corpus["input"]

    resolved: list[Resolved] = []
    for plan in STEP_ORDER:
        block = dict(steps_cfg.get(plan.key) or {})
        enabled = bool(block.pop("enabled", False))
        out = str(root / plan.subdir)

        if plan.key == "ingest":
            step_cfg = {
                "input": corpus["input"],
                "output_dir": out,
                "text_field": corpus.get("text_field", "text"),
                # A corpus that already carries an id keeps it; otherwise one is
                # minted from content, because a positional id does not survive
                # resharding and every downstream id claim would go stale.
                "id_from": corpus.get("id_field") if corpus.get("id_field_in_source") else None,
                # Minting from text alone would give two documents with identical
                # text but different URLs one id — defensible for a crawl, wrong
                # for a corpus where the URL is part of what a document is. So the
                # recipe is stated rather than inferred, and a URL carried in
                # metadata_fields joins it by default.
                "id_fields": corpus.get("id_fields")
                or [
                    f
                    for f in ("url", corpus.get("text_field", "text"))
                    if f == corpus.get("text_field", "text") or f in (corpus.get("metadata_fields") or [])
                ],
                "id_prefix": corpus.get("id_prefix") or "",
                "source_from": corpus.get("source_field_in_source"),
                "source": corpus.get("source_value"),
                "keep_fields": [f for f in (corpus.get("metadata_fields") or []) if f not in ("id", "source")],
            }
        elif plan.key == "profile":
            step_cfg = {
                **shared,
                # The UNFILTERED corpus on purpose: profiling the output measures
                # the gates after they have already removed what they remove.
                "input_glob": read_from,
                "output_dir": out,
                "language": corpus.get("language"),
                "langpack_dir": corpus.get("langpack_dir"),
            }
        elif plan.key == "filter":
            metadata_fields = list(corpus.get("metadata_fields") or [])
            if ingest_on:
                # These are produced by ingest independently of their raw
                # source names and must survive Curator's reader.
                metadata_fields.append("id")
                if shared["source_field"]:
                    metadata_fields.append("source")
            step_cfg = {
                **shared,
                "input_glob": read_from,
                "output_dir": out,
                "metadata_fields": list(dict.fromkeys(metadata_fields)),
                "emit_manifest": paths["manifest"],
                "emit_ledger": paths["ledger"],
            }
        elif plan.key == "audit":
            step_cfg = {
                "source_field": shared["source_field"],
                "target_glob": f"{paths['corpus']}/**/*.jsonl",
                "output_dir": out,
                # Derived, not asked for again. These are the two nulls that
                # silently disable the completeness claim when they disagree.
                "declared_manifest": paths["manifest"],
                "ledger_glob": paths["ledger"],
                "reference_glob": read_from,
                "digest_root": paths["corpus"],
            }
        elif plan.key == "subset":
            step_cfg = {
                **shared,
                "input_glob": f"{paths['corpus']}/**/*.jsonl",
                "output_dir": out,
            }
        else:  # decontamination
            step_cfg = {
                "text_field": shared["text_field"],
                "id_field": shared["id_field"],
                "train_glob": f"{paths['corpus']}/**/*.jsonl",
                "output_dir": out,
                "work_dir": str(root / plan.subdir / "cache"),
            }
            if block.get("holdout"):
                step_cfg["holdout_glob"] = block.pop("holdout")

        # Whatever the author wrote for this step wins over the derivation, so a
        # flow config is never a cage: an escape hatch that needs a second file
        # is not an escape hatch.
        step_cfg.update(block)
        resolved.append(Resolved(plan=plan, enabled=enabled, config=step_cfg))

    return resolved, paths


# -- preflight ----------------------------------------------------------------


def _artifact_exists(path: str) -> bool:
    """Whether an artifact a disabled producer would have written is there.

    A plain file is checked directly; a corpus reference goes through the shared
    resolver so a directory and a glob answer the same way.
    """
    p = Path(path)
    if p.is_file():
        return True
    return bool(integrity.expand_inputs(path))


def _stale_corpus_warnings(paths: dict[str, str]) -> list[str]:
    """Whether a reused corpus came from a run that actually finished."""
    manifest_path = Path(paths["manifest"])
    if not manifest_path.is_file():
        return [
            "reusing a corpus with no run_manifest beside it, so there is nothing to say "
            "whether the run that wrote it finished. A corpus left by a killed writer "
            "parses cleanly and is simply short."
        ]
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return [f"reusing a corpus whose manifest at {manifest_path} could not be read"]
    if not run_manifest.is_complete(document):
        return [
            f"reusing a corpus whose manifest at {manifest_path} has no completed_at: the "
            "run that wrote it did not reach its write barrier, so the corpus is a partial "
            "one. Re-run steps.filter rather than measuring against it."
        ]
    return []


def preflight(cfg: dict, resolved: list[Resolved], paths: dict[str, str]) -> list[str]:
    """Refuse everything refusable before the first step does any work.

    A flow that fails forty minutes in, after the filter has rewritten the
    corpus, is worse than six separate commands — the whole point of running
    them together is that the run either starts or explains why not.
    """
    problems: list[str] = []
    warnings: list[str] = []
    enabled = {r.plan.key for r in resolved if r.enabled}

    # A typo under steps: would otherwise drop a whole stage the author asked
    # for, and the run would report success having never attempted it.
    unknown = sorted(set(cfg.get("steps") or {}) - set(STEPS_BY_KEY))
    if unknown:
        problems.append(f"unknown step(s) under steps: {unknown}. Known: {sorted(STEPS_BY_KEY)}.")

    if not enabled:
        problems.append("no steps are enabled; set steps.<name>.enabled: true for at least one")

    for resolved_step in resolved:
        if not resolved_step.enabled:
            continue
        for artifact in resolved_step.plan.needs:
            producer = next((p.key for p in STEP_ORDER if artifact in p.produces), None)
            if producer in enabled:
                continue
            # The producer is disabled. Reuse a previous run's artifact only if
            # it is actually there; otherwise say which step would have made it,
            # rather than letting the step report an empty-looking result.
            if _artifact_exists(paths[artifact]):
                resolved_step.notes.append(
                    f"reusing {artifact} from a previous run at {paths[artifact]} (steps.{producer}.enabled is false)"
                )
                if artifact == "corpus":
                    # A corpus left behind by a filter run that DIED parses
                    # perfectly and is the wrong length. The manifest beside it
                    # is the only thing that can tell the two apart, and its
                    # absent completed_at is exactly that signal.
                    warnings.extend(_stale_corpus_warnings(paths))
                continue
            problems.append(
                f"steps.{resolved_step.plan.key} needs {artifact!r}, which steps.{producer} "
                f"produces, but that step is disabled and nothing is at {paths[artifact]}. "
                f"Enable steps.{producer}, or point at a previous run's output_root."
            )

        # Optional inputs degrade the step's report rather than stopping it, so
        # they are named instead of refused. The audit's two are the ones that
        # read as clean results: no manifest means completeness is merely
        # informational, no ledger means "nobody recorded why records left" —
        # not "no records left".
        for artifact in resolved_step.plan.optional:
            producer = next((p.key for p in STEP_ORDER if artifact in p.produces), None)
            if producer in enabled or _artifact_exists(paths[artifact]):
                continue
            if artifact == "ledger":
                warnings.append(
                    f"steps.{resolved_step.plan.key} runs without a ledger (steps.{producer} "
                    "is disabled), so it will report attribution.available: false. That is "
                    "'nobody recorded why records left', not 'no records left'."
                )
            elif artifact == "manifest":
                warnings.append(
                    f"steps.{resolved_step.plan.key} runs without a producer manifest "
                    f"(steps.{producer} is disabled), so row counts are informational and "
                    "completeness is not claimed."
                )
            else:
                warnings.append(
                    f"steps.{resolved_step.plan.key} runs without {artifact!r} "
                    f"(steps.{producer} is disabled), so its report will be reduced."
                )

    # corpus.source_field names the column every step AFTER ingest reads. What
    # ingest WRITES comes from source_field_in_source (a column in the raw data)
    # or source_value (a constant). Naming only the first is the natural thing to
    # write and produces no source column at all: every downstream step then
    # falls back to "each shard is its own source" and reports per-shard figures
    # that read exactly like per-corpus ones.
    corpus_block = cfg.get("corpus") or {}
    ingest_on = any(r.plan.key == "ingest" and r.enabled for r in resolved)
    if (
        ingest_on
        and corpus_block.get("source_field")
        and not corpus_block.get("source_field_in_source")
        and not corpus_block.get("source_value")
    ):
        problems.append(
            f"corpus.source_field is {corpus_block['source_field']!r}, so every step after "
            "ingest reads that column — but ingest is enabled and has nothing to write into "
            "it. Set corpus.source_field_in_source to the column the RAW data carries, or "
            "corpus.source_value to a constant. Without one of those the column is absent "
            "and per-source figures silently describe shards instead."
        )

    # A score column subset stratifies on only exists if the filter wrote it.
    subset_cfg = next((r for r in resolved if r.plan.key == "subset"), None)
    filter_cfg = next((r for r in resolved if r.plan.key == "filter"), None)
    if subset_cfg and subset_cfg.enabled and subset_cfg.config.get("quality_score_field"):
        column = subset_cfg.config["quality_score_field"]
        mode = (filter_cfg.config.get("mode") if filter_cfg else None) or "filter"
        if filter_cfg and filter_cfg.enabled:
            if mode == "filter":
                problems.append(
                    f"steps.subset.quality_score_field is {column!r}, but steps.filter.mode is "
                    "'filter', which discards scores after use. Set mode to 'annotate' or 'both' "
                    "so the column reaches the output, or drop quality_score_field."
                )
            elif not cfg.get("approve") and not filter_cfg.config.get("heuristic_filters"):
                # mode alone does not produce a column. The score columns are
                # written by the policy's signals, so with no policy there are no
                # signals and annotate/both write nothing — which is exactly the
                # documented first run.
                problems.append(
                    f"steps.subset.quality_score_field is {column!r}, but no policy is "
                    f"configured, so steps.filter writes no score columns at all — mode "
                    f"{mode!r} only decides what happens to the scores a policy produces. "
                    "Add an approve block, or drop quality_score_field."
                )

    # An optional artifact that will not exist must be UNSET, not merely warned
    # about: curate/audit reads declared_manifest unconditionally, so leaving the
    # derived path in place turns the case preflight just called legitimate into
    # a FileNotFoundError. Standalone curate/audit ships these as null and
    # degrades cleanly; the flow has to reproduce that, not just describe it.
    for resolved_step in resolved:
        if not resolved_step.enabled:
            continue
        for artifact, key in (("manifest", "declared_manifest"), ("ledger", "ledger_glob")):
            if artifact not in resolved_step.plan.optional:
                continue
            producer = next((p.key for p in STEP_ORDER if artifact in p.produces), None)
            if producer not in enabled and not _artifact_exists(paths[artifact]):
                resolved_step.config[key] = None

    # curate/nemo_curator does not clear its output directory, and a run whose
    # output differs does not overwrite the previous shard, so a second run lands
    # beside the first. Everything downstream then reads the union: the ledger
    # counts more documents out than in, the audit reports the surplus
    # unexplained, and subset refuses on duplicate ids. Three symptoms of one
    # stale directory, and none of them name it — which is why this is refused
    # here, while it is still one sentence. It bites the two-run approval
    # workflow specifically: run 1 measures and filters, run 2 approves and
    # filters again into the same place.
    if "filter" in enabled:
        stale = integrity.expand_inputs(f"{paths['corpus']}/**/*.jsonl")
        if stale:
            problems.append(
                f"steps.filter would write into {paths['corpus']}, which already holds "
                f"{len(stale)} corpus shard(s) from an earlier run (e.g. "
                f"{Path(stale[0]).name}). The step does not clear the directory, so this run "
                "would add a second corpus rather than replace the first, and every step "
                f"after it would read both. Delete {paths['corpus']}, or set output_root to "
                "somewhere new."
            )

    # An approval nothing applies is the shape of a run someone believes is
    # filtering and is not. Cheap to say, and the alternative is a clean-looking
    # corpus that never had the policy anywhere near it.
    if cfg.get("approve") and "filter" not in enabled:
        warnings.append(
            "approve is set but steps.filter is disabled, so the policy is promoted and "
            "written but nothing applies it. Enable steps.filter, or drop approve if you "
            "only meant to record the decision."
        )
    if cfg.get("approve") and "profile" in enabled:
        problems.append(
            "approve is set while steps.profile is enabled. Profiling and approval are two "
            "runs on purpose: otherwise this run can replace candidate_policies.yaml with "
            "measurements nobody reviewed while carrying the old approval block forward. "
            "Run profile first, review its output, then disable steps.profile before approving."
        )

    profile_cfg = next((r for r in resolved if r.plan.key == "profile" and r.enabled), None)
    if profile_cfg:
        language = profile_cfg.config.get("language")
        if not isinstance(language, str) or not language:
            problems.append(
                "steps.profile requires corpus.language (or a per-step override) as a "
                "non-empty BCP-47 tag; there is no safe default language."
            )
        pack_root = profile_cfg.config.get("langpack_dir")
        if pack_root is None or str(pack_root).strip() in {"", "bundled"}:
            problems.append(
                "steps.profile requires an explicit langpack_dir. Nemotron ships no "
                "production language packs; point corpus.langpack_dir or "
                "steps.profile.langpack_dir at the reviewed pack used for this corpus."
            )

    if "decontamination" in enabled:
        decon = next(r for r in resolved if r.plan.key == "decontamination")
        if not decon.config.get("holdout_glob"):
            problems.append(
                "steps.decontamination needs steps.decontamination.holdout: the split it protects cannot be guessed"
            )
        if not decon.config.get("skip_similarity"):
            warnings.append(
                "steps.decontamination runs Curator's GPU MinHash/LSH stages. Set "
                "skip_similarity: true to run the exact source-identity pass on CPU alone; "
                "the report then says overlap was NOT measured rather than reporting none."
            )

    if problems:
        raise FlowConfigError("the flow cannot run as configured:\n  - " + "\n  - ".join(problems))
    return warnings


# -- the approval gate --------------------------------------------------------


def materialise_policy(
    cfg: dict,
    resolved: list[Resolved],
    paths: dict[str, str],
    *,
    dry_run: bool = False,
    defer_corpus_verification: bool = False,
) -> list[str]:
    """Turn the authored ``approve:`` block into an approved policy on disk.

    The block is the deliberate act. It cannot be written on a first run — the
    candidates it selects thresholds from do not exist yet — and it goes through
    :func:`policy.promote`, which is the only function in ``steps/curate`` allowed
    to mark a policy approved and which checks the bound direction, that the
    signal was actually profiled, and that a corpus fingerprint is present.
    """
    approve = cfg.get("approve")
    if not approve:
        return []

    candidates_path = Path(approve.get("from") or paths["candidates"])
    if not candidates_path.is_file():
        profile_on = any(r.plan.key == "profile" and r.enabled for r in resolved)
        if profile_on:
            raise FlowConfigError(
                f"approve.from points at {candidates_path}, which does not exist yet — and "
                "steps.profile is enabled in this same config, which cannot help: the policy "
                "is promoted before any step runs, so the candidates this run is about to "
                "measure are not there to approve from. Profiling and approving are two runs "
                "on purpose. Run this once with approve unset, read the profile, then set "
                "approve and disable steps.profile."
            )
        raise FlowConfigError(
            f"approve.from points at {candidates_path}, which does not exist. Run the flow "
            "once with steps.profile enabled to measure the corpus before approving a "
            "threshold for it."
        )
    candidate = yaml.safe_load(candidates_path.read_text(encoding="utf-8")) or {}

    thresholds = approve.get("thresholds")
    if not thresholds:
        raise FlowConfigError("approve.thresholds is required: an approval that gates nothing is not one")

    # Only what the author actually wrote. approver/date/evidence are optional
    # and carried through verbatim when present; a block full of nulls would
    # record nothing while looking like provenance.
    approval = {
        key: approve[key] for key in ("method", "approver", "date", "evidence") if approve.get(key) is not None
    }
    document, warnings = cast(
        tuple[dict[str, Any], list[str]],
        policy_module.promote(candidate, thresholds=thresholds, approval=approval),
    )

    corpus = _corpus(cfg)
    declared = (document.get("corpus") or {}).get("fingerprint")
    if approve.get("verify_corpus", True) and defer_corpus_verification:
        warnings.append(
            "corpus fingerprint verification is deferred until ingest has committed this "
            "run's prepared corpus, immediately before the filter can apply the policy"
        )
    elif approve.get("verify_corpus", True):
        # The corpus the thresholds will be applied to, which is the one the
        # profile measured: with ingest enabled that is the ingested JSONL, not
        # the raw input the flow started from. Verifying corpus.input instead
        # compares two different corpora, and since the reader is JSONL-only it
        # refused every approval over a parquet source.
        filter_step = next((r for r in resolved if r.plan.key == "filter"), None)
        verified = (filter_step.config.get("input_glob") if filter_step else None) or corpus["input"]
        verified_text_field = filter_step.config.get("text_field") if filter_step else corpus.get("text_field", "text")
        verified_id_field = filter_step.config.get("id_field") if filter_step else corpus.get("id_field")
        if not isinstance(verified_text_field, str) or not verified_text_field:
            raise FlowConfigError("the corpus text_field used for approval verification must be a non-empty string")
        if verified_id_field is not None and (not isinstance(verified_id_field, str) or not verified_id_field):
            raise FlowConfigError("the corpus id_field used for approval verification must be a non-empty string")
        try:
            actual = integrity.corpus_fingerprint(verified, verified_text_field, verified_id_field)
        except integrity.UnreadableCorpusError as exc:
            raise FlowConfigError(
                f"the approval cannot be verified against {verified}: {exc} Enable "
                "steps.ingest so the corpus is read into JSONL first, or set "
                "approve.verify_corpus: false to approve without the check."
            ) from exc
        if declared and actual != declared:
            raise FlowConfigError(
                f"the approval was granted against corpus {declared}, but {verified} "
                f"currently fingerprints to {actual}. Thresholds and the signature that "
                "approved them do not transfer to different data — re-profile this corpus, "
                "or set approve.verify_corpus: false if you accept that they describe "
                "something else."
            )
        warnings.append(f"corpus fingerprint verified against the data: {actual}")

    destination = Path(paths["approved_policy"])
    for resolved_step in resolved:
        if resolved_step.plan.key == "filter":
            block = dict(resolved_step.config.get("heuristic_filters") or {})
            existing = block.get("approved_policy")
            if existing and Path(existing).resolve() != destination.resolve():
                raise FlowConfigError(
                    "approve promotes a policy to "
                    f"{destination}, but steps.filter.heuristic_filters.approved_policy points "
                    f"at {existing}. The flow cannot truthfully report which policy it applied; "
                    "remove the per-step path and let approve wire the promoted policy."
                )
            pack_root = block.get("langpack_dir") or corpus.get("langpack_dir")
            pack_signals = sorted(
                {
                    str(entry.get("signal"))
                    for entry in document.get("thresholds") or []
                    if entry.get("signal") in signal_registry.PACK_SIGNALS
                }
            )
            pack_root_configured = pack_root is not None and str(pack_root).strip() not in {
                "",
                "bundled",
            }
            if pack_signals and not pack_root_configured:
                raise FlowConfigError(
                    f"the approved thresholds use language-pack signals {pack_signals}, but "
                    "no langpack_dir is configured. Nemotron ships no production packs; set "
                    "corpus.langpack_dir or steps.filter.heuristic_filters.langpack_dir to "
                    "the reviewed pack root used during profiling."
                )
            if pack_root_configured:
                block["langpack_dir"] = pack_root
            block["approved_policy"] = str(destination)
            resolved_step.config["heuristic_filters"] = block

    if dry_run or defer_corpus_verification:
        # Everything above already ran: promote()'s checks and the corpus
        # fingerprint comparison when the input already exists. A preview must
        # not leave an approved policy behind, and a deferred verification must
        # not publish one before ingest proves which corpus this run will use.
        return warnings
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(destination)
    return warnings


# -- execution ----------------------------------------------------------------


def step_runner(key: str) -> Callable[[dict], dict]:
    """The ``run(cfg)`` seam every curate step exposes, imported on demand.

    Late so a flow that does not enable ``filter`` never imports Curator, and a
    flow that does not enable ``decontamination`` never touches the GPU path.
    """
    if key == "ingest":
        from nemotron.steps.curate.scripts import run_ingest

        return run_ingest.run
    if key == "profile":
        from nemotron.steps.curate.scripts import run_profile

        return run_profile.run
    if key == "filter":
        from nemotron.steps.curate.nemo_curator import step as nemo_curator_step

        return cast(Callable[[dict], dict], nemo_curator_step.run)
    if key == "audit":
        from nemotron.steps.curate.scripts import run_audit

        return run_audit.run
    if key == "subset":
        from nemotron.steps.curate.scripts import run_subset

        return run_subset.run
    if key == "decontamination":
        from nemotron.steps.curate.scripts import run_decontamination

        return run_decontamination.run
    raise FlowConfigError(f"unknown step {key!r}")


def plan(cfg: dict, *, dry_run: bool = False) -> tuple[list[Resolved], dict[str, str], list[str]]:
    """Derive, refuse, and write ``flow_plan.json`` — without running any step.

    Shared by ``run`` and ``--plan`` so the preview cannot describe a different
    run from the one that would happen.
    """
    root = Path(cfg.get("output_root") or "./output/curate")
    plan_path = root / "flow_plan.json"
    plan_path.unlink(missing_ok=True)
    plan_path.with_name(f".{plan_path.name}.tmp").unlink(missing_ok=True)

    resolved, paths = derive(cfg)
    warnings = preflight(cfg, resolved, paths)
    enabled = {step.plan.key for step in resolved if step.enabled}
    defer_verification = bool(cfg.get("approve") and {"ingest", "filter"} <= enabled)
    warnings += materialise_policy(
        cfg,
        resolved,
        paths,
        dry_run=dry_run,
        defer_corpus_verification=defer_verification,
    )

    _write_json_atomic(
        plan_path,
        {
            "schema_version": SCHEMA_VERSION,
            "output_root": str(root),
            "dry_run": dry_run,
            "artifacts": paths,
            "steps": [
                {
                    "key": r.plan.key,
                    "step_id": r.plan.step_id,
                    "enabled": r.enabled,
                    "notes": r.notes,
                    "config": r.config,
                }
                for r in resolved
            ],
            "warnings": warnings,
        },
    )
    return resolved, paths, warnings


def run(cfg: dict) -> dict[str, Any]:
    """Run every enabled step in dependency order and return one report.

    Does not raise ``SystemExit``: the uniform seam, so a caller — including a
    future flow of flows — keeps control of what a failure means.
    """
    started_at = run_manifest.utc_now()
    root = Path(cfg.get("output_root") or "./output/curate")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "flow_report.json"
    report_path.unlink(missing_ok=True)
    report_path.with_name(f".{report_path.name}.tmp").unlink(missing_ok=True)
    resolved, paths, warnings = plan(cfg)

    for warning in warnings:
        logger.warning(warning)

    results: list[dict[str, Any]] = []
    failure: BaseException | None = None
    for resolved_step in resolved:
        if not resolved_step.enabled:
            results.append({"step_id": resolved_step.plan.step_id, "status": "disabled"})
            continue
        logger.info("running %s", resolved_step.plan.step_id)
        try:
            if (
                resolved_step.plan.key == "filter"
                and cfg.get("approve")
                and any(step.plan.key == "ingest" and step.enabled for step in resolved)
            ):
                # Planning cannot verify the corpus ingest has not written yet.
                # Do it after ingest commits and before the first consumer can
                # apply the policy; this closes the old-output/new-input race.
                for warning in materialise_policy(cfg, resolved, paths):
                    if warning not in warnings:
                        warnings.append(warning)
                        logger.warning(warning)
            report = step_runner(resolved_step.plan.key)(resolved_step.config)
        except BaseException as exc:  # noqa: BLE001 - re-raised after the report
            # A run that dies mid-flow has already written some steps' artifacts
            # and not others. Letting the exception escape before the report is
            # written leaves that state on disk with nothing saying how far it
            # got — and a previous run's flow_report.json sitting next to it,
            # describing a different run entirely.
            results.append(
                {
                    "step_id": resolved_step.plan.step_id,
                    "status": "failed",
                    "output_dir": resolved_step.config.get("output_dir"),
                    "notes": resolved_step.notes,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            failure = exc
            break
        step_warnings = report.get("warnings") if isinstance(report, dict) else None
        results.append(
            {
                "step_id": resolved_step.plan.step_id,
                "status": "ok",
                "output_dir": resolved_step.config.get("output_dir"),
                "notes": resolved_step.notes,
                # Carried up rather than collapsed: a flow that "succeeded" while
                # its audit failed is the same silent success the audit exists to
                # catch. Same for each step's own warnings.
                "passed": report.get("passed") if isinstance(report, dict) else None,
                "warnings": list(step_warnings or []),
            }
        )

    audit_result = next((r for r in results if r["step_id"] == "curate/audit"), None)
    filter_result = next((r for r in results if r["step_id"] == "curate/nemo_curator"), None)
    # Presence of an approve block is not application. When steps.filter is
    # disabled the policy is promoted and written and nothing runs it, and a
    # report claiming otherwise is the specific lie this field would tell.
    policy_applied = bool(cfg.get("approve") and filter_result and filter_result["status"] == "ok")
    report = {
        "schema_version": SCHEMA_VERSION,
        "step_id": "curate/flow",
        "started_at": started_at,
        "completed_at": run_manifest.utc_now(),
        "output_root": str(root),
        "artifacts": paths,
        "steps": results,
        "warnings": warnings,
        "audit_passed": audit_result.get("passed") if audit_result else None,
        "status": "failed" if failure is not None else "ok",
        "policy_applied": policy_applied,
        "policy_promoted": bool(cfg.get("approve")),
    }
    _write_json_atomic(report_path, report)
    ran = [r["step_id"] for r in results if r["status"] == "ok"]
    if failure is not None:
        failed = next(r["step_id"] for r in results if r["status"] == "failed")
        print(
            f"curate/flow: {failed} failed after {len(ran)} step(s); "
            f"flow_report.json records how far it got -> {root}",
            file=sys.stderr,
        )
        raise failure
    print(f"curate/flow: ran {len(ran)} step(s) -> {root}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the curate flow from one config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="write flow_plan.json and print what would run, without running it",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    try:
        if args.plan:
            resolved, _paths, warnings = plan(cfg, dry_run=True)
            for resolved_step in resolved:
                mark = "run " if resolved_step.enabled else "skip"
                print(f"  [{mark}] {resolved_step.plan.step_id}")
                for note in resolved_step.notes:
                    print(f"         {note}")
            for warning in warnings:
                print(f"  WARNING {warning}")
            root = Path(cfg.get("output_root") or "./output/curate")
            print(f"  wrote {root / 'flow_plan.json'} (no step was run)")
            return
        report = run(cfg)
    except FlowConfigError as exc:
        print(f"curate/flow: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except policy_module.PolicyNotPromotableError as exc:
        print(f"curate/flow: approve block refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    raise SystemExit(0 if report.get("audit_passed") is not False else 1)


if __name__ == "__main__":
    main()
