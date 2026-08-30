#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/curate/nemo_curator"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Lightweight JSONL curation via NeMo Curator."""

from __future__ import annotations

import argparse
import glob as globlib
import os
from ast import literal_eval
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

from nemotron.steps.curate.runtime import manifest as run_manifest

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def keep_language(value: str, allowed: set[str]) -> bool:
    """Whether a FastText language label is one the config asked for.

    ``FastTextLangId.score_document`` returns ``str([score, lang_code])`` with the
    label exactly as the model emitted it — ``lid.176`` emits lowercase ISO codes.
    The comparison is case-folded because the config's codes are upper-cased on
    load, and an exact match would then reject every document while looking like
    a filter that simply found nothing.

    A label may also carry a script suffix (``zh_Hans``); matching the part before
    the underscore lets ``ZH`` select those without enumerating every variant.
    """
    score, lang_code = literal_eval(value)
    if score < 0.0:
        return False
    label = str(lang_code).casefold()
    wanted = {code.casefold() for code in allowed}
    return label in wanted or label.split("_", 1)[0] in wanted


def reader_fields(cfg: dict) -> list[str]:
    """Columns to project at the reader.

    ``JsonlReader`` treats ``fields`` as a projection, so anything left out here
    is gone for the rest of the pipeline. Listing ``metadata_fields`` is the only
    way a document's own ``id`` — the one identifier that survives resharding —
    reaches the output. Absent key reproduces the historical text-only read.
    """
    declared = list(cfg.get("metadata_fields") or [])

    # A field named for the manifest but absent from the projection is read away
    # at the first stage, so the manifest would come back with no per-source
    # counts and nothing would say why. Carry them rather than fail: the user
    # asked for the column by naming it.
    for key in ("id_field", "source_field"):
        value = cfg.get(key)
        if value and value not in declared:
            declared.append(value)

    fields = [cfg["text_field"], *declared]
    seen: set[str] = set()
    return [f for f in fields if not (f in seen or seen.add(f))]


#: Extensions the reader will pick up from a directory. Curator's own discovery
#: is used when it is importable; this list is the fallback, and it is wider than
#: ``.jsonl`` because a directory of ``.json`` line-files is a common shape and
#: missing it would report a manifest input count of zero for a run that read
#: everything.
JSONL_EXTENSIONS = (".jsonl", ".json")


def resolve_inputs(input_glob: str | list[str]) -> list[str]:
    """Expand the reader's ``file_paths`` the way the reader itself would.

    The manifest's input counts are only meaningful if this sees the same files
    the pipeline read. Curator's own ``get_all_file_paths_under`` is preferred
    for exactly that reason; the local fallback exists so the manifest can still
    be written when it is unavailable.
    """
    patterns = [input_glob] if isinstance(input_glob, str) else list(input_glob)
    paths: list[str] = []

    try:
        from nemo_curator.utils.file_utils import get_all_file_paths_under
    except Exception:  # noqa: BLE001 - the fallback below is the point
        get_all_file_paths_under = None

    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(globlib.glob(pattern, recursive=True))
            continue
        candidate = Path(pattern)
        if not candidate.is_dir():
            paths.append(pattern)
            continue
        if get_all_file_paths_under is not None:
            paths.extend(
                get_all_file_paths_under(
                    str(candidate),
                    recurse_subdirectories=True,
                    keep_extensions=list(JSONL_EXTENSIONS),
                )
            )
        else:
            paths.extend(
                str(p)
                for p in sorted(candidate.rglob("*"))
                if p.is_file() and p.suffix in JSONL_EXTENSIONS
            )

    return sorted({p for p in paths if Path(p).is_file()})


#: Minimum FastText confidence for a language prediction to be trusted. Matches
#: Curator's own ``FastTextLangId`` default so the two agree; ``language_codes``
#: decides WHICH language, this decides how sure the model has to be.
DEFAULT_LANGID_SCORE = 0.3

#: Passed to MultilingualDomainClassifier instead of accepting its default.
#:
#: Curator's tokenizer truncates IN PLACE — ``df[text] = df[text].str.slice(0,
#: max_chars)`` at stages/text/models/tokenizer.py:160 — on the DataFrame that
#: continues to the writer. So the classifier's reading window silently becomes
#: the delivered corpus. Its default of 2000 rewrote 53.4% of one Vietnamese run,
#: destroying 29.7 million characters, while the ledger, manifest and audit all
#: reported success: every one of them counts rows, none reads content.
#:
#: None disables the slice entirely. The model still bounds what it reads, at the
#: token level, inside its own tokenizer — where truncation belongs, because it
#: does not touch the corpus.
DOMAIN_MAX_CHARS = None

#: Requirements a signal may declare that are satisfied by loading a language
#: pack. Every one of these is also a pack *capability* name, which is why the
#: pack itself is the thing that supplies them.
PACK_REQUIREMENTS: frozenset[str] = frozenset(
    {
        "boilerplate_hits",
        "diacritic_ratio",
        "script_ratio",
        "sentence_end_ratio",
        "stopword_ratio",
        "stopword_ratio_folded",
    }
)

#: Everything this step knows how to supply. A signal declaring anything else is
#: refused by name rather than reaching ``Signal.build`` and failing on a missing
#: keyword argument — the failure mode that let ``token_count`` through.
KNOWN_REQUIREMENTS: frozenset[str] = PACK_REQUIREMENTS | {"tokenizer"}


#: Which bound keys a policy may set, per signal direction. Enforced rather than
#: inferred: ``min``/``max`` zipped positionally onto ``threshold_params`` maps a
#: ``max:`` bound onto a ``min_*`` parameter and silently inverts the gate, so a
#: policy meaning "drop above 0.9" keeps exactly the documents it meant to drop.
BOUND_KEYS: dict[str, tuple[str, ...]] = {
    "min": ("min",),
    "max": ("max",),
    "interval": ("min", "max"),
}


def visible_gpu_count() -> int:
    """How many GPUs this process was actually given.

    ``CUDA_VISIBLE_DEVICES`` is what the scheduler hands a job, so it is the
    honest answer — not what the machine physically holds, which a shared node
    would over-report.
    """
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        return 0
    return len([d for d in devices.split(",") if d.strip() not in ("", "-1")])


def ray_client_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg.get("ray") or {})
    if "num_cpus" not in kwargs and os.environ.get("NEMOTRON_CURATOR_RAY_NUM_CPUS"):
        kwargs["num_cpus"] = int(os.environ["NEMOTRON_CURATOR_RAY_NUM_CPUS"])
    if "num_gpus" not in kwargs:
        # RayClient defaults num_gpus to None, which starts Ray declaring ZERO
        # GPUs however many the job was given. A pipeline holding a classifier
        # then dies with "requires 1.0 but only 0 are available" — after the
        # reader has already run. Declaring what the scheduler allocated is the
        # correct default; ray.num_gpus in config still overrides it.
        override = os.environ.get("NEMOTRON_CURATOR_RAY_NUM_GPUS")
        count = int(override) if override else visible_gpu_count()
        if count:
            kwargs["num_gpus"] = count
    return kwargs


def text_filter_stages():
    """Return Filter/ScoreFilter across supported NeMo Curator releases."""
    try:
        from nemo_curator.stages.text.modules import Filter, ScoreFilter
    except ImportError:
        from nemo_curator.stages.text.filters import Filter, ScoreFilter
    return Filter, ScoreFilter


def score_stage():
    """Return Score across supported NeMo Curator releases."""
    try:
        from nemo_curator.stages.text.modules import Score
    except ImportError:
        from nemo_curator.stages.text.filters import Score
    return Score


def build_pipeline(cfg: dict) -> tuple[Any, list[str]]:
    """Construct the Curator pipeline for this config, without running it.

    Split out from :func:`run` so the pipeline's *shape* can be inspected and
    tested without a Ray cluster, and so a caller orchestrating several steps can
    fail on a bad config before any of them starts work.
    """
    mode = cfg.get("mode", "filter")
    allowed_languages = {code.upper() for code in cfg.get("language_codes") or []}
    models = cfg.get("models") or {}
    quality_filters = cfg.get("quality_filters") or {}

    pipeline = Pipeline(name="curate_nemo_curator")
    pipeline.add_stage(JsonlReader(file_paths=cfg["input_glob"], fields=reader_fields(cfg)))
    if allowed_languages:
        Filter, ScoreFilter = text_filter_stages()
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        pipeline.add_stage(
            ScoreFilter(
                FastTextLangId(
                    model_path=models["fasttext_langid"],
                    # Curator's own default, not 0.0. A fallback of 0.0 disables
                    # the confidence gate entirely while the config says nothing,
                    # so a run that named language_codes would keep every document
                    # the model was unsure about — and the shipped default.yaml
                    # said 0.3, so the flow and the standalone step disagreed.
                    min_langid_score=quality_filters.get("min_langid_score", DEFAULT_LANGID_SCORE),
                ),
                text_field=cfg["text_field"],
                score_field="language",
            )
        )
        # Named, not a lambda. Curator derives the stage name from the callable's
        # __name__ (score_filter.py:348), and the ledger's per-gate attribution
        # is keyed on stage names — so a lambda produced an entry reading
        # "filter_fn: 5134", which tells a reader nothing about what removed
        # their documents. The two language stages are the confidence gate and
        # the code gate, and the ledger should say which is which.
        def language_code(value: str) -> bool:
            return keep_language(value, allowed_languages)

        # Curator's Filter defaults its stage name to the literal "filter_fn"
        # (score_filter.py:24). The ledger's per-gate attribution is keyed on
        # stage names, so leaving it produced an entry reading "filter_fn: 5134"
        # — a number with no way to tell what removed those documents. The two
        # language stages are the confidence gate and the code gate; the ledger
        # has to say which is which.
        # Curator's Filter hardcodes its stage name to "filter_fn"
        # (score_filter.py:24, and __post_init__ overwrites whatever the
        # constructor is given). The ledger's per-gate attribution is keyed on
        # stage names, so that produced an entry reading "filter_fn: 5134" — a
        # count with no way to tell what removed those documents. The two
        # language stages are the confidence gate and the code gate; the ledger
        # has to say which is which, so the name is set after construction.
        language_stage = Filter(filter_fn=language_code, filter_field="language")
        language_stage.name = "language_code"
        pipeline.add_stage(language_stage)

    has_word_filter = any(key in quality_filters for key in ("min_words", "max_words"))
    if has_word_filter:
        if not all(key in quality_filters for key in ("min_words", "max_words")):
            raise ValueError("quality_filters must set both min_words and max_words to enable WordCountFilter")
        _, ScoreFilter = text_filter_stages()
        from nemo_curator.stages.text.filters.heuristic import WordCountFilter

        pipeline.add_stage(
            ScoreFilter(
                WordCountFilter(
                    min_words=quality_filters["min_words"],
                    max_words=quality_filters["max_words"],
                ),
                text_field=cfg["text_field"],
            )
        )
    # Two ways to reach the classifier, because "which domains do I keep" is a
    # decision that needs evidence, and the evidence is the labels themselves.
    # Filtering on a first run drops most of a corpus with nothing to justify it;
    # annotating writes the label and the probability and drops nothing, so the
    # next run can gate on a distribution somebody looked at.
    if cfg.get("domains") or cfg.get("annotate_domains"):
        from nemo_curator.stages.text.classifiers import MultilingualDomainClassifier

        domain_kwargs: dict[str, Any] = {
            "text_field": cfg["text_field"],
            # None means label everything and drop nothing.
            "filter_by": cfg.get("domains") or None,
            "cache_dir": models.get("hf_cache_dir"),
            # Never the default. See DOMAIN_MAX_CHARS.
            "max_chars": cfg.get("domain_max_chars", DOMAIN_MAX_CHARS),
        }
        # Passed only when asked for. Naming it records how confident the
        # prediction was — an argmax label with no strength cannot be gated on
        # later — but passing it unconditionally, even as None, changes the
        # constructed call for every config that already used `domains`.
        if cfg.get("domain_score_field"):
            domain_kwargs["score_field"] = cfg["domain_score_field"]

        pipeline.add_stage(MultilingualDomainClassifier(**domain_kwargs))
    policy_warnings: list[str] = []

    pipeline.add_stage(JsonlWriter(path=cfg["output_dir"]))
    return pipeline, policy_warnings


def run(cfg: dict) -> dict[str, Any]:
    """Curate the corpus and return a report describing what the run did.

    The uniform entry point every curate step exposes. Does not raise
    ``SystemExit``: a caller running several steps decides what a failure means.
    """
    mode = cfg.get("mode", "filter")

    if cfg.get("dataset"):
        snapshot_download(**cfg["dataset"])

    # Curator's writer names shards by content hash, so a second run into the
    # same directory ADDS to it rather than replacing it. Measured: one corpus
    # ended with 31,689 rows out of 20,000 in, mixing a policy run with an
    # earlier no-policy one, and the two were indistinguishable by filename.
    #
    # Refusing rather than deleting: this step does not own every file under
    # output_dir, and silently removing a previous run's corpus is worse than
    # stopping. The ledger and audit do catch it afterwards, but only after the
    # work has been done twice.
    existing = sorted(Path(cfg["output_dir"]).rglob("*.jsonl")) if cfg.get("output_dir") else []
    if existing:
        raise ValueError(
            f"{cfg['output_dir']} already holds {len(existing)} .jsonl file(s) from a previous "
            f"run, e.g. {existing[0].name}. Curator's writer names shards by content hash, so "
            "this run would add to them rather than replace them and the corpus would be a "
            "mixture of two policies. Remove the directory, or point output_dir somewhere new."
        )

    pipeline, policy_warnings = build_pipeline(cfg)
    for warning in policy_warnings:
        print(f"curate/nemo_curator: WARNING {warning}")

    manifest_path = cfg.get("emit_manifest")
    ledger_path = cfg.get("emit_ledger")
    started_at = run_manifest.utc_now()

    ray_client = RayClient(**ray_client_kwargs(cfg))
    ray_client.start()
    try:
        # The returned tasks carry Curator's per-stage counters, which is where
        # the ledger's per-gate breakdown comes from.
        tasks = pipeline.run()
    except BaseException:
        # A manifest without completed_at is how an auditor learns the run died
        # rather than inferring completion from whatever files happen to exist.
        if manifest_path:
            emit_manifest(cfg, manifest_path, started_at, completed=False)
        if ledger_path:
            emit_ledger(cfg, ledger_path, completed=False)
        raise
    finally:
        ray_client.stop()

    # Everything below is accounting over a corpus that is already written, so
    # none of it may fail the run: a missing attribute costs the breakdown, not
    # the output.
    stage_counts = collect_stage_counts(tasks or [])
    stage_names = [getattr(stage, "name", "") for stage in getattr(pipeline, "stages", [])]

    if manifest_path:
        emit_manifest(
            cfg,
            manifest_path,
            started_at,
            completed=True,
            stage_names=stage_names,
            stage_counts=stage_counts,
        )
    if ledger_path:
        emit_ledger(
            cfg, ledger_path, completed=True, stage_names=stage_names, stage_counts=stage_counts
        )

    return {
        "step_id": "curate/nemo_curator",
        "started_at": started_at,
        "completed_at": run_manifest.utc_now(),
        "mode": mode,
        "output_dir": cfg["output_dir"],
        "warnings": list(policy_warnings),
        "artifacts": {
            k: v
            for k, v in (("run_manifest", manifest_path), ("curation_ledger", ledger_path))
            if v
        },
    }


#: What ``filtered`` is attributed to when the per-gate breakdown could not be
#: reconciled. Not a placeholder: it is the honest name for what this step then
#: knows — how many documents entered and left, and nothing in between.
UNATTRIBUTED = "unattributed"

ATTRIBUTION_NOTE = (
    "Per-gate removal counts come from Curator's own per-stage item counters "
    "(StagePerfStats.num_items_processed, collected via TaskPerfUtils.collect_stage_metrics) and "
    "are published only when they sum to the removal measured independently by counting the input "
    "and output files on disk. When the two disagree the breakdown is discarded and everything is "
    "recorded as 'unattributed', because a plausible-looking attribution that does not reconcile "
    "is worse than declaring the breakdown unavailable."
)

UNRECONCILED_NOTE = (
    "per-stage counters were collected but did not sum to the {observed} documents the corpus "
    "actually lost; the breakdown was discarded rather than published"
)


def collect_stage_counts(tasks: list) -> dict[str, int]:
    """How many documents each Curator stage processed, by stage name.

    Curator tracks this already — every ``ProcessingStage`` accumulates a
    ``StagePerfStats`` on the tasks flowing through it — so nothing is
    instrumented here. ``collect_stage_metrics`` handles the awkward part: a
    batch that fans out into several tasks carries the same stats object on each,
    and it deduplicates by identity so the fan-out is not counted twice.

    Returns ``{}`` rather than raising when the executor reports nothing. That is
    a reason to fall back to an unattributed ledger, not to fail a run whose
    output is already written.
    """
    if not tasks:
        return {}
    try:
        from nemo_curator.tasks.utils import TaskPerfUtils

        collected = TaskPerfUtils.collect_stage_metrics(tasks)
    except Exception:  # noqa: BLE001 - telemetry must never fail a completed run
        return {}

    counts: dict[str, int] = {}
    for stage_name, metrics in (collected or {}).items():
        values = metrics.get("num_items_processed")
        if values is None:
            continue
        counts[str(stage_name)] = int(sum(float(v) for v in values))
    return counts


def attribute_removals(
    stage_names: list[str], counts: dict[str, int], observed_removed: int
) -> dict[str, int] | None:
    """Turn per-stage item counts into per-gate removal counts.

    A filter chain is a sequence of stages each fed by the previous one, so what
    a stage removed is what it received minus what the next stage received. The
    counts are matched to declared stage names by prefix because a
    ``CompositeStage`` decomposes into differently-named parts at execution.

    Returns ``None`` — meaning "record this as unattributed" — whenever the
    result cannot be trusted: fewer than two stages reported, or the removals do
    not sum to ``observed_removed``, which was measured by counting the files on
    disk and is therefore independent of anything the pipeline reported about
    itself. That reconciliation is the whole safeguard. Without it a stage
    silently dropping rows would produce a breakdown that looks complete.
    """
    ordered: list[tuple[str, int]] = []
    for declared in stage_names:
        for name, count in counts.items():
            if name == declared or name.startswith(declared):
                ordered.append((declared, count))
                break

    if len(ordered) < 2:  # noqa: PLR2004 - a diff needs two points
        return None

    removals = {
        name: ordered[i][1] - ordered[i + 1][1]
        for i, (name, _) in enumerate(ordered[:-1])
        if ordered[i][1] - ordered[i + 1][1] > 0
    }
    if sum(removals.values()) != observed_removed:
        return None
    return removals


def emit_ledger(
    cfg: dict,
    path: str,
    *,
    completed: bool,
    stage_names: list[str] | None = None,
    stage_counts: dict[str, int] | None = None,
) -> None:
    """Write the run's record accounting.

    The producer half of ``curation_ledger``. ``curate/audit`` consumes it to tell
    a record removed on purpose from one lost to a swallowed exception — a
    distinction nothing in the input and output alone can supply.

    What it can and cannot say is stated in the ledger itself rather than left to
    be inferred: see :data:`ATTRIBUTION_NOTE`.
    """
    from nemotron.steps.curate.runtime import ledger as ledger_module

    source_field = cfg.get("source_field")
    input_counts = run_manifest.count_jsonl(resolve_inputs(cfg["input_glob"]), source_field)
    output_files = sorted(str(p) for p in Path(cfg["output_dir"]).rglob("*.jsonl"))
    output_counts = run_manifest.count_jsonl(output_files, source_field)

    entry = ledger_module.StageLedger(stage="curate/nemo_curator")
    entry.add_input(input_counts["row_count"])
    entry.add_success(output_counts["row_count"])
    removed = input_counts["row_count"] - output_counts["row_count"]
    if removed > 0:
        per_gate = attribute_removals(list(stage_names or []), dict(stage_counts or {}), removed)
        if per_gate:
            for gate, count in per_gate.items():
                entry.add_filtered(gate, count)
        else:
            entry.add_filtered(UNATTRIBUTED, removed)
            # Distinguish "no counters reached us" from "counters reached us and
            # disagreed with the files on disk". The second is a finding.
            if stage_counts:
                entry.notes["unreconciled"] = UNRECONCILED_NOTE.format(observed=removed)
    entry.notes["attribution"] = ATTRIBUTION_NOTE
    entry.notes["completed"] = completed
    entry.notes["mode"] = cfg.get("mode", "filter")

    if removed < 0:
        # More rows out than in. Balancing this by inventing a negative filtered
        # count would hide it; recording it as an unprocessable unit makes the
        # audit say so out loud.
        entry.add_quarantined(
            cfg["output_dir"],
            f"output has {-removed} more rows than input; the run did not only remove documents",
            0,
        )

    # A run that died has an unbalanced ledger by definition, and that is the
    # signal an auditor needs — refusing to write it would destroy the evidence.
    entry.write(path, require_balanced=completed and removed >= 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate JSONL text with NeMo Curator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run(cfg)


def emit_manifest(
    cfg: dict,
    path: str,
    started_at: str,
    *,
    completed: bool,
    stage_names: list[str] | None = None,
    stage_counts: dict[str, int] | None = None,
) -> None:
    """Write the run manifest after the pipeline's own write barrier.

    Row counting happens here rather than inside the pipeline because Curator's
    writers return file paths without row counts — ``BaseWriter.process`` logs
    ``task.num_items`` and discards it — so the wrapper has no per-stage figure
    to carry forward from the writer.

    The per-gate breakdown does reach here, from Curator's per-stage counters. It
    is reconciled independently of the ledger, against this function's own disk
    counts: if the two artifacts ever disagreed, that disagreement would itself
    be the finding, and sharing one computation between them would hide it.
    """
    source_field = cfg.get("source_field")
    output_files = sorted(str(p) for p in Path(cfg["output_dir"]).rglob("*.jsonl"))

    input_counts = run_manifest.count_jsonl(resolve_inputs(cfg["input_glob"]), source_field)
    output_counts = run_manifest.count_jsonl(output_files, source_field)

    declared = None
    removed = input_counts.get("row_count", 0) - output_counts.get("row_count", 0)
    if removed > 0:
        per_gate = attribute_removals(list(stage_names or []), dict(stage_counts or {}), removed)
        if per_gate:
            declared = {
                "attribution": run_manifest.ATTRIBUTION_DECLARED,
                "rows_absent_from_output": removed,
                "filtered": per_gate,
                "failed": None,
                "quarantined": None,
            }

    document = run_manifest.build_manifest(
        step_id="curate/nemo_curator",
        config=cfg,
        started_at=started_at,
        input_glob=cfg["input_glob"],
        input_counts=input_counts,
        output_counts=output_counts,
        id_field=cfg.get("id_field"),
        source_field=source_field,
        completed_at=run_manifest.utc_now() if completed else None,
        declared=declared,
    )
    run_manifest.write_manifest(path, document)


if __name__ == "__main__":
    main()
