"""Cross-wording stability analysis for a frozen BFCL release (SOV-862).

The question is whether benchmark conclusions move when only the user wording
moves. Answering it properly needs the same skeleton scored under both a human
and a model wording, paired at task level, because with three or four models a
ranking carries almost no power and the sharper question is which individual
tasks flip verdict.

A frozen release usually cannot answer that on its own. Dedup and balancing
publish one variant per skeleton, so the published table holds human-worded and
model-worded tasks but never both wordings of the same skeleton. This module
therefore reports three separate things and refuses to blur them:

- a *paired replicate* floor, measured between two scored runs over the same
  task set, which says how often a verdict flips for reasons that have nothing
  to do with wording;
- an *unpaired* wording contrast over the published table, which is real
  measurement but confounded, because which skeletons received which wording
  was decided by balancing rather than by assignment;
- the *paired wording* design itself, which is reported as unavailable, naming
  the artifact that is missing, rather than approximated by the unpaired result.

A stability conclusion is only called `stable` or `unstable` when the paired
wording design exists. Until then the honest answer is `underpowered`, which
the SOV-862 acceptance criterion accepts as an outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_statistics import (
    StatisticsError,
    critical_z,
    mcnemar_exact_p,
    newcombe_difference_interval,
    round_statistic,
    two_proportion_test,
)

CROSS_WORDING_CONTRACT_VERSION: Final = "1.0"

HUMAN_WORDING_SOURCE: Final = "template"
MODEL_WORDING_SOURCE: Final = "model"
WORDING_SOURCES: Final = (HUMAN_WORDING_SOURCE, MODEL_WORDING_SOURCE)

STABILITY_CONCLUSIONS: Final = ("stable", "unstable", "underpowered", "not_measured")

# Strata the per-policy readout is grouped by. `turn_policy` is the axis
# SOV-862 asks for; the others locate a confound rather than a conclusion.
STRATUM_FIELDS: Final = ("turn_policy", "category", "difficulty")


class CrossWordingError(ValueError):
    """An input cannot support a trustworthy wording-stability statement."""


@dataclass(frozen=True)
class ScoredRun:
    """One evaluation run over the published table."""

    run_id: str
    task_results: Path
    role: str


@dataclass(frozen=True)
class CrossWordingInputs:
    published: Path
    rendered_conversations: Path
    primary_run: ScoredRun
    output_dir: Path
    replicate_runs: tuple[ScoredRun, ...] = field(default=())
    confidence_level: float = 0.95


def _round(value: float) -> float:
    try:
        return round_statistic(value)
    except StatisticsError as exc:
        raise CrossWordingError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _semantic_hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()}"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _read_columns(path: Path, columns: Sequence[str], label: str) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CrossWordingError(f"{label} does not exist: {resolved}")
    try:
        table = pq.read_table(resolved, columns=list(columns))
    except (OSError, ValueError, KeyError) as exc:
        raise CrossWordingError(f"{label} is missing required columns {list(columns)}: {exc}") from exc
    return cast(list[dict[str, Any]], table.to_pylist())


def _load_published(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    rows = _read_columns(path, ("task_id", *STRATUM_FIELDS), "published benchmark")
    if not rows:
        raise CrossWordingError("published benchmark is empty")
    published: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise CrossWordingError("every published row needs a non-empty task_id")
        if task_id in published:
            raise CrossWordingError(f"published benchmark repeats task_id {task_id}")
        published[task_id] = {stratum: row[stratum] for stratum in STRATUM_FIELDS}
    return published, _file_hash(path.expanduser().resolve())


def _load_wording_sources(path: Path, task_ids: set[str]) -> tuple[dict[str, str], str]:
    """Bind every published task to the wording that produced its user turns."""
    rows = _read_columns(
        path,
        ("task_id", "base_task_id", "source"),
        "rendered conversations",
    )
    sources: dict[str, str] = {}
    base_by_task: dict[str, Any] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id not in task_ids:
            continue
        source = row["source"]
        if source not in WORDING_SOURCES:
            raise CrossWordingError(
                f"task {task_id} declares an unknown wording source {source!r}; "
                f"expected one of {list(WORDING_SOURCES)}"
            )
        if task_id in sources and sources[task_id] != source:
            raise CrossWordingError(f"task {task_id} maps to two wording sources")
        sources[task_id] = source
        base_by_task[task_id] = row["base_task_id"]
    missing = sorted(task_ids - set(sources))
    if missing:
        raise CrossWordingError(
            "rendered conversations do not cover every published task; "
            f"{len(missing)} missing, first: {missing[0]}"
        )
    return sources, _file_hash(path.expanduser().resolve())


def _paired_wording_availability(
    path: Path,
    task_ids: set[str],
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """Report whether any skeleton is published under both wordings."""
    rows = _read_columns(path, ("task_id", "base_task_id", "source"), "rendered conversations")
    published_base = {row["base_task_id"] for row in rows if row["task_id"] in task_ids}
    wordings_by_base: dict[Any, set[str]] = defaultdict(set)
    for row in rows:
        if row["task_id"] in task_ids:
            wordings_by_base[row["base_task_id"]].add(sources[row["task_id"]])
    both = sorted(str(base) for base, found in wordings_by_base.items() if len(found) > 1)
    available_unscored = {
        row["base_task_id"]
        for row in rows
        if row["task_id"] not in task_ids and row["base_task_id"] in published_base
    }
    return {
        "status": "available" if both else "not_available",
        "paired_base_task_count": len(both),
        "published_base_task_count": len(published_base),
        "counterpart_wording_rendered_but_unscored": len(available_unscored),
        "missing_artifact": (
            None
            if both
            else (
                "an evaluation of the counterpart wording of each published skeleton; "
                "the rendered variants exist in the release but were never scored"
            )
        ),
    }


def _load_scored_run(run: ScoredRun, task_ids: set[str]) -> tuple[dict[str, bool], dict[str, Any]]:
    rows = _read_columns(
        run.task_results,
        ("task_id", "task_success", "non_candidate_stop", "failure_codes"),
        f"{run.role} run {run.run_id} task results",
    )
    verdicts: dict[str, bool] = {}
    failures: dict[str, int] = defaultdict(int)
    non_candidate_stops = 0
    for row in rows:
        task_id = row["task_id"]
        if task_id in verdicts:
            raise CrossWordingError(f"{run.run_id} repeats task_id {task_id}")
        success = row["task_success"]
        if not isinstance(success, bool):
            raise CrossWordingError(f"{run.run_id} task {task_id} has a non-boolean task_success")
        verdicts[task_id] = success
        if row["non_candidate_stop"]:
            non_candidate_stops += 1
        for code in row["failure_codes"] or ():
            failures[str(code)] += 1
    if set(verdicts) != task_ids:
        extra = sorted(set(verdicts) - task_ids)
        absent = sorted(task_ids - set(verdicts))
        raise CrossWordingError(
            f"{run.run_id} does not score exactly the published task set "
            f"({len(absent)} unscored, {len(extra)} unexpected)"
        )
    metadata = {
        "run_id": run.run_id,
        "role": run.role,
        "task_results_file": run.task_results.name,
        "task_results_hash": _file_hash(run.task_results.expanduser().resolve()),
        "scored_tasks": len(verdicts),
        "successes": sum(verdicts.values()),
        "non_candidate_stops": non_candidate_stops,
        "failure_codes": dict(sorted(failures.items())),
    }
    return verdicts, metadata


def _paired_comparison(
    left: Mapping[str, bool],
    right: Mapping[str, bool],
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Agreement and exact McNemar over tasks scored by both runs."""
    both_pass = both_fail = left_only = right_only = 0
    flipped: list[str] = []
    for task_id in task_ids:
        a, b = left[task_id], right[task_id]
        if a and b:
            both_pass += 1
        elif not a and not b:
            both_fail += 1
        elif a:
            left_only += 1
            flipped.append(task_id)
        else:
            right_only += 1
            flipped.append(task_id)
    total = len(task_ids)
    discordant = left_only + right_only
    return {
        "paired_tasks": total,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "left_only_pass": left_only,
        "right_only_pass": right_only,
        "discordant_pairs": discordant,
        "agreement": _round((both_pass + both_fail) / total),
        "verdict_flip_rate": _round(discordant / total),
        "net_success_delta": _round((right_only - left_only) / total),
        "mcnemar_exact_p": _round(mcnemar_exact_p(left_only, right_only)),
        "flipped_task_ids": sorted(flipped),
    }


def _group_contrast(
    counts: Mapping[str, tuple[int, int]],
    z: float,
) -> dict[str, Any]:
    """Contrast model wording against human wording within one group."""
    human = counts.get(HUMAN_WORDING_SOURCE, (0, 0))
    model = counts.get(MODEL_WORDING_SOURCE, (0, 0))
    record: dict[str, Any] = {
        "human_successes": human[0],
        "human_tasks": human[1],
        "model_successes": model[0],
        "model_tasks": model[1],
        "human_success_rate": None,
        "model_success_rate": None,
        "absolute_delta": None,
        "confidence_interval": None,
        "p_value": None,
        "status": "not_comparable",
    }
    if human[1] == 0 or model[1] == 0:
        record["reason"] = (
            "only one wording is published in this group, so no contrast exists"
        )
        return record
    record["human_success_rate"] = _round(human[0] / human[1])
    record["model_success_rate"] = _round(model[0] / model[1])
    record["absolute_delta"] = _round(model[0] / model[1] - human[0] / human[1])
    low, high = newcombe_difference_interval(model, human, z)
    record["confidence_interval"] = [_round(low), _round(high)]
    record["p_value"] = _round(two_proportion_test(model, human))
    record["status"] = "separated_from_zero" if not low <= 0.0 <= high else "overlaps_zero"
    return record


def _wording_contrast(
    verdicts: Mapping[str, bool],
    sources: Mapping[str, str],
    published: Mapping[str, Mapping[str, Any]],
    z: float,
) -> dict[str, Any]:
    overall: dict[str, tuple[int, int]] = {}
    for source in WORDING_SOURCES:
        tasks = [task_id for task_id, value in sources.items() if value == source]
        overall[source] = (sum(verdicts[task_id] for task_id in tasks), len(tasks))

    strata: dict[str, list[dict[str, Any]]] = {}
    for stratum in STRATUM_FIELDS:
        grouped: dict[Any, dict[str, list[int]]] = defaultdict(
            lambda: {source: [0, 0] for source in WORDING_SOURCES}
        )
        for task_id, source in sources.items():
            bucket = grouped[published[task_id][stratum]][source]
            bucket[0] += int(verdicts[task_id])
            bucket[1] += 1
        strata[stratum] = [
            {"group": str(group)}
            | _group_contrast(
                {source: (value[0], value[1]) for source, value in counts.items()},
                z,
            )
            for group, counts in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
    return {
        "design": "unpaired_observational",
        "confound": (
            "which skeletons were published under which wording was decided by dedup "
            "and balancing, not by assignment, so a group difference mixes the wording "
            "effect with the selection that produced the groups"
        ),
        "overall": _group_contrast(overall, z),
        "strata": strata,
    }


def _saturated_groups(contrast: Mapping[str, Any]) -> list[str]:
    """Groups where every task passes or every task fails under both wordings.

    A group at a floor or ceiling cannot move under any intervention, so a zero
    delta there is not evidence that wording does not matter.
    """
    saturated: list[str] = []
    for entries in contrast["strata"].values():
        for entry in entries:
            if entry["status"] == "not_comparable":
                continue
            successes = entry["human_successes"] + entry["model_successes"]
            tasks = entry["human_tasks"] + entry["model_tasks"]
            if successes in (0, tasks):
                saturated.append(entry["group"])
    return sorted(set(saturated))


def _stability_conclusion(
    paired_wording: Mapping[str, Any],
    replicate: Mapping[str, Any] | None,
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    findings: list[str] = []
    if replicate is not None:
        findings.append(
            f"a replicate of the same task set by the same model flips "
            f"{replicate['discordant_pairs']} of {replicate['paired_tasks']} verdicts "
            f"(flip rate {replicate['verdict_flip_rate']}), which is the floor any "
            "wording effect has to clear"
        )
    separated = [
        f"{stratum}={entry['group']} (delta {entry['absolute_delta']})"
        for stratum, entries in contrast["strata"].items()
        for entry in entries
        if entry["status"] == "separated_from_zero"
    ]
    if separated:
        findings.append(
            "unpaired groups whose interval excludes zero, which locates where a "
            f"paired run should look first: {separated}"
        )
    saturated = _saturated_groups(contrast)
    if saturated:
        findings.append(
            "groups saturated at a floor or ceiling under both wordings, where a zero "
            f"delta carries no information: {saturated}"
        )
    if paired_wording["status"] != "available":
        return {
            "conclusion": "underpowered",
            "reason": (
                "no skeleton is published under both wordings, so the paired design "
                "SOV-862 specifies cannot be evaluated; the unpaired contrast is "
                "confounded by publication selection and cannot substitute for it"
            ),
            "findings": findings,
            "required_to_conclude": [
                cast(str, paired_wording["missing_artifact"]),
                "a per-policy paired readout over the same pinned evaluator and scoring config",
            ],
        }
    return {
        "conclusion": "not_measured",
        "reason": (
            "paired wording data is available but this contract has not been given a "
            "paired scored run to analyse"
        ),
        "findings": findings,
        "required_to_conclude": [
            "supply the counterpart-wording run as a paired input",
        ],
    }


def build_cross_wording_report(inputs: CrossWordingInputs) -> dict[str, Any]:
    """Measure what the frozen release can support about wording stability."""
    z = critical_z(inputs.confidence_level)
    published, published_hash = _load_published(inputs.published)
    task_ids = set(published)
    sources, rendered_hash = _load_wording_sources(inputs.rendered_conversations, task_ids)
    paired_wording = _paired_wording_availability(inputs.rendered_conversations, task_ids, sources)

    primary_verdicts, primary_meta = _load_scored_run(inputs.primary_run, task_ids)
    replicates: list[dict[str, Any]] = []
    replicate_summary: dict[str, Any] | None = None
    ordered_tasks = sorted(task_ids)
    seen_runs = {inputs.primary_run.run_id}
    for run in inputs.replicate_runs:
        if run.run_id in seen_runs:
            raise CrossWordingError(f"replicate run_id {run.run_id} is not unique")
        seen_runs.add(run.run_id)
        verdicts, metadata = _load_scored_run(run, task_ids)
        if metadata["task_results_hash"] == primary_meta["task_results_hash"]:
            raise CrossWordingError(
                f"replicate {run.run_id} is byte-identical to the primary run, "
                "so it cannot measure a replicate floor"
            )
        comparison = _paired_comparison(primary_verdicts, verdicts, ordered_tasks)
        replicates.append({"run": metadata, "paired": comparison})
        if replicate_summary is None or comparison["discordant_pairs"] > replicate_summary["discordant_pairs"]:
            replicate_summary = comparison

    contrast = _wording_contrast(primary_verdicts, sources, published, z)
    conclusion = _stability_conclusion(paired_wording, replicate_summary, contrast)

    report: dict[str, Any] = {
        "schema_version": CROSS_WORDING_CONTRACT_VERSION,
        "report_hash": None,
        "source": {
            "published_file": inputs.published.name,
            "published_hash": published_hash,
            "published_tasks": len(task_ids),
            "published_task_ids_hash": _semantic_hash(ordered_tasks),
            "rendered_conversations_file": inputs.rendered_conversations.name,
            "rendered_conversations_hash": rendered_hash,
        },
        "policy": {
            "confidence_level": inputs.confidence_level,
            "human_wording_source": HUMAN_WORDING_SOURCE,
            "model_wording_source": MODEL_WORDING_SOURCE,
            "causal_claim": False,
        },
        "wording_distribution": {
            source: sum(1 for value in sources.values() if value == source)
            for source in WORDING_SOURCES
        },
        "primary_run": primary_meta,
        "paired_wording_design": paired_wording,
        "replicate_floor": {
            "status": "measured" if replicates else "not_measured",
            "runs": replicates,
        },
        "unpaired_wording_contrast": contrast,
        "stability_conclusion": conclusion,
    }
    report["report_hash"] = _semantic_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    validate_cross_wording_report(report)
    return report


def validate_cross_wording_report(report: Mapping[str, Any]) -> None:
    """Re-derive every claim the report makes about itself."""
    document = dict(report)
    if document.get("schema_version") != CROSS_WORDING_CONTRACT_VERSION:
        raise CrossWordingError("cross-wording report schema_version is unsupported")
    claimed = document.get("report_hash")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        raise CrossWordingError("cross-wording report_hash must be sha256:<64 hex>")
    unsigned = {key: value for key, value in document.items() if key != "report_hash"}
    if claimed != _semantic_hash(unsigned):
        raise CrossWordingError("cross-wording report_hash mismatch")
    conclusion = document["stability_conclusion"]["conclusion"]
    if conclusion not in STABILITY_CONCLUSIONS:
        raise CrossWordingError(f"unsupported stability conclusion: {conclusion}")
    if document["policy"]["causal_claim"] is not False:
        raise CrossWordingError("this contract cannot publish a causal claim")
    if (
        document["paired_wording_design"]["status"] != "available"
        and conclusion in ("stable", "unstable")
    ):
        raise CrossWordingError(
            "a stability verdict requires the paired wording design; the unpaired "
            "contrast cannot support one"
        )


def render_cross_wording_markdown(report: Mapping[str, Any]) -> str:
    """Render the reviewer-facing stability readout."""
    validate_cross_wording_report(report)
    source = report["source"]
    conclusion = report["stability_conclusion"]
    paired = report["paired_wording_design"]
    contrast = report["unpaired_wording_contrast"]
    lines = [
        "# BFCL cross-wording stability analysis",
        "",
        f"- Report hash: `{report['report_hash']}`",
        f"- Published tasks: {source['published_tasks']}",
        f"- Published table hash: `{source['published_hash']}`",
        f"- Primary run: `{report['primary_run']['run_id']}`",
        f"- Stability conclusion: **{conclusion['conclusion']}**",
        "",
        "## Why this is the conclusion",
        "",
        f"- {conclusion['reason']}",
    ]
    for finding in conclusion["findings"]:
        lines.append(f"- Finding: {finding}")
    for requirement in conclusion["required_to_conclude"]:
        lines.append(f"- Required to conclude: {requirement}")

    lines.extend(
        [
            "",
            "## Paired wording design",
            "",
            f"- Status: `{paired['status']}`",
            f"- Published skeletons: {paired['published_base_task_count']}",
            f"- Skeletons published under both wordings: {paired['paired_base_task_count']}",
            "- Counterpart wordings rendered but never scored: "
            f"{paired['counterpart_wording_rendered_but_unscored']}",
        ]
    )

    lines.extend(["", "## Replicate verdict-flip floor", ""])
    if report["replicate_floor"]["status"] != "measured":
        lines.append("- Not measured: no replicate run was supplied.")
    else:
        lines.extend(
            [
                "| Replicate | Paired | Agreement | Flip rate | Discordant | Net delta | McNemar p |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report["replicate_floor"]["runs"]:
            paired_item = item["paired"]
            lines.append(
                "| `{run}` | {n} | {agree} | {flip} | {disc} | {net} | {p} |".format(
                    run=item["run"]["run_id"],
                    n=paired_item["paired_tasks"],
                    agree=paired_item["agreement"],
                    flip=paired_item["verdict_flip_rate"],
                    disc=paired_item["discordant_pairs"],
                    net=paired_item["net_success_delta"],
                    p=paired_item["mcnemar_exact_p"],
                )
            )

    overall = contrast["overall"]
    lines.extend(
        [
            "",
            "## Unpaired wording contrast",
            "",
            f"- Design: `{contrast['design']}`",
            f"- Confound: {contrast['confound']}",
            "",
            "| Group | Human | Model | Delta | 95% CI | p | Status |",
            "|---|---|---|---:|---|---:|---|",
            "| **overall** | {h}/{hn} | {m}/{mn} | {d} | {ci} | {p} | {s} |".format(
                h=overall["human_successes"],
                hn=overall["human_tasks"],
                m=overall["model_successes"],
                mn=overall["model_tasks"],
                d=overall["absolute_delta"],
                ci=(
                    "n/a"
                    if overall["confidence_interval"] is None
                    else f"[{overall['confidence_interval'][0]}, {overall['confidence_interval'][1]}]"
                ),
                p="n/a" if overall["p_value"] is None else overall["p_value"],
                s=overall["status"],
            ),
        ]
    )
    for entry in contrast["strata"]["turn_policy"]:
        lines.append(
            "| `{group}` | {h}/{hn} | {m}/{mn} | {d} | {ci} | {p} | {s} |".format(
                group=entry["group"],
                h=entry["human_successes"],
                hn=entry["human_tasks"],
                m=entry["model_successes"],
                mn=entry["model_tasks"],
                d="n/a" if entry["absolute_delta"] is None else entry["absolute_delta"],
                ci=(
                    "n/a"
                    if entry["confidence_interval"] is None
                    else f"[{entry['confidence_interval'][0]}, {entry['confidence_interval'][1]}]"
                ),
                p="n/a" if entry["p_value"] is None else entry["p_value"],
                s=entry["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- The unpaired contrast is a measurement, not an assignment. It cannot "
            "separate the wording effect from the balancing that chose the groups.",
            "- Causal claim: no.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_cross_wording_report(
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown atomically; never replace different report bytes."""
    validate_cross_wording_report(report)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_bytes = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = render_cross_wording_markdown(report).encode("utf-8")
    json_path = root / "cross_wording_report.json"
    markdown_path = root / "cross_wording_report.md"
    for path, content in ((json_path, json_bytes), (markdown_path, markdown_bytes)):
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise CrossWordingError(f"refusing to replace a different report: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return json_path, markdown_path


__all__ = [
    "CROSS_WORDING_CONTRACT_VERSION",
    "CrossWordingError",
    "CrossWordingInputs",
    "STABILITY_CONCLUSIONS",
    "STRATUM_FIELDS",
    "WORDING_SOURCES",
    "ScoredRun",
    "build_cross_wording_report",
    "render_cross_wording_markdown",
    "validate_cross_wording_report",
    "write_cross_wording_report",
]
