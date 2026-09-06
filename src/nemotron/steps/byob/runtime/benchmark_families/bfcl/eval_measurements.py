"""Turn BFCL evaluation artifacts into ablation-ladder measurements.

SOV-866 requires the ablation summary to speak to task success, failure codes,
cost and latency. Those four families only exist once a candidate model has been
scored, and the numbers live scattered across an evaluation run: per-task
verdicts in the results table, token usage and per-call latency in the candidate
IO cache, published aggregates in the evaluation report.

Transcribing them by hand would put unverifiable numbers into a
content-addressed report, so this module derives them and records the hash of
every file it read. Two properties matter more than convenience:

- it *reconciles* the task success rate it computes from the per-task table
  against the rate the evaluation report already published, and refuses to emit
  anything when they disagree, because that disagreement means the artifacts
  supplied do not belong to the same run;
- it refuses to average over missing data. A completion without token usage, or
  a selected attempt without a latency sample, aborts the extraction instead of
  contributing an implicit zero.

Latency is reported but flagged. It was observed against a shared inference
endpoint under unknown concurrent load, so it characterises that afternoon's
serving conditions, not the pipeline. The flag travels with the number.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_statistics import (
    StatisticsError,
    round_statistic,
)

EVAL_MEASUREMENT_VERSION: Final = "1.0"

# The rate recomputed here and the rate the evaluation report published must
# agree to the last place a proportion of this size can carry.
_RECONCILIATION_TOLERANCE: Final = 1e-9

LATENCY_CAVEAT: Final = (
    "observed against a shared inference endpoint under unknown concurrent load, so "
    "it describes the serving conditions of that run and not a property of the "
    "pipeline; it must not be compared across runs measured at different times"
)


class EvalMeasurementError(ValueError):
    """An evaluation run cannot support trustworthy ladder measurements."""


@dataclass(frozen=True)
class EvalMeasurementInputs:
    run_id: str
    task_results: Path
    candidate_io_cache: Path
    eval_report: Path
    output_dir: Path


def _round(value: float) -> float:
    try:
        return round_statistic(value)
    except StatisticsError as exc:
        raise EvalMeasurementError(str(exc)) from exc


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


def _resolve(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EvalMeasurementError(f"{label} does not exist: {resolved}")
    return resolved


def _nearest_rank(sorted_values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile: no interpolation, so no platform drift."""
    if not sorted_values:
        raise EvalMeasurementError("a percentile needs at least one sample")
    index = math.ceil(quantile * len(sorted_values)) - 1
    return sorted_values[min(max(index, 0), len(sorted_values) - 1)]


def _load_task_results(path: Path) -> tuple[dict[str, bool], dict[str, int]]:
    resolved = _resolve(path, "task results")
    try:
        rows = pq.read_table(
            resolved, columns=["task_id", "task_success", "failure_codes"]
        ).to_pylist()
    except (OSError, ValueError, KeyError) as exc:
        raise EvalMeasurementError(f"task results are missing required columns: {exc}") from exc
    if not rows:
        raise EvalMeasurementError("task results are empty")
    verdicts: dict[str, bool] = {}
    failure_codes: dict[str, int] = defaultdict(int)
    for row in rows:
        task_id = row["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise EvalMeasurementError("every scored row needs a non-empty task_id")
        if task_id in verdicts:
            raise EvalMeasurementError(f"task results repeat task_id {task_id}")
        success = row["task_success"]
        if not isinstance(success, bool):
            raise EvalMeasurementError(f"task {task_id} has a non-boolean task_success")
        verdicts[task_id] = success
        for code in row["failure_codes"] or ():
            failure_codes[str(code)] += 1
    return verdicts, dict(sorted(failure_codes.items()))


def _load_usage_and_latency(path: Path) -> tuple[int, int, list[float], int]:
    """Sum token usage and collect the latency of each selected attempt."""
    resolved = _resolve(path, "candidate IO cache")
    prompt_tokens = completion_tokens = 0
    latencies: list[float] = []
    completions = 0
    with resolved.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalMeasurementError(
                    f"candidate IO cache line {number} is not valid JSON"
                ) from exc
            if record.get("record_type") != "completion":
                continue
            completions += 1
            outcome = record.get("payload", {}).get("outcome")
            if not isinstance(outcome, dict):
                raise EvalMeasurementError(f"completion on line {number} has no outcome")
            response = outcome.get("response")
            if not isinstance(response, dict):
                raise EvalMeasurementError(f"completion on line {number} has no response")
            usage = response.get("usage")
            if not isinstance(usage, dict):
                raise EvalMeasurementError(
                    f"completion on line {number} reports no token usage; refusing to "
                    "sum a cost that would treat it as zero"
                )
            for field, target in (("prompt_tokens", "prompt"), ("completion_tokens", "completion")):
                value = usage.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise EvalMeasurementError(
                        f"completion on line {number} has a non-integer {field}"
                    )
                if target == "prompt":
                    prompt_tokens += value
                else:
                    completion_tokens += value
            selected = response.get("selected_attempt")
            attempts = outcome.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise EvalMeasurementError(f"completion on line {number} records no attempt")
            chosen = [item for item in attempts if item.get("attempt_index") == selected]
            if len(chosen) != 1:
                raise EvalMeasurementError(
                    f"completion on line {number} does not identify exactly one selected attempt"
                )
            latency = chosen[0].get("latency_s")
            if not isinstance(latency, (int, float)) or isinstance(latency, bool):
                raise EvalMeasurementError(
                    f"completion on line {number} has no latency sample; refusing to "
                    "summarise a latency distribution with a hole in it"
                )
            if not math.isfinite(float(latency)) or float(latency) < 0.0:
                raise EvalMeasurementError(f"completion on line {number} has an invalid latency")
            latencies.append(float(latency))
    if completions == 0:
        raise EvalMeasurementError("candidate IO cache holds no completion record")
    return prompt_tokens, completion_tokens, sorted(latencies), completions


def _published_task_success(path: Path) -> tuple[float, int, int, str]:
    resolved = _resolve(path, "evaluation report")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvalMeasurementError(f"evaluation report is not valid JSON: {resolved}") from exc
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise EvalMeasurementError(
            "reconciliation needs an evaluation report with exactly one candidate"
        )
    candidate = candidates[0]
    metric = (candidate.get("metrics") or {}).get("task_success_rate")
    if not isinstance(metric, dict) or metric.get("value") is None:
        raise EvalMeasurementError("evaluation report publishes no task_success_rate to reconcile")
    return (
        float(metric["value"]),
        int(metric["numerator"]),
        int(metric["denominator"]),
        str(candidate.get("canonical_id", "")),
    )


def build_eval_measurements(inputs: EvalMeasurementInputs) -> dict[str, Any]:
    """Derive the evaluation-side ladder measurements for one scored run."""
    verdicts, failure_codes = _load_task_results(inputs.task_results)
    prompt_tokens, completion_tokens, latencies, completions = _load_usage_and_latency(
        inputs.candidate_io_cache
    )
    published_rate, published_numerator, published_denominator, canonical_id = (
        _published_task_success(inputs.eval_report)
    )

    successes = sum(verdicts.values())
    scored = len(verdicts)
    if (successes, scored) != (published_numerator, published_denominator):
        raise EvalMeasurementError(
            f"task success disagrees with the evaluation report: table says "
            f"{successes}/{scored}, report says {published_numerator}/{published_denominator}; "
            "these artifacts do not describe the same run"
        )
    if abs(successes / scored - published_rate) > _RECONCILIATION_TOLERANCE:
        raise EvalMeasurementError(
            "recomputed task success rate does not match the published rate"
        )

    report: dict[str, Any] = {
        "schema_version": EVAL_MEASUREMENT_VERSION,
        "report_hash": None,
        "run_id": inputs.run_id,
        "candidate_canonical_id": canonical_id,
        "source": {
            "task_results_file": inputs.task_results.name,
            "task_results_hash": _file_hash(_resolve(inputs.task_results, "task results")),
            "candidate_io_cache_file": inputs.candidate_io_cache.name,
            "candidate_io_cache_hash": _file_hash(
                _resolve(inputs.candidate_io_cache, "candidate IO cache")
            ),
            "eval_report_file": inputs.eval_report.name,
            "eval_report_hash": _file_hash(_resolve(inputs.eval_report, "evaluation report")),
        },
        "task_set": {
            "task_count": scored,
            "task_ids_hash": _semantic_hash(sorted(verdicts)),
        },
        "reconciliation": {
            "recomputed_successes": successes,
            "published_successes": published_numerator,
            "recomputed_denominator": scored,
            "published_denominator": published_denominator,
            "agrees": True,
        },
        "measurements": [
            {
                "metric_id": "task_success_rate",
                "kind": "proportion",
                "numerator": successes,
                "denominator": scored,
            },
            {"metric_id": "input_tokens", "kind": "deterministic", "value": prompt_tokens},
            {"metric_id": "output_tokens", "kind": "deterministic", "value": completion_tokens},
            {
                "metric_id": "latency_p50_ms",
                "kind": "deterministic",
                "value": _round(_nearest_rank(latencies, 0.50) * 1000.0),
            },
            {
                "metric_id": "latency_p95_ms",
                "kind": "deterministic",
                "value": _round(_nearest_rank(latencies, 0.95) * 1000.0),
            },
        ],
        "failure_codes": failure_codes,
        "latency_context": {
            "samples": len(latencies),
            "model_calls": completions,
            "environment_dependent": True,
            "caveat": LATENCY_CAVEAT,
        },
    }
    report["report_hash"] = _semantic_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    validate_eval_measurements(report)
    return report


def validate_eval_measurements(report: Mapping[str, Any]) -> None:
    """Re-derive every claim the extraction makes about itself."""
    document = dict(report)
    if document.get("schema_version") != EVAL_MEASUREMENT_VERSION:
        raise EvalMeasurementError("eval measurement schema_version is unsupported")
    claimed = document.get("report_hash")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        raise EvalMeasurementError("eval measurement report_hash must be sha256:<64 hex>")
    unsigned = {key: value for key, value in document.items() if key != "report_hash"}
    if claimed != _semantic_hash(unsigned):
        raise EvalMeasurementError("eval measurement report_hash mismatch")
    if document["reconciliation"]["agrees"] is not True:
        raise EvalMeasurementError("refusing to publish measurements that failed reconciliation")
    if document["latency_context"]["environment_dependent"] is not True:
        raise EvalMeasurementError(
            "latency measured against a shared endpoint cannot be declared "
            "environment-independent"
        )
    metric_ids = [item["metric_id"] for item in document["measurements"]]
    if len(metric_ids) != len(set(metric_ids)):
        raise EvalMeasurementError("eval measurements repeat a metric")


def render_eval_measurements_markdown(report: Mapping[str, Any]) -> str:
    validate_eval_measurements(report)
    latency = report["latency_context"]
    lines = [
        f"# BFCL evaluation measurements — `{report['run_id']}`",
        "",
        f"- Report hash: `{report['report_hash']}`",
        f"- Candidate: `{report['candidate_canonical_id']}`",
        f"- Scored tasks: {report['task_set']['task_count']}",
        f"- Model calls: {latency['model_calls']}",
        "",
        "## Measurements",
        "",
        "| Metric | Kind | Value |",
        "|---|---|---:|",
    ]
    for item in report["measurements"]:
        value = (
            f"{item['numerator']}/{item['denominator']}"
            if item["kind"] == "proportion"
            else item["value"]
        )
        lines.append(f"| `{item['metric_id']}` | {item['kind']} | {value} |")
    lines.extend(
        [
            "",
            "## Failure codes",
            "",
            "| Code | Tasks |",
            "|---|---:|",
        ]
    )
    for code, count in report["failure_codes"].items():
        lines.append(f"| `{code}` | {count} |")
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            "The task success rate recomputed from the per-task table agrees with the rate "
            "the evaluation report published "
            f"({report['reconciliation']['recomputed_successes']}/"
            f"{report['reconciliation']['recomputed_denominator']}).",
            "",
            "## Latency caveat",
            "",
            f"- {latency['caveat']}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_eval_measurements(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    validate_eval_measurements(report)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"eval_measurements_{report['run_id']}"
    json_bytes = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown_bytes = render_eval_measurements_markdown(report).encode("utf-8")
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    for path, content in ((json_path, json_bytes), (markdown_path, markdown_bytes)):
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise EvalMeasurementError(f"refusing to replace a different report: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return json_path, markdown_path


def ladder_measurements(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the measurement records in the shape a ladder arm declares."""
    validate_eval_measurements(report)
    return [dict(cast(Mapping[str, Any], item)) for item in report["measurements"]]


__all__ = [
    "EVAL_MEASUREMENT_VERSION",
    "LATENCY_CAVEAT",
    "EvalMeasurementError",
    "EvalMeasurementInputs",
    "build_eval_measurements",
    "ladder_measurements",
    "render_eval_measurements_markdown",
    "validate_eval_measurements",
    "write_eval_measurements",
]
