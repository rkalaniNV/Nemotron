"""Tests for the BFCL evaluation-measurement extraction contract (SOV-866)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval_measurements import (
    EVAL_MEASUREMENT_VERSION,
    EvalMeasurementError,
    EvalMeasurementInputs,
    build_eval_measurements,
    ladder_measurements,
    render_eval_measurements_markdown,
    validate_eval_measurements,
    write_eval_measurements,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = REPO_ROOT.parent
EVALUATIONS = WORKSPACE / "release-candidate" / "sov867-clean-52907cc" / "evaluations"

_PASSING = 6
_TOTAL = 10


def _task_results(path: Path, *, successes: int = _PASSING, total: int = _TOTAL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "task_id": f"task-{index:03d}",
            "task_success": index < successes,
            "failure_codes": [] if index < successes else ["arguments.mismatch"],
        }
        for index in range(total)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _completion(
    index: int,
    *,
    prompt: int = 100,
    completion: int = 20,
    latency: float | None = 0.5,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {"attempt_index": 0, "latency_s": latency}
    if latency is None:
        del attempt["latency_s"]
    return {
        "record_type": "completion",
        "payload": {
            "outcome": {
                "status": "completed",
                "attempts": [attempt],
                "response": {
                    "selected_attempt": 0,
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                    },
                },
            }
        },
    }


def _io_cache(path: Path, records: list[dict[str, Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = records if records is not None else [_completion(index) for index in range(_TOTAL)]
    # A request record is interleaved, as the real cache does, to prove the
    # extraction only counts completions.
    lines = [json.dumps({"record_type": "request", "payload": {"request": {}}})]
    lines.extend(json.dumps(record) for record in payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _eval_report(path: Path, *, successes: int = _PASSING, total: int = _TOTAL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "canonical_id": "huggingface:acme/model@abc",
                        "metrics": {
                            "task_success_rate": {
                                "numerator": successes,
                                "denominator": total,
                                "value": successes / total,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path, **overrides: Any) -> EvalMeasurementInputs:
    defaults: dict[str, Any] = {
        "run_id": "run-a",
        "task_results": _task_results(tmp_path / "eval_task_results.parquet"),
        "candidate_io_cache": _io_cache(tmp_path / "candidate_io_cache.jsonl"),
        "eval_report": _eval_report(tmp_path / "eval_report.json"),
        "output_dir": tmp_path / "out",
    }
    return EvalMeasurementInputs(**(defaults | overrides))


def _metric(report: dict[str, Any], metric_id: str) -> dict[str, Any]:
    return next(item for item in report["measurements"] if item["metric_id"] == metric_id)


def test_extraction_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = build_eval_measurements(inputs)
    second = build_eval_measurements(inputs)

    assert first == second
    assert first["schema_version"] == EVAL_MEASUREMENT_VERSION
    validate_eval_measurements(first)


def test_measurements_are_derived_from_the_artifacts(tmp_path: Path) -> None:
    report = build_eval_measurements(_inputs(tmp_path))

    assert _metric(report, "task_success_rate")["numerator"] == _PASSING
    assert _metric(report, "task_success_rate")["denominator"] == _TOTAL
    assert _metric(report, "input_tokens")["value"] == 100 * _TOTAL
    assert _metric(report, "output_tokens")["value"] == 20 * _TOTAL
    assert _metric(report, "latency_p50_ms")["value"] == pytest.approx(500.0)
    assert report["failure_codes"] == {"arguments.mismatch": _TOTAL - _PASSING}
    assert report["latency_context"]["model_calls"] == _TOTAL


def test_a_mismatched_evaluation_report_fails_closed(tmp_path: Path) -> None:
    """Pointing at the wrong evaluation directory must not produce numbers."""
    report = _eval_report(tmp_path / "other_report.json", successes=9)
    with pytest.raises(EvalMeasurementError, match="do not describe the same run"):
        build_eval_measurements(_inputs(tmp_path, eval_report=report))


def test_missing_token_usage_is_refused_rather_than_counted_as_zero(tmp_path: Path) -> None:
    records = [_completion(index) for index in range(_TOTAL)]
    del records[3]["payload"]["outcome"]["response"]["usage"]
    cache = _io_cache(tmp_path / "no_usage.jsonl", records)
    with pytest.raises(EvalMeasurementError, match="reports no token usage"):
        build_eval_measurements(_inputs(tmp_path, candidate_io_cache=cache))


def test_missing_latency_sample_is_refused(tmp_path: Path) -> None:
    records = [_completion(index) for index in range(_TOTAL)]
    records[5] = _completion(5, latency=None)
    cache = _io_cache(tmp_path / "no_latency.jsonl", records)
    with pytest.raises(EvalMeasurementError, match="no latency sample"):
        build_eval_measurements(_inputs(tmp_path, candidate_io_cache=cache))


def test_an_ambiguous_selected_attempt_is_refused(tmp_path: Path) -> None:
    records = [_completion(index) for index in range(_TOTAL)]
    records[2]["payload"]["outcome"]["attempts"].append({"attempt_index": 0, "latency_s": 9.0})
    cache = _io_cache(tmp_path / "ambiguous.jsonl", records)
    with pytest.raises(EvalMeasurementError, match="exactly one selected attempt"):
        build_eval_measurements(_inputs(tmp_path, candidate_io_cache=cache))


def test_an_empty_cache_is_refused(tmp_path: Path) -> None:
    cache = _io_cache(tmp_path / "empty.jsonl", [])
    with pytest.raises(EvalMeasurementError, match="no completion record"):
        build_eval_measurements(_inputs(tmp_path, candidate_io_cache=cache))


def test_a_multi_candidate_report_cannot_be_reconciled(tmp_path: Path) -> None:
    path = tmp_path / "two_candidates.json"
    path.write_text(
        json.dumps({"candidates": [{"metrics": {}}, {"metrics": {}}]}), encoding="utf-8"
    )
    with pytest.raises(EvalMeasurementError, match="exactly one candidate"):
        build_eval_measurements(_inputs(tmp_path, eval_report=path))


def test_percentiles_use_nearest_rank(tmp_path: Path) -> None:
    latencies = [0.1, 0.2, 0.3, 0.4, 1.0]
    records = [_completion(index, latency=value) for index, value in enumerate(latencies)]
    report = build_eval_measurements(
        _inputs(
            tmp_path,
            task_results=_task_results(tmp_path / "five.parquet", successes=3, total=5),
            candidate_io_cache=_io_cache(tmp_path / "five.jsonl", records),
            eval_report=_eval_report(tmp_path / "five.json", successes=3, total=5),
        )
    )

    # ceil(0.50 * 5) - 1 = 2 -> 0.3s; ceil(0.95 * 5) - 1 = 4 -> 1.0s
    assert _metric(report, "latency_p50_ms")["value"] == pytest.approx(300.0)
    assert _metric(report, "latency_p95_ms")["value"] == pytest.approx(1000.0)


def test_tampering_is_detected_and_flags_cannot_be_dropped(tmp_path: Path) -> None:
    report = build_eval_measurements(_inputs(tmp_path))

    tampered = copy.deepcopy(report)
    tampered["failure_codes"] = {}
    with pytest.raises(EvalMeasurementError, match="report_hash mismatch"):
        validate_eval_measurements(tampered)

    import hashlib

    forged = copy.deepcopy(report)
    forged["latency_context"]["environment_dependent"] = False
    unsigned = {key: value for key, value in forged.items() if key != "report_hash"}
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    forged["report_hash"] = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    with pytest.raises(EvalMeasurementError, match="cannot be declared"):
        validate_eval_measurements(forged)


def test_ladder_measurements_are_in_the_shape_an_arm_declares(tmp_path: Path) -> None:
    measurements = ladder_measurements(build_eval_measurements(_inputs(tmp_path)))

    assert {item["metric_id"] for item in measurements} == {
        "task_success_rate",
        "input_tokens",
        "output_tokens",
        "latency_p50_ms",
        "latency_p95_ms",
    }
    for item in measurements:
        assert item["kind"] in ("proportion", "deterministic")
        expected = (
            {"metric_id", "kind", "numerator", "denominator"}
            if item["kind"] == "proportion"
            else {"metric_id", "kind", "value"}
        )
        assert set(item) == expected


def test_writer_refuses_to_replace_a_different_report(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = build_eval_measurements(inputs)
    json_path, markdown_path = write_eval_measurements(report, inputs.output_dir)

    assert json_path.name == "eval_measurements_run-a.json"
    assert write_eval_measurements(report, inputs.output_dir) == (json_path, markdown_path)
    markdown_path.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(EvalMeasurementError, match="refusing to replace"):
        write_eval_measurements(report, inputs.output_dir)


def test_markdown_carries_the_reconciliation_and_the_caveat(tmp_path: Path) -> None:
    markdown = render_eval_measurements_markdown(build_eval_measurements(_inputs(tmp_path)))

    assert "## Reconciliation" in markdown
    assert "shared inference endpoint" in markdown
    assert "`arguments.mismatch`" in markdown


def test_contract_document_matches_the_implemented_contract() -> None:
    text = (
        REPO_ROOT
        / "src"
        / "nemotron"
        / "steps"
        / "byob"
        / "references"
        / "bfcl-eval-measurements-contract.md"
    ).read_text(encoding="utf-8")

    assert f"contract version is `{EVAL_MEASUREMENT_VERSION}`" in text
    for metric_id in (
        "task_success_rate",
        "input_tokens",
        "output_tokens",
        "latency_p50_ms",
        "latency_p95_ms",
    ):
        assert f"`{metric_id}`" in text, metric_id


@pytest.mark.skipif(
    not (EVALUATIONS / "gptoss-120b-8k" / "candidate_io_cache.jsonl").is_file(),
    reason="frozen Banking VN evaluation artifacts are not present in this checkout",
)
def test_frozen_evaluation_measurements_are_pinned(tmp_path: Path) -> None:
    """Golden test: the published SOV-866 evaluation numbers must stay reproducible."""
    report = build_eval_measurements(
        EvalMeasurementInputs(
            run_id="gptoss-120b-8k",
            task_results=EVALUATIONS / "gptoss-120b-8k" / "eval_task_results.parquet",
            candidate_io_cache=EVALUATIONS / "gptoss-120b-8k" / "candidate_io_cache.jsonl",
            eval_report=EVALUATIONS / "gptoss-120b-8k" / "eval_report.json",
            output_dir=tmp_path,
        )
    )

    assert report["report_hash"] == (
        "sha256:2fc0606a26c308d41ead661b8a1656ee283f295b74d1a374338c79b72649f3bf"
    )
    assert _metric(report, "task_success_rate")["numerator"] == 1060
    assert _metric(report, "input_tokens")["value"] == 1804887
    assert _metric(report, "output_tokens")["value"] == 412893
    assert report["reconciliation"]["agrees"] is True
    assert sum(report["failure_codes"].values()) == 1933
