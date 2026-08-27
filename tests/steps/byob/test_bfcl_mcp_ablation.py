from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.mcp.ablation import (
    AblationError,
    build_ablation_report,
    load_ablation_input,
    write_ablation_report,
)

_DIGEST = "sha256:" + "a" * 64


def _input() -> dict:
    common = {
        "validation_pass_rate": 1.0,
        "tool_coverage": 1.0,
        "replay_stability": 1.0,
        "benchmark_rows": 100,
        "evaluation_score": 0.75,
        "evaluation_score_stderr": 0.01,
    }
    flow_effort = {
        "manual": (20, 120.0, 30.0),
        "llm_backend": (8, 40.0, 45.0),
        "llm_mcp": (3, 20.0, 50.0),
    }
    observations = []
    sequence = 1
    for repetition in range(1, 4):
        for flow in ("manual", "llm_backend", "llm_mcp"):
            fields, authoring, review = flow_effort[flow]
            observations.append(
                {
                    **common,
                    "flow": flow,
                    "repetition": repetition,
                    "sequence": sequence,
                    "run_digest": f"sha256:{sequence:064x}",
                    "user_authored_fields": fields,
                    "authoring_minutes": authoring,
                    "review_minutes": review,
                }
            )
            sequence += 1
    return {
        "schema_version": "bfcl-onboarding-ablation-input-v2",
        "experiment_id": "inventory-v1",
        "domain_artifact_digest": _DIGEST,
        "evaluator_model": "test/evaluator@1",
        "evaluation_config_digest": _DIGEST,
        "held_out_policy_digest": _DIGEST,
        "repetitions_per_flow": 3,
        "observations": observations,
    }


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_three_flow_ablation_is_deterministic_and_uses_manual_baseline(
    tmp_path: Path,
) -> None:
    source = load_ablation_input(_write(tmp_path / "input.json", _input()))

    first = build_ablation_report(source)
    second = build_ablation_report(source)
    output = write_ablation_report(first, tmp_path / "report.json")

    assert first == second
    assert output.is_file()
    assert [item["flow"] for item in first["flows"]] == [
        "manual",
        "llm_backend",
        "llm_mcp",
    ]
    assert first["flows"][2]["delta_vs_manual"]["total_human_minutes"] == -80.0
    assert first["flows"][2]["repetitions"] == 3
    assert len(first["flows"][2]["runs"]) == 3
    assert first["comparison_contract"]["causal_claim"] is False


def test_ablation_requires_exactly_one_of_each_comparable_flow(tmp_path: Path) -> None:
    document = _input()
    document["observations"][2]["flow"] = "manual"

    with pytest.raises(AblationError, match="repetitions 1, 2, and 3"):
        load_ablation_input(_write(tmp_path / "duplicate-flow.json", document))


def test_ablation_refuses_duplicate_keys_nonfinite_values_and_digest_drift(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"bfcl-onboarding-ablation-input-v1",'
        '"schema_version":"bfcl-onboarding-ablation-input-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(AblationError, match="repeats JSON key"):
        load_ablation_input(duplicate)

    document = _input()
    document["observations"][0]["authoring_minutes"] = float("nan")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(AblationError, match="non-finite"):
        load_ablation_input(nonfinite)

    report = build_ablation_report(
        load_ablation_input(_write(tmp_path / "valid.json", _input()))
    )
    report["flows"][0]["benchmark_rows"] = 99
    with pytest.raises(AblationError, match="report_digest mismatch"):
        write_ablation_report(report, tmp_path / "report.json")


def test_protocol_requires_complete_order_unique_runs_and_uniform_scoring(
    tmp_path: Path,
) -> None:
    bad_sequence = _input()
    bad_sequence["observations"][0]["sequence"] = 2
    with pytest.raises(AblationError, match="sequence"):
        load_ablation_input(_write(tmp_path / "sequence.json", bad_sequence))

    duplicate_run = _input()
    duplicate_run["observations"][0]["run_digest"] = duplicate_run["observations"][1][
        "run_digest"
    ]
    with pytest.raises(AblationError, match="run_digest must be unique"):
        load_ablation_input(_write(tmp_path / "duplicate-run.json", duplicate_run))

    partial_scores = _input()
    partial_scores["observations"][0]["evaluation_score"] = None
    partial_scores["observations"][0]["evaluation_score_stderr"] = None
    with pytest.raises(AblationError, match="all nine"):
        load_ablation_input(_write(tmp_path / "partial-score.json", partial_scores))
