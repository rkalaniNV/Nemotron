from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.ablation_aggregation import (
    ABLATION_AGGREGATION_VERSION,
    ARM_STATUSES,
    METRIC_DEFINITIONS,
    RECOMMENDATIONS,
    REQUIRED_FAMILIES,
    AblationAggregationError,
    AggregationInputs,
    build_ablation_summary,
    load_ablation_aggregation_input,
    render_ablation_summary_markdown,
    validate_ablation_summary,
    write_ablation_summary,
)

_CONTRACT_DOC = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "references"
    / "bfcl-ablation-aggregation-contract.md"
)

_TASK_SET = "sha256:" + "a" * 64
_EVIDENCE_HASH = "sha256:" + "b" * 64


def _evidence(kind: str = "report") -> dict[str, Any]:
    return {"kind": kind, "locator": "results/A0/baseline_report.md", "content_hash": _EVIDENCE_HASH}


def _baseline_arm() -> dict[str, Any]:
    return {
        "arm_id": "A0",
        "title": "Human baseline",
        "ticket": "SOV-859",
        "status": "measured",
        "intervention": "none",
        "evidence": [_evidence()],
        "task_set_hash": _TASK_SET,
        "cost_context": {"currency": "USD", "pricing_snapshot": "2026-09-01"},
        "measurements": [
            {"metric_id": "authoring_lines", "kind": "deterministic", "value": 1642},
            {"metric_id": "distinct_surface_count", "kind": "deterministic", "value": 17},
            {"metric_id": "validation_pass_rate", "kind": "proportion", "numerator": 22, "denominator": 22},
            {"metric_id": "task_success_rate", "kind": "proportion", "numerator": 900, "denominator": 1392},
            {"metric_id": "cost_amount", "kind": "deterministic", "value": 10.0},
        ],
        "failure_codes": {"MISSING_TOOL_CALL": 300, "WRONG_TOOL_ORDER": 192},
        "truth_preservation_gates": [
            {
                "gate_id": "a0_gold_gate",
                "verdict": "passed",
                "sensitive_to_intervention": True,
                "rationale": "executable replay validates the unmodified pack",
            }
        ],
        "limitations": ["single domain"],
    }


def _simplification_arm() -> dict[str, Any]:
    return {
        "arm_id": "A1",
        "title": "Deterministic simplification",
        "ticket": "SOV-860",
        "status": "measured",
        "intervention": "derive mechanically inferable pack fields",
        "evidence": [_evidence("equivalence_report")],
        "task_set_hash": _TASK_SET,
        "cost_context": {"currency": "USD", "pricing_snapshot": "2026-09-01"},
        "measurements": [
            {"metric_id": "authoring_lines", "kind": "deterministic", "value": 1386},
            {"metric_id": "distinct_surface_count", "kind": "deterministic", "value": 17},
            {"metric_id": "validation_pass_rate", "kind": "proportion", "numerator": 22, "denominator": 22},
            {"metric_id": "task_success_rate", "kind": "proportion", "numerator": 900, "denominator": 1392},
            {"metric_id": "cost_amount", "kind": "deterministic", "value": 10.0},
        ],
        "failure_codes": {"MISSING_TOOL_CALL": 300, "WRONG_TOOL_ORDER": 192},
        "truth_preservation_gates": [
            {
                "gate_id": "a1_equivalence",
                "verdict": "passed",
                "sensitive_to_intervention": True,
                "rationale": "gate covers compiled milestones, which the intervention rewrites",
            }
        ],
        "limitations": ["three compiler rules are untested in this pack"],
    }


def _input(**overrides: Any) -> dict[str, Any]:
    document = {
        "schema_version": ABLATION_AGGREGATION_VERSION,
        "experiment_id": "bfcl-oracle-authoring-ladder-v1",
        "baseline_arm_id": "A0",
        "policy": {"confidence_level": 0.95, "multiple_comparison": "holm"},
        "arms": [_baseline_arm(), _simplification_arm()],
    }
    document.update(overrides)
    return document


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _build(tmp_path: Path, document: dict[str, Any] | None = None) -> dict[str, Any]:
    source = _write(tmp_path / "input.json", document if document is not None else _input())
    return build_ablation_summary(
        AggregationInputs(ablation_input=source, output_dir=tmp_path / "out")
    )


def _comparison(report: dict[str, Any], arm_id: str, metric_id: str) -> dict[str, Any]:
    arm = next(item for item in report["arms"] if item["arm_id"] == arm_id)
    return next(item for item in arm["comparisons"] if item["metric_id"] == metric_id)


def _resign(report: dict[str, Any]) -> str:
    """Re-sign a mutated report, so a test probes a rule and not the hash."""
    unsigned = {key: value for key, value in report.items() if key != "report_hash"}
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def test_aggregation_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    first = _build(tmp_path)
    second = _build(tmp_path)

    assert first == second
    validate_ablation_summary(first)
    json_path, markdown_path = write_ablation_summary(first, tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8")) == first
    assert markdown_path.read_text(encoding="utf-8") == render_ablation_summary_markdown(first)
    # Rewriting identical bytes is allowed; the summary is content-addressed.
    write_ablation_summary(first, tmp_path / "out")


def test_deterministic_metric_delta_is_exact_and_directional(tmp_path: Path) -> None:
    report = _build(tmp_path)

    lines = _comparison(report, "A1", "authoring_lines")
    assert lines["absolute_delta"] == -256.0
    assert lines["relative_delta"] == pytest.approx(-0.1559074300, abs=1e-9)
    # Fewer authored lines is an improvement, and a deterministic count has no
    # sampling error to rule out.
    assert lines["verdict"] == "material_improvement"
    assert lines["p_value"] is None

    surfaces = _comparison(report, "A1", "distinct_surface_count")
    assert surfaces["absolute_delta"] == 0.0
    assert surfaces["verdict"] == "no_material_change"


def test_identical_proportions_are_not_reported_as_change(tmp_path: Path) -> None:
    report = _build(tmp_path)

    success = _comparison(report, "A1", "task_success_rate")
    assert success["absolute_delta"] == 0.0
    assert success["verdict"] == "no_material_change"
    low, high = success["confidence_interval"]
    assert low <= 0.0 <= high


def test_small_proportion_shift_is_separated_from_noise(tmp_path: Path) -> None:
    document = _input()
    # A one-task difference on 1392 tasks is not distinguishable from noise.
    document["arms"][1]["measurements"][3]["numerator"] = 901
    noisy = _build(tmp_path / "noise", document)
    assert _comparison(noisy, "A1", "task_success_rate")["verdict"] == "no_material_change"

    document = _input()
    document["arms"][1]["measurements"][3]["numerator"] = 1150
    material = _build(tmp_path / "material", document)
    shift = _comparison(material, "A1", "task_success_rate")
    assert shift["verdict"] == "material_improvement"
    assert shift["adjusted_p_value"] < 0.05


def test_underpowered_repeated_measures_are_inconclusive(tmp_path: Path) -> None:
    document = _input()
    for arm, observations in (
        (document["arms"][0], [30.0, 14.6, 10.8]),
        (document["arms"][1], [6.8, 17.4, 12.3]),
    ):
        arm["measurements"].append(
            {"metric_id": "authoring_minutes", "kind": "repeated", "observations": observations}
        )

    minutes = _comparison(_build(tmp_path, document), "A1", "authoring_minutes")
    assert minutes["verdict"] == "inconclusive"
    assert any("underpowered" in note for note in minutes["notes"])


def test_a_gate_that_cannot_fail_downgrades_the_recommendation(tmp_path: Path) -> None:
    document = _input()
    arm = document["arms"][1]
    arm["truth_preservation_gates"] = [
        {
            "gate_id": "a2_frozen_verdict",
            "verdict": "passed",
            "sensitive_to_intervention": False,
            "rationale": "gates task_id and expected_tool_calls, neither of which this arm touches",
        }
    ]

    report = _build(tmp_path, document)
    recommendation = next(item for item in report["arms"] if item["arm_id"] == "A1")["recommendation"]

    assert recommendation["recommendation"] == "adopt_with_conditions"
    assert any("cannot fail" in condition for condition in recommendation["conditions"])


def test_failed_gate_rejects_the_arm_despite_effort_savings(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["truth_preservation_gates"][0]["verdict"] = "failed"

    report = _build(tmp_path, document)
    arm = next(item for item in report["arms"] if item["arm_id"] == "A1")

    assert arm["recommendation"]["recommendation"] == "reject"
    assert report["summary"]["release_readiness"] == "blocked"


def test_material_regression_rejects_the_arm(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["measurements"][2] = {
        "metric_id": "validation_pass_rate",
        "kind": "proportion",
        "numerator": 400,
        "denominator": 1000,
    }
    document["arms"][0]["measurements"][2] = {
        "metric_id": "validation_pass_rate",
        "kind": "proportion",
        "numerator": 900,
        "denominator": 1000,
    }

    report = _build(tmp_path, document)
    arm = next(item for item in report["arms"] if item["arm_id"] == "A1")

    assert _comparison(report, "A1", "validation_pass_rate")["verdict"] == "material_regression"
    assert arm["recommendation"]["recommendation"] == "reject"


def test_deferred_arm_is_insufficient_evidence_not_zero(tmp_path: Path) -> None:
    document = _input()
    document["arms"].append(
        {
            "arm_id": "A3",
            "title": "LLM task proposal",
            "ticket": "SOV-863",
            "status": "deferred",
            "intervention": "let the model propose task semantics",
            "deferral_reason": "A3 starts only after A2 is clean",
        }
    )

    report = _build(tmp_path, document)
    arm = next(item for item in report["arms"] if item["arm_id"] == "A3")

    assert arm["recommendation"]["recommendation"] == "insufficient_evidence"
    assert arm["comparisons"] == []
    assert arm["failure_codes"]["status"] == "not_measured"
    assert report["summary"]["release_readiness"] == "incomplete"


def _partial_arm(**overrides: Any) -> dict[str, Any]:
    arm = {
        "arm_id": "STEP4",
        "title": "Cross-wording paired evaluation",
        "ticket": "SOV-862",
        "status": "partially_measured",
        "intervention": "score one skeleton under both wordings, paired at task level",
        "deferral_reason": "no skeleton is published under both wordings, so no paired delta exists",
        "evidence": [_evidence("cross_wording_report")],
        "task_set_hash": "sha256:" + "c" * 64,
        "measurements": [
            {"metric_id": "task_success_rate", "kind": "proportion", "numerator": 1060, "denominator": 1392}
        ],
        "limitations": ["the available wording contrast is unpaired and confounded"],
    }
    arm.update(overrides)
    return arm


def test_partial_arm_keeps_its_numbers_and_withholds_the_comparison(tmp_path: Path) -> None:
    document = _input()
    document["arms"].append(_partial_arm())

    report = _build(tmp_path, document)
    arm = next(item for item in report["arms"] if item["arm_id"] == "STEP4")

    assert arm["recommendation"]["recommendation"] == "insufficient_evidence"
    assert arm["comparisons"] == []
    assert [item["metric_id"] for item in arm["arm_local_measurements"]] == ["task_success_rate"]
    assert arm["arm_local_measurements"][0]["value"] == pytest.approx(1060 / 1392)
    assert arm["arm_local_measurements"][0]["comparison_withheld_because"]
    assert report["summary"]["arms_partially_measured"] == 1


def test_partial_measurements_do_not_close_a_coverage_gap(tmp_path: Path) -> None:
    """A withheld comparison must never read as readiness."""
    document = _input()
    for arm in document["arms"]:
        arm["measurements"] = [
            item for item in arm["measurements"] if item["metric_id"] != "task_success_rate"
        ]
    document["arms"].append(_partial_arm())

    report = _build(tmp_path, document)
    coverage = report["coverage"]

    assert "task_success" in coverage["unmeasured_families"]
    assert "task_success" not in coverage["measured_families"]
    assert coverage["families_measured_without_comparison"] == ["task_success"]
    assert report["summary"]["release_readiness"] == "incomplete"


def test_a_partial_arm_scoring_a_foreign_task_set_is_not_rejected(tmp_path: Path) -> None:
    """The status exists so an incomparable task set stops the delta, not the arm."""
    document = _input()
    document["arms"].append(_partial_arm())
    report = _build(tmp_path, document)

    baseline_hash = report["baseline"]["arm_id"]
    assert baseline_hash == "A0"
    # A `measured` arm on the same foreign task set fails closed instead.
    document["arms"][-1] = _partial_arm(status="measured")
    del document["arms"][-1]["deferral_reason"]
    with pytest.raises(AblationAggregationError, match="different task set than the baseline"):
        _build(tmp_path / "measured", document)


def test_partial_arm_must_declare_a_reason_and_a_measurement(tmp_path: Path) -> None:
    without_reason = _partial_arm()
    del without_reason["deferral_reason"]
    document = _input()
    document["arms"].append(without_reason)
    with pytest.raises(AblationAggregationError, match="missing required fields"):
        _build(tmp_path / "no_reason", document)

    document = _input()
    document["arms"].append(_partial_arm(measurements=[]))
    with pytest.raises(AblationAggregationError, match="at least one measurement"):
        _build(tmp_path / "no_measurement", document)


def test_a_forged_partial_comparison_is_refused(tmp_path: Path) -> None:
    document = _input()
    document["arms"].append(_partial_arm())
    report = _build(tmp_path, document)

    forged = json.loads(json.dumps(report))
    arm = next(item for item in forged["arms"] if item["arm_id"] == "STEP4")
    arm["comparisons"] = [_comparison(report, "A1", "task_success_rate")]
    forged["report_hash"] = _resign(forged)
    with pytest.raises(AblationAggregationError, match="publishes a baseline comparison"):
        validate_ablation_summary(forged)


def test_a_measured_arm_cannot_smuggle_in_arm_local_measurements(tmp_path: Path) -> None:
    report = _build(tmp_path)
    forged = json.loads(json.dumps(report))
    arm = next(item for item in forged["arms"] if item["arm_id"] == "A1")
    arm["arm_local_measurements"] = [
        {"metric_id": "latency_p50_ms", "family": "latency", "kind": "deterministic", "value": 1.0}
    ]
    forged["report_hash"] = _resign(forged)
    with pytest.raises(AblationAggregationError, match="must not report arm-local measurements"):
        validate_ablation_summary(forged)


def test_trade_off_names_what_a_gain_was_purchased_with(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["truth_preservation_gates"].append(
        {
            "gate_id": "a1_intent_check",
            "verdict": "not_run",
            "sensitive_to_intervention": True,
            "rationale": "recall against injected intent shifts was never measured",
        }
    )

    report = _build(tmp_path, document)
    trade = next(item for item in report["arms"] if item["arm_id"] == "A1")["trade_offs"]

    assert trade["verdict"] == "gain_with_unpriced_risk"
    assert [gain["metric_id"] for gain in trade["gains"]] == ["authoring_lines"]
    assert trade["measured_costs"] == []
    assert "a1_intent_check" in trade["unverified_by_gates"]
    assert "latency" in trade["unpriced_families"]


def test_trade_off_reports_a_regression_as_a_measured_cost(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["measurements"] = [
        {"metric_id": "authoring_lines", "kind": "deterministic", "value": 1386},
        {"metric_id": "validation_pass_rate", "kind": "proportion", "numerator": 11, "denominator": 22},
    ]
    document["arms"][0]["measurements"] = [
        {"metric_id": "authoring_lines", "kind": "deterministic", "value": 1642},
        {"metric_id": "validation_pass_rate", "kind": "proportion", "numerator": 22, "denominator": 22},
    ]
    for arm in document["arms"]:
        arm.pop("cost_context", None)
        arm.pop("failure_codes", None)

    report = _build(tmp_path, document)
    arm = next(item for item in report["arms"] if item["arm_id"] == "A1")

    assert "validation_pass_rate" in [cost["metric_id"] for cost in arm["trade_offs"]["measured_costs"]]
    # A regression rejects the arm, so the trade-off must not claim a gain stands.
    assert arm["recommendation"]["recommendation"] == "reject"


def test_a_recommendation_cannot_claim_a_gain_it_never_measured(tmp_path: Path) -> None:
    report = _build(tmp_path)
    forged = json.loads(json.dumps(report))
    arm = next(item for item in forged["arms"] if item["arm_id"] == "A1")
    arm["trade_offs"]["gains"] = []
    forged["report_hash"] = _resign(forged)

    with pytest.raises(AblationAggregationError, match="without a single material gain"):
        validate_ablation_summary(forged)


def test_token_cost_needs_no_pricing_snapshot(tmp_path: Path) -> None:
    """Demanding a snapshot for a token count would invite an invented one."""
    document = _input()
    for arm in document["arms"]:
        arm.pop("cost_context", None)
        arm["measurements"] = [
            item for item in arm["measurements"] if item["metric_id"] != "cost_amount"
        ]
        arm["measurements"].append(
            {"metric_id": "input_tokens", "kind": "deterministic", "value": 1000}
        )

    report = _build(tmp_path, document)

    assert "cost" in report["coverage"]["measured_families"]


def test_a_priced_cost_still_needs_a_pricing_snapshot(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1].pop("cost_context")

    with pytest.raises(AblationAggregationError, match="priced cost without cost_context"):
        _build(tmp_path, document)


def test_a_partial_arm_publishes_its_own_failure_profile(tmp_path: Path) -> None:
    document = _input()
    # Strip the baseline profile, so the partial arm is the only failure-code
    # evidence in the ladder and the report has to surface it or lose it.
    for arm in document["arms"]:
        arm.pop("failure_codes", None)
    document["arms"].append(
        _partial_arm(failure_codes={"arguments.mismatch": 307, "text_turn.mismatch": 160})
    )

    report = _build(tmp_path, document)
    failure = next(item for item in report["arms"] if item["arm_id"] == "STEP4")["failure_codes"]

    assert failure["status"] == "arm_only"
    assert failure["missing_in"] == ["baseline"]
    assert [code["code"] for code in failure["codes"]] == [
        "arguments.mismatch",
        "text_turn.mismatch",
    ]
    assert all(code["baseline_count"] is None and code["delta"] is None for code in failure["codes"])
    assert "failure_codes" in report["coverage"]["families_measured_without_comparison"]


def test_metric_missing_from_one_arm_is_not_measured(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["measurements"] = [
        item for item in document["arms"][1]["measurements"] if item["metric_id"] != "distinct_surface_count"
    ]

    report = _build(tmp_path, document)
    surfaces = _comparison(report, "A1", "distinct_surface_count")

    assert surfaces["verdict"] == "not_measured"
    assert surfaces["arm_value"] is None
    assert surfaces["baseline_value"] is not None
    arm = next(item for item in report["arms"] if item["arm_id"] == "A1")
    assert any("distinct_surface_count" in condition for condition in arm["recommendation"]["conditions"])


def test_unmeasured_required_families_are_reported_as_gaps(tmp_path: Path) -> None:
    report = _build(tmp_path)

    assert "latency" in report["coverage"]["unmeasured_families"]
    assert "task_success" in report["coverage"]["measured_families"]
    assert report["summary"]["release_readiness"] == "incomplete"


def test_incomparable_task_sets_and_pricing_fail_closed(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["task_set_hash"] = "sha256:" + "c" * 64
    with pytest.raises(AblationAggregationError, match="different task set"):
        _build(tmp_path / "tasks", document)

    document = _input()
    document["arms"][1]["cost_context"] = {"currency": "EUR", "pricing_snapshot": "2026-09-01"}
    with pytest.raises(AblationAggregationError, match="priced differently"):
        _build(tmp_path / "cost", document)


def test_input_loading_refuses_malformed_and_unsafe_documents(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": "1.0", "schema_version": "1.0"}', encoding="utf-8")
    with pytest.raises(AblationAggregationError, match="repeats JSON key"):
        load_ablation_aggregation_input(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    document = _input()
    nonfinite.write_text(
        json.dumps(document).replace('"value": 1386', '"value": NaN'),
        encoding="utf-8",
    )
    with pytest.raises(AblationAggregationError, match="non-finite constant"):
        load_ablation_aggregation_input(nonfinite)

    unregistered = _input()
    unregistered["arms"][1]["measurements"].append(
        {"metric_id": "operator_confidence", "kind": "deterministic", "value": 1.0}
    )
    with pytest.raises(AblationAggregationError, match="unregistered metric"):
        load_ablation_aggregation_input(_write(tmp_path / "unregistered.json", unregistered))

    zero_denominator = _input()
    zero_denominator["arms"][1]["measurements"][2]["denominator"] = 0
    with pytest.raises(AblationAggregationError, match="denominator must be positive"):
        load_ablation_aggregation_input(_write(tmp_path / "zero.json", zero_denominator))

    unmeasured_baseline = _input()
    unmeasured_baseline["arms"][0] = {
        "arm_id": "A0",
        "title": "Human baseline",
        "ticket": "SOV-859",
        "status": "blocked",
        "intervention": "none",
        "deferral_reason": "artifacts not imported",
    }
    with pytest.raises(AblationAggregationError, match="baseline arm must be measured"):
        load_ablation_aggregation_input(_write(tmp_path / "baseline.json", unmeasured_baseline))


def test_declared_evidence_hash_must_match_a_file_that_exists(tmp_path: Path) -> None:
    document = _input()
    results = tmp_path / "results" / "A0"
    results.mkdir(parents=True)
    (results / "baseline_report.md").write_text("# baseline\n", encoding="utf-8")

    with pytest.raises(AblationAggregationError, match="content hash does not match"):
        load_ablation_aggregation_input(_write(tmp_path / "input.json", document))


def test_threshold_overrides_require_a_rationale(tmp_path: Path) -> None:
    document = _input()
    document["policy"]["practical_threshold_overrides"] = [
        {"metric_id": "authoring_lines", "relative_threshold": 0.5}
    ]
    with pytest.raises(AblationAggregationError, match="rationale"):
        load_ablation_aggregation_input(_write(tmp_path / "override.json", document))

    document["policy"]["practical_threshold_overrides"] = [
        {
            "metric_id": "authoring_lines",
            "relative_threshold": 0.5,
            "rationale": "only a halving of authoring effort would change adoption",
        }
    ]
    report = _build(tmp_path / "applied", document)
    assert _comparison(report, "A1", "authoring_lines")["verdict"] == "no_material_change"
    assert report["policy"]["practical_threshold_overrides"][0]["relative_threshold"] == 0.5


def test_tampered_summary_is_refused(tmp_path: Path) -> None:
    report = _build(tmp_path)

    tampered = json.loads(json.dumps(report))
    tampered["summary"]["release_readiness"] = "ready"
    with pytest.raises(AblationAggregationError, match="report_hash mismatch"):
        validate_ablation_summary(tampered)

    swapped = json.loads(json.dumps(report))
    swapped["arms"][0]["recommendation"]["recommendation"] = "reject"
    swapped["report_hash"] = report["report_hash"]
    with pytest.raises(AblationAggregationError, match="report_hash mismatch"):
        validate_ablation_summary(swapped)


def test_written_summary_cannot_be_replaced_by_different_bytes(tmp_path: Path) -> None:
    report = _build(tmp_path)
    output = tmp_path / "out"
    write_ablation_summary(report, output)
    (output / "ablation_summary.md").write_text("# edited\n", encoding="utf-8")

    with pytest.raises(AblationAggregationError, match="refusing to replace"):
        write_ablation_summary(report, output)


def test_markdown_names_the_recommendation_and_claim_boundary(tmp_path: Path) -> None:
    markdown = render_ablation_summary_markdown(_build(tmp_path))

    assert "# BFCL ablation summary and release recommendation" in markdown
    assert "Release readiness" in markdown
    assert "Causal claim: no." in markdown
    assert "`authoring_lines`" in markdown


def test_cli_reports_readiness_and_exits_nonzero_when_blocked(tmp_path: Path) -> None:
    document = _input()
    document["arms"][1]["truth_preservation_gates"][0]["verdict"] = "failed"
    source = _write(tmp_path / "input.json", document)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemotron.steps.byob.scripts.aggregate_bfcl_ablation",
            "--input",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["release_readiness"] == "blocked"
    assert payload["recommendations"]["A1"] == "reject"
    assert (tmp_path / "out" / "ablation_summary.json").is_file()


def test_contract_document_matches_the_implemented_contract() -> None:
    text = _CONTRACT_DOC.read_text(encoding="utf-8")

    assert f"contract version is `{ABLATION_AGGREGATION_VERSION}`" in text
    for family in REQUIRED_FAMILIES:
        assert f"`{family}`" in text, family
    for recommendation in RECOMMENDATIONS:
        assert f"`{recommendation}`" in text, recommendation
    for status in ARM_STATUSES:
        assert f"`{status}`" in text, status
    # A registry the document does not describe is a registry a reviewer cannot
    # audit, so every family the metrics use has to be named.
    assert {definition.family for definition in METRIC_DEFINITIONS.values()} <= {
        family for family in (*REQUIRED_FAMILIES, "effort")
    }


_REAL_LADDER = Path(__file__).resolve().parents[3].parent / "results" / "ablation-ladder.json"


@pytest.mark.skipif(
    not _REAL_LADDER.is_file(),
    reason="the A0-A4 ladder input is not present in this checkout",
)
def test_published_ladder_summary_is_pinned(tmp_path: Path) -> None:
    """Golden test: the published SOV-866 summary must stay reproducible."""
    report = build_ablation_summary(
        AggregationInputs(ablation_input=_REAL_LADDER, output_dir=tmp_path)
    )

    assert report["report_hash"] == (
        "sha256:ac5139d4fde727e3001bc08102683af5e9e279501d8496d8bc2f040160105c64"
    )
    assert report["summary"]["recommendations"] == {
        "A1": "adopt",
        "A2": "adopt_with_conditions",
        "A3A4": "insufficient_evidence",
        "STEP4": "insufficient_evidence",
        # STEP4B executed the paired design STEP4 could only record as missing.
        # It still recommends nothing on its own: its control is internal, so the
        # contract has no baseline delta to turn into a release decision.
        "STEP4B": "insufficient_evidence",
    }
    assert report["summary"]["release_readiness"] == "incomplete"
    # Every family SOV-866 asks about is now spoken to, and the four that only a
    # scored run can produce are reported without a baseline delta.
    assert report["coverage"]["families_measured_without_comparison"] == [
        "cost",
        "failure_codes",
        "latency",
        "task_success",
    ]
    assert report["coverage"]["unmeasured_families"] == [
        "cost",
        "failure_codes",
        "latency",
        "task_success",
    ]


def test_cli_fails_closed_on_an_invalid_input(tmp_path: Path) -> None:
    broken = _write(tmp_path / "input.json", {"schema_version": "0.9"})

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemotron.steps.byob.scripts.aggregate_bfcl_ablation",
            "--input",
            str(broken),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "ablation_aggregation_failed" in completed.stderr
    assert not (tmp_path / "out").exists()
