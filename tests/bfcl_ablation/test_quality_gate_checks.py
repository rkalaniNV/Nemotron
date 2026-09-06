from __future__ import annotations

from pathlib import Path

import yaml

from bfcl_ablation import common
from bfcl_ablation.quality_gate.artifacts import load_artifacts, load_thresholds
from bfcl_ablation.quality_gate.checks import run_quality_gate, wilson_interval
from bfcl_ablation.quality_gate.labels import (
    build_review_expectations,
    build_review_queue,
    label_coverage,
)
from bfcl_ablation.quality_gate.schema import AuditCheck, ThresholdPolicy, rollup

QUALITY_ROOT = common.ABLATION_ROOT / "quality_gate"


def _policy() -> ThresholdPolicy:
    return ThresholdPolicy.model_validate(
        yaml.safe_load((QUALITY_ROOT / "defaults.yaml").read_text(encoding="utf-8"))
    )


def _current_audit() -> tuple[dict, list[AuditCheck], dict]:
    artifacts, inventory = load_artifacts(common.RESULTS)
    policy, provenance = load_thresholds(QUALITY_ROOT / "defaults.yaml")
    queue = build_review_queue(artifacts)
    coverage = label_coverage(
        queue,
        policy,
        ["no human label file supplied"],
        build_review_expectations(artifacts),
    )
    metrics, checks = run_quality_gate(
        artifacts=artifacts,
        inventory=inventory,
        policy=policy,
        threshold_provenance=provenance,
        label_coverage=coverage,
        labels_path=None,
    )
    return metrics, checks, coverage


def test_wilson_interval_is_bounded_and_none_without_samples() -> None:
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 100) == (0.0, 0.037)
    low, high = wilson_interval(7, 323) or (0.0, 0.0)
    assert 0.0 <= low < 7 / 323 < high <= 1.0


def test_rollup_prefers_known_failure_then_missing_evidence() -> None:
    base = {
        "arm": "a0",
        "dimension": "release_readiness",
        "claim": "claim",
        "detail": "detail",
    }
    conditional = AuditCheck(check_id="conditional", status="CONDITIONAL", **base)
    inconclusive = AuditCheck(check_id="inconclusive", status="INCONCLUSIVE", **base)
    failed = AuditCheck(check_id="failed", status="FAIL", **base)
    assert rollup([conditional], "release_readiness") == "CONDITIONAL"
    assert rollup([conditional, inconclusive], "release_readiness") == "INCONCLUSIVE"
    assert rollup([conditional, inconclusive, failed], "release_readiness") == "FAIL"


def test_current_artifacts_reconcile_and_keep_semantics_inconclusive() -> None:
    metrics, checks, coverage = _current_audit()
    by_id = {check.check_id: check for check in checks}
    assert metrics["rollup"]["integrity"] == "PASS"
    assert metrics["rollup"]["study_validity"] == "INCONCLUSIVE"
    assert metrics["publication_decision"] == "NOT_READY"
    assert by_id["A4-TRIAL-RECONCILIATION"].status == "PASS"
    assert by_id["A5-PAIR-RECONCILIATION"].status == "PASS"
    assert by_id["A2-FROZEN-SCOPE"].status == "PASS"
    assert by_id["A2-FROZEN-NOT-SEMANTIC"].status == "CONDITIONAL"
    assert by_id["A2-HUMAN-SEMANTICS-STUDY_VALIDITY"].status == "INCONCLUSIVE"
    assert coverage["prevalence_sample"]["required"] == 51
    assert coverage["intent_shift_controls"]["required"] == 17


def test_a6_reports_an_identified_interval_not_a_point_estimate() -> None:
    metrics, checks, _ = _current_audit()
    by_id = {check.check_id: check for check in checks}
    bounds = metrics["derived"]["a6_all_gate_blind_bounds"]
    assert bounds == {
        "lower_count": 4,
        "upper_count": 49,
        "observable": 107,
        "lower_rate": 0.0374,
        "upper_rate": 0.4579,
    }
    assert by_id["A6-BLIND-BOUNDS"].status == "INCONCLUSIVE"
    assert by_id["A6-RELEASE-BLIND-RATE"].status == "INCONCLUSIVE"
    assert by_id["A6-BLIND-BOUNDS"].denominator == 107


def test_missing_artifacts_are_inconclusive_not_implicit_pass(tmp_path: Path) -> None:
    artifacts, inventory = load_artifacts(tmp_path)
    policy, provenance = load_thresholds(QUALITY_ROOT / "defaults.yaml")
    queue = build_review_queue(artifacts)
    coverage = label_coverage(queue, policy, ["no artifacts"])
    metrics, checks = run_quality_gate(
        artifacts=artifacts,
        inventory=inventory,
        policy=policy,
        threshold_provenance=provenance,
        label_coverage=coverage,
        labels_path=None,
    )
    by_id = {check.check_id: check for check in checks}
    assert by_id["G-ARTIFACTS"].status == "INCONCLUSIVE"
    assert metrics["rollup"]["integrity"] == "INCONCLUSIVE"
    assert metrics["rollup"]["study_validity"] == "INCONCLUSIVE"
