from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bfcl_ablation.quality_gate.labels import (
    build_review_expectations,
    build_review_queue,
    label_coverage,
    merge_review_labels,
)
from bfcl_ablation.quality_gate.schema import HumanReviewFile, ThresholdPolicy

ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = ROOT / "bfcl_ablation" / "quality_gate"


def _thresholds() -> ThresholdPolicy:
    return ThresholdPolicy.model_validate(
        yaml.safe_load((QUALITY_ROOT / "defaults.yaml").read_text(encoding="utf-8"))
    )


def _review_payload() -> dict:
    return {
        "schema_version": "1.0",
        "rubric_version": "1.0",
        "pack_id": "banking_vn",
        "language": "vi",
        "reviewers": [
            {"reviewer_id": "r1", "display_name": None},
            {"reviewer_id": "r2", "display_name": None},
        ],
        "items": [
            {
                "item_id": "pair-1",
                "kind": "paraphrase_pair",
                "source_arm": "a2",
                "source_ref": "row:1",
                "template_id": "template-1",
                "task_id": None,
                "variant_index": 1,
                "reference_text": "Kiểm tra tài khoản {account_id}.",
                "candidate_text": "Xem giúp tài khoản {account_id}.",
                "context": {"sampling_stratum": "deterministic_prevalence"},
                "labels": [
                    {
                        "reviewer_id": reviewer,
                        "intent_preserved": True,
                        "acceptable_for_benchmark": True,
                        "required_tools": ["get_account"],
                        "turn_policy": "single_turn",
                        "mutant_classification": None,
                        "severity": "none",
                        "notes": None,
                    }
                    for reviewer in ("r1", "r2")
                ],
                "adjudication": None,
            }
        ],
    }


def test_example_human_review_file_is_valid() -> None:
    payload = yaml.safe_load(
        (QUALITY_ROOT / "labels" / "example.v1.yaml").read_text(encoding="utf-8")
    )
    review = HumanReviewFile.model_validate(payload)
    assert review.schema_version == "1.0"
    assert len(review.items) == 1
    assert len(review.items[0].labels) == 2


def test_contract_rejects_unknown_fields() -> None:
    payload = _review_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HumanReviewFile.model_validate(payload)


def test_contract_rejects_implicit_type_coercion() -> None:
    payload = _review_payload()
    payload["items"][0]["variant_index"] = "1"
    with pytest.raises(ValidationError, match="int_type"):
        HumanReviewFile.model_validate(payload)


def test_contract_rejects_duplicate_reviewers_and_missing_intent_label() -> None:
    duplicate = _review_payload()
    duplicate["reviewers"].append({"reviewer_id": "r1", "display_name": None})
    with pytest.raises(ValidationError, match="unique reviewer_id"):
        HumanReviewFile.model_validate(duplicate)

    missing_intent = _review_payload()
    missing_intent["items"][0]["labels"][0]["intent_preserved"] = None
    with pytest.raises(ValidationError, match="require intent_preserved"):
        HumanReviewFile.model_validate(missing_intent)


def test_label_coverage_requires_independent_reviewers() -> None:
    review = HumanReviewFile.model_validate(_review_payload())
    coverage = label_coverage(review, _thresholds(), [])
    assert coverage["complete"] is True
    assert coverage["prevalence_sample"] == {
        "errors": 0,
        "reviewed": 1,
        "required": 1,
    }

    incomplete_payload = _review_payload()
    incomplete_payload["items"][0]["labels"].pop()
    incomplete = HumanReviewFile.model_validate(incomplete_payload)
    coverage = label_coverage(incomplete, _thresholds(), [])
    assert coverage["complete"] is False
    assert coverage["items_complete"] == 0


def test_merge_detects_changed_immutable_review_text() -> None:
    queue_payload = _review_payload()
    queue_payload["reviewers"] = []
    queue_payload["items"][0]["labels"] = []
    queue = HumanReviewFile.model_validate(queue_payload)
    supplied = HumanReviewFile.model_validate(deepcopy(_review_payload()))
    supplied.items[0].candidate_text = "Nội dung đã bị thay đổi."
    merged, issues = merge_review_labels(queue, supplied)
    assert any("immutable fields" in issue for issue in issues)
    assert merged.items[0].labels == []


def test_review_queue_uses_prevalence_controls_and_diagnostics() -> None:
    artifacts = {
        "a2_metrics": {
            "intent_check": {
                "rows": {
                    "canonical": [
                        {
                            "template_id": "t1",
                            "variant_index": 0,
                            "text": "Câu gốc",
                        }
                    ],
                    "paraphrases": [
                        {
                            "template_id": "t1",
                            "variant_index": 1,
                            "text": "Câu một",
                            "agrees": True,
                            "expected_tools": ["tool_a"],
                            "predicted_tools": ["tool_a"],
                        },
                        {
                            "template_id": "t1",
                            "variant_index": 9,
                            "text": "Câu đổi ý",
                            "agrees": False,
                            "expected_tools": ["tool_a"],
                            "predicted_tools": ["tool_b"],
                        },
                    ],
                    "shifts": [
                        {
                            "template_id": "t1",
                            "text": "Control đổi ý",
                            "agrees": False,
                        }
                    ],
                }
            }
        }
    }
    queue = build_review_queue(artifacts, sample_per_template=1)
    expectations = build_review_expectations(artifacts, sample_per_template=1)
    strata = sorted(expectations[item.item_id]["sampling_stratum"] for item in queue.items)
    assert strata == sorted(
        [
            "deterministic_prevalence",
            "checker_disagreement_diagnostic",
            "intent_shift_control",
        ]
    )
    assert all(item.context == {} for item in queue.items)
    assert all(item.source_ref.startswith("surface:") for item in queue.items)
    assert all(item.variant_index is None for item in queue.items)
