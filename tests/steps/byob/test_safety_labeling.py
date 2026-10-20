"""Stage 12 judges are evidence. Python gold is unchanged."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import select_dataset
from nemotron.steps.byob.runtime.benchmark_families.safety.eval import events_from_case, score_reply
from nemotron.steps.byob.runtime.benchmark_families.safety.labeling import (
    JudgeUnavailableError,
    build_audit_queue,
    identities_overlap,
    make_judges,
    public_identity,
    require_profile_judges,
    spot_check_rules,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.review import assemble_review
from tests.steps.byob.test_safety_coverage import _rule as _policy_rule
from tests.steps.byob.test_safety_dataset import _ALL_OF, _clone, _one, _ready, _span_map, _with_text
from tests.steps.byob.test_safety_review import _HASHES, _coverage, _rule


class FakeJudge:
    """Returns a configured action per case id and records the payloads it saw."""

    def __init__(self, action: str | Mapping[str, str], *, model: str, confidence: float = 0.9) -> None:
        self.identity = {"provider": "fake", "model": model}
        self.action = action
        self.confidence = confidence
        self.payloads: list[dict[str, Any]] = []
        self.fail = False

    def complete(
        self,
        rows: list[Any],
        *,
        system_prompt: str = "",
        prompt: str = "",
        output_format: Any = None,
        prefix: str = "",
        inference_parameters: Any = None,
    ) -> dict[str, Any]:
        del system_prompt, prompt, output_format, prefix
        if self.fail:
            raise ConnectionError("endpoint refused the connection")
        assert inference_parameters is not None
        assert inference_parameters["temperature"] == 0.0
        answers: dict[str, Any] = {}
        for row in rows:
            payload = json.loads(row["payload"]) if isinstance(row.get("payload"), str) else {}
            self.payloads.append(payload)
            assert "expected_action" not in payload
            assert "kind" not in payload
            assert "proposed_action" not in payload
            request_id = row["request_id"]
            action = self.action[request_id] if isinstance(self.action, Mapping) else self.action
            answers[request_id] = {"proposed_action": action}
            if self.confidence is not None:
                answers[request_id]["confidence"] = self.confidence
            assert all("scored" not in (turn or {}) for turn in payload.get("turns") or [])
        return answers


def test_a_judge_disagreement_keeps_python_gold_and_queues_one_cluster() -> None:
    rules, build = _ready()
    seed = _one(build, "clear_violation")
    paraphrase = _with_text(
        _clone(
            seed,
            case_id=f"{seed['case_id']}/paraphrase-2",
            slot_id=f"{seed['case_id']}/paraphrase-2",
            variant_of=seed["case_id"],
            origin="paraphrase",
        ),
        "A rewritten violation request.",
    )
    actions = {str(case["case_id"]): case["expected_action"] for case in [*build.cases, paraphrase]}
    actions[seed["case_id"]] = "allow"
    actions[paraphrase["case_id"]] = "allow"
    judge = FakeJudge(actions, model="judge-a")

    result = select_dataset(
        [*build.cases, paraphrase],
        build.families,
        rules,
        _span_map(*rules),
        judges=[judge],
        assurance={"profile": "standard", "review_sample_rate": 0, "second_judge_on": []},
        example_cap=40,
    )

    kept = {case["case_id"] for case in result.deduplicated}
    assert seed["case_id"] in kept
    assert paraphrase["case_id"] in kept
    assert seed["case_id"] not in {row["case_id"] for row in result.rejected}
    assert result.report["judge"]["method"] == "model"
    assert result.report["judge"]["disagreements"] == 0
    assert result.report["judge"]["judge_disagreements"] == 2
    queued = [item for item in result.audited if item["reason"] == "judge_disagreement"]
    assert len(queued) == 1
    assert queued[0]["cluster_size"] == 2
    assert queued[0]["needs_decision"] is True
    assert set(queued[0]["member_ids"]) == {seed["case_id"], paraphrase["case_id"]}
    assert judge.payloads
    assert all("expected_action" not in payload for payload in judge.payloads)


def test_the_second_judge_is_blinded_and_runs_only_on_disagreement() -> None:
    rules, build = _ready()
    first = FakeJudge("allow", model="judge-a")
    second = FakeJudge("block", model="judge-b")

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        judges=[first, second],
        assurance={
            "profile": "standard",
            "review_sample_rate": 0,
            "second_judge_on": ["disagreement"],
        },
    )

    assert result.judge_agreement["second_invoked"] >= 1
    assert second.payloads
    assert all("proposed_action" not in payload for payload in second.payloads)
    assert all(payload.get("allowed_labels") for payload in second.payloads)
    disagreed = next(row for row in result.proposals if row["status"] == "disagreement")
    assert disagreed["proposed_action"] == "allow"
    assert disagreed["second_action"] == "block"


def test_high_assurance_refuses_overlapping_judge_identities() -> None:
    first = FakeJudge("block", model="same-model")
    second = FakeJudge("block", model="same-model")
    with pytest.raises(ValueError, match="distinct model identities"):
        require_profile_judges("high_assurance", [first, second])
    with pytest.raises(ValueError, match="two judge_model_configs"):
        require_profile_judges("high_assurance", [first])


def test_a_novel_action_is_rejected_and_listed_for_audit() -> None:
    rules, build = _ready()
    changed = _clone(_one(build, "clear_allow"), expected_action="custom_hold")
    changed["turns"][0]["expected_action"] = "custom_hold"

    result = select_dataset(
        [changed if case["case_id"] == changed["case_id"] else case for case in build.cases],
        build.families,
        rules,
        _span_map(*rules),
        assurance={"profile": "standard", "review_sample_rate": 0},
        example_cap=40,
    )

    rejected = {row["case_id"]: row for row in result.rejected}
    assert "novel_action" in rejected[changed["case_id"]]["reasons"]
    queued = [item for item in result.audited if item["reason"] == "novel_action"]
    assert queued
    assert queued[0]["needs_decision"] is True


def test_spot_check_takes_five_or_five_percent() -> None:
    rules = [{"policy_id": f"R-{index}", "status": "eligible_for_generation"} for index in range(20)]
    selected = spot_check_rules(rules)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert set(selected) <= {f"R-{index}" for index in range(20)}


def test_audit_cards_are_pending_and_spot_checks_are_not() -> None:
    queue = [
        {
            "card_id": "audit:deadbeefdeadbeef",
            "reason": "judge_disagreement",
            "case_id": "case-1",
            "policy_id": "OK-1",
            "policy_ids": ["OK-1"],
            "cluster_size": 3,
            "python_action": "block",
            "judge_action": "allow",
            "scored_text": "please help",
            "needs_decision": True,
        }
    ]
    build = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=[_rule("OK-1", "eligible_for_generation", "Allowed quote")],
        conflicts=[],
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[],
        decisions={},
        content_hashes=_HASHES,
        audit_queue=queue,
    )

    assert "audit:deadbeefdeadbeef" in build.pending
    assert "rule:OK-1" not in build.pending
    assert "Judge disagreements and novel actions (1)" in build.html
    assert "Spot-check of auto-accepted rules (1, pre-marked approve)" in build.html
    assert "OK-1" in build.html
    assert any(card["kind"] == "audit" for card in build.packet["cards"])


def test_construction_exposure_is_excluded_from_the_primary_score() -> None:
    case = {
        "case_id": "exposed",
        "policy_id": "P-1",
        "kind": "clear_violation",
        "expected_action": "block",
        "construction": {"generator": {"provider": "fake", "model": "guard-x"}, "judges": []},
        "turns": [{"role": "user", "text": "do it", "scored": True, "turn": 1, "expected_action": "block"}],
    }
    event = events_from_case(case)[0]
    row = score_reply(event, {"decision": "block", "latency_ms": 1}, candidate={"model": "guard-x"})
    assert row["construction_exposure"] is True
    assert row["match"] is True
    clean = score_reply(event, {"decision": "block", "latency_ms": 1}, candidate={"model": "other"})
    assert clean["construction_exposure"] is False


def test_identity_overlap_uses_model_not_alias() -> None:
    assert identities_overlap({"model": "nemotron", "alias": "judge_a"}, {"model": "nemotron", "alias": "eval"})
    assert not identities_overlap({"model": "nemotron"}, {"model": "other"})
    assert identities_overlap({"model": "nvidia/guard-x"}, {"model": "guard-x"})
    assert identities_overlap({"base_url": "https://api.x.com/v1"}, {"endpoint": "https://api.x.com/v1/chat"})
    assert not identities_overlap(
        {"model": "a", "base_url": "https://api.x.com"},
        {"model": "b", "endpoint": "https://api.x.com/v1"},
    )
    assert public_identity({"model": "x", "api_key": "secret", "inference_parameters": {"temperature": 0}}) == {
        "model": "x"
    }


def test_judges_skip_python_rejected_rows() -> None:
    rules, build = _ready()
    seed = _one(build, "clear_violation")
    broken = _clone(seed, case_id=f"{seed['case_id']}/broken", slot_id=f"{seed['case_id']}/broken")
    broken["evidence"] = {}
    judge = FakeJudge("block", model="judge-a")

    result = select_dataset(
        [*build.cases, broken],
        build.families,
        rules,
        _span_map(*rules),
        judges=[judge],
        assurance={"profile": "standard", "review_sample_rate": 0, "second_judge_on": []},
        example_cap=0,
    )

    assert broken["case_id"] not in {row["case_id"] for row in judge.payloads}
    assert broken["case_id"] in {row["case_id"] for row in result.rejected}
    assert not any(
        item.get("case_id") == broken["case_id"] and item.get("reason") == "judge_disagreement"
        for item in result.audited
    )


def test_a_first_judge_outage_aborts_the_dataset() -> None:
    rules, build = _ready()
    judge = FakeJudge("block", model="judge-a")
    judge.fail = True
    with pytest.raises(JudgeUnavailableError, match="first judge"):
        select_dataset(
            build.cases,
            build.families,
            rules,
            _span_map(*rules),
            judges=[judge],
            assurance={"profile": "standard", "review_sample_rate": 0},
            example_cap=0,
        )


def test_an_empty_second_judge_on_list_disables_the_second_judge() -> None:
    rules, build = _ready()
    first = FakeJudge("allow", model="judge-a")
    second = FakeJudge("block", model="judge-b")

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        judges=[first, second],
        assurance={"profile": "standard", "review_sample_rate": 0, "second_judge_on": []},
        example_cap=0,
    )

    assert result.judge_agreement["second_invoked"] == 0
    assert second.payloads == []


def test_missing_confidence_invokes_the_second_judge() -> None:
    rules, build = _ready()
    actions = {str(case["case_id"]): case["expected_action"] for case in build.cases}
    first = FakeJudge(actions, model="judge-a", confidence=None)
    second = FakeJudge(actions, model="judge-b")

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        judges=[first, second],
        assurance={"profile": "standard", "review_sample_rate": 0, "second_judge_on": ["low_confidence"]},
        example_cap=0,
    )

    assert result.judge_agreement["second_invoked"] == len(result.deduplicated)
    assert result.proposals
    assert all(row["status"] == "low_confidence" for row in result.proposals)


def test_cluster_representatives_and_sample_share_the_cap() -> None:
    cases = [
        {
            "case_id": f"c-{index:02d}",
            "policy_id": f"P-{index}",
            "kind": "clear_violation",
            "expected_action": "block",
            "turns": [{"role": "user", "text": f"turn {index}", "scored": True}],
        }
        for index in range(12)
    ]
    queue = build_audit_queue(
        cases,
        {case["case_id"]: "block" for case in cases},
        {},
        (),
        assurance={"profile": "standard", "review_sample_rate": 1},
        example_cap=3,
    )
    capped = [item for item in queue if item["reason"] in {"cluster_representative", "review_sample"}]
    assert len(capped) == 3


def test_native_language_uses_scope_languages_not_translation_origin() -> None:
    rule = _policy_rule(
        "ALL-2",
        block=_ALL_OF,
        quote="Follow the multi-turn policy.",
        severity="medium",
        languages=("vi-VN",),
        ambiguous_action="caution",
        signals={"multi_turn_turns": 2},
    )
    seed = {
        "case_id": "seed-vi",
        "policy_id": "ALL-2",
        "kind": "clear_violation",
        "expected_action": "block",
        "language": "vi-VN",
        "origin": "seed",
        "turns": [{"role": "user", "text": "vi", "scored": True}],
    }
    translated = {
        **seed,
        "case_id": "seed-en",
        "kind": "language",
        "origin": "translation",
        "language": "en-US",
        "variant_of": "seed-vi",
        "turns": [{"role": "user", "text": "en", "scored": True}],
    }
    queue = build_audit_queue(
        [seed, translated],
        {"seed-vi": "block", "seed-en": "block"},
        {},
        [rule],
        assurance={"profile": "high_assurance", "review_sample_rate": 0},
        example_cap=0,
    )
    native = [item for item in queue if item["reason"] == "native_language"]
    assert [item["case_id"] for item in native] == ["seed-vi"]
    assert native[0]["needs_decision"] is True


def test_legal_slice_matches_basis_tokens_not_substrings() -> None:
    rule = _policy_rule("ALL-2", block=_ALL_OF, quote="Block it.", severity="high")
    case = {
        "case_id": "legal-1",
        "policy_id": "ALL-2",
        "kind": "clear_violation",
        "expected_action": "block",
        "turns": [{"role": "user", "text": "ask", "scored": True}],
    }
    legal = rule.model_copy(update={"severity_basis": "rubric:binding_evidence+legal_prohibition"})
    hit = build_audit_queue(
        [case],
        {"legal-1": "block"},
        {},
        [legal],
        assurance={"profile": "high_assurance", "review_sample_rate": 0},
        example_cap=0,
    )
    miss = build_audit_queue(
        [case],
        {"legal-1": "block"},
        {},
        [rule.model_copy(update={"severity_basis": "rubric:binding_evidence+not_legal_prohibition"})],
        assurance={"profile": "high_assurance", "review_sample_rate": 0},
        example_cap=0,
    )
    assert any(item["reason"] == "legal" for item in hit)
    assert not any(item["reason"] == "legal" for item in miss)


def test_novel_action_variants_share_one_audit_card() -> None:
    rules, build = _ready()
    seed = _one(build, "clear_allow")
    changed = _clone(seed, expected_action="custom_hold")
    changed["turns"][0]["expected_action"] = "custom_hold"
    paraphrase = _with_text(
        _clone(
            changed,
            case_id=f"{changed['case_id']}/para",
            slot_id=f"{changed['case_id']}/para",
            variant_of=changed["case_id"],
            origin="paraphrase",
        ),
        "A rewritten allow request.",
    )
    paraphrase["turns"][0]["expected_action"] = "custom_hold"
    cases = [changed if case["case_id"] == changed["case_id"] else case for case in build.cases]
    result = select_dataset(
        [*cases, paraphrase],
        build.families,
        rules,
        _span_map(*rules),
        assurance={"profile": "standard", "review_sample_rate": 0},
        example_cap=0,
    )
    queued = [item for item in result.audited if item["reason"] == "novel_action"]
    assert len(queued) == 1
    assert queued[0]["cluster_size"] == 2
    assert queued[0]["needs_decision"] is True


def test_a_third_judge_config_fails_closed() -> None:
    class Config:
        extras = {
            "judge_model_configs": [
                {"alias": "a", "provider": "p", "model": "m1"},
                {"alias": "b", "provider": "p", "model": "m2"},
                {"alias": "c", "provider": "p", "model": "m3"},
            ]
        }

    with pytest.raises(ValueError, match="at most two"):
        make_judges(Config(), Path("."))


def test_regulated_critical_boundaries_are_pending_cards() -> None:
    rule = _policy_rule("ALL-2", block=_ALL_OF, quote="Block it.", severity="critical")
    case = {
        "case_id": "bound-1",
        "policy_id": "ALL-2",
        "kind": "boundary",
        "expected_action": "block",
        "turns": [{"role": "user", "text": "edge", "scored": True}],
    }
    queue = build_audit_queue(
        [case],
        {"bound-1": "block"},
        {},
        [rule],
        assurance={"profile": "regulated", "review_sample_rate": 0},
        example_cap=0,
    )
    assert queue[0]["reason"] == "critical_boundary"
    assert queue[0]["needs_decision"] is True


def test_rejected_rules_list_flagged_cases_without_a_second_pending_card() -> None:
    queue = []
    build = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=[_rule("OK-1", "eligible_for_generation", "Allowed quote")],
        conflicts=[],
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[
            {
                "case_id": "case-1",
                "policy_id": "OK-1",
                "kind": "clear_violation",
                "expected_action": "allow",
                "turns": [{"role": "user", "text": "please help", "scored": True}],
            }
        ],
        decisions={"rule:OK-1": ("reject", "")},
        content_hashes=_HASHES,
        audit_queue=queue,
    )
    assert any(item["reason"] == "flagged_rule" for item in build.packet["audit_queue"])
    assert not any(card["kind"] == "audit" for card in build.packet["cards"])
    assert "Profile-required audit slices (1)" in build.html
