"""Policy Rule IR, condition operators, action order, and confidence routing."""

import copy
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.safety.actions import (
    ACTION_ORDER,
    RuleMatch,
    aggregate,
    condition_matches,
    expected_action,
    rule_matches,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import (
    CONFIDENCE_WEIGHTS,
    ConfidenceSignals,
    Thresholds,
    canonical_text,
    comma_or_semicolon,
    compute_confidence,
    fold_latin,
    operator_explicitness,
    quote_in_span,
    quote_match_key,
    route_rule,
    route_rule_from_span,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import (
    ConditionGroup,
    PolicyRule,
    RuleValidationError,
    normalize_candidate,
    policy_rule_json_schema,
    validate_rule,
)

QUOTE = "The assistant must not explain how to organize a protest against the state."


def candidate(**rule: Any) -> dict[str, Any]:
    body = {"actor": "assistant", "action": "provide_content", "object": "anti_state_activity", **rule}
    return {
        "policy_id": "VN-ANTI-STATE-001",
        "rule": body,
        "severity": "high",
        "testability": "testable",
        "provenance": {"document_id": "POLICY", "source_quote": QUOTE},
    }


def signals(**overrides: Any) -> ConfidenceSignals:
    values = {"quote_match": 1.0, "schema_completeness": 1.0, "operator_explicitness": 1.0}
    return ConfidenceSignals(**{**values, **overrides})


def normalized(raw: dict[str, Any], **kwargs: Any) -> PolicyRule:
    result = normalize_candidate(raw, **kwargs)
    assert result.errors == []
    assert result.rule is not None
    return result.rule


# Normalization: the documented examples


def test_a_multi_leg_bare_block_if_needs_a_decision() -> None:
    rule = normalized(candidate(block_if=["instructional_or_organizing_verb", "prohibited_target"]))

    assert rule.rule.block_if is None
    assert rule.unresolved_conditions["block_if"] == ("instructional_or_organizing_verb", "prohibited_target")

    routed = route_rule(rule, signals(), quote_found=True)
    assert routed.rule.status == "needs_decision"


def test_explicit_all_of_is_stored_unchanged() -> None:
    rule = normalized(
        candidate(
            block_if={
                "all_of": ["instructional_or_organizing_verb", "prohibited_target"],
                "operator_source": "explicit",
            }
        )
    )

    assert rule.rule.block_if == ConditionGroup(all_of=["instructional_or_organizing_verb", "prohibited_target"])


@pytest.mark.parametrize("name", ["allow_if", "escalate_if", "require_confirmation_if"])
def test_bare_list_without_a_default_needs_a_decision(name: str) -> None:
    raw = candidate(block_if=["prohibited_target"], **{name: ["condition_a", "condition_b"]})

    rule = normalized(raw)

    assert getattr(rule.rule, name) is None
    assert rule.unresolved_conditions == {name: ("condition_a", "condition_b")}
    routed = route_rule(rule, signals(), quote_found=True)
    assert routed.rule.status == "needs_decision"
    assert any(name in reason for reason in routed.reasons)


@pytest.mark.parametrize(
    "quote",
    [
        "Chặn khi yêu cầu đồng thời có động từ hướng dẫn và mục tiêu bị cấm.",
        "Block when the request has all of the following components.",
        "Chặn khi đồng thời có đủ hai điều kiện.",
        "Chặn khi có một trong các dấu hiệu sau.",
        "Block when any of the following signals appear.",
        "Bloquer lorsque toutes les conditions suivantes s'appliquent.",
        "仅当条件甲与条件乙同时成立时才阻止请求。",
        "Block if none of the following conditions apply.",
        "Chặn nếu không có điều kiện nào sau đây.",
        "Block when all of the following apply, or any of the following personas.",
    ],
)
def test_source_wording_does_not_choose_an_operator(quote: str) -> None:
    raw = candidate(block_if=["instructional_or_organizing_verb", "prohibited_target"], allow_if=["news", "academic"])
    raw["provenance"]["source_quote"] = quote

    rule = normalized(raw, operator_text={"block_if": quote})

    assert rule.rule.block_if is None
    assert rule.rule.allow_if is None
    assert rule.unresolved_conditions == {
        "block_if": ("instructional_or_organizing_verb", "prohibited_target"),
        "allow_if": ("news", "academic"),
    }
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


def test_shared_operator_wording_is_not_assigned_across_conditions() -> None:
    raw = candidate(
        block_if=["verb", "target"],
        allow_if={"any_of": ["news"]},
    )
    raw["provenance"]["source_quote"] = "Block when all of the following components appear."

    rule = normalized(raw)

    assert rule.rule.block_if is None
    assert rule.rule.allow_if is not None
    assert rule.unresolved_conditions == {"block_if": ("verb", "target")}
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


@pytest.mark.parametrize(
    "operator_text",
    [
        {"blok_if": "all of the following"},
        {"block_if": 1},
    ],
)
def test_condition_evidence_mapping_is_strict(operator_text: dict[str, Any]) -> None:
    result = normalize_candidate(candidate(block_if=["verb", "target"]), operator_text=operator_text)

    assert result.rule is None
    assert result.errors


def test_one_bare_allow_list_without_a_structured_operator_needs_a_decision() -> None:
    raw = candidate(allow_if=["news", "academic"])
    raw["provenance"]["source_quote"] = "Allow when any of the following apply."

    rule = normalized(raw)

    assert rule.rule.allow_if is None
    assert rule.unresolved_conditions == {"allow_if": ("news", "academic")}


def test_incidental_or_does_not_choose_an_operator() -> None:
    raw = candidate(block_if=["weapon", "explosive"], allow_if=["news", "academic"])
    raw["provenance"]["source_quote"] = "Không được hướng dẫn chế tạo vũ khí hoặc chất nổ."

    rule = normalized(raw)

    assert rule.rule.block_if is None
    assert rule.unresolved_conditions["block_if"] == ("weapon", "explosive")
    assert rule.rule.allow_if is None
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


def test_clear_all_of_frame_does_not_choose_an_operator() -> None:
    raw = candidate(block_if=["verb", "target"])
    raw["provenance"]["source_quote"] = "Chặn khi đồng thời có đủ điều kiện."

    rule = normalized(raw)

    assert rule.rule.block_if is None
    assert rule.unresolved_conditions["block_if"] == ("verb", "target")
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


def test_mixed_wording_does_not_choose_an_operator() -> None:
    raw = candidate(block_if=["verb", "target", "persona"])
    raw["provenance"]["source_quote"] = "Block when all of the following apply, or any of the following personas."

    rule = normalized(raw)

    assert rule.rule.block_if is None
    assert rule.unresolved_conditions == {"block_if": ("verb", "target", "persona")}
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


def test_operator_text_string_does_not_choose_an_operator() -> None:
    rule = normalized(
        candidate(block_if=["verb", "target"]), operator_text="Chỉ chặn khi đủ các thành phần sau cùng xuất hiện."
    )

    assert rule.rule.block_if is None
    assert rule.unresolved_conditions["block_if"] == ("verb", "target")


def test_one_leg_block_if_stays_default_any_of_when_quote_has_operator_wording() -> None:
    raw = candidate(block_if=["target"])
    raw["provenance"]["source_quote"] = "Block when all of the following apply."

    rule = normalized(raw, operator_text={"block_if": raw["provenance"]["source_quote"]})

    assert rule.rule.block_if is not None
    assert rule.rule.block_if.operator == "any_of"
    assert rule.rule.block_if.operator_source == "default_any_of"
    assert rule.unresolved_conditions == {}


def test_model_self_score_and_claimed_status_are_not_kept() -> None:
    raw = candidate(block_if=["target"])
    raw.update({"model_confidence": 0.99, "self_score": 0.98, "status": "eligible_for_generation"})
    raw["policy_id_candidate"] = raw.pop("policy_id")

    result = normalize_candidate(raw)

    assert result.rule is not None
    assert result.rule.policy_id == "VN-ANTI-STATE-001"
    assert result.rule.status == "needs_decision"
    assert {flag.code for flag in result.flags} >= {"model_value_dropped", "status_reset"}
    assert "model_confidence" not in result.rule.model_dump()


def test_model_filled_global_jurisdiction_is_cleared_and_needs_a_decision() -> None:
    raw = candidate(block_if=["target"])
    raw["scope"] = {"jurisdictions": ["global"]}

    rule = normalized(raw)

    assert rule.scope.jurisdictions == ("unspecified",)
    assert rule.unsupported_fields == ("scope.jurisdictions",)
    routed = route_rule(rule, signals(), quote_found=True)
    assert routed.rule.status == "needs_decision"


def test_missing_metadata_is_unspecified() -> None:
    rule = normalized(candidate(block_if=["target"]))

    assert (rule.version, rule.owner, rule.effective_date) == ("unspecified", "unspecified", "unspecified")
    assert rule.scope.jurisdictions == ("unspecified",)


def test_normalization_does_not_mutate_its_input() -> None:
    raw = candidate(block_if=["target"], allow_if=["news"])
    before = copy.deepcopy(raw)

    normalize_candidate(raw)

    assert raw == before


def test_invalid_candidate_returns_errors_instead_of_raising() -> None:
    result = normalize_candidate(candidate(block_if={"all_of": ["a"], "any_of": ["b"]}))

    assert result.rule is None
    assert any("exactly one of all_of, any_of, or none_of" in error for error in result.errors)


# Schema validation of stored rules


def stored(**rule: Any) -> dict[str, Any]:
    return {**candidate(**rule), "status": "needs_decision"}


@pytest.mark.parametrize(
    ("block_if", "message"),
    [
        (["target"], "valid dictionary"),
        ({"all_of": ["a"], "any_of": ["b"]}, "exactly one of all_of, any_of, or none_of"),
        ({"operator_source": "explicit"}, "exactly one of all_of, any_of, or none_of"),
        ({"any_of": []}, "at least one condition"),
        ({"any_of": ["a", "a"]}, "repeats a condition"),
        ({"all_of": ["a", "b"], "operator_source": "default_any_of"}, "only valid with any_of"),
        ({"any_of": ["Not Snake Case"]}, "must start with a letter"),
        ({"one_of": ["a"]}, "Extra inputs are not permitted"),
    ],
)
def test_stored_condition_lists_must_declare_one_operator(block_if: Any, message: str) -> None:
    with pytest.raises(RuleValidationError) as caught:
        validate_rule(stored(block_if=block_if))

    assert message in str(caught.value)


def test_only_block_if_may_use_the_default_operator() -> None:
    with pytest.raises(RuleValidationError, match="only block_if has a default"):
        validate_rule(stored(allow_if={"any_of": ["news"], "operator_source": "default_any_of"}))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rule": {"actor": "a", "action": "b", "object": "c"}}, "at least one condition list"),
        ({"surprise": True}, "Extra inputs are not permitted"),
        ({"version": 2.0}, "valid string"),
        ({"scope": {"jurisdictions": ["Global"]}}, "jurisdiction 'global' is not allowed"),
        ({"severity": "extreme"}, "severity"),
        ({"decision": {"violation_action": "ban"}}, "violation_action"),
        ({"confidence": 1.5}, "less than or equal to 1"),
        ({"confidence": "0.9"}, "valid number"),
        ({"provenance": {"document_id": "D", "source_quote": " "}}, "at least 1 character"),
        ({"provenance": {"document_id": "D", "source_quote": "q", "document_sha256": "abc"}}, "should match pattern"),
    ],
)
def test_invalid_stored_rules_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(RuleValidationError) as caught:
        validate_rule({**stored(block_if={"any_of": ["target"]}), **change})

    assert message in str(caught.value)


@pytest.mark.parametrize(
    "change",
    [
        {"unresolved_conditions": {"allow_if": ["news"]}},
        {"unsupported_fields": ["scope.jurisdictions"]},
        {"confidence": None},
        {"testability": "partially_testable"},
    ],
)
def test_eligible_rule_cannot_carry_open_questions(change: dict[str, Any]) -> None:
    data = {**stored(block_if={"any_of": ["target"]}), "status": "eligible_for_generation", "confidence": 0.95}

    with pytest.raises(RuleValidationError, match="eligible_for_generation rule cannot have"):
        validate_rule({**data, **change})


def test_condition_cannot_be_both_stored_and_unresolved() -> None:
    with pytest.raises(RuleValidationError, match="both stored and unresolved"):
        validate_rule({**stored(block_if={"any_of": ["target"]}), "unresolved_conditions": {"block_if": ["target"]}})


def test_customer_extensions_are_kept() -> None:
    rule = validate_rule({**stored(block_if={"any_of": ["target"]}), "extensions": {"risk_owner": "compliance"}})

    assert rule.extensions == {"risk_owner": "compliance"}


def test_stored_rule_round_trips() -> None:
    rule = normalized(candidate(block_if=["verb", "target"], allow_if={"any_of": ["news_reporting"]}))

    assert validate_rule(rule.model_dump()) == rule
    assert validate_rule(rule.model_dump(mode="json", exclude_none=True)) == rule


def test_json_schema_is_exported_for_hand_authored_rules() -> None:
    schema = policy_rule_json_schema()

    assert "ConditionGroup" in schema["$defs"]
    assert "policy_id" in schema["required"]
    assert [item["required"] for item in schema["$defs"]["ConditionGroup"]["oneOf"]] == [
        ["all_of"],
        ["any_of"],
        ["none_of"],
    ]


def test_a_non_latin_condition_name_is_a_valid_stored_leg() -> None:
    from jsonschema import Draft202012Validator

    data = stored(block_if={"any_of": ["禁止"], "operator_source": "single_condition"})

    assert validate_rule(data).rule.block_if.legs == ("禁止",)
    assert list(Draft202012Validator(policy_rule_json_schema()).iter_errors(data)) == []


def test_a_mixed_script_condition_name_is_a_valid_stored_leg() -> None:
    rule = validate_rule(stored(block_if={"any_of": ["条件a"], "operator_source": "single_condition"}))

    assert rule.rule.block_if.legs == ("条件a",)


@pytest.mark.parametrize("name", ["？？？", "—", "条件-a"])
def test_unicode_punctuation_is_not_a_stored_leg(name: str) -> None:
    with pytest.raises(RuleValidationError, match="must start with a letter"):
        validate_rule(stored(block_if={"any_of": [name], "operator_source": "single_condition"}))


@pytest.mark.parametrize(
    "group",
    [
        {"all_of": None},
        {"any_of": []},
        {"none_of": ["a", "a"]},
        {"all_of": ["a"], "any_of": None},
    ],
)
def test_json_schema_rejects_invalid_operator_values(group: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    data = stored(block_if=group)

    assert list(Draft202012Validator(policy_rule_json_schema()).iter_errors(data))


# Condition evaluation and action order


@pytest.mark.parametrize(
    ("group", "facts", "expected"),
    [
        ({"all_of": ["a", "b"]}, {"a", "b"}, True),
        ({"all_of": ["a", "b"]}, {"a"}, False),
        ({"any_of": ["a", "b"]}, {"b"}, True),
        ({"any_of": ["a", "b"]}, set(), False),
        ({"none_of": ["a", "b"]}, set(), True),
        ({"none_of": ["a", "b"]}, {"a"}, False),
    ],
)
def test_condition_matches(group: dict[str, list[str]], facts: set[str], expected: bool) -> None:
    assert condition_matches(ConditionGroup(**group), facts) is expected


def test_escalate_beats_block() -> None:
    decision = aggregate(
        [
            RuleMatch("BANK-CONFIRM-002", "block", "high"),
            RuleMatch("BANK-FRAUD-001", "escalate", "critical"),
        ]
    )

    assert decision.action == "escalate"
    assert decision.primary_policy_id == "BANK-FRAUD-001"
    assert decision.contributing_policy_ids == ("BANK-FRAUD-001", "BANK-CONFIRM-002")
    assert decision.overridden_policy_ids == ("BANK-CONFIRM-002",)


@pytest.mark.parametrize(("higher", "lower"), list(zip(ACTION_ORDER, ACTION_ORDER[1:], strict=False)))
def test_each_action_beats_the_next_one(higher: str, lower: str) -> None:
    for matches in (
        [RuleMatch("A", lower, "low"), RuleMatch("B", higher, "low")],
        [RuleMatch("A", higher, "low"), RuleMatch("B", lower, "low")],
    ):
        assert aggregate(matches).action == higher


def test_caution_only_when_nothing_higher_applies() -> None:
    assert aggregate([RuleMatch("A", "caution", "medium"), RuleMatch("B", "allow", "low")]).action == "caution"
    assert aggregate([RuleMatch("A", "caution", "medium"), RuleMatch("B", "block", "low")]).action == "block"


def test_primary_policy_is_highest_severity_then_lowest_id() -> None:
    decision = aggregate(
        [
            RuleMatch("RULE-C", "block", "high"),
            RuleMatch("RULE-B", "block", "critical"),
            RuleMatch("RULE-A", "block", "high"),
        ]
    )

    assert decision.primary_policy_id == "RULE-B"
    assert decision.contributing_policy_ids == ("RULE-B", "RULE-A", "RULE-C")


def test_no_applicable_rule_is_not_applicable() -> None:
    decision = aggregate([])

    assert decision.action == "not_applicable"
    assert decision.primary_policy_id is None


@pytest.mark.parametrize(("action", "severity"), [("not_supported", "low"), ("block", "extreme")])
def test_unknown_action_or_severity_is_refused(action: str, severity: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        RuleMatch("A", action, severity)


def test_all_of_missing_leg_is_not_a_block() -> None:
    block = validate_rule(stored(block_if={"all_of": ["verb", "target"]}))
    allow = validate_rule(
        {**stored(allow_if={"any_of": ["target"]}), "policy_id": "VN-ANTI-STATE-001-ALLOW", "severity": "low"}
    )

    assert expected_action([block, allow], {"verb", "target"}, include_review=True).action == "block"
    assert expected_action([block, allow], {"target"}, include_review=True).action == "allow"
    assert expected_action([block, allow], {"verb"}, include_review=True).action == "not_applicable"


def test_one_leg_default_any_of_blocks_and_a_bare_pair_does_not() -> None:
    one = normalized(candidate(block_if=["verb"]))
    pair = normalized(candidate(block_if=["verb", "target"]))

    assert expected_action([one], {"verb"}, include_review=True).action == "block"
    assert pair.rule.block_if is None
    assert expected_action([pair], {"verb"}, include_review=True).action == "not_applicable"


def test_exception_turns_the_rule_off() -> None:
    rule = validate_rule({**stored(block_if={"any_of": ["target"]}), "exceptions": ["academic_context"]})

    assert rule_matches(rule, {"target"})[0].action == "block"
    assert rule_matches(rule, {"target", "academic_context"}) == []


def test_decision_block_names_the_action_for_each_condition() -> None:
    rule = validate_rule(
        {
            **stored(block_if={"any_of": ["target"]}, escalate_if={"any_of": ["violence"]}),
            "decision": {"violation_action": "caution", "critical_action": "escalate"},
        }
    )

    assert [match.action for match in rule_matches(rule, {"target", "violence"})] == ["caution", "escalate"]


# Confidence


def test_confidence_weights_sum_to_one() -> None:
    assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


def test_documented_confidence_example() -> None:
    assert compute_confidence(signals(operator_explicitness=0.5)) == 0.96


def test_one_pass_counts_as_full_agreement() -> None:
    assert compute_confidence(signals()) == 1.0
    assert compute_confidence(signals(pass_agreement=0.5)) == 0.86
    assert compute_confidence(signals(schema_completeness=0.0)) == 0.72


def test_one_leg_default_any_of_does_not_lower_operator_explicitness() -> None:
    rule = normalized(candidate(block_if=["target"]))

    assert rule.rule.block_if is not None
    assert rule.rule.block_if.operator_source == "default_any_of"
    assert operator_explicitness(rule) == 1.0


def test_an_authored_multi_leg_default_any_of_scores_zero_and_stays_eligible() -> None:
    rule = validate_rule(
        {
            **candidate(block_if={"any_of": ["verb", "target"], "operator_source": "default_any_of"}),
            "confidence": 0.95,
        }
    )

    assert operator_explicitness(rule) == 0.0
    routed = route_rule(rule, signals(operator_explicitness=0.0), quote_found=True)

    assert routed.rule.confidence == 0.92
    assert routed.rule.status == "eligible_for_generation"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "1"])
def test_signals_must_be_scores(value: Any) -> None:
    with pytest.raises(ValueError):
        signals(quote_match=value)


@pytest.mark.parametrize(
    ("quote", "span", "found"),
    [
        (QUOTE, f"Section 5. {QUOTE} Section 6.", True),
        ("The assistant must not  explain\nhow", QUOTE, True),
        ("The assistant must not describe how", QUOTE, False),
        ("Không được hướng dẫn", "Không được hưởng dẫn tổ chức.", False),
        ("not", "do not help", True),
        ("not", "cannot help", False),
        ("don't", "don\u2019t stop", True),
        ("ab", "see a\u200bb now", True),
        ("", QUOTE, False),
        ("不得协助", "系统不得协助用户制作武器。", True),
        ("card", "discard data", False),
        ("thẻ", "Không cung cấp thẻ.", True),
        ("thẻ", "Không cung cấp thẻcăn.", False),
    ],
)
def test_quote_in_span(quote: str, span: str, found: bool) -> None:
    assert quote_in_span(quote, span) is found


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("hướng dẫn hoạt động khủng bố.", "hướng dẫn hoạt động khủng bố", True),
        ("foo.", "foo。", True),
        ("禁止。", "禁止", True),
        ("Block it!", "Block it", True),
        ("Block it?", "Block it", True),
        ("Block it…", "Block it", True),
        ("Block it！", "Block it", True),
        ("Block it？", "Block it", True),
        ("Block it．", "Block it", True),
        ("Block it?!", "Block it", True),
        ("هل هذا؟", "هل هذا", True),
        ("တားမြစ်၊", "တားမြစ်", True),
        ("བཀག།", "བཀག", True),
        ("A. Nguyen", "A. Nguyen", True),
        ("hướng dẫn hoạt động khủng bố.", "hướng dẫn bạo loạn vũ trang", False),
        ("redis, kafka,", "redis, kafka", False),
        ("list (a)", "list (a", False),
        ("...", "???", False),
    ],
)
def test_quote_match_key_ignores_trailing_other_punctuation(left: str, right: str, same: bool) -> None:
    assert (quote_match_key(left) == quote_match_key(right)) is same


def test_canonical_text_keeps_a_trailing_period() -> None:
    assert canonical_text("hướng dẫn hoạt động khủng bố.").endswith(".")


@pytest.mark.parametrize(
    ("char", "listed"),
    [
        (",", True),
        ("，", True),
        (";", True),
        ("；", True),
        (".", False),
        ("。", False),
        (")", False),
    ],
)
def test_comma_or_semicolon_uses_unicode_name_tokens(char: str, listed: bool) -> None:
    assert comma_or_semicolon(char) is listed


def test_fold_latin_maps_non_decomposing_letters() -> None:
    assert fold_latin("đĐøØłŁ") == "dDoOlL"
    assert fold_latin("Hướng") == "Huong"


# Routing


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({}, "eligible_for_generation"),
        ({"operator_explicitness": 0.0}, "eligible_for_generation"),
        ({"schema_completeness": 0.5}, "needs_decision"),
        ({"schema_completeness": 0.0}, "needs_decision"),
        ({"pass_agreement": 0.0}, "needs_decision"),
        ({"schema_completeness": 0.0, "operator_explicitness": 0.0}, "quarantined"),
        ({"schema_completeness": 0.0, "pass_agreement": 0.0}, "quarantined"),
    ],
)
def test_confidence_thresholds_route_the_rule(overrides: dict[str, float], status: str) -> None:
    rule = normalized(candidate(block_if=["target"]))

    assert route_rule(rule, signals(**overrides), quote_found=True).rule.status == status


def test_threshold_boundaries_are_inclusive_below() -> None:
    rule = normalized(candidate(block_if=["target"]))
    exactly_eligible = Thresholds(eligible=0.9, decision=0.7)

    at_eligible = route_rule(rule, signals(schema_completeness=9 / 14), quote_found=True, thresholds=exactly_eligible)
    assert at_eligible.rule.confidence == 0.9
    assert at_eligible.rule.status == "eligible_for_generation"

    at_decision = route_rule(
        rule,
        signals(schema_completeness=0.0, operator_explicitness=0.75),
        quote_found=True,
        thresholds=exactly_eligible,
    )
    assert at_decision.rule.confidence == 0.7
    assert at_decision.rule.status == "needs_decision"


def test_quote_outside_the_span_is_quarantined_even_with_high_scores() -> None:
    rule = normalized(candidate(block_if=["target"]))

    routed = route_rule(rule, signals(), quote_found=False)

    assert routed.rule.status == "quarantined"
    assert routed.rule.confidence == 0.64
    assert "source quote is not in the cited span" in routed.reasons


def test_route_from_span_uses_one_authoritative_quote_check() -> None:
    rule = normalized(candidate(block_if=["target"]))

    found = route_rule_from_span(rule, signals(quote_match=0.0), f"Before. {QUOTE} After.")
    missing = route_rule_from_span(rule, signals(quote_match=1.0), "A different sentence.")

    assert found.rule.confidence == 1.0
    assert found.rule.status == "eligible_for_generation"
    assert missing.rule.confidence == 0.64
    assert missing.rule.status == "quarantined"


@pytest.mark.parametrize(
    ("change", "kwargs", "status"),
    [
        ({"testability": "partially_testable"}, {}, "needs_decision"),
        ({"testability": "needs_clarification"}, {}, "quarantined"),
        ({"testability": "non_testable"}, {}, "quarantined"),
        ({}, {"has_conflict": True}, "needs_decision"),
        ({"status": "rejected"}, {}, "rejected"),
    ],
)
def test_routing_reasons_beyond_confidence(change: dict[str, Any], kwargs: dict[str, Any], status: str) -> None:
    rule = normalized(candidate(block_if=["target"])).model_copy(update=change)

    assert (
        route_rule(PolicyRule.model_validate(rule.model_dump()), signals(), quote_found=True, **kwargs).rule.status
        == status
    )


def test_unassessed_testability_needs_a_decision() -> None:
    raw = candidate(block_if=["target"])
    raw.pop("testability")
    rule = normalized(raw)

    routed = route_rule(rule, signals(), quote_found=True)

    assert routed.rule.status == "needs_decision"
    assert "testability has not been assessed" in routed.reasons


def test_one_leg_default_any_of_is_not_a_routing_reason() -> None:
    rule = normalized(candidate(block_if=["verb"]))

    routed = route_rule(rule, signals(), quote_found=True)

    assert routed.rule.rule.block_if is not None
    assert routed.rule.rule.block_if.operator_source == "default_any_of"
    assert routed.rule.status == "eligible_for_generation"


def test_published_confidence_is_computed_not_copied_from_the_model() -> None:
    raw = candidate(block_if=["target"])
    raw["model_confidence"] = 0.99
    rule = normalized(raw)

    routed = route_rule(rule, signals(schema_completeness=0.0, operator_explicitness=0.0), quote_found=True)

    assert routed.rule.confidence == 0.64
    assert routed.rule.status == "quarantined"


@pytest.mark.parametrize(("eligible", "decision"), [(0.6, 0.7), (1.1, 0.7), (0.9, -0.1)])
def test_invalid_thresholds_are_refused(eligible: float, decision: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        Thresholds(eligible=eligible, decision=decision)


# Robustness of untrusted extractor output


@pytest.mark.parametrize("value", ["allow_if", 3, ["allow_if"]])
def test_malformed_unresolved_conditions_return_errors(value: Any) -> None:
    result = normalize_candidate({**candidate(block_if=["target"]), "unresolved_conditions": value})

    assert result.rule is None
    assert result.errors
    assert any("unresolved_conditions" in error for error in result.errors)


@pytest.mark.parametrize(
    "scope",
    [
        {"jurisdictions": [" global "]},
        {"jurisdictions": "global"},
        {"jurisdiction": "global"},
        {"jurisdiction": "Global", "jurisdictions": ["VN"]},
    ],
)
def test_global_jurisdiction_shapes_are_cleared_for_review(scope: dict[str, Any]) -> None:
    rule = normalized({**candidate(block_if=["target"]), "scope": scope})

    assert "global" not in {value.casefold() for value in rule.scope.jurisdictions}
    assert "scope.jurisdictions" in rule.unsupported_fields
    assert route_rule(rule, signals(), quote_found=True).rule.status == "needs_decision"


def test_empty_jurisdiction_is_unspecified() -> None:
    rule = normalized({**candidate(block_if=["target"]), "scope": {"jurisdictions": [], "domains": []}})

    assert rule.scope.jurisdictions == ("unspecified",)
    assert rule.scope.domains == ("unspecified",)
    assert rule.unsupported_fields == ()


def test_a_string_unsupported_field_is_not_split_into_letters() -> None:
    rule = normalized(
        {**candidate(block_if=["target"]), "unsupported_fields": "note", "scope": {"jurisdictions": ["global"]}}
    )

    assert "n" not in rule.unsupported_fields
    assert rule.unsupported_fields == ("scope.jurisdictions",)


def test_supplied_confidence_is_dropped_because_routing_computes_it() -> None:
    result = normalize_candidate({**candidate(block_if=["target"]), "confidence": 0.99})

    assert result.rule is not None
    assert result.rule.confidence is None
    assert "confidence_dropped" in [flag.code for flag in result.flags]


def test_nested_model_score_does_not_drop_the_rule() -> None:
    result = normalize_candidate(candidate(block_if=["target"], model_confidence=0.2))

    assert result.errors == []
    assert result.rule is not None
    assert "model_confidence" not in result.rule.model_dump()


def test_yaml_date_and_uppercase_digest_are_stored() -> None:
    import datetime

    raw = candidate(block_if=["target"])
    raw["effective_date"] = datetime.date(2024, 1, 1)
    raw["provenance"]["document_sha256"] = "AB" * 32

    rule = normalized(raw)

    assert rule.effective_date == "2024-01-01"
    assert rule.provenance.document_sha256 == "ab" * 32


def test_reviewer_rejection_survives_normalization() -> None:
    raw = {**candidate(block_if={"any_of": ["target"]}), "status": "rejected"}

    rule = normalized(raw)
    routed = route_rule(rule, signals(), quote_found=True)

    assert rule.status == "rejected"
    assert routed.rule.status == "rejected"


def test_stored_sequences_cannot_be_mutated() -> None:
    rule = validate_rule(
        {
            **stored(block_if={"any_of": ["target"]}),
            "exceptions": ["academic_context"],
            "unsupported_fields": ["scope.jurisdictions"],
            "extensions": {"risk_owner": "compliance"},
        }
    )

    with pytest.raises(TypeError):
        rule.exceptions[0] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        rule.unsupported_fields[0] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        rule.extensions["risk_owner"] = "other"
    with pytest.raises(TypeError):
        rule.unresolved_conditions["allow_if"] = ("news",)


def test_every_matching_policy_is_contributing() -> None:
    decision = aggregate(
        [
            RuleMatch("VN-PRIVACY-001", "block", "critical"),
            RuleMatch("INTERNAL-FRAUD-001", "escalate", "critical"),
            RuleMatch("SOP-TRANSFER-001", "block", "critical"),
        ]
    )

    assert decision.action == "escalate"
    assert decision.primary_policy_id == "INTERNAL-FRAUD-001"
    assert decision.contributing_policy_ids == ("INTERNAL-FRAUD-001", "SOP-TRANSFER-001", "VN-PRIVACY-001")
    assert decision.overridden_policy_ids == ("SOP-TRANSFER-001", "VN-PRIVACY-001")


def test_primary_policy_comes_from_the_winning_action() -> None:
    decision = aggregate(
        [RuleMatch("CRITICAL-BLOCK", "block", "critical"), RuleMatch("LOW-ESCALATE", "escalate", "low")]
    )

    assert decision.action == "escalate"
    assert decision.primary_policy_id == "LOW-ESCALATE"
    assert decision.contributing_policy_ids == ("LOW-ESCALATE", "CRITICAL-BLOCK")


def test_rejected_and_quarantined_rules_do_not_set_the_label() -> None:
    blocked = validate_rule(stored(block_if={"any_of": ["target"]}))

    assert (
        expected_action([blocked.model_copy(update={"status": "quarantined"})], {"target"}).action == "not_applicable"
    )
    assert expected_action([blocked.model_copy(update={"status": "rejected"})], {"target"}).action == "not_applicable"
    assert expected_action([blocked], {"target"}).action == "not_applicable"
    assert expected_action([blocked], {"target"}, include_review=True).action == "block"


def test_ambiguous_action_is_used_when_the_event_is_ambiguous() -> None:
    rule = validate_rule({**stored(block_if={"any_of": ["target"]}), "decision": {"ambiguous_action": "caution"}})

    assert rule_matches(rule, {"target"}, ambiguous=True)[0].action == "caution"
    assert rule_matches(rule, set(), ambiguous=True) == []
    assert expected_action([rule], {"target"}, include_review=True).action == "block"
    assert (
        expected_action([rule], {"target"}, include_review=True, ambiguous_rules={"VN-ANTI-STATE-001"}).action
        == "caution"
    )


def test_quote_found_must_be_a_boolean() -> None:
    rule = normalized(candidate(block_if=["target"]))

    with pytest.raises(ValueError, match="quote_found"):
        route_rule(rule, signals(), quote_found="false")  # type: ignore[arg-type]


def test_a_condition_name_string_is_not_treated_as_facts() -> None:
    rule = normalized(candidate(block_if=["target"]))

    with pytest.raises(TypeError, match="facts"):
        expected_action([rule], "target", include_review=True)  # type: ignore[arg-type]
