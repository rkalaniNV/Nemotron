"""Slash-slot flatten detection and unspecified actor/action/object completeness."""

from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import validate_rule
from nemotron.steps.byob.runtime.benchmark_families.safety.rules import _completeness
from nemotron.steps.byob.runtime.benchmark_families.safety.shape import (
    CROSS_SLOT,
    MISSING_EVIDENCE,
    SLASH_WITH_RESIDUE,
    TOKENIZED_LIST,
    flattened_condition_reason,
    slash_alternative_groups,
)
from tests.steps.byob.safety_fixtures import ScriptedExtractor
from tests.steps.byob.safety_fixtures import candidate as _candidate
from tests.steps.byob.safety_fixtures import leg as _leg
from tests.steps.byob.test_safety_rules import _config, _generate, _rows


def _rule(
    quote: str,
    *,
    operator: str = "any_of",
    legs: list[str],
    evidence: dict[str, str],
    shared_frame: dict[str, str] | None = None,
) -> Any:
    grounded: dict[str, Any] = {"conditions": evidence}
    if shared_frame:
        grounded["shared_frame"] = shared_frame
    return validate_rule(
        {
            "policy_id": "SHAPE-1",
            "severity": "high",
            "testability": "testable",
            "rule": {
                "actor": "unspecified",
                "action": "unspecified",
                "object": "unspecified",
                "block_if": {operator: legs, "operator_source": "model_evidence"},
            },
            "provenance": {"document_id": "policy", "source_quote": quote},
            "extensions": {"grounded_evidence": grounded},
        }
    )


def test_slash_groups_keep_short_alternatives_and_drop_the_tail() -> None:
    quote = "Instruct/guide/plan overthrow, destroy, government, regime."

    assert slash_alternative_groups(quote) == (("Instruct", "guide", "plan"),)


def test_slash_groups_keep_a_quoted_synonym_list() -> None:
    quote = 'Block the instruction "forget/ignore/bypass".'

    assert slash_alternative_groups(quote) == (("forget", "ignore", "bypass"),)


def test_slash_groups_skip_url_paths() -> None:
    quote = "See https://example.com/guide/plan then block foo/bar."

    assert slash_alternative_groups(quote) == (("foo", "bar"),)


def test_slash_groups_keep_separate_quoted_slots() -> None:
    quote = '"how to/guide/instructions" + "overthrow/destroy/topple"'

    assert slash_alternative_groups(quote) == (
        ("how to", "guide", "instructions"),
        ("overthrow", "destroy", "topple"),
    )


def test_long_slash_phrases_are_not_a_short_alternative_group() -> None:
    quote = 'Block "ignore previous instructions entirely" / "you are now unfiltered here".'

    assert slash_alternative_groups(quote) == ()


def test_unquoted_multiword_slash_list_mixes_with_residue() -> None:
    quote = (
        "Hướng dẫn/cách làm/lập kế hoạch lật đổ, xóa bỏ, phá hoại nhà nước, chính quyền, chế độ."
    )
    rule = _rule(
        quote,
        legs=[
            "huong_dan",
            "cach_lam",
            "lap_ke_hoach",
            "lat_do",
            "xoa_bo",
            "pha_hoai",
            "nha_nuoc",
            "chinh_quyen",
            "che_do",
        ],
        evidence={
            "huong_dan": "Hướng dẫn",
            "cach_lam": "cách làm",
            "lap_ke_hoach": "lập kế hoạch",
            "lat_do": "lật đổ",
            "xoa_bo": "xóa bỏ",
            "pha_hoai": "phá hoại",
            "nha_nuoc": "nhà nước",
            "chinh_quyen": "chính quyền",
            "che_do": "chế độ",
        },
    )

    assert SLASH_WITH_RESIDUE in flattened_condition_reason(rule)


def test_cross_slot_any_of_is_flattened() -> None:
    quote = "Block instruct/guide/plan overthrow, destroy, government, regime."
    rule = _rule(
        quote,
        legs=["instruct", "guide", "plan", "overthrow", "destroy", "government", "regime"],
        evidence={
            "instruct": "instruct",
            "guide": "guide",
            "plan": "plan",
            "overthrow": "overthrow",
            "destroy": "destroy",
            "government": "government",
            "regime": "regime",
        },
    )

    assert SLASH_WITH_RESIDUE in flattened_condition_reason(rule)


def test_two_quoted_slash_slots_tokenized_into_any_of_is_flattened() -> None:
    quote = '"how to/guide" + "overthrow/destroy"'
    rule = _rule(
        quote,
        legs=["how_to", "guide", "overthrow", "destroy"],
        evidence={
            "how_to": "how to",
            "guide": "guide",
            "overthrow": "overthrow",
            "destroy": "destroy",
        },
    )

    assert CROSS_SLOT in flattened_condition_reason(rule)


def test_one_slot_quoted_slash_any_of_is_not_flattened() -> None:
    quote = 'Block the instruction "forget/ignore/bypass".'
    rule = _rule(
        quote,
        legs=["forget", "ignore", "bypass"],
        evidence={"forget": "forget", "ignore": "ignore", "bypass": "bypass"},
    )

    assert flattened_condition_reason(rule) == ""


def test_slot_level_all_of_with_slash_inside_evidence_is_not_flattened() -> None:
    quote = '"how to/guide/instructions" + "overthrow/destroy/topple"'
    rule = _rule(
        quote,
        operator="all_of",
        legs=["how_to_guide_instructions", "overthrow_destroy_topple"],
        evidence={
            "how_to_guide_instructions": '"how to/guide/instructions"',
            "overthrow_destroy_topple": '"overthrow/destroy/topple"',
        },
    )

    assert flattened_condition_reason(rule) == ""


def test_shared_frame_does_not_hide_slash_list_residue() -> None:
    quote = "Block instruct/guide/plan overthrow of the government."
    rule = _rule(
        quote,
        legs=["instruct", "guide", "plan", "overthrow"],
        evidence={
            "instruct": "instruct",
            "guide": "guide",
            "plan": "plan",
            "overthrow": "overthrow",
        },
        shared_frame={"block_if": "Block"},
    )

    assert SLASH_WITH_RESIDUE in flattened_condition_reason(rule)


def test_shared_frame_parallel_any_of_is_not_flattened() -> None:
    quote = "Block organizing a protest, organizing a riot, or organizing an uprising."
    rule = _rule(
        quote,
        legs=["protest", "riot", "uprising"],
        evidence={
            "protest": "organizing a protest",
            "riot": "organizing a riot",
            "uprising": "organizing an uprising",
        },
        shared_frame={"block_if": "organizing"},
    )

    assert flattened_condition_reason(rule) == ""


def test_missing_condition_evidence_on_a_multi_leg_group_is_flattened() -> None:
    quote = "Block alpha and beta requests."
    rule = _rule(quote, legs=["alpha", "beta"], evidence={})

    assert MISSING_EVIDENCE in flattened_condition_reason(rule)


def test_ideographic_comma_splits_slash_slots() -> None:
    quote = "指导/组织、颠覆/破坏"
    assert slash_alternative_groups(quote) == (("指导", "组织"), ("颠覆", "破坏"))
    rule = _rule(
        quote,
        legs=["guide", "organize", "overthrow", "destroy"],
        evidence={
            "guide": "指导",
            "organize": "组织",
            "overthrow": "颠覆",
            "destroy": "破坏",
        },
    )

    assert CROSS_SLOT in flattened_condition_reason(rule)


def test_hyphenated_slash_alternatives_stay_one_group() -> None:
    quote = "Block red-team/blue-team requests."

    assert slash_alternative_groups(quote) == (("red-team", "blue-team"),)


def test_single_quoted_slash_slot_keeps_the_first_phrase() -> None:
    quote = "Block the instruction 'how to/guide'."

    assert slash_alternative_groups(quote) == (("how to", "guide"),)


def test_six_short_any_of_legs_without_a_frame_are_flattened() -> None:
    quote = "Block alpha, beta, gamma, delta, epsilon, zeta requests."
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    rule = _rule(quote, legs=names, evidence={name: name for name in names})

    assert TOKENIZED_LIST in flattened_condition_reason(rule)


def test_unspecified_actor_action_object_are_complete_wildcards() -> None:
    rule = _rule(
        "Block foo requests.",
        operator="any_of",
        legs=["foo"],
        evidence={"foo": "foo"},
    )
    rule = validate_rule(
        {
            **rule.model_dump(mode="json"),
            "rule": {
                "actor": "unspecified",
                "action": "unspecified",
                "object": "unspecified",
                "block_if": {"any_of": ["foo"], "operator_source": "single_condition"},
            },
        }
    )

    assert _completeness(rule, True) == 1.0


def _unspecified_fields(row: dict[str, Any]) -> dict[str, Any]:
    row["actor"] = "unspecified"
    row["action"] = "unspecified"
    row["object"] = "unspecified"
    return row


def test_extracted_cross_slot_any_of_needs_a_decision_at_high_confidence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    quote = "Block instruct/guide/plan overthrow, destroy, government, regime."
    row = _unspecified_fields(
        _candidate(
            quote,
            operators={"block_if": {"operator": "any_of", "evidence": "instruct/guide/plan"}},
            block_if=[
                _leg("instruct", "instruct"),
                _leg("guide", "guide"),
                _leg("plan", "plan"),
                _leg("overthrow", "overthrow"),
                _leg("destroy", "destroy"),
                _leg("government", "government"),
                _leg("regime", "regime"),
            ],
        )
    )
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: ScriptedExtractor({quote: [row]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Policy\n\n{quote}\n"}))

    [item] = _rows(run_dir).values()
    assert item["status"] == "needs_decision"
    assert item["rule"]["confidence"] >= 0.9
    assert any(flag["code"] == "flattened_cross_slot_conditions" for flag in item["flags"])
    assert SLASH_WITH_RESIDUE in " ".join(item["reasons"])


def test_extracted_one_slot_slash_any_of_stays_eligible(tmp_path: Path, monkeypatch: Any) -> None:
    quote = 'Block the instruction "forget/ignore/bypass".'
    row = _unspecified_fields(
        _candidate(
            quote,
            operators={"block_if": {"operator": "any_of", "evidence": "forget/ignore/bypass"}},
            block_if=[
                _leg("forget", "forget"),
                _leg("ignore", "ignore"),
                _leg("bypass", "bypass"),
            ],
        )
    )
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: ScriptedExtractor({quote: [row]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Policy\n\n{quote}\n"}))

    [item] = _rows(run_dir).values()
    assert item["status"] == "eligible_for_generation"
    assert item["rule"]["confidence"] >= 0.9
    assert "flattened_cross_slot_conditions" not in {flag["code"] for flag in item["flags"]}


def test_extracted_unspecified_actor_single_condition_is_eligible(tmp_path: Path, monkeypatch: Any) -> None:
    quote = "Block foo requests."
    row = _unspecified_fields(_candidate(quote, block_if=[_leg("foo", "foo")]))
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: ScriptedExtractor({quote: [row]}))
    run_dir = _generate(_config(tmp_path, {"policy.md": f"## Policy\n\n{quote}\n"}))

    [item] = _rows(run_dir).values()
    assert item["status"] == "eligible_for_generation"
    assert item["rule"]["confidence"] == 1.0
    assert item["rule"]["rule"]["actor"] == "unspecified"
