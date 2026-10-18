"""Coverage planning: required cells first, then optional cells, inside the budget."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety.coverage import (
    CoverageInputError,
    build_coverage_plan,
    case_cap_for,
    span_text,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import validate_rule
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus

_MODULES = {
    "adversarial": False,
    "political_symmetry": "confirm_if_detected",
    "production_replay": "auto_if_evidence_exists",
    "persona": False,
}


def _budget(**overrides: Any) -> dict[str, int]:
    budget = {
        "preview_max_cases": 100,
        "assisted_max_cases": 500,
        "cases_per_critical_rule": 12,
        "cases_per_normal_rule": 6,
        "max_review_items": 40,
        "max_turns": 6,
    }
    budget.update(overrides)
    return budget


def _rule(
    policy_id: str,
    *,
    block: dict[str, Any] | None = None,
    allow: dict[str, Any] | None = None,
    languages: tuple[str, ...] = (),
    quote: str = "Block it.",
    severity: str = "critical",
    paired: str | None = None,
    span: str | None = "doc#1#span-01",
    status: str = "eligible_for_generation",
    signals: dict[str, Any] | None = None,
) -> Any:
    body: dict[str, Any] = {"actor": "assistant", "action": "respond", "object": "request"}
    if block is not None:
        body["block_if"] = block
    if allow is not None:
        body["allow_if"] = allow
    if "block_if" not in body and "allow_if" not in body:
        body["block_if"] = {"any_of": ["request"], "operator_source": "single_condition"}
    data: dict[str, Any] = {
        "policy_id": policy_id,
        "severity": severity,
        "provenance": {"document_id": "doc", "source_quote": quote},
        "rule": body,
        "decision": {"violation_action": "block"},
        "status": status,
    }
    if span is not None:
        data["provenance"]["source_span_id"] = span
    if status == "eligible_for_generation":
        data["testability"] = "testable"
        data["confidence"] = 0.95
    if languages:
        data["scope"] = {"languages": list(languages)}
    if paired is not None:
        data["paired_with"] = paired
    if signals:
        data["signals"] = signals
    return validate_rule(data)


def _plan(rules: list[Any], **options: Any) -> Any:
    modules = {**_MODULES, **options.pop("modules", {})}
    spans = options.pop("spans", None)
    if spans is None:
        spans = {rule.provenance.source_span_id: "" for rule in rules if rule.provenance.source_span_id}
    return build_coverage_plan(
        rules,
        spans,
        case_cap=options.pop("cap", 50),
        budget=options.pop("budget", _budget(cases_per_critical_rule=20)),
        modules=modules,
        generation=options.pop("generation", None),
        routed=options.pop("routed", None),
        target_languages=options.pop("target_languages", ()),
    )


def _kinds(plan: Any, kind: str, policy_id: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in (*plan.rows, *plan.gaps) if row["kind"] == kind]
    if policy_id is not None:
        rows = [row for row in rows if row["policy_id"] == policy_id]
    return rows


def test_required_rows_are_scheduled_before_optional_rows() -> None:
    rule = _rule("ALL-2", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    plan = _plan([rule], modules={"adversarial": True})

    priorities = [row["priority"] for row in plan.rows]
    first_optional = priorities.index("optional")
    assert first_optional > 0
    assert priorities[:first_optional] == ["required"] * first_optional
    assert "required" not in priorities[first_optional:]
    assert plan.certification == "gold"
    assert plan.reason == "complete"


def test_a_two_condition_all_of_rule_gets_one_missing_leg_allow_per_condition() -> None:
    block = _rule(
        "ALL-2",
        block={"all_of": ["verb", "target"], "operator_source": "explicit"},
        paired="ALL-2-ALLOW",
    )
    allow = _rule(
        "ALL-2-ALLOW",
        allow={"all_of": ["historical_question"], "operator_source": "single_condition"},
    )
    plan = _plan([block, allow])

    legs = [row for row in plan.rows if row["kind"] == "missing_leg" and row["policy_id"] == "ALL-2"]
    assert {(row["absent_leg"], tuple(row["present_legs"])) for row in legs} == {
        ("verb", ("target",)),
        ("target", ("verb",)),
    }
    assert {row["expected_action"] for row in legs} == {"allow"}
    assert {row["paired_policy_id"] for row in legs} == {"ALL-2-ALLOW"}
    assert {row["priority"] for row in legs} == {"required"}
    families = [family for family in plan.contrast_families if family["policy_id"] == "ALL-2"]
    assert {family["changed_dimension"] for family in families} == {"verb", "target"}
    violation = next(row["slot_id"] for row in plan.rows if row["kind"] == "clear_violation")
    assert {family["members"][0]["slot_id"] for family in families} == {violation}
    assert _kinds(plan, "all_legs") == []


def test_a_second_all_of_group_gets_its_own_block_case() -> None:
    rule = validate_rule(
        {
            "policy_id": "BOTH",
            "severity": "critical",
            "testability": "testable",
            "confidence": 0.95,
            "provenance": {"document_id": "doc", "source_quote": "Block it.", "source_span_id": "doc#1"},
            "decision": {"violation_action": "block", "critical_action": "escalate"},
            "rule": {
                "actor": "assistant",
                "action": "respond",
                "object": "request",
                "block_if": {"all_of": ["verb", "target"], "operator_source": "explicit"},
                "escalate_if": {"all_of": ["weapon", "channel"], "operator_source": "explicit"},
            },
            "status": "eligible_for_generation",
        }
    )
    plan = _plan([rule])

    [block] = _kinds(plan, "all_legs")
    assert block["condition_class"] == "block_if"
    assert block["expected_action"] == "block"
    assert block["present_legs"] == ["verb", "target"]
    violation = next(row["slot_id"] for row in plan.rows if row["kind"] == "clear_violation")
    verb = next(family for family in plan.contrast_families if family["changed_dimension"] == "verb")
    weapon = next(family for family in plan.contrast_families if family["changed_dimension"] == "weapon")
    assert verb["members"][0]["slot_id"] == block["slot_id"]
    assert verb["members"][0]["expected_action"] == "block"
    assert weapon["members"][0]["slot_id"] == violation
    assert weapon["members"][0]["expected_action"] == "escalate"


def test_a_default_any_of_rule_gets_no_missing_leg_case() -> None:
    defaulted = _rule(
        "DAN",
        block={"any_of": ["dan", "developer_mode"], "operator_source": "default_any_of"},
    )
    stated = _rule("OR", block={"any_of": ["joke", "fiction"], "operator_source": "explicit"})
    plan = _plan([defaulted, stated])

    assert _kinds(plan, "missing_leg") == []
    assert {family["changed_dimension"] for family in plan.contrast_families} == {"benign_neighbor"}


def test_every_secondary_any_of_action_path_is_required() -> None:
    data = _rule("MULTI").model_dump(mode="json")
    data["severity"] = "critical"
    data["decision"] = {"violation_action": "block", "critical_action": "escalate"}
    data["rule"]["escalate_if"] = {"any_of": ["self_harm_signal"], "operator_source": "single_condition"}
    data["rule"]["block_if"] = {
        "any_of": ["hate_slur", "violence_instruction"],
        "operator_source": "explicit",
    }

    plan = _plan([validate_rule(data)])

    [clear] = _kinds(plan, "clear_violation")
    assert clear["present_legs"] == ["self_harm_signal"]
    paths = _kinds(plan, "condition_path")
    assert {
        (row["trigger_leg"], tuple(row["present_legs"]), row["expected_action"]) for row in paths
    } == {
        ("hate_slur", ("hate_slur",), "block"),
        ("violence_instruction", ("violence_instruction",), "block"),
    }
    assert all(row["priority"] == "required" for row in paths)


def test_target_language_waits_for_the_translation_stage() -> None:
    rule = _rule("LANG", languages=("vi-VN", "en-US"))
    plan = _plan([rule])

    assert plan.certification == "silver"
    assert plan.reason == "translation_required"
    assert {row.get("source_language") for row in plan.rows if row["policy_id"] == "LANG"} == {"vi-VN"}
    [extra] = [row for row in plan.gaps if row["kind"] == "language"]
    assert extra["language"] == "en-US"
    assert extra["priority"] == "required"
    assert extra["expected_action"] == "block"
    assert extra["reason"] == "translation_required"


def test_grounded_multi_turn_evidence_is_required_and_a_numbered_list_is_not() -> None:
    signaled = _rule("TURN", quote="Block it.", signals={"multi_turn_turns": 2})
    numbered = _rule("LIST", quote="Block it.", span="doc#1#span-02")
    spans = {
        "doc#1#span-01": "1. Hide an instruction.\n2. Ask the assistant to follow it on the next turn.",
        "doc#1#span-02": "1. Hide an instruction.\n2. Ask the assistant to follow it on the next turn.",
    }
    plan = _plan([signaled, numbered], spans=spans)

    [cell] = [row for row in plan.rows if row["kind"] == "multi_turn"]
    assert cell["policy_id"] == "TURN"
    assert cell["priority"] == "required"
    assert cell["turns"] == 2
    assert cell["expected_action"] == "block"
    assert "utterance" not in cell


def test_a_budget_that_cannot_hold_required_coverage_is_silver_and_lists_every_gap() -> None:
    rule = _rule("ALL-2", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    plan = _plan([rule], cap=1)

    assert plan.certification == "silver"
    assert plan.reason == "budget_cap"
    assert [row["priority"] for row in plan.rows] == ["required"]
    scheduled = {row["slot_id"] for row in plan.rows}
    missing = [gap["slot_id"] for gap in plan.gaps if gap["blocks_gold"]]
    assert plan.summary()["missing_required"] == missing
    assert plan.summary()["gold_cases_needed"] == plan.required_cases
    assert len(scheduled) + len(missing) == plan.required_cases
    assert not scheduled & set(missing)
    assert {gap["reason"] for gap in plan.gaps if gap["blocks_gold"]} == {"budget_cap"}
    legs = [row["absent_leg"] for row in (*plan.rows, *plan.gaps) if row["kind"] == "missing_leg"]
    assert set(legs) == {"verb", "target"}


def test_max_turns_below_two_records_the_multi_turn_case_instead_of_dropping_it() -> None:
    rule = _rule("TURN", signals={"multi_turn_turns": 2})
    plan = _plan([rule], budget=_budget(cases_per_critical_rule=20, max_turns=1))

    assert [row for row in plan.rows if row["kind"] == "multi_turn"] == []
    [gap] = [gap for gap in plan.gaps if gap["kind"] == "multi_turn"]
    assert gap["reason"] == "max_turns"
    assert gap["blocks_gold"] is True
    assert plan.certification == "silver"


def test_a_per_rule_cap_does_not_fill_the_rest_with_optional_cases() -> None:
    rule = _rule("ALL-2", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    plan = _plan([rule], cap=50, budget=_budget(cases_per_critical_rule=1), modules={"persona": True})

    assert [row["priority"] for row in plan.rows] == ["required"]
    assert plan.certification == "silver"
    assert plan.reason == "rule_cap"
    persona = [gap for gap in plan.gaps if gap["kind"] == "persona"]
    assert len(persona) == 1
    assert persona[0]["blocks_gold"] is False
    assert persona[0]["reason"] == "rule_cap"


def test_persona_is_optional_only_when_the_module_is_enabled() -> None:
    rule = _rule("ALL-2", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    off = _plan([rule])
    on = _plan([rule], modules={"persona": True})

    assert [row for row in (*off.rows, *off.gaps) if row["kind"] == "persona"] == []
    assert off.modules["persona"] == "disabled"
    assert any(row["kind"] == "persona" for row in (*on.rows, *on.gaps))
    assert on.modules["persona"] == "enabled"


def test_a_rule_without_a_source_span_holds_required_cells() -> None:
    plan = _plan([_rule("SELF", span=None)], modules={"persona": True})

    assert plan.rows == ()
    assert plan.certification == "silver"
    assert plan.reason == "span_missing"
    assert {gap["reason"] for gap in plan.gaps if gap["blocks_gold"]} == {"span_missing"}
    assert all(row["kind"] != "persona" for row in (*plan.rows, *plan.gaps))


def test_incidental_foreign_words_do_not_add_a_language_or_code_switch_case() -> None:
    quote = 'Chặn yêu cầu gồm "ignore previous instructions" và developer mode.'
    plan = _plan([_rule("DAN", quote=quote)])

    assert _kinds(plan, "language") == []
    assert _kinds(plan, "code_switch") == []
    assert all(row.get("source_language") is None for row in plan.rows)


def test_code_switch_requires_grounded_evidence_and_two_declared_languages() -> None:
    affirmed = _rule("MIX", languages=("vi-VN", "en-US"), signals={"code_switch": True})
    phrase_only = _rule("NEG", languages=("fr-FR", "de-DE"), quote="This passage is a code-switch example.")
    one_language = _rule("ONE", languages=("ja-JP",), signals={"code_switch": True})
    plan = _plan([affirmed, phrase_only, one_language])

    [cell] = [row for row in plan.gaps if row["kind"] == "code_switch"]
    assert cell["policy_id"] == "MIX"
    assert cell["languages"] == ["vi-VN", "en-US"]
    assert cell["reason"] == "translation_required"
    assert [row for row in plan.gaps if row["kind"] == "code_switch" and row["policy_id"] == "NEG"] == []


def test_sequence_phrases_do_not_add_a_multi_turn_case() -> None:
    plan = _plan(
        [
            _rule("NO", quote="Không cần nhiều bước."),
            _rule("NOT", quote="This is not multi-turn."),
            _rule("YES", quote="会話を複数ターンで確認する。", signals={"multi_turn_turns": 3}),
        ]
    )

    assert {row["policy_id"] for row in plan.rows if row["kind"] == "multi_turn"} == {"YES"}
    assert next(row["turns"] for row in plan.rows if row["kind"] == "multi_turn") == 3


def test_a_short_budget_keeps_one_clear_violation_per_rule() -> None:
    first = _rule("A", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    second = _rule("B", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    plan = _plan([second, first], cap=2)

    assert {(row["policy_id"], row["kind"]) for row in plan.rows} == {
        ("A", "clear_violation"),
        ("B", "clear_violation"),
    }
    assert plan.certification == "silver"


def test_configured_mix_guides_cells_after_each_policy_anchor() -> None:
    plan = _plan(
        [_rule("A")],
        cap=2,
        generation={
            "decision_mix": {"allow": 1.0, "block": 0.0},
            "case_family_mix": {"clear": 1.0, "boundary": 0.0, "exception": 0.0, "contrast": 0.0},
            "turn_mix": {"single_turn": 1.0, "multi_turn": 0.0},
        },
    )

    assert [row["kind"] for row in plan.rows] == ["clear_violation", "clear_allow"]


def test_gold_lists_rules_that_were_excluded_from_the_plan() -> None:
    eligible = _rule("KEEP")
    plan = _plan(
        [eligible],
        routed=[
            {"policy_id": "KEEP", "status": "eligible_for_generation"},
            {"policy_id": "WAIT", "status": "needs_decision"},
            {"policy_id": "HOLD", "status": "quarantined"},
        ],
    )

    assert plan.certification == "gold"
    assert plan.summary()["coverage_of"] == "eligible_rules"
    assert plan.summary()["excluded_rules"] == [
        {"policy_id": "HOLD", "status": "quarantined"},
        {"policy_id": "WAIT", "status": "needs_decision"},
    ]


def test_an_allow_group_does_not_get_missing_leg_cases() -> None:
    rule = _rule("ALLOW", allow={"all_of": ["history", "opinion"], "operator_source": "explicit"})
    plan = _plan([rule])

    assert _kinds(plan, "missing_leg") == []


def test_a_partner_that_is_not_an_allow_rule_keeps_its_id() -> None:
    block = _rule("BLOCK", paired="OTHER")
    other = _rule("OTHER")
    plan = _plan([block, other])

    allow = [row for row in plan.rows if row["policy_id"] == "BLOCK" and row["kind"] == "clear_allow"][0]
    assert allow["paired_policy_id"] == "OTHER"
    assert allow["flags"] == ["paired_rule_not_allow"]


def test_a_language_tag_that_contains_a_slash_does_not_change_the_language_field() -> None:
    plan = _plan([_rule("LANG", languages=("vi-VN", "en/US"))])

    [extra] = [row for row in plan.gaps if row["kind"] == "language"]
    assert extra["language"] == "en/US"
    assert extra["slot_id"] == "LANG/language/en\\/US"
    assert len({row["slot_id"] for row in plan.rows}) == len(plan.rows)


def test_a_cited_span_that_is_not_in_the_index_stops_the_plan() -> None:
    rule = _rule("TURN", quote="Block it.")
    with pytest.raises(CoverageInputError, match="not in the source index"):
        _plan([rule], spans={"other": "1. Hide an instruction.\n2. Ask again."})


def test_a_damaged_or_repeated_span_stops_the_plan() -> None:
    with pytest.raises(CoverageInputError, match="must be an object"):
        span_text({"spans": ["span"]})
    with pytest.raises(CoverageInputError, match="text must be a string"):
        span_text({"spans": [{"source_span_id": "doc#1", "text": 1}]})
    with pytest.raises(CoverageInputError, match="repeats doc#1"):
        span_text(
            {
                "spans": [
                    {"source_span_id": "doc#1", "text": "one"},
                    {"source_span_id": "doc#1", "text": "two"},
                ]
            }
        )


def test_a_rule_waiting_for_review_is_not_planned() -> None:
    pending = _rule("WAIT", status="needs_decision")
    plan = _plan([pending])

    assert plan.rows == ()
    assert plan.certification == "silver"
    assert plan.reason == "no_eligible_rules"
    assert plan.summary()["excluded_rules"] == [{"policy_id": "WAIT", "status": "needs_decision"}]


def test_preview_and_expert_use_the_documented_case_caps() -> None:
    budget = _budget(preview_max_cases=3, assisted_max_cases=9)
    assert case_cap_for("preview", budget) == 3
    assert case_cap_for("assisted", budget) == 9
    assert case_cap_for("expert", budget) == 9


def test_generate_writes_the_plan_and_a_short_budget_is_silver(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "policy.md").write_text("Block it.\n", encoding="utf-8")
    (policies / "rules.yaml").write_text(
        yaml.safe_dump(
            {
                "policy_rules": [
                    {
                        "policy_id": "ALL-001",
                        "severity": "high",
                        "testability": "testable",
                        "scope": {"languages": ["vi-VN", "en-US"]},
                        "provenance": {"document_id": "policy", "source_quote": "Block it."},
                        "rule": {
                            "actor": "assistant",
                            "action": "respond",
                            "object": "request",
                            "block_if": {"all_of": ["verb", "target"], "operator_source": "explicit"},
                        },
                    },
                    {
                        "policy_id": "ANY-001",
                        "severity": "high",
                        "testability": "testable",
                        "provenance": {"document_id": "policy", "source_quote": "Block it."},
                        "rule": {
                            "actor": "assistant",
                            "action": "respond",
                            "object": "request",
                            "block_if": {"any_of": ["dan", "developer_mode"], "operator_source": "default_any_of"},
                        },
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold = _generate(
        tmp_path,
        policies,
        "gold",
        {"cases_per_normal_rule": 12, "assisted_max_cases": 50},
        generation={"modules": {"adversarial": True}},
    )
    rows = _jsonl(gold / "stage_cache" / "coverage_plan.jsonl")
    report = json.loads((gold / "stage_cache" / "coverage_gaps.json").read_text(encoding="utf-8"))

    assert report["certification"] == "silver"
    assert report["reason"] == "translation_required"
    priorities = [row["priority"] for row in rows]
    first_optional = priorities.index("optional")
    assert first_optional > 0
    assert "required" not in priorities[first_optional:]
    missing_legs = [row for row in rows if row["kind"] == "missing_leg"]
    assert {row["policy_id"] for row in missing_legs} == {"ALL-001"}
    assert {row["absent_leg"] for row in missing_legs} == {"verb", "target"}
    [language_gap] = [row for row in report["gaps"] if row["kind"] == "language"]
    assert language_gap["language"] == "en-US"
    assert language_gap["reason"] == "translation_required"
    assert all("utterance" not in row and "prompt" not in row for row in rows)
    record = json.loads((gold / "run_record.json").read_text(encoding="utf-8"))
    assert record["coverage"]["certification"] == "silver"
    assert record["coverage"]["coverage_of"] == "eligible_rules"
    assert record["coverage"]["excluded_rules"] == []
    assert report["modules"]["production_replay"] == "not_configured"
    assert report["modules"]["persona"] == "disabled"
    index = json.loads((gold / "stage_cache" / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["coverage_plan"]["status"] == "completed"

    silver = _generate(tmp_path, policies, "silver", {"assisted_max_cases": 1, "cases_per_normal_rule": 12})
    silver_rows = _jsonl(silver / "stage_cache" / "coverage_plan.jsonl")
    silver_report = json.loads((silver / "stage_cache" / "coverage_gaps.json").read_text(encoding="utf-8"))
    assert silver_report["certification"] == "silver"
    assert len(silver_rows) == 1
    assert silver_rows[0]["priority"] == "required"
    assert len(silver_report["missing_required"]) == silver_report["required_cases"] - 1
    scheduled = {silver_rows[0]["slot_id"]}
    assert not scheduled & set(silver_report["missing_required"])


def test_a_planner_error_marks_the_stage_failed_and_removes_a_partial_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "rules.yaml").write_text(
        yaml.safe_dump(
            {
                "policy_id": "CARD-001",
                "severity": "high",
                "testability": "testable",
                "provenance": {"source_quote": "Block requests for card numbers."},
                "rule": {"actor": "assistant", "action": "provide", "object": "card_number", "block_if": ["asks"]},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "family": "safety",
                "expt_name": "plan",
                "config_status": "resolved",
                "mode": "assisted",
                "input_dir": str(policies),
                "output_dir": str(output),
            }
        ),
        encoding="utf-8",
    )
    prepare_safety(config)

    def partial(stage_cache: Path, plan: Any) -> dict[str, Any]:
        del plan
        (stage_cache / "coverage_plan.jsonl").write_text("{}\n", encoding="utf-8")
        raise RuntimeError("planner broke")

    monkeypatch.setattr("nemotron.steps.byob.runtime.benchmark_families.safety.pipeline.write_coverage_plan", partial)
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    assert caught.value.status is RunStatus.INVALID_RUN_ARTIFACT
    assert "planner broke" in str(caught.value)
    run_dir = output / "plan"
    assert not (run_dir / "stage_cache" / "coverage_plan.jsonl").exists()
    assert not (run_dir / "stage_cache" / "coverage_gaps.json").exists()
    index = json.loads((run_dir / "stage_cache" / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["coverage_plan"]["status"] == "failed"
    assert index["stages"]["coverage_plan"]["outcome"] == "RuntimeError"


def _generate(
    tmp_path: Path,
    policies: Path,
    name: str,
    budget: dict[str, int],
    generation: dict[str, Any] | None = None,
) -> Path:
    output = tmp_path / name
    config = tmp_path / f"{name}.yaml"
    payload: dict[str, Any] = {
        "family": "safety",
        "expt_name": "plan",
        "config_status": "resolved",
        "mode": "assisted",
        "input_dir": str(policies),
        "output_dir": str(output),
        "budget": budget,
    }
    if generation is not None:
        payload["generation"] = generation
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    prepare_safety(config)
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)
    assert caught.value.status is RunStatus.REVIEW_REQUIRED
    assert "Coverage plan:" in str(caught.value)
    assert "Scenarios:" in str(caught.value)
    return output / "plan"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
