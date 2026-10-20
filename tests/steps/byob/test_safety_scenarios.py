"""Scenario seeds, contrast reuse, and rejection of a changed case."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety.ir import validate_rule
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.scenarios import (
    ScenarioInputError,
    build_scenarios,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from tests.steps.byob.test_safety_coverage import _plan, _rule

_ALL_OF = {"all_of": ["verb", "target"], "operator_source": "explicit"}


def _cases(rules: list[Any], proposals: list[dict[str, Any]] | None = None, **options: Any) -> Any:
    plan = _plan(rules, **options)
    return plan, build_scenarios(
        plan.rows,
        plan.contrast_families,
        rules,
        proposals,
        unscheduled=[gap["slot_id"] for gap in plan.gaps],
    )


def _proposal(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot_id": case["slot_id"],
        "case_id": case["case_id"],
        "policy_id": case["policy_id"],
        "kind": case["kind"],
        "expected_action": case["expected_action"],
        "present_legs": list(case["present_legs"]),
        "absent_legs": list(case["absent_legs"]),
        "evidence": dict(case["evidence"]),
        "turns": [dict(turn) for turn in case["turns"]],
    }


def _one(build: Any, kind: str, policy_id: str | None = None) -> dict[str, Any]:
    rows = [case for case in build.cases if case["kind"] == kind]
    if policy_id is not None:
        rows = [case for case in rows if case["policy_id"] == policy_id]
    assert len(rows) == 1
    return rows[0]


def test_a_seed_names_its_rule_and_keeps_the_source_quote() -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan_rows, build = _cases([rule])

    assert build.summary()["rejected"] == 0
    violation = _one(build, "clear_violation")
    assert violation["case_id"] == violation["slot_id"]
    assert violation["policy_id"] == "ALL-2"
    assert violation["origin"] == "seed"
    assert violation["evidence"]["source_quote"] == "Block it."
    assert violation["evidence"]["document_id"] == "doc"
    assert violation["evidence"]["source_span_id"] == "doc#1#span-01"
    assert violation["present_legs"] == ["verb", "target"]
    assert violation["absent_legs"] == []
    assert violation["turns"][0]["text"] == "Block it."


def test_a_seed_copies_the_planner_source_language() -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.", languages=("vi-VN",))
    _plan_rows, build = _cases([rule])

    assert {case.get("source_language") for case in build.cases} == {"vi-VN"}
    assert "language" not in _one(build, "clear_violation")


def test_a_missing_leg_case_removes_one_condition_and_is_reused() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, build = _cases([rule])

    missing = [case for case in build.cases if case["kind"] == "missing_leg"]
    assert {(tuple(case["absent_legs"]), tuple(case["present_legs"])) for case in missing} == {
        (("verb",), ("target",)),
        (("target",), ("verb",)),
    }
    assert {case["expected_action"] for case in missing} == {"allow"}
    assert len({case["case_id"] for case in build.cases}) == len(build.cases)
    families = [family for family in build.families if family["changed_dimension"] in {"verb", "target"}]
    assert len(families) == 2
    for family in families:
        allow = next(member for member in family["members"] if member["role"] == "missing_leg")
        case = next(case for case in missing if case["absent_legs"] == [family["changed_dimension"]])
        assert allow["case_id"] == case["case_id"] == case["slot_id"]
        assert allow["reused"] is True
        assert "text" not in allow
        assert "turns" not in allow
        outside = case["turns"][0]["text"].replace(rule.provenance.source_quote, " ")
        assert family["changed_dimension"] not in outside
    assert build.summary()["reused_missing_leg"] == 2
    probes = {row["case_id"] for row in build.overrefusal}
    assert {case["case_id"] for case in missing} <= probes
    assert _one(build, "clear_violation")["case_id"] not in probes


def test_a_quote_that_names_the_removed_leg_stays_in_the_evidence() -> None:
    quote = "Block when verb and target are both present."
    rule = _rule("ALL-2", block=_ALL_OF, quote=quote)
    _plan_rows, build = _cases([rule])

    case = next(case for case in build.cases if case["absent_legs"] == ["verb"])
    assert case["evidence"]["source_quote"] == quote
    outside = case["turns"][0]["text"].replace(quote, " ")
    assert "verb" not in outside
    assert "target" in case["turns"][0]["text"]


def test_each_scored_multi_turn_keeps_the_planned_decision() -> None:
    rule = _rule("MT", block=_ALL_OF, quote="Follow the multi-turn policy.", signals={"multi_turn_turns": 2})
    _plan_rows, build = _cases([rule])

    case = _one(build, "multi_turn")
    assert case["expected_action"] == "block"
    assert len(case["turns"]) >= 2
    assert [turn["turn"] for turn in case["turns"]] == list(range(1, len(case["turns"]) + 1))
    assert {turn["scored"] for turn in case["turns"]} == {True}
    assert {turn["expected_action"] for turn in case["turns"]} == {"block"}
    assert [row for row in build.cases if row["kind"] == "boundary"] == []


def test_an_unknown_policy_id_is_rejected_and_not_replaced_by_a_seed() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, seeds = _cases([rule])
    missing = _one_of(seeds, "missing_leg")
    bad = _proposal(missing)
    bad["policy_id"] = "NO-SUCH"

    build = build_scenarios(plan.rows, plan.contrast_families, [rule], [bad])

    assert missing["slot_id"] not in {case["slot_id"] for case in build.cases}
    assert build.rejections[0]["reasons"] == ["unknown_policy_id", "changed_meaning"]
    assert {case["origin"] for case in build.cases} == {"seed"}
    allow = next(
        member
        for family in build.families
        if family["changed_dimension"] == missing["absent_leg"]
        for member in family["members"]
        if member["role"] == "missing_leg"
    )
    assert allow["case_id"] is None
    assert allow["generation"] == "rejected"
    assert allow["reused"] is False


def test_a_changed_action_quote_or_new_condition_is_rejected() -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    plan, seeds = _cases([rule])
    violation = _one(seeds, "clear_violation")

    changed_action = _proposal(violation)
    changed_action["expected_action"] = "allow"
    changed_action["turns"][0]["expected_action"] = "allow"
    changed_quote = _proposal(violation)
    changed_quote["evidence"] = {**violation["evidence"], "source_quote": "Block something else."}
    new_condition = _proposal(violation)
    new_condition["present_legs"] = ["verb", "target", "password"]
    dropped_leg = _proposal(violation)
    dropped_leg["present_legs"] = ["verb"]
    dropped_leg["turns"][0]["text"] = "Block it. Conditions kept: verb."
    second_id = _proposal(violation)
    second_id["case_id"] = "ALL-2/clear_violation/again"

    proposals = (changed_action, changed_quote, new_condition, dropped_leg, second_id)
    expected = [["changed_meaning"]] * 4 + [["duplicate_case"]]
    for proposal, reason in zip(proposals, expected, strict=True):
        build = build_scenarios(plan.rows, plan.contrast_families, [rule], [proposal])
        assert build.rejections[0]["reasons"] == reason
        assert violation["slot_id"] not in {case["slot_id"] for case in build.cases}


def test_wording_beyond_the_seed_sentence_is_rejected() -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    plan, seeds = _cases([rule])
    violation = _one(seeds, "clear_violation")
    extra = _proposal(violation)
    extra["turns"][0]["text"] += " Also block requests about weapons."

    build = build_scenarios(plan.rows, plan.contrast_families, [rule], [extra])

    assert build.rejections[0]["reasons"] == ["changed_meaning"]
    assert violation["slot_id"] not in {case["slot_id"] for case in build.cases}


def test_a_missing_leg_that_drops_or_keeps_the_wrong_condition_is_rejected() -> None:
    rule = _rule(
        "ALL-3",
        block={"all_of": ["verb", "target", "channel"], "operator_source": "explicit"},
    )
    plan, seeds = _cases([rule])
    case = next(row for row in seeds.cases if row["absent_legs"] == ["verb"])
    drops_another = _proposal(case)
    drops_another["present_legs"] = ["target"]
    drops_another["turns"][0]["text"] = "Block it. Conditions kept: target."
    keeps_absent = _proposal(case)
    keeps_absent["present_legs"] = ["target", "channel", "verb"]
    keeps_absent["turns"][0]["text"] = "Block it. Conditions kept: target, channel, verb."

    dropped = build_scenarios(plan.rows, plan.contrast_families, [rule], [drops_another])
    kept = build_scenarios(plan.rows, plan.contrast_families, [rule], [keeps_absent])

    assert dropped.rejections[0]["reasons"] == ["invalid_missing_leg"]
    assert kept.rejections[0]["reasons"] == ["invalid_missing_leg"]
    assert case["slot_id"] not in {row["slot_id"] for row in (*dropped.cases, *kept.cases)}


def test_a_scored_turn_without_a_decision_is_rejected() -> None:
    rule = _rule("MT", block=_ALL_OF, quote="Follow the multi-turn policy.", signals={"multi_turn_turns": 2})
    plan, seeds = _cases([rule])
    case = _one(seeds, "multi_turn")
    missing_decision = _proposal(case)
    del missing_decision["turns"][1]["expected_action"]
    short = _proposal(case)
    short["turns"] = case["turns"][:1]

    missing = build_scenarios(plan.rows, plan.contrast_families, [rule], [missing_decision])
    truncated = build_scenarios(plan.rows, plan.contrast_families, [rule], [short])

    assert missing.rejections[0]["reasons"] == ["missing_turn_decision"]
    assert truncated.rejections[0]["reasons"] == ["invalid_multi_turn"]
    assert case["slot_id"] not in {row["slot_id"] for row in (*missing.cases, *truncated.cases)}


def test_an_any_of_subset_or_a_reordered_leg_list_is_rejected() -> None:
    any_of = _rule("DAN", block={"any_of": ["dan", "developer_mode"], "operator_source": "default_any_of"})
    plan, seeds = _cases([any_of])
    assert [case for case in seeds.cases if case["kind"] == "missing_leg"] == []
    violation = _one(seeds, "clear_violation")
    subset = _proposal(violation)
    subset["present_legs"] = ["dan"]
    subset["turns"][0]["text"] = "Block it. Conditions kept: dan."
    exact = _proposal(violation)

    rejected = build_scenarios(plan.rows, plan.contrast_families, [any_of], [subset])
    accepted = build_scenarios(plan.rows, plan.contrast_families, [any_of], [exact])

    assert rejected.rejections[0]["reasons"] == ["changed_meaning"]
    assert violation["slot_id"] not in {case["slot_id"] for case in rejected.cases}
    stored = _one(accepted, "clear_violation")
    assert stored["origin"] == "proposal"
    assert stored["present_legs"] == ["dan", "developer_mode"]
    assert stored["turns"][0]["text"] == violation["turns"][0]["text"]
    again = build_scenarios(plan.rows, plan.contrast_families, [any_of], [_proposal(stored)])
    assert _one(again, "clear_violation")["origin"] == "proposal"

    ordered = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    ordered_plan, ordered_seeds = _cases([ordered])
    clear = _one(ordered_seeds, "clear_violation")
    reordered = _proposal(clear)
    reordered["present_legs"] = ["target", "verb"]
    reordered["turns"][0]["text"] = "Block it. Conditions kept: target, verb."
    build = build_scenarios(ordered_plan.rows, ordered_plan.contrast_families, [ordered], [reordered])
    assert build.rejections[0]["reasons"] == ["changed_meaning"]
    assert clear["slot_id"] not in {case["slot_id"] for case in build.cases}


def test_an_unscheduled_contrast_member_does_not_grow_a_new_case() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, build = _cases([rule], cap=1)

    assert len(build.cases) == 1
    unscheduled = [
        member
        for family in build.families
        for member in family["members"]
        if member["generation"] == "unscheduled"
    ]
    assert unscheduled
    assert {member["case_id"] for member in unscheduled} == {None}


def test_a_second_group_reuses_a_block_case_that_contains_only_its_legs() -> None:
    rule = _both_groups()
    _plan_rows, build = _cases([rule])

    verb = next(family for family in build.families if family["changed_dimension"] == "verb")
    block = next(case for case in build.cases if case["slot_id"] == verb["members"][0]["slot_id"])
    allow = next(case for case in build.cases if case["slot_id"] == verb["members"][1]["slot_id"])
    assert block["kind"] == "all_legs"
    assert block["present_legs"] == ["verb", "target"]
    assert block["absent_legs"] == []
    assert block["expected_action"] == "block"
    assert allow["present_legs"] == ["target"]
    assert allow["absent_legs"] == ["verb"]
    assert verb["members"][1]["reused"] is True
    assert verb["members"][1]["case_id"] == allow["case_id"]
    weapon = next(family for family in build.families if family["changed_dimension"] == "weapon")
    escalate = next(case for case in build.cases if case["slot_id"] == weapon["members"][0]["slot_id"])
    assert escalate["kind"] == "clear_violation"
    assert escalate["present_legs"] == ["weapon", "channel"]


def test_a_none_of_violation_records_the_absent_conditions() -> None:
    rule = validate_rule(_stored("NONE", block={"none_of": ["consent", "order"], "operator_source": "explicit"}))
    _plan_rows, build = _cases([rule])

    violation = _one(build, "clear_violation")
    allowed = _one(build, "clear_allow")
    assert violation["present_legs"] == []
    assert violation["absent_legs"] == ["consent", "order"]
    assert violation["turns"][0]["text"] == rule.provenance.source_quote
    assert allowed["present_legs"] == ["consent", "order"]
    assert allowed["absent_legs"] == []
    assert allowed["expected_action"] == "allow"
    assert allowed["turns"][0]["text"] == rule.provenance.source_quote
    family = build.families[0]
    assert [member["expected_action"] for member in family["members"]] == ["block", "allow"]


def test_an_allow_rule_uses_its_allow_conditions_as_facts() -> None:
    rule = _rule(
        "ALLOW",
        allow={"all_of": ["consent", "age_verified"], "operator_source": "explicit"},
        quote="Allow it.",
    )
    _plan_rows, build = _cases([rule])

    allowed = _one(build, "clear_allow")
    assert allowed["present_legs"] == ["consent", "age_verified"]
    assert allowed["absent_legs"] == []
    assert allowed["expected_action"] == "allow"
    assert allowed["turns"][0]["text"] == "Allow it."


def test_an_exception_case_keeps_the_rule_exception_names() -> None:
    rule = validate_rule({**_stored("EX"), "exceptions": ["historical_question"]})
    _plan_rows, build = _cases([rule])

    case = _one(build, "exception")
    assert case["exceptions"] == ["historical_question"]
    assert case["present_legs"] == ["request", "historical_question"]
    assert case["turns"][0]["text"] == rule.provenance.source_quote
    assert case["expected_action"] == "allow"


def test_a_plan_row_that_does_not_match_its_rule_stops_the_build() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, _seeds = _cases([rule])
    missing_action = dict(next(row for row in plan.rows if row["kind"] == "clear_violation"))
    del missing_action["expected_action"]
    unknown_action = dict(missing_action)
    unknown_action["expected_action"] = "permit"
    ghost = dict(next(row for row in plan.rows if row["kind"] == "clear_violation"))
    ghost["policy_id"] = "GHOST"
    ghost["slot_id"] = "GHOST/clear_violation"
    excused = validate_rule({**_stored("EX"), "exceptions": ["historical_question"]})
    excused_plan = _plan([excused])
    wrong_rows = [
        {**row, "exceptions": ["other_exception"]} if row["kind"] == "exception" else row
        for row in excused_plan.rows
    ]

    with pytest.raises(ScenarioInputError, match="needs an expected_action"):
        build_scenarios([missing_action], [], [rule])
    with pytest.raises(ScenarioInputError, match="unknown expected_action"):
        build_scenarios([unknown_action], [], [rule])
    with pytest.raises(ScenarioInputError, match="unknown policy"):
        build_scenarios([ghost], [], [rule])
    with pytest.raises(ScenarioInputError, match="exceptions"):
        build_scenarios(wrong_rows, [], [excused])


def test_a_contrast_family_that_disagrees_with_its_cases_stops_the_build() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, _seeds = _cases([rule])
    family = dict(plan.contrast_families[0])
    family["members"] = [dict(member) for member in family["members"]]
    family["members"][1]["expected_action"] = "block"

    with pytest.raises(ScenarioInputError, match="disagrees"):
        build_scenarios(plan.rows, [family], [rule])


def test_a_missing_leg_keeps_the_action_of_a_condition_that_still_matches() -> None:
    data = _stored("S", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    data["severity"] = "critical"
    data["decision"] = {"violation_action": "block", "critical_action": "escalate"}
    data["rule"]["escalate_if"] = {"any_of": ["target"], "operator_source": "explicit"}
    _plan_rows, build = _cases([validate_rule(data)])

    by_absent = {tuple(case["absent_legs"]): case for case in build.cases if case["kind"] == "missing_leg"}
    assert by_absent[("verb",)]["present_legs"] == ["target"]
    assert by_absent[("verb",)]["expected_action"] == "escalate"
    assert by_absent[("target",)]["present_legs"] == ["verb"]
    assert by_absent[("target",)]["expected_action"] == "allow"
    probes = {row["case_id"] for row in build.overrefusal}
    assert by_absent[("verb",)]["case_id"] not in probes
    assert by_absent[("target",)]["case_id"] in probes


def test_a_missing_leg_keeps_another_rules_action() -> None:
    left = _rule("A", block=_ALL_OF)
    right = _rule("C", block={"any_of": ["target"], "operator_source": "explicit"})
    _plan_rows, build = _cases([left, right])

    missing = next(case for case in build.cases if case["policy_id"] == "A" and case["absent_legs"] == ["verb"])
    assert missing["present_legs"] == ["target"]
    assert missing["expected_action"] == "block"
    assert missing["case_id"] not in {row["case_id"] for row in build.overrefusal}


def test_a_missing_leg_ignores_a_rule_for_a_different_policy_object() -> None:
    left = _stored("A", block=_ALL_OF)
    right = _stored("C", block={"any_of": ["target"], "operator_source": "explicit"})
    left["rule"]["object"] = "bank_transfer"
    right["rule"]["object"] = "job_application"
    _plan_rows, build = _cases([validate_rule(left), validate_rule(right)])

    missing = next(case for case in build.cases if case["policy_id"] == "A" and case["absent_legs"] == ["verb"])
    assert missing["present_legs"] == ["target"]
    assert missing["expected_action"] == "allow"
    assert missing["case_id"] in {row["case_id"] for row in build.overrefusal}


def test_a_none_of_leg_stays_present_when_the_case_must_not_fire_it() -> None:
    data = _stored("B", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    data["severity"] = "critical"
    data["decision"] = {"violation_action": "block", "critical_action": "escalate"}
    data["rule"]["escalate_if"] = {"none_of": ["consent"], "operator_source": "explicit"}
    _plan_rows, build = _cases([validate_rule(data)])

    block = _one(build, "all_legs")
    assert block["present_legs"] == ["verb", "target", "consent"]
    assert block["expected_action"] == "block"
    missing = next(case for case in build.cases if case["absent_legs"] == ["verb"])
    assert missing["present_legs"] == ["target", "consent"]
    assert missing["expected_action"] == "allow"
    assert _one(build, "clear_allow")["present_legs"] == ["consent"]
    assert _one(build, "clear_violation")["absent_legs"] == ["consent"]
    assert _one(build, "clear_violation")["present_legs"] == []


def test_a_boundary_uses_the_facts_that_make_the_rule_match() -> None:
    data = _stored("D", block=_ALL_OF)
    data["decision"]["ambiguous_action"] = "caution"
    _plan_rows, build = _cases([validate_rule(data)])

    boundary = _one(build, "boundary")
    assert boundary["present_legs"] == ["verb", "target"]
    assert boundary["expected_action"] == "caution"


def test_a_plan_row_with_a_null_action_stops_the_build() -> None:
    rule = _rule("NULL")
    plan = _plan([rule])
    rows = [dict(row) for row in plan.rows]
    rows[0]["expected_action"] = None

    with pytest.raises(ScenarioInputError, match="empty expected_action"):
        build_scenarios(rows, plan.contrast_families, [rule], None)


def test_a_contrast_slot_that_is_not_in_the_plan_or_the_gaps_stops_the_build() -> None:
    rule = _rule("ALL-2", block=_ALL_OF)
    plan, _seeds = _cases([rule])
    family = dict(plan.contrast_families[0])
    family["members"] = [dict(member) for member in family["members"]]
    family["members"][0]["slot_id"] = "no-such-slot"

    with pytest.raises(ScenarioInputError, match="unknown slot"):
        build_scenarios(
            plan.rows,
            [family],
            [rule],
            unscheduled=[gap["slot_id"] for gap in plan.gaps],
        )


def test_a_repeated_plan_slot_is_refused() -> None:
    row = {"slot_id": "ALL-2/clear_allow", "policy_id": "ALL-2", "kind": "clear_allow", "expected_action": "allow"}
    with pytest.raises(ScenarioInputError, match="repeats slot"):
        build_scenarios([row, dict(row)], [], [])


def test_generate_writes_one_missing_leg_case_and_reuses_it(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "policy.md").write_text("Block it.\n", encoding="utf-8")
    (policies / "rules.yaml").write_text(
        yaml.safe_dump(
            {
                "policy_id": "ALL-001",
                "severity": "high",
                "testability": "testable",
                "provenance": {"document_id": "policy", "source_quote": "Block it."},
                "rule": {
                    "actor": "assistant",
                    "action": "respond",
                    "object": "request",
                    "block_if": {"all_of": ["verb", "target"], "operator_source": "explicit"},
                },
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
                "expt_name": "scenarios",
                "config_status": "resolved",
                "mode": "assisted",
                "input_dir": str(policies),
                "output_dir": str(output),
                "budget": {"cases_per_normal_rule": 12, "assisted_max_cases": 50},
            }
        ),
        encoding="utf-8",
    )
    prepare_safety(config)
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    assert caught.value.status is RunStatus.REVIEW_REQUIRED
    assert "Coverage plan:" in str(caught.value)
    assert "Scenarios:" in str(caught.value)
    run_dir = output / "scenarios"
    cache = run_dir / "stage_cache"
    cases = _jsonl(cache / "scenario_candidates.jsonl")
    families = _jsonl(cache / "contrast_families.jsonl")
    missing = [case for case in cases if case["kind"] == "missing_leg"]
    assert {tuple(case["absent_legs"]) for case in missing} == {("verb",), ("target",)}
    assert len({case["case_id"] for case in cases}) == len(cases)
    for family in families:
        if family["changed_dimension"] not in {"verb", "target"}:
            continue
        allow = next(member for member in family["members"] if member["role"] == "missing_leg")
        assert sum(case["case_id"] == allow["case_id"] for case in cases) == 1
        assert allow["reused"] is True
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["scenarios"]["rejected"] == 0
    assert record["scenarios"]["accepted"] == len(cases)
    assert record["scenarios"]["reused_missing_leg"] == 2
    index = json.loads((cache / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["scenarios"]["status"] == "completed"


def test_a_scenario_error_marks_the_stage_failed_and_removes_partial_files(
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
                "expt_name": "scenarios",
                "config_status": "resolved",
                "mode": "assisted",
                "input_dir": str(policies),
                "output_dir": str(output),
            }
        ),
        encoding="utf-8",
    )
    prepare_safety(config)

    def partial(stage_cache: Path, build: Any) -> dict[str, Any]:
        del build
        (stage_cache / "scenario_candidates.jsonl").write_text("{}\n", encoding="utf-8")
        raise RuntimeError("scenarios broke")

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.safety.pipeline.write_scenarios",
        partial,
    )
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    assert caught.value.status is RunStatus.INVALID_RUN_ARTIFACT
    assert "scenarios broke" in str(caught.value)
    cache = output / "scenarios" / "stage_cache"
    assert (cache / "coverage_plan.jsonl").exists()
    assert not (cache / "scenario_candidates.jsonl").exists()
    assert not (cache / "contrast_families.jsonl").exists()
    assert not (cache / "scenario_rejections.jsonl").exists()
    assert not (cache / "overrefusal_cases.jsonl").exists()
    assert not (cache / "paraphrase_report.json").exists()
    index = json.loads((cache / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["coverage_plan"]["status"] == "completed"
    assert index["stages"]["scenarios"]["status"] == "failed"
    assert index["stages"]["scenarios"]["outcome"] == "RuntimeError"


def _stored(policy_id: str, *, block: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "severity": "high",
        "testability": "testable",
        "confidence": 0.95,
        "provenance": {"document_id": "doc", "source_quote": "Block it.", "source_span_id": "doc#1"},
        "decision": {"violation_action": "block"},
        "rule": {
            "actor": "assistant",
            "action": "respond",
            "object": "request",
            "block_if": block or {"any_of": ["request"], "operator_source": "single_condition"},
        },
        "status": "eligible_for_generation",
    }


def _both_groups() -> Any:
    data = _stored("BOTH", block={"all_of": ["verb", "target"], "operator_source": "explicit"})
    data["severity"] = "critical"
    data["decision"] = {"violation_action": "block", "critical_action": "escalate"}
    data["rule"]["escalate_if"] = {"all_of": ["weapon", "channel"], "operator_source": "explicit"}
    return validate_rule(data)


def _one_of(build: Any, kind: str) -> dict[str, Any]:
    return next(case for case in build.cases if case["kind"] == kind)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
