"""Review packet, capped examples, and the publication tier."""

from pathlib import Path

import pytest

from nemotron.steps.byob.runtime.benchmark_families.safety.publish import publication_tier
from nemotron.steps.byob.runtime.benchmark_families.safety.review import (
    CSV_COLUMNS,
    DECISION_COLUMNS,
    DECISIONS_NAME,
    assemble_review,
    card_is_pending,
    conflict_card_id,
    load_decisions,
    withheld_case_ids,
    write_decisions,
)

_HASHES = {
    "deduplicated_cases_sha256": "a" * 64,
    "split_manifest_sha256": "b" * 64,
    "eligible_rules_sha256": "c" * 64,
    "routed_rules_sha256": "d" * 64,
    "conflict_report_sha256": "e" * 64,
    "coverage_gaps_sha256": "f" * 64,
    "contrast_families_sha256": "1" * 64,
    "validation_report_sha256": "2" * 64,
}


def _rule(policy_id: str, status: str, quote: str, *reasons: str) -> dict:
    return {
        "policy_id": policy_id,
        "status": status,
        "reasons": list(reasons),
        "confidence": 0.4,
        "rule": {"provenance": {"source_quote": quote}},
    }


def _case(case_id: str, policy_id: str, text: str) -> dict:
    return {
        "case_id": case_id,
        "policy_id": policy_id,
        "kind": "clear_violation",
        "expected_action": "block",
        "turns": [{"text": text}],
    }


def _coverage() -> dict:
    return {
        "certification": "gold",
        "reason": "complete",
        "required_cases": 1,
        "scheduled_cases": 1,
        "missing_required": [],
        "capacity": {
            "case_cap": 10,
            "planned_candidates": 1,
            "additional_distinct_cells": 2,
            "final_case_upper_bound": 3,
        },
        "quality_gate_preflight": {"enabled": True, "possible": True, "failures": []},
    }


def test_the_page_lists_every_unresolved_item_when_examples_are_capped() -> None:
    conflicts = [
        {
            "type": "contradictory_rules",
            "policy_ids": ["NEED-1"],
            "detail": "both allows and blocks",
            "resolution": "unresolved",
            "case_context": ["asks"],
        },
        {
            "type": "definition_conflict",
            "policy_ids": ["NEED-2"],
            "detail": "term defined twice",
            "resolution": "unresolved",
            "case_context": [],
        },
    ]
    build = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=[
            _rule("NEED-1", "needs_decision", "Quote one", "low_confidence"),
            _rule("NEED-2", "needs_decision", "Quote two", "conflict"),
            _rule("OK-1", "eligible_for_generation", "Allowed quote"),
        ],
        conflicts=conflicts,
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[_case(f"case-{index}", "OK-1", f"example {index} unique") for index in range(5)],
        decisions={},
        content_hashes=_HASHES,
    )

    page = build.html
    assert "Generation capacity preflight" in page
    assert "Final-case upper bound" in page
    assert "NEED-1" in page
    assert "NEED-2" in page
    assert "Quote one" in page
    assert "Quote two" in page
    assert "both allows and blocks" in page
    assert "term defined twice" in page
    assert "Showing 3 of 5 cases." in page
    assert "case-0" in page
    assert "case-3" not in page
    assert "case-4" not in page
    assert "example 4 unique" not in page
    assert "Cards: 5. Pending: 4. Estimated review time: 3 minutes." in page
    assert "review_decisions.csv" in page
    assert "4 cards need a decision." in page
    assert "does not generate new cases" in page
    assert "CSV row: rule:NEED-1" in page
    conflict_ids = {conflict_card_id(row) for row in conflicts}
    assert set(build.pending) == {"rule:NEED-1", "rule:NEED-2", *conflict_ids}
    assert "rule:OK-1" not in build.pending
    assert build.packet["content"]["review_html_sha256"]
    assert "conflict:0" not in build.pending


def test_reject_and_edit_withhold_a_rule_and_its_contrast_family() -> None:
    rules = [_rule("OK-1", "eligible_for_generation", "Block it.")]
    conflicts = [
        {
            "type": "contradictory_rules",
            "policy_ids": ["OK-1", "OK-2"],
            "detail": "overlap",
            "resolution": "unresolved",
            "case_context": [],
        }
    ]
    rejected = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=rules,
        conflicts=conflicts,
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[],
        decisions={conflict_card_id(conflicts[0]): ("reject", "")},
        content_hashes=_HASHES,
    )
    edited = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=rules,
        conflicts=[],
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[],
        decisions={"rule:OK-1": ("edit", "")},
        content_hashes=_HASHES,
    )

    assert rejected.withheld_policies == {"OK-1", "OK-2"}
    assert rejected.pending == ()
    assert edited.pending == ("rule:OK-1",)
    assert edited.withheld_policies == frozenset()
    families = [{"policy_id": "OK-1", "members": [{"case_id": "keep"}, {"case_id": "partner"}]}]
    cases = [
        {"case_id": "keep", "policy_id": "OK-1"},
        {"case_id": "partner", "policy_id": "OK-2"},
        {"case_id": "other", "policy_id": "OK-3"},
    ]
    assert withheld_case_ids(cases, families, frozenset({"OK-1"})) == {"keep", "partner"}


def test_example_cap_is_configurable_and_unknown_decision_cards_are_refused() -> None:
    values = dict(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=[_rule("OK-1", "eligible_for_generation", "Allowed quote")],
        conflicts=[],
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[_case(f"case-{index}", "OK-1", f"example {index}") for index in range(4)],
        content_hashes=_HASHES,
    )

    build = assemble_review(**values, decisions={}, example_cap=1)

    assert build.packet["example_cap"] == 1
    assert build.packet["examples"][0]["shown"] == 1
    assert "Showing 1 of 4 cases." in build.html
    try:
        assemble_review(**values, decisions={"rule:REMOVED": ("approve", "")})
    except ValueError as exc:
        assert "unknown cards: rule:REMOVED" in str(exc)
    else:
        raise AssertionError("a stale decision card was accepted")


def test_publication_tier_follows_coverage_review_and_preview() -> None:
    complete = dict(
        coverage_certification="gold",
        coverage_reason="complete",
        split_complete=True,
        split_reason="complete",
        development=2,
        private=2,
        withheld=False,
        blocking=False,
    )
    assert publication_tier(mode="assisted", **complete) == ("gold", "complete")
    assert publication_tier(mode="preview", **complete) == ("preview", "preview")
    assert publication_tier(mode="assisted", **{**complete, "blocking": True, "coverage_reason": "budget_cap"}) == (
        "silver",
        "budget_cap",
    )
    assert publication_tier(mode="assisted", **{**complete, "withheld": True}) == ("silver", "review_removed_cases")
    assert publication_tier(mode="assisted", **{**complete, "review_limitations": True}) == (
        "silver",
        "review_unresolved_items",
    )
    assert publication_tier(mode="assisted", **{**complete, "development": 0, "split_reason": "no_cases"}) == (
        "silver",
        "no_cases",
    )
    assert publication_tier(mode="assisted", **{**complete, "quality_gates_passed": False}) == (
        "silver",
        "minimum_dataset_quality_not_met",
    )
    assert publication_tier(
        mode="assisted",
        **{**complete, "development": 0, "split_reason": "one_group", "quality_gates_passed": False},
    ) == ("silver", "one_group")


def test_conflict_card_ids_stay_with_the_same_conflict_when_the_list_grows() -> None:
    original = {
        "type": "contradictory_rules",
        "policy_ids": ["B-2", "A-1"],
        "detail": "overlap",
        "resolution": "unresolved",
        "case_context": ["asks"],
    }
    extra = {
        "type": "definition_conflict",
        "policy_ids": ["C-3"],
        "detail": "term defined twice",
        "resolution": "unresolved",
        "case_context": [],
    }
    first = assemble_review(**_review_args(conflicts=[original]), decisions={})
    grown = assemble_review(**_review_args(conflicts=[extra, original]), decisions={})

    original_id = conflict_card_id(original)
    assert original_id in {card["card_id"] for card in first.packet["cards"]}
    assert original_id in {card["card_id"] for card in grown.packet["cards"]}
    assert original_id == conflict_card_id({**original, "policy_ids": ["A-1", "B-2"]})
    assert original_id != conflict_card_id(extra)


def test_pending_cards_cannot_be_pre_accepted_and_edit_notes_must_be_nonempty() -> None:
    values = _review_args(
        rules=[
            _rule("NEED-1", "needs_decision", "Quote", "low_confidence"),
            _rule("OK-1", "eligible_for_generation", "Allowed"),
        ],
        conflicts=[
            {
                "type": "contradictory_rules",
                "policy_ids": ["OK-1"],
                "detail": "overlap",
                "resolution": "unresolved",
                "case_context": [],
            }
        ],
    )
    conflict_id = conflict_card_id(values["conflicts"][0])
    try:
        assemble_review(**values, decisions={conflict_id: ("pre_accepted", "")})
    except ValueError as exc:
        assert f"{conflict_id} cannot use pre_accepted" in str(exc)
    else:
        raise AssertionError("pre_accepted on a conflict was accepted")
    try:
        assemble_review(**values, decisions={"rule:NEED-1": ("pre_accepted", "")})
    except ValueError as exc:
        assert "rule:NEED-1 cannot use pre_accepted" in str(exc)
    else:
        raise AssertionError("pre_accepted on a needs_decision rule was accepted")

    edited = assemble_review(**values, decisions={"rule:OK-1": ("edit", "   ")})
    assert "rule:OK-1" in edited.pending
    assert conflict_id in edited.pending
    assert edited.withheld_policies == frozenset()

    try:
        assemble_review(
            **values,
            decisions={"rule:OK-1": ("approve", "")},
            decision_metadata={
                "rule:OK-1": {"kind": "conflict", "policy_id": "OK-1", "rule_status": "eligible_for_generation"}
            },
        )
    except ValueError as exc:
        assert "rule:OK-1 kind 'conflict' does not match the current review card." in str(exc)
    else:
        raise AssertionError("mismatched decision metadata was accepted")


def test_decision_sheet_round_trips_through_the_canonical_writer(tmp_path: Path) -> None:
    path = tmp_path / DECISIONS_NAME
    write_decisions(
        path,
        [
            {
                "card_id": "rule:RULE-1",
                "kind": "rule",
                "policy_id": "RULE-1",
                "rule_status": "eligible_for_generation",
                "decision": "approve",
                "note": "",
                "summary": "extra keys must not become columns",
            }
        ],
    )

    text = path.read_text(encoding="utf-8")
    assert "\r" not in text
    assert text.splitlines()[0] == ",".join(CSV_COLUMNS)
    assert text.splitlines()[0].startswith(",".join(DECISION_COLUMNS))
    assert "summary" not in text.splitlines()[0]
    decisions, metadata = load_decisions(path, tmp_path)
    assert decisions == {"rule:RULE-1": ("approve", "")}
    assert metadata["rule:RULE-1"]["kind"] == "rule"
    assert not card_is_pending({"decision": "approve", "note": ""})


def test_pending_cards_show_the_quote_and_extracted_meaning() -> None:
    build = assemble_review(
        **_review_args(
            rules=[
                {
                    "policy_id": "NEED-1",
                    "status": "needs_decision",
                    "reasons": ["confidence 0.86 is below 0.9"],
                    "rule": {
                        "title": "Block terror instructions",
                        "severity": "high",
                        "decision": {"violation_action": "block"},
                        "provenance": {
                            "source_quote": "do not teach terror",
                            "section": "topic-x",
                        },
                        "rule": {"block_if": {"any_of": ["teach_terror"]}},
                    },
                }
            ]
        ),
        decisions={},
    )

    page = build.html
    assert "Block terror instructions" in page
    assert "do not teach terror" in page
    assert "block_if any of: teach_terror" in page
    assert "confidence 0.86 is below 0.9" in page
    assert "does not generate new cases" in page
    header = build.csv_text.splitlines()[0]
    assert header == ",".join(CSV_COLUMNS)
    row = next(line for line in build.csv_text.splitlines() if line.startswith("rule:NEED-1"))
    assert "Block terror instructions" in row
    assert "do not teach terror" in row
    assert "block" in row


def test_an_allow_if_rule_shows_allow_not_the_violation_action() -> None:
    build = assemble_review(
        **_review_args(
            rules=[
                {
                    "policy_id": "ALLOW-1",
                    "status": "needs_decision",
                    "reasons": ["allow rule has 14 possible acting partners in its section"],
                    "rule": {
                        "title": "Allow general knowledge",
                        "severity": "low",
                        "decision": {"violation_action": "block"},
                        "provenance": {"source_quote": "Linux là gì"},
                        "rule": {"allow_if": {"all_of": ["general_knowledge"]}},
                    },
                }
            ]
        ),
        decisions={},
    )

    page = build.html
    assert "Allow general knowledge" in page
    assert "allow_if all of: general_knowledge" in page
    row = next(line for line in build.csv_text.splitlines() if line.startswith("rule:ALLOW-1"))
    assert ",allow," in row
    assert page.count(">allow<") >= 1


def test_a_six_column_decision_sheet_still_loads(tmp_path: Path) -> None:
    path = tmp_path / DECISIONS_NAME
    path.write_text(
        "card_id,kind,policy_id,rule_status,decision,note\nrule:RULE-1,rule,RULE-1,needs_decision,approve,\n",
        encoding="utf-8",
    )

    decisions, metadata = load_decisions(path, tmp_path)

    assert decisions == {"rule:RULE-1": ("approve", "")}
    assert metadata["rule:RULE-1"]["kind"] == "rule"


def test_decision_sheet_refuses_an_empty_or_blank_card_list(tmp_path: Path) -> None:
    path = tmp_path / DECISIONS_NAME
    with pytest.raises(ValueError, match="has no cards to write"):
        write_decisions(path, [])
    with pytest.raises(ValueError, match="without a card_id"):
        write_decisions(path, [{"kind": "rule", "decision": "approve"}])
    assert not path.exists()


def _review_args(**overrides: object) -> dict:
    values: dict = dict(
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
        content_hashes=_HASHES,
    )
    values.update(overrides)
    return values
