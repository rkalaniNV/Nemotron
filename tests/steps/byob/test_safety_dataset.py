"""Labels, deterministic checks, exact deduplication, and the development split."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety import dataset as dataset_module
from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety import pipeline as pipeline_module
from nemotron.steps.byob.runtime.benchmark_families.safety.confidence import canonical_text
from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import (
    DatasetInputError,
    evaluate_quality_gates,
    load_jsonl_objects,
    select_dataset,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.deduplication import (
    run_mcq_semantic_dedup,
    scored_turn_text,
    semantic_partitions,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import validate_rule
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from tests.steps.byob.safety_fixtures import FIVE_TOPIC, FIVE_TOPIC_REPLIES, ScriptedExtractor
from tests.steps.byob.test_safety_coverage import _rule
from tests.steps.byob.test_safety_rules import _config
from tests.steps.byob.test_safety_scenarios import _cases

_ALL_OF = {"all_of": ["verb", "target"], "operator_source": "explicit"}


@pytest.fixture
def use(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(extractor: Any) -> Any:
        monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: extractor)
        return extractor

    return install


def test_unrealized_surface_drops_the_allow_and_keeps_the_block() -> None:
    rules, build = _ready()
    missing = next(case for case in build.cases if case.get("absent_legs") == ["verb"])
    flagged = _clone(missing, surface_required=True)
    cases = [flagged if case["case_id"] == flagged["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    rejected = {row["case_id"]: row for row in result.rejected}
    assert rejected[flagged["case_id"]]["reasons"] == ["unrealized_surface"]
    assert rejected[flagged["case_id"]]["stage"] == "surface"
    kept = {case["case_id"] for case in result.deduplicated}
    assert _one(build, "clear_violation")["case_id"] in kept
    assert flagged["case_id"] not in kept
    other = next(case for case in build.cases if case.get("absent_legs") == ["target"])
    assert other["case_id"] in kept
    assert _one(build, "clear_allow")["case_id"] in kept
    assert _one(build, "persona")["case_id"] in kept
    assert result.report["reasons"]["unrealized_surface"] == 1
    assert result.manifest["unrealized_surfaces"] == [
        {
            "case_id": flagged["case_id"],
            "policy_id": flagged["policy_id"],
            "kind": "missing_leg",
            "reason": "unrealized_surface",
        }
    ]


def test_an_unrealized_variant_or_optional_allow_does_not_block_gold() -> None:
    rules, build = _ready()
    missing = next(case for case in build.cases if case.get("absent_legs") == ["verb"])
    variant = _clone(
        missing,
        case_id=f"{missing['case_id']}/paraphrase-2",
        slot_id=f"{missing['case_id']}/paraphrase-2",
        variant_of=missing["case_id"],
        surface_variant=2,
        surface_required=True,
    )
    optional = _clone(
        _one(build, "clear_allow"),
        case_id="optional-allow",
        slot_id="optional-allow",
        priority="optional",
        surface_required=True,
    )

    result = select_dataset([*build.cases, variant, optional], build.families, rules, _span_map(*rules))

    rejected = {row["case_id"]: row["reasons"] for row in result.rejected}
    assert rejected[variant["case_id"]] == ["unrealized_surface"]
    assert rejected["optional-allow"] == ["unrealized_surface"]
    assert result.manifest["unrealized_surfaces"] == []


def test_a_realized_allow_surface_is_kept() -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    realized = _clone(
        _with_text(allow, "Ask a historical question about the same topic."),
        origin="paraphrase",
        surface_required=True,
    )
    cases = [realized if case["case_id"] == allow["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    kept = {case["case_id"] for case in result.deduplicated}
    assert allow["case_id"] in kept
    assert result.summary()["rejected"] == 0


def test_accepted_cases_point_at_a_rule_and_stay_in_development() -> None:
    rules, build = _ready()
    spans = _span_map(*rules)
    result = _select(rules, build)
    again = select_dataset(list(reversed(build.cases)), build.families, rules, spans)

    assert result.summary()["rejected"] == 0
    assert result.manifest["complete"] is True
    assert result.manifest["reason"] == "complete"
    assert set(result.manifest["splits"]) == {"development"}
    assert len(result.manifest["splits"]["development"]) == len(result.deduplicated)
    assert again.manifest["splits"] == result.manifest["splits"]
    assert result.report["judge"]["method"] == "deterministic"
    assert result.report["judge"]["model"] is None
    assert result.report["judge"]["disagreements"] == 0
    assert result.report["judge"]["judge_disagreements"] == 0
    assert result.report["judge"]["skipped_reason"] == "no_judge_model_configs"
    assert result.report["semantic_dedup"] == {"status": "not_configured"}
    assert result.manifest["public_test"] == "not_assigned"
    by_split = result.manifest["balance"]["by_split"]
    assert by_split["development"]["cases"] == len(result.manifest["splits"]["development"])
    for case in result.deduplicated:
        assert case["policy_id"] == "ALL-2"
        assert case["evidence"]["source_span_id"] in spans
        label = next(row for row in result.labels if row["case_id"] == case["case_id"])
        assert label["status"] == "confirmed"
        assert label["proposed_action"] == label["aggregated_action"] == case["expected_action"]
    multi = _one(build, "multi_turn")
    assert len(multi["turns"]) >= 2
    assert {turn["expected_action"] for turn in multi["turns"]} == {multi["expected_action"]}
    _assert_pairs_share_a_split(result, build.families)


def test_quality_gates_count_only_deduplicated_development_cases() -> None:
    rules, build = _ready()
    requirements = {
        "min_total_cases": 100,
        "min_development_cases": 1,
        "min_cases_per_policy": 1,
        "min_cases_per_action": {"allow": 1, "block": 1},
    }

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        quality_gates=requirements,
    )

    gates = result.manifest["quality_gates"]
    assert gates["enabled"] is True
    assert gates["passed"] is False
    assert gates["observed"]["total_cases"] == len(result.deduplicated)
    assert gates["observed"]["development_cases"] == len(result.manifest["splits"]["development"])
    assert gates["observed"]["by_policy"]["ALL-2"] == len(result.deduplicated)
    assert gates["failures"] == [
        {
            "gate": "min_total_cases",
            "required": 100,
            "actual": len(result.deduplicated),
            "missing": 100 - len(result.deduplicated),
        }
    ]
    assert result.report["quality_gates"] == gates
    assert result.summary()["quality_gates_passed"] is False
    assert gates["on_failure"] == "silver"


def test_quality_gates_count_each_language_tag_on_a_case() -> None:
    rows = [
        {
            "case_id": "switch",
            "policy_id": "P-1",
            "expected_action": "block",
            "languages": ["vi", "en"],
            "split": "development",
        },
        {
            "case_id": "joined",
            "policy_id": "P-1",
            "expected_action": "allow",
            "language": "VI,EN",
            "split": "development",
        },
    ]

    gates = evaluate_quality_gates(
        rows,
        {
            "min_cases_per_language": {"VI": 2, "en": 2},
            "min_cases_per_action": {"Allow": 1, "block": 1},
        },
    )

    assert gates["passed"] is True
    assert gates["observed"]["by_language"]["vi"] == 2
    assert gates["observed"]["by_language"]["en"] == 2
    assert gates["observed"]["by_action"]["allow"] == 1
    assert gates["observed"]["by_action"]["block"] == 1


def test_quality_gates_reject_damaged_requirement_types() -> None:
    with pytest.raises(DatasetInputError, match="min_total_cases must be a non-negative whole number"):
        evaluate_quality_gates([], {"min_total_cases": True})
    with pytest.raises(DatasetInputError, match="min_cases_per_language.vi must be"):
        evaluate_quality_gates([], {"min_cases_per_language": {"vi": -1}})
    with pytest.raises(DatasetInputError, match="repeats"):
        evaluate_quality_gates([], {"min_cases_per_language": {"vi": 1, "VI": 2}})
    with pytest.raises(DatasetInputError, match="on_failure must be silver or fail"):
        evaluate_quality_gates([], {"on_failure": "retry"})


def test_quality_gates_record_on_failure_mode() -> None:
    failed = evaluate_quality_gates([], {"min_total_cases": 1, "on_failure": "fail"})
    assert failed["passed"] is False
    assert failed["on_failure"] == "fail"
    defaulted = evaluate_quality_gates([], {})
    assert defaulted["passed"] is True
    assert defaulted["on_failure"] == "silver"


def test_an_exact_duplicate_keeps_the_lowest_case_id() -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    higher = _clone(allow, case_id="zzz-duplicate", slot_id="zzz-duplicate")
    lower = _clone(allow, case_id="000-duplicate", slot_id="000-duplicate")
    translated = _clone(allow, case_id="lang-duplicate", slot_id="lang-duplicate", language="en-US")
    spans = _span_map(*rules)

    higher_result = select_dataset([*build.cases, higher, translated], build.families, rules, spans)
    kept = {case["case_id"] for case in higher_result.deduplicated}
    assert allow["case_id"] in kept
    assert "zzz-duplicate" not in kept
    assert "lang-duplicate" in kept
    assert higher_result.manifest["dropped_duplicates"] == [
        {"case_id": "zzz-duplicate", "kept_case_id": allow["case_id"]}
    ]

    lower_result = select_dataset([*build.cases, lower], build.families, rules, spans)
    assert {case["case_id"] for case in lower_result.deduplicated} >= {"000-duplicate"}
    assert allow["case_id"] not in {case["case_id"] for case in lower_result.deduplicated}
    assert lower_result.manifest["dropped_duplicates"] == [
        {"case_id": allow["case_id"], "kept_case_id": "000-duplicate"}
    ]


def test_a_family_that_does_not_change_the_facts_collapses() -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    first = _clone(allow, case_id="pair-a", slot_id="pair-a")
    second = _clone(allow, case_id="pair-b", slot_id="pair-b")
    family = {
        "family_id": "contrast/pair",
        "policy_id": "ALL-2",
        "members": [
            {"slot_id": "pair-a", "case_id": "pair-a", "role": "violation", "generation": "accepted"},
            {"slot_id": "pair-b", "case_id": "pair-b", "role": "allow", "generation": "accepted"},
        ],
    }

    paired = select_dataset([first, second], [family], rules, _span_map(*rules))

    assert {case["case_id"] for case in paired.deduplicated} == {"pair-a"}
    assert paired.summary()["duplicates_removed"] == 1
    result = _select(rules, build)
    kept = {case["case_id"] for case in result.deduplicated}
    assert _one(build, "clear_violation")["case_id"] in kept
    assert allow["case_id"] in kept
    assert _one(build, "persona")["case_id"] in kept
    _assert_pairs_share_a_split(result, build.families)


def test_a_changed_label_rejects_every_family_that_shares_the_case() -> None:
    rules, build = _ready()
    changed = _clone(next(case for case in build.cases if case["absent_legs"] == ["verb"]))
    changed["expected_action"] = "block"
    changed["turns"][0]["expected_action"] = "block"
    cases = [changed if case["case_id"] == changed["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    rejected = {row["case_id"]: row for row in result.rejected}
    assert "label_disagreement" in rejected[changed["case_id"]]["reasons"]
    violation = _one(build, "clear_violation")
    other = next(case for case in build.cases if case["absent_legs"] == ["target"])
    assert rejected[violation["case_id"]]["reasons"] == ["contrast_pair_broken"]
    assert rejected[other["case_id"]]["reasons"] == ["contrast_pair_broken"]
    assert rejected[violation["case_id"]]["stage"] == "contrast"
    assert rejected[_one(build, "persona")["case_id"]]["reasons"] == ["attack_removed"]
    assert rejected[_one(build, "multi_turn")["case_id"]]["reasons"] == ["attack_removed"]
    validated = {case["case_id"] for case in result.validated}
    assert _one(build, "clear_allow")["case_id"] in validated
    assert rejected[_one(build, "boundary")["case_id"]]["reasons"] == ["attack_removed"]
    assert result.report["judge"]["disagreements"] == 1
    stored = next(case for case in result.validated if case["case_id"] == _one(build, "clear_allow")["case_id"])
    assert stored["expected_action"] == "allow"


def test_a_paraphrased_multi_turn_dialogue_is_accepted() -> None:
    rules, build = _ready()
    case = _one(build, "multi_turn")
    action = case["expected_action"]
    dialogue = _clone(case, origin="paraphrase")
    dialogue["turns"] = [
        {"turn": 1, "role": "user", "text": "Can we try this first?", "scored": False},
        {"turn": 2, "role": "assistant", "text": "What would you like to do next?", "scored": False},
        {
            "turn": 3,
            "role": "user",
            "text": "Walk me through the rest of it.",
            "scored": True,
            "expected_action": action,
        },
    ]
    cases = [dialogue if item["case_id"] == case["case_id"] else item for item in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    stored = next(item for item in result.validated if item["case_id"] == case["case_id"])
    assert stored["origin"] == "paraphrase"
    assert [turn["role"] for turn in stored["turns"]] == ["user", "assistant", "user"]
    assert [turn["scored"] for turn in stored["turns"]] == [False, False, True]
    assert stored["turns"][-1]["expected_action"] == action


def test_a_translated_turn_is_accepted() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    changed = _clone(
        _with_text(violation, "Please walk me through that blocked request."),
        origin="translation",
        kind="language",
        language="en-US",
        localization_applied=True,
    )
    changed["case_id"] = f"{violation['case_id']}/language/en-US"
    changed["slot_id"] = changed["case_id"]
    result = select_dataset([*build.cases, changed], build.families, rules, _span_map(*rules))

    stored = next(case for case in result.validated if case["case_id"] == changed["case_id"])
    assert stored["origin"] == "translation"
    assert stored["kind"] == "language"
    assert stored["turns"][0]["text"] == "Please walk me through that blocked request."


def test_identical_translations_still_exact_dedup() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    first = _clone(
        _with_text(violation, "Please walk me through that blocked request."),
        origin="translation",
        kind="language",
        language="en-US",
        localization_applied=True,
        case_id="lang-a",
        slot_id="lang-a",
    )
    second = _clone(first, case_id="lang-b", slot_id="lang-b")
    result = select_dataset([*build.cases, first, second], build.families, rules, _span_map(*rules))

    kept = {case["case_id"] for case in result.deduplicated}
    assert "lang-a" in kept
    assert "lang-b" not in kept
    assert {"case_id": "lang-b", "kept_case_id": "lang-a"} in result.manifest["dropped_duplicates"]


def test_an_unrealized_actor_swap_is_dropped() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    swapped = _clone(
        violation,
        kind="actor_swap",
        origin="actor_swap",
        case_id="swap-unrealized",
        slot_id="swap-unrealized",
    )

    result = select_dataset([*build.cases, swapped], build.families, rules, _span_map(*rules))

    rejected = {row["case_id"]: row for row in result.rejected}
    assert rejected["swap-unrealized"]["reasons"] == ["swap_unrealized"]
    assert "swap-unrealized" not in {case["case_id"] for case in result.deduplicated}
    assert result.manifest["unrealized_surfaces"] == []
    assert result.manifest["complete"] is True


def test_an_applied_actor_swap_is_accepted() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    swapped = _clone(
        _with_text(violation, "Follow the multi-turn policy for candidate B."),
        kind="actor_swap",
        origin="actor_swap",
        swap_applied=True,
        swapped_from=violation["case_id"],
        swap_from="candidate A",
        swap_to="candidate B",
        swap_axis="candidate",
        case_id="swap-applied",
        slot_id="swap-applied",
    )

    result = select_dataset([*build.cases, swapped], build.families, rules, _span_map(*rules))

    kept = {case["case_id"] for case in result.deduplicated}
    assert "swap-applied" in kept
    stored = next(case for case in result.validated if case["case_id"] == "swap-applied")
    assert stored["origin"] == "actor_swap"
    assert stored["turns"][0]["text"] == "Follow the multi-turn policy for candidate B."


def test_identical_actor_swaps_still_exact_dedup() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    first = _clone(
        _with_text(violation, "Follow the multi-turn policy for candidate B."),
        kind="actor_swap",
        origin="actor_swap",
        swap_applied=True,
        case_id="swap-a",
        slot_id="swap-a",
    )
    second = _clone(first, case_id="swap-b", slot_id="swap-b")
    result = select_dataset([*build.cases, first, second], build.families, rules, _span_map(*rules))

    kept = {case["case_id"] for case in result.deduplicated}
    assert "swap-a" in kept
    assert "swap-b" not in kept
    assert {"case_id": "swap-b", "kept_case_id": "swap-a"} in result.manifest["dropped_duplicates"]


def test_semantic_near_dup_keeps_an_actor_swap(tmp_path: Path) -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    swapped = _clone(
        _with_text(violation, "Follow the multi-turn policy for candidate B."),
        kind="actor_swap",
        origin="actor_swap",
        swap_applied=True,
        language=violation.get("language") or "en",
        case_id="swap-dup",
        slot_id="swap-dup",
    )
    result = select_dataset(
        [*build.cases, swapped],
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(),
        work_dir=tmp_path,
        semantic_runner=_runner(["swap-dup"]),
    )
    kept = {case["case_id"] for case in result.deduplicated}
    assert "swap-dup" in kept
    assert result.report["semantic_dedup"]["protected"] >= 1
    assert result.report["semantic_dedup"]["removed"] == 0


def test_semantic_near_dup_keeps_a_translation(tmp_path: Path) -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    translated = _clone(
        _with_text(violation, "Please walk me through that blocked request."),
        origin="translation",
        kind="language",
        language=violation.get("language") or "en",
        localization_applied=True,
        case_id="translated-dup",
        slot_id="translated-dup",
    )
    result = select_dataset(
        [*build.cases, translated],
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(),
        work_dir=tmp_path,
        semantic_runner=_runner(["translated-dup"]),
    )
    kept = {case["case_id"] for case in result.deduplicated}
    assert "translated-dup" in kept
    assert result.report["semantic_dedup"]["protected"] >= 1
    assert result.report["semantic_dedup"]["removed"] == 0


def test_a_paraphrased_turn_is_accepted() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    changed = _clone(_with_text(violation, "Please walk me through that blocked request."), origin="paraphrase")
    cases = [changed if case["case_id"] == violation["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    assert violation["case_id"] in {case["case_id"] for case in result.validated}
    stored = next(case for case in result.validated if case["case_id"] == violation["case_id"])
    assert stored["origin"] == "paraphrase"
    assert stored["turns"][0]["text"] == "Please walk me through that blocked request."


def test_a_rewritten_turn_is_rejected() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    changed = _with_text(violation, "Please allow this.")
    cases = [changed if case["case_id"] == violation["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    row = next(item for item in result.rejected if item["case_id"] == violation["case_id"])
    assert row["reasons"] == ["turn_text_mismatch"]
    assert violation["case_id"] not in {case["case_id"] for case in result.validated}
    missing = next(case for case in build.cases if case["absent_legs"] == ["verb"])
    rejected = {item["case_id"]: item["reasons"] for item in result.rejected}
    assert rejected[missing["case_id"]] == ["contrast_pair_broken"]
    assert rejected[_one(build, "persona")["case_id"]] == ["attack_removed"]


def test_a_turn_decision_that_differs_from_the_case_is_rejected() -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    changed = _clone(allow)
    changed["turns"][0]["expected_action"] = "block"
    cases = [changed if case["case_id"] == allow["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    row = next(item for item in result.rejected if item["case_id"] == allow["case_id"])
    assert row["reasons"] == ["turn_label_disagreement"]
    assert allow["case_id"] not in {case["case_id"] for case in result.validated}


def test_a_quote_outside_its_span_is_rejected() -> None:
    rules, build = _ready()
    spans = {rule.provenance.source_span_id: "unrelated policy text" for rule in rules}

    result = select_dataset(build.cases, build.families, rules, spans)

    assert result.summary()["validated"] == 0
    assert result.manifest["reason"] == "no_cases"
    assert result.manifest["splits"] == {"development": []}
    assert all("quote_not_in_span" in row["reasons"] for row in result.rejected)


def test_an_unknown_policy_and_a_missing_span_are_rejected() -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    unknown = _clone(allow, policy_id="NOPE")
    missing = _clone(_one(build, "boundary"))
    missing["evidence"] = {**missing["evidence"], "source_span_id": "missing-span"}
    cases = []
    for case in build.cases:
        if case["case_id"] == allow["case_id"]:
            cases.append(unknown)
        elif case["case_id"] == missing["case_id"]:
            cases.append(missing)
        else:
            cases.append(case)

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    rejected = {row["case_id"]: row["reasons"] for row in result.rejected}
    assert rejected[unknown["case_id"]] == ["unknown_policy"]
    assert "span_missing" in rejected[missing["case_id"]]
    assert _one(build, "clear_violation")["case_id"] in {case["case_id"] for case in result.validated}


def test_a_rule_without_a_span_cannot_enter_the_dataset() -> None:
    rule = validate_rule(
        {
            "policy_id": "BARE",
            "severity": "high",
            "testability": "testable",
            "confidence": 0.95,
            "provenance": {"document_id": "doc", "source_quote": "Block it."},
            "decision": {"violation_action": "block"},
            "rule": {
                "actor": "assistant",
                "action": "respond",
                "object": "request",
                "block_if": _ALL_OF,
            },
            "status": "eligible_for_generation",
        }
    )
    build = _build([rule])

    assert build.cases == ()
    injected = _one(_build([_rule("BARE", block=_ALL_OF)]), "clear_violation")
    result = select_dataset([injected], [], [rule], {})

    assert result.summary()["validated"] == 0
    assert result.manifest["reason"] == "no_cases"
    assert {tuple(row["reasons"]) for row in result.rejected} == {("span_missing",)}


def test_one_group_completes_in_development() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")

    result = select_dataset([violation], [], rules, _span_map(*rules))

    assert result.manifest["complete"] is True
    assert result.manifest["reason"] == "complete"
    assert result.manifest["splits"]["development"] == [violation["case_id"]]


def test_a_case_with_no_action_is_rejected() -> None:
    rules, build = _ready()
    boundary = _clone(_one(build, "boundary"), expected_action=None)
    boundary["turns"][0]["expected_action"] = None
    cases = [boundary if case["case_id"] == boundary["case_id"] else case for case in build.cases]

    result = select_dataset(cases, build.families, rules, _span_map(*rules))

    label = next(row for row in result.labels if row["case_id"] == boundary["case_id"])
    assert label["status"] == "rejected"
    assert label["proposed_action"] is None
    assert label["aggregated_action"] == "caution"
    rejected = next(row for row in result.rejected if row["case_id"] == boundary["case_id"])
    assert rejected["reasons"] == ["invalid_case"]
    assert boundary["case_id"] not in {case["case_id"] for case in result.validated}


def test_a_case_that_copies_the_violation_stays_on_its_side() -> None:
    rule = validate_rule(
        {
            "policy_id": "ALL-2",
            "severity": "critical",
            "testability": "testable",
            "confidence": 0.95,
            "scope": {"languages": ["vi-VN", "en-US"]},
            "provenance": {"document_id": "doc", "source_quote": "Block it.", "source_span_id": "doc#1#span-01"},
            "decision": {"violation_action": "block", "ambiguous_action": "caution"},
            "exceptions": ["historical_question"],
            "rule": {
                "actor": "assistant",
                "action": "respond",
                "object": "request",
                "block_if": _ALL_OF,
            },
            "status": "eligible_for_generation",
        }
    )
    rules = [rule]
    build = _build(rules)

    result = _select(rules, build)

    violation = _one(build, "clear_violation")["case_id"]
    _assert_ids_share_a_split(
        result,
        [violation, _one(build, "exception")["case_id"], _one(build, "boundary")["case_id"]],
    )
    assert result.manifest["complete"] is True
    kept = {case["case_id"]: case for case in result.deduplicated}
    assert kept[violation]["language"] == "vi-VN"
    assert [case for case in build.cases if case["kind"] == "language"] == []
    assert set(result.manifest["balance"]["by_language"]) == {"vi-VN"}


def test_a_benign_contrast_pair_stays_on_one_side() -> None:
    rule = _rule("ANY-1", block={"any_of": ["request"], "operator_source": "single_condition"})
    rules = [rule]
    build = _build(rules)

    result = _select(rules, build)

    assert result.manifest["complete"] is True
    assert result.manifest["reason"] == "complete"
    assert result.manifest["splits"]["development"]
    _assert_pairs_share_a_split(result, build.families)


def test_damaged_candidates_stop_the_build() -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    family = {
        "family_id": "contrast/x",
        "members": [{"slot_id": "missing", "case_id": "missing", "generation": "accepted"}],
    }

    with pytest.raises(DatasetInputError, match="must be a list"):
        select_dataset("nope", [], rules, {})
    with pytest.raises(DatasetInputError, match="repeat"):
        select_dataset([violation, dict(violation)], [], rules, _span_map(*rules))
    with pytest.raises(DatasetInputError, match="not in the candidates"):
        select_dataset([violation], [family], rules, _span_map(*rules))
    with pytest.raises(DatasetInputError, match="line 1"):
        load_jsonl_objects("{", "scenario_candidates.jsonl")


def test_generate_records_only_development(tmp_path: Path, use: Any) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    prepare_safety(config)

    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    message = str(caught.value)
    assert caught.value.status is RunStatus.REVIEW_REQUIRED
    assert "Coverage plan:" in message
    assert "Scenarios:" in message
    assert "Dataset:" in message
    run_dir = tmp_path / "out" / "customer-safety-v1"
    cache = run_dir / "stage_cache"
    manifest = json.loads((cache / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    development = set(manifest["splits"]["development"])
    assert set(manifest["splits"]) == {"development"}
    assert development
    for family in _jsonl(cache / "contrast_families.jsonl"):
        ids = [member["case_id"] for member in family["members"] if member.get("generation") == "accepted"]
        if len(ids) < 2:
            continue
        assert all(case_id in development for case_id in ids)
    index = json.loads((cache / "source_index.json").read_text(encoding="utf-8"))
    spans = {span["source_span_id"] for span in index["spans"]}
    for case in _jsonl(cache / "deduplicated_cases.jsonl"):
        assert case["policy_id"]
        assert case["evidence"]["source_span_id"] in spans
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["dataset"]["split_complete"] is True
    assert json.loads((cache / "cache_index.json").read_text(encoding="utf-8"))["stages"]["dataset"]["status"] == (
        "completed"
    )


def test_backfill_fills_room_from_healthy_policies_and_skips_rule_level_rejections(
    tmp_path: Path, use: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _five_topic_with_gates(tmp_path / "base", use)
    base_cases = _jsonl(base / "deduplicated_cases.jsonl")
    base_plan = _jsonl(base / "coverage_plan.jsonl")
    assert len(base_plan) == 10
    broken_quote = next(case for case in base_cases if case["policy_id"] == base_plan[0]["policy_id"])["evidence"][
        "source_quote"
    ]
    _break_quote(monkeypatch, broken_quote)

    cache = _five_topic_with_gates(tmp_path / "broken", use)

    rejected = _jsonl(cache / "rejected_cases.jsonl")
    broken = {row["policy_id"] for row in rejected if "quote_not_in_span" in row["reasons"]}
    assert broken
    gaps = json.loads((cache / "coverage_gaps.json").read_text(encoding="utf-8"))
    added = set(gaps["backfill"]["added_slot_ids"])
    assert added
    plan = _jsonl(cache / "coverage_plan.jsonl")
    backfilled = [row for row in plan if row.get("backfill") is True]
    assert {row["slot_id"] for row in backfilled} == added
    assert not {row["policy_id"] for row in backfilled} & broken
    survivors = _jsonl(cache / "deduplicated_cases.jsonl")
    assert not {case["policy_id"] for case in survivors} & broken
    kept_before = sum(1 for case in base_cases if case["policy_id"] not in broken)
    assert kept_before < len(survivors) <= 10


def test_a_backfill_round_that_does_not_lower_the_shortfall_is_rolled_back(
    tmp_path: Path, use: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _five_topic_with_gates(tmp_path / "base", use)
    base_cases = _jsonl(base / "deduplicated_cases.jsonl")
    first_policy = _jsonl(base / "coverage_plan.jsonl")[0]["policy_id"]
    _break_quote(
        monkeypatch,
        next(case for case in base_cases if case["policy_id"] == first_policy)["evidence"]["source_quote"],
    )
    monkeypatch.setattr(pipeline_module, "_current_shortfall", lambda stage_cache, run_dir: 10**9)

    cache = _five_topic_with_gates(tmp_path / "broken", use)

    plan = _jsonl(cache / "coverage_plan.jsonl")
    gaps = json.loads((cache / "coverage_gaps.json").read_text(encoding="utf-8"))
    assert len(plan) == 10
    assert all(row.get("backfill") is not True for row in plan)
    assert "backfill" not in gaps
    scheduled = {row["slot_id"] for row in plan}
    assert {case["slot_id"] for case in _jsonl(cache / "scenario_candidates.jsonl")} == scheduled
    index = json.loads((cache / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["coverage_plan"]["scheduled_cases"] == 10
    assert "backfill_rounds" not in index["stages"]["coverage_plan"]


def _five_topic_with_gates(root: Path, use: Any) -> Path:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    root.mkdir()
    config = _config(root, {"five-topic.md": FIVE_TOPIC})
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["budget"] = {"assisted_max_cases": 10}
    data["quality_gates"] = {"min_total_cases": 10}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    prepare_safety(config)
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)
    assert caught.value.status is RunStatus.REVIEW_REQUIRED, str(caught.value)
    return root / "out" / "customer-safety-v1" / "stage_cache"


def _break_quote(monkeypatch: pytest.MonkeyPatch, quote: str) -> None:
    original = dataset_module.quote_in_span

    def patched(candidate: str, span: str) -> bool:
        if canonical_text(candidate) == canonical_text(quote):
            return False
        return original(candidate, span)

    monkeypatch.setattr(dataset_module, "quote_in_span", patched)


def test_a_dataset_error_marks_the_stage_failed_and_keeps_scenarios(
    tmp_path: Path, use: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    prepare_safety(config)

    def partial(stage_cache: Path, build: Any) -> dict[str, Any]:
        del build
        (stage_cache / "split_manifest.json").write_text("{}\n", encoding="utf-8")
        raise RuntimeError("dataset broke")

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.safety.pipeline.write_dataset",
        partial,
    )
    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    assert caught.value.status is RunStatus.INVALID_RUN_ARTIFACT
    assert "dataset broke" in str(caught.value)
    cache = tmp_path / "out" / "customer-safety-v1" / "stage_cache"
    assert (cache / "coverage_plan.jsonl").is_file()
    assert (cache / "scenario_candidates.jsonl").is_file()
    assert (cache / "contrast_families.jsonl").is_file()
    assert not (cache / "split_manifest.json").exists()
    assert not (cache / "validated_cases.jsonl").exists()
    assert not (cache / "label_checks.jsonl").exists()
    index = json.loads((cache / "cache_index.json").read_text(encoding="utf-8"))
    assert index["stages"]["coverage_plan"]["status"] == "completed"
    assert index["stages"]["scenarios"]["status"] == "completed"
    assert index["stages"]["dataset"]["status"] == "failed"
    assert index["stages"]["dataset"]["outcome"] == "RuntimeError"


def test_scored_turn_text_uses_the_triggering_user_turn() -> None:
    assert (
        scored_turn_text(
            {
                "turns": [
                    {"role": "user", "text": "Context only.", "scored": False},
                    {"role": "assistant", "text": "Go on.", "scored": False},
                    {"role": "user", "text": "Now block this request.", "scored": True},
                ]
            }
        )
        == "Now block this request."
    )
    assert scored_turn_text({"turns": [{"role": "user", "text": "  only user  "}]}) == "only user"
    assert scored_turn_text({"turns": []}) == ""


def test_semantic_near_dup_drops_an_unprotected_paraphrase(tmp_path: Path) -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    extra = _with_text(
        _clone(allow, case_id="near-dup", slot_id="near-dup", origin="paraphrase"),
        "A nearby allowed request that is not a greeting.",
    )

    result = select_dataset(
        [*build.cases, extra],
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(),
        work_dir=tmp_path,
        semantic_runner=_runner(["near-dup"]),
    )

    kept = {case["case_id"] for case in result.deduplicated}
    assert allow["case_id"] in kept
    assert "near-dup" not in kept
    assert result.report["semantic_dedup"]["status"] == "applied"
    assert result.report["semantic_dedup"]["removed"] == 1
    dropped = next(row for row in result.rejected if row["case_id"] == "near-dup")
    assert dropped["reasons"] == ["semantic_duplicate"]
    assert "kept_case_id" not in dropped
    assert {"case_id": "near-dup"} in result.manifest["dropped_duplicates"]


def test_semantic_near_dup_keeps_a_complete_contrast_pair(tmp_path: Path) -> None:
    rules, build = _ready()
    violation = _one(build, "clear_violation")
    missing = next(case for case in build.cases if case["kind"] == "missing_leg")

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(),
        work_dir=tmp_path,
        semantic_runner=_runner([missing["case_id"]]),
    )

    kept = {case["case_id"] for case in result.deduplicated}
    assert missing["case_id"] in kept
    assert violation["case_id"] in kept
    assert result.report["semantic_dedup"]["protected"] >= 1
    assert result.report["semantic_dedup"]["removed"] == 0


def test_disabled_semantic_dedup_does_not_call_the_runner(tmp_path: Path) -> None:
    rules, build = _ready()

    def runner(cases: Any, *, work_dir: Path, settings: Mapping[str, Any]) -> dict[str, str]:
        del cases, work_dir, settings
        raise AssertionError("disabled semantic runner must not run")

    result = select_dataset(
        build.cases,
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(enabled=False),
        work_dir=tmp_path,
        semantic_runner=runner,
    )

    assert result.report["semantic_dedup"]["status"] == "disabled"


def test_unavailable_semantic_dedup_fails_closed(tmp_path: Path) -> None:
    rules, build = _ready()

    def runner(cases: Any, *, work_dir: Path, settings: Mapping[str, Any]) -> list[str]:
        del cases, work_dir, settings
        raise ImportError("nemo_curator")

    with pytest.raises(DatasetInputError, match="semantic deduplication failed: nemo_curator"):
        select_dataset(
            build.cases,
            build.families,
            rules,
            _span_map(*rules),
            semantic_dedup=_semantic_settings(),
            work_dir=tmp_path,
            semantic_runner=runner,
        )


def test_marked_semantic_dups_can_be_kept(tmp_path: Path) -> None:
    rules, build = _ready()
    allow = _one(build, "clear_allow")
    extra = _with_text(
        _clone(allow, case_id="near-dup", slot_id="near-dup", origin="paraphrase"),
        "A nearby allowed request that is not a greeting.",
    )

    result = select_dataset(
        [*build.cases, extra],
        build.families,
        rules,
        _span_map(*rules),
        semantic_dedup=_semantic_settings(remove_duplicates=False),
        work_dir=tmp_path,
        semantic_runner=_runner(["near-dup"]),
    )

    assert "near-dup" in {case["case_id"] for case in result.deduplicated}
    assert result.report["semantic_dedup"]["marked"] == 1
    assert result.report["semantic_dedup"]["removed"] == 0


def _semantic_settings(**overrides: Any) -> dict[str, Any]:
    settings = {
        "enabled": True,
        "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
        "n_clusters": 1,
        "eps": 0.07,
        "remove_duplicates": True,
    }
    settings.update(overrides)
    return settings


def test_semantic_partitions_separate_action_and_language() -> None:
    cases = [
        {
            "case_id": "vi-allow-a",
            "expected_action": "allow",
            "language": "vi",
            "turns": [{"role": "user", "text": "Câu A", "scored": True}],
        },
        {
            "case_id": "vi-allow-b",
            "expected_action": "allow",
            "language": "VI",
            "turns": [{"role": "user", "text": "Câu B", "scored": True}],
        },
        {
            "case_id": "vi-block",
            "expected_action": "block",
            "language": "vi",
            "turns": [{"role": "user", "text": "Câu A", "scored": True}],
        },
        {
            "case_id": "en-allow",
            "expected_action": "allow",
            "language": "en",
            "turns": [{"role": "user", "text": "Câu A", "scored": True}],
        },
    ]

    partitions = semantic_partitions(cases)

    assert {case["case_id"] for case in partitions[("allow", "vi")]} == {
        "vi-allow-a",
        "vi-allow-b",
    }
    assert [case["case_id"] for case in partitions[("block", "vi")]] == ["vi-block"]
    assert [case["case_id"] for case in partitions[("allow", "en")]] == ["en-allow"]


def test_semantic_adapter_needs_no_backend_when_every_partition_is_singleton(tmp_path: Path) -> None:
    cases = [
        {
            "case_id": "allow",
            "expected_action": "allow",
            "language": "vi",
            "turns": [{"role": "user", "text": "Cùng bề mặt", "scored": True}],
        },
        {
            "case_id": "block",
            "expected_action": "block",
            "language": "vi",
            "turns": [{"role": "user", "text": "Cùng bề mặt", "scored": True}],
        },
    ]

    assert (
        run_mcq_semantic_dedup(
            cases,
            work_dir=tmp_path,
            settings=_semantic_settings(),
        )
        == ()
    )


@pytest.mark.parametrize(
    ("duplicates", "message"),
    [
        (["unknown"], "unknown duplicate ids"),
        (["same", "same"], "repeated a duplicate id"),
        ([1], "invalid duplicate id"),
    ],
)
def test_semantic_runner_rejects_invalid_duplicate_ids(
    tmp_path: Path,
    duplicates: list[Any],
    message: str,
) -> None:
    rules, build = _ready()

    with pytest.raises(DatasetInputError, match=message):
        select_dataset(
            build.cases,
            build.families,
            rules,
            _span_map(*rules),
            semantic_dedup=_semantic_settings(),
            work_dir=tmp_path,
            semantic_runner=_runner(duplicates),
        )


def _runner(duplicates: list[Any]) -> Any:
    def run(cases: Any, *, work_dir: Path, settings: Mapping[str, Any]) -> list[Any]:
        del cases, work_dir, settings
        return list(duplicates)

    return run


def _ready() -> tuple[list[Any], Any]:
    rules = [
        _rule(
            "ALL-2",
            block=_ALL_OF,
            quote="Follow the multi-turn policy.",
            signals={"multi_turn_turns": 2},
            ambiguous_action="caution",
        )
    ]
    return rules, _cases(rules, modules={"persona": True})[1]


def _build(rules: list[Any]) -> Any:
    _plan, build = _cases(rules)
    return build


def _select(rules: list[Any], build: Any) -> Any:
    return select_dataset(build.cases, build.families, rules, _span_map(*rules))


def _span_map(*rules: Any) -> dict[str, str]:
    return {
        rule.provenance.source_span_id: f"{rule.provenance.source_quote} End."
        for rule in rules
        if rule.provenance.source_span_id
    }


def _one(build: Any, kind: str) -> dict[str, Any]:
    rows = [case for case in build.cases if case["kind"] == kind]
    assert len(rows) == 1
    return rows[0]


def _clone(case: dict[str, Any], **changes: Any) -> dict[str, Any]:
    copied = json.loads(json.dumps(case))
    copied.update(changes)
    return copied


def _with_text(case: dict[str, Any], text: str) -> dict[str, Any]:
    copied = _clone(case)
    for turn in copied["turns"]:
        turn["text"] = text
    return copied


def _assert_pairs_share_a_split(result: Any, families: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    for family in families:
        ids = [member["case_id"] for member in family["members"] if member.get("generation") == "accepted"]
        if len(ids) < 2:
            continue
        _assert_ids_share_a_split(result, ids)


def _assert_ids_share_a_split(result: Any, case_ids: list[str]) -> None:
    sides = [
        name for name, listed in result.manifest["splits"].items() if any(case_id in listed for case_id in case_ids)
    ]
    assert len(sides) == 1
    assert all(case_id in result.manifest["splits"][sides[0]] for case_id in case_ids)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
