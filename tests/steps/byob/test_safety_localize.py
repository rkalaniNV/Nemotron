"""Stage 10: translate held language and code-switch cells after paraphrase."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety.coverage import realize_held_cells
from nemotron.steps.byob.runtime.benchmark_families.safety.localize import (
    LOCALIZE_TEMPERATURE,
    apply_localizations,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.review import assemble_review
from tests.steps.byob.safety_fixtures import scripted_complete_answer
from tests.steps.byob.test_safety_coverage import _budget, _generate, _plan, _rule
from tests.steps.byob.test_safety_review import _HASHES, _coverage
from tests.steps.byob.test_safety_scenarios import _cases


class FakeCompleter:
    identity = {
        "provider": "fake",
        "model": "localize",
        "inference_parameters": {"temperature": 0.8, "max_tokens": 4096},
    }

    def __init__(self, replies: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.requested: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.inference_calls: list[Any] = []

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
        del system_prompt, prompt
        self.inference_calls.append(inference_parameters)
        if self.fail:
            raise ConnectionError("endpoint refused the connection")
        answers: dict[str, Any] = {}
        for row in rows:
            request_id = row["request_id"]
            self.requested.append(request_id)
            payload = json.loads(row["payload"]) if isinstance(row.get("payload"), str) else {}
            self.payloads.append(payload)
            if request_id in self.replies:
                answers[request_id] = self.replies[request_id]
                continue
            structured = scripted_complete_answer(row, output_format=output_format, prefix=prefix)
            if structured is not None:
                answers[request_id] = structured
                continue
            answers[request_id] = {
                "turns": [{"role": "user", "text": f"Translated request for {request_id}."}],
            }
        return answers


def test_no_model_keeps_the_translation_gap() -> None:
    plan, build = _cases([_rule("LANG", languages=("vi-VN", "en-US"), quote="Chặn yêu cầu ALL-2.")])

    updated, report = apply_localizations(build, plan.summary(), None, Path("unused"))

    assert report["status"] == "skipped"
    assert report["skipped_reason"] == "no_model"
    assert report["applied"] == 0
    assert len(updated.cases) == len(build.cases)
    assert all(case.get("kind") != "language" for case in updated.cases)


def test_a_translated_language_cell_keeps_gold_fields(tmp_path: Path) -> None:
    quote = "Chặn yêu cầu ALL-2 ngay."
    plan, build = _cases([_rule("LANG", languages=("vi-VN",), quote=quote)], target_languages=["en-US"])
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {slot: {"turns": [{"role": "user", "text": "Please block that ALL-2 request immediately."}]}}
    )

    updated, report = apply_localizations(
        build,
        plan.summary(),
        completer,
        tmp_path,
        country="Vietnam",
        planned_rows=plan.rows,
        case_cap=50,
    )

    assert report["status"] == "applied"
    assert report["applied"] == 1
    assert report["realized_slot_ids"] == [slot]
    assert report["inference_parameters"]["temperature"] == LOCALIZE_TEMPERATURE
    [case] = [row for row in updated.cases if row["kind"] == "language"]
    assert case["origin"] == "translation"
    assert case["language"] == "en-US"
    assert case["source_language"] == "vi-VN"
    assert case["localization_applied"] is True
    assert case["translated_from"]
    assert case["evidence"]["source_quote"] == quote
    assert case["present_legs"] == ["request"]
    assert case["turns"][0]["text"] == "Please block that ALL-2 request immediately."
    assert completer.payloads[0]["task_type"] == "translate"
    assert completer.payloads[0]["country"] == "Vietnam"
    assert "ALL-2" in completer.payloads[0]["protected_tokens"]


def test_identity_and_quote_copies_keep_the_gap(tmp_path: Path) -> None:
    quote = "Block it now."
    plan, build = _cases([_rule("LANG", languages=("vi-VN",), quote=quote)], target_languages=["en-US"])
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    rewritten = []
    for case in build.cases:
        copied = json.loads(json.dumps(case))
        if copied["kind"] == "clear_violation":
            copied["origin"] = "paraphrase"
            copied["turns"][0]["text"] = "Walk me through that blocked request."
        rewritten.append(copied)
    build = replace(build, cases=tuple(rewritten))
    source = next(case for case in build.cases if case["kind"] == "clear_violation")
    completer = FakeCompleter({slot: {"turns": [{"role": "user", "text": source["turns"][0]["text"]}]}})

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "identity_copy"
    assert all(case.get("kind") != "language" for case in updated.cases)

    completer = FakeCompleter({slot: {"turns": [{"role": "user", "text": quote}]}})
    _updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)
    assert report["failures"][0]["reason"] == "quote_copy"


def test_a_dropped_identifier_is_rejected(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("vi-VN",), quote="Chặn yêu cầu ALL-2.")],
        target_languages=["en-US"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter({slot: {"turns": [{"role": "user", "text": "Please block that request."}]}})

    _updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "dropped_identifier"


def test_english_is_rejected_for_a_vietnamese_cell(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block CASE-7 now.")],
        target_languages=["vi-VN"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {
            slot: {"turns": [{"role": "user", "text": "Please block CASE-7 immediately."}]},
            f"{slot}#language": {
                "matches": False,
                "evidence": [{"language": "en", "evidence": "Please block CASE-7 immediately."}],
            },
        }
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "wrong_language"
    assert all(case.get("kind") != "language" for case in updated.cases)
    assert f"{slot}#language" in completer.requested


def test_an_unknown_language_prefix_is_rejected_without_a_language_model(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block ALL-2.")],
        target_languages=["xx-XX"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {slot: {"turns": [{"role": "user", "text": "Please block that ALL-2 request immediately."}]}}
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "wrong_language"
    assert all(case.get("kind") != "language" for case in updated.cases)
    assert all(not request_id.endswith("#language") for request_id in completer.requested)


def test_language_scripts_can_accept_an_unlisted_prefix(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block ALL-2.")],
        target_languages=["xx-XX"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {slot: {"turns": [{"role": "user", "text": "Please block that ALL-2 request immediately."}]}}
    )

    updated, report = apply_localizations(
        build,
        plan.summary(),
        completer,
        tmp_path,
        language_scripts={"xx": ["LATIN"]},
    )

    assert report["applied"] == 1
    [case] = [row for row in updated.cases if row["kind"] == "language"]
    assert case["language"] == "xx-XX"
    assert f"{slot}#language" in completer.requested


def test_identifiers_must_stay_on_the_matching_turn(tmp_path: Path) -> None:
    uuid = "550e8400-e29b-41d4-a716-446655440000"
    quote = f"Chặn ticket {uuid}."
    plan, build = _cases([_rule("LANG", languages=("vi-VN",), quote=quote)], target_languages=["en-US"])
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    rewritten = []
    for case in build.cases:
        copied = json.loads(json.dumps(case))
        if copied["kind"] == "clear_violation":
            copied["origin"] = "paraphrase"
            copied["turns"] = [
                {
                    "turn": 1,
                    "role": "user",
                    "text": f"Xem ABC123 và {uuid}.",
                    "scored": False,
                },
                {"turn": 2, "role": "assistant", "text": "Tôi thấy rồi.", "scored": False},
                {
                    "turn": 3,
                    "role": "user",
                    "text": "Đóng acc9999 giúp tôi.",
                    "scored": True,
                    "expected_action": "block",
                },
            ]
        rewritten.append(copied)
    build = replace(build, cases=tuple(rewritten))
    moved = FakeCompleter(
        {
            slot: {
                "turns": [
                    {"role": "user", "text": "Please look at that ticket."},
                    {"role": "assistant", "text": "I see it."},
                    {"role": "user", "text": f"Close ABC123 {uuid} and acc9999."},
                ]
            }
        }
    )

    _updated, report = apply_localizations(build, plan.summary(), moved, tmp_path)
    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "dropped_identifier"

    kept = FakeCompleter(
        {
            slot: {
                "turns": [
                    {"role": "user", "text": f"Please look at ABC123 and {uuid}."},
                    {"role": "assistant", "text": "I see it."},
                    {"role": "user", "text": "Please close acc9999 for me."},
                ]
            }
        }
    )
    updated, report = apply_localizations(build, plan.summary(), kept, tmp_path)
    assert report["applied"] == 1
    [case] = [row for row in updated.cases if row["kind"] == "language"]
    assert uuid in case["turns"][0]["text"]
    assert "ABC123" in case["turns"][0]["text"]
    assert "acc9999" in case["turns"][-1]["text"]


def test_code_switch_uses_the_declared_languages(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("MIX", languages=("vi-VN", "en-US"), signals={"code_switch": True}, quote="Chặn yêu cầu ALL-2.")]
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "code_switch")
    language_slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {
            slot: {"turns": [{"role": "user", "text": "Help me chặn yêu cầu ALL-2 in this chat."}]},
            language_slot: {"turns": [{"role": "user", "text": "Please block that ALL-2 request."}]},
        }
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path, planned_rows=plan.rows)

    assert slot in report["realized_slot_ids"]
    [case] = [row for row in updated.cases if row["kind"] == "code_switch"]
    assert case["origin"] == "translation"
    assert case["languages"] == ["vi-VN", "en-US"]
    assert next(payload["task_type"] for payload in completer.payloads if payload["kind"] == "code_switch") == (
        "code_switch"
    )


def test_no_diacritic_extra_is_added_when_marks_remain(tmp_path: Path) -> None:
    url = "https://ví.example/case"
    email = "ops@ví.example"
    plan, build = _cases([_rule("LANG", languages=("en-US",), quote="Block ALL-2.")], target_languages=["vi-VN"])
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    completer = FakeCompleter(
        {
            slot: {
                "turns": [
                    {
                        "role": "user",
                        "text": f"Hãy chặn yêu cầu ALL-2 tại {url} và {email} ngay lập tức.",
                    }
                ]
            }
        }
    )

    updated, report = apply_localizations(
        build,
        plan.summary(),
        completer,
        tmp_path,
        planned_rows=plan.rows,
        case_cap=50,
    )

    assert report["no_diacritics_added"] == 1
    extras = [case for case in updated.cases if case.get("localization_variant") == "no_diacritics"]
    assert len(extras) == 1
    assert extras[0]["origin"] == "translation"
    stripped = extras[0]["turns"][0]["text"]
    assert "ă" not in stripped
    assert "ALL-2" in stripped
    assert url in stripped
    assert email in stripped
    assert extras[0]["slot_id"] not in report["realized_slot_ids"]


def test_cached_translation_is_kept_when_the_batch_fails(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block ALL-2.")],
        target_languages=["vi-VN", "fr-FR"],
    )
    vi_slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language" and gap.get("language") == "vi-VN")
    fr_slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language" and gap.get("language") == "fr-FR")
    first = FakeCompleter({vi_slot: {"turns": [{"role": "user", "text": "Hãy chặn yêu cầu ALL-2 ngay lập tức."}]}})

    _warmed, first_report = apply_localizations(build, plan.summary(), first, tmp_path, planned_rows=plan.rows)
    assert vi_slot in first_report["realized_slot_ids"]
    assert fr_slot not in first_report["realized_slot_ids"]

    failed = FakeCompleter(fail=True)
    updated, report = apply_localizations(build, plan.summary(), failed, tmp_path, planned_rows=plan.rows)

    assert report["skipped_reason"] == "model_unavailable"
    assert report["cached"] >= 1
    assert report["requested"] == 1
    assert vi_slot in report["realized_slot_ids"]
    assert fr_slot not in report["realized_slot_ids"]
    languages = {case["language"] for case in updated.cases if case.get("kind") == "language"}
    assert languages == {"vi-VN"}


def test_whole_turn_language_evidence_is_rejected(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block CASE-7 now.")],
        target_languages=["vi-VN"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    text = "Please block CASE-7 immediately."
    completer = FakeCompleter(
        {
            slot: {"turns": [{"role": "user", "text": text}]},
            f"{slot}#language": {
                "matches": True,
                "evidence": [{"language": "vi", "evidence": text}],
            },
        }
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "wrong_language"
    assert all(case.get("kind") != "language" for case in updated.cases)


def test_short_latin_language_evidence_is_rejected(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block CASE-7 now.")],
        target_languages=["vi-VN"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    text = "Please block CASE-7 immediately."
    completer = FakeCompleter(
        {
            slot: {"turns": [{"role": "user", "text": text}]},
            f"{slot}#language": {
                "matches": True,
                "evidence": [{"language": "vi", "evidence": "Pl"}],
            },
        }
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path)

    assert report["applied"] == 0
    assert report["failures"][0]["reason"] == "wrong_language"
    assert all(case.get("kind") != "language" for case in updated.cases)


def test_shared_code_switch_evidence_is_rejected(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("MIX", languages=("vi-VN", "en-US"), signals={"code_switch": True}, quote="Chặn yêu cầu ALL-2.")]
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "code_switch")
    text = "Help me chặn yêu cầu ALL-2 in this chat."
    completer = FakeCompleter(
        {
            slot: {"turns": [{"role": "user", "text": text}]},
            f"{slot}#language": {
                "matches": True,
                "evidence": [
                    {"language": "vi", "evidence": "Help me"},
                    {"language": "en", "evidence": "Help me"},
                ],
            },
        }
    )

    updated, report = apply_localizations(build, plan.summary(), completer, tmp_path, planned_rows=plan.rows)

    assert slot not in report["realized_slot_ids"]
    assert any(item["slot_id"] == slot and item["reason"] == "wrong_language" for item in report["failures"])
    assert all(case.get("kind") != "code_switch" for case in updated.cases)


def test_cached_whole_turn_language_check_is_rejected(tmp_path: Path) -> None:
    plan, build = _cases(
        [_rule("LANG", languages=("en-US",), quote="Block ALL-2.")],
        target_languages=["vi-VN"],
    )
    slot = next(gap["slot_id"] for gap in plan.gaps if gap["kind"] == "language")
    first = FakeCompleter({slot: {"turns": [{"role": "user", "text": "Hãy chặn yêu cầu ALL-2 ngay lập tức."}]}})
    _updated, report = apply_localizations(build, plan.summary(), first, tmp_path, planned_rows=plan.rows)
    assert slot in report["realized_slot_ids"]

    for path in (tmp_path / "localize").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        text = next(turn["text"] for turn in data["turns"] if turn["role"] == "user")
        data["language_check"] = {"matches": True, "evidence": [{"language": "vi", "evidence": text}]}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    failed = FakeCompleter(fail=True)
    updated, report = apply_localizations(build, plan.summary(), failed, tmp_path, planned_rows=plan.rows)

    assert report["requested"] == 0
    assert slot not in report["realized_slot_ids"]
    assert report["failures"][0]["reason"] == "wrong_language"
    assert all(case.get("kind") != "language" for case in updated.cases)


def test_realize_held_cells_recertifies_gold_then_restores_a_failed_round() -> None:
    plan = _plan([_rule("LANG", languages=("vi-VN", "en-US"))])
    [gap] = [row for row in plan.gaps if row["kind"] == "language"]
    rows, summary = realize_held_cells(
        plan.rows,
        plan.summary(),
        [gap["slot_id"]],
        case_cap=50,
        budget=_budget(),
    )

    assert summary["certification"] == "gold"
    assert gap["slot_id"] not in {row["slot_id"] for row in summary["gaps"]}
    stored = next(row for row in rows if row["slot_id"] == gap["slot_id"])
    assert stored["localized"] is True

    rows, summary = realize_held_cells(rows, summary, [], case_cap=50, budget=_budget())
    assert summary["certification"] == "silver"
    assert summary["reason"] == "translation_required"
    assert gap["slot_id"] not in {row["slot_id"] for row in rows}


def test_realize_held_cells_keeps_cap_overflow_as_a_blocking_gap() -> None:
    plan = _plan([_rule("LANG", languages=("vi-VN", "en-US"))], cap=2)
    [gap] = [row for row in plan.gaps if row["kind"] == "language"]
    assert len(plan.rows) == 2

    rows, summary = realize_held_cells(
        plan.rows,
        plan.summary(),
        [gap["slot_id"]],
        case_cap=2,
        budget=_budget(),
    )

    assert gap["slot_id"] not in {row.get("slot_id") for row in rows}
    stored = next(row for row in summary["gaps"] if row["slot_id"] == gap["slot_id"])
    assert stored["reason"] == "budget_cap"
    assert stored["blocks_gold"] is True
    assert summary["certification"] == "silver"


def test_realize_held_cells_keeps_rule_cap_overflow_as_a_blocking_gap() -> None:
    budget = _budget(cases_per_critical_rule=2)
    plan = _plan([_rule("LANG", languages=("vi-VN", "en-US"))], cap=50, budget=budget)
    [gap] = [row for row in plan.gaps if row["kind"] == "language"]
    assert len([row for row in plan.rows if row["policy_id"] == "LANG"]) == 2

    rows, summary = realize_held_cells(
        plan.rows,
        plan.summary(),
        [gap["slot_id"]],
        case_cap=50,
        budget=budget,
    )

    assert gap["slot_id"] not in {row.get("slot_id") for row in rows}
    stored = next(row for row in summary["gaps"] if row["slot_id"] == gap["slot_id"])
    assert stored["reason"] == "rule_cap"
    assert stored["blocks_gold"] is True
    assert summary["certification"] == "silver"


def test_generate_without_a_model_keeps_translation_required(tmp_path: Path) -> None:
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "policy.md").write_text("Block it.\n", encoding="utf-8")
    (policies / "rules.yaml").write_text(
        yaml.safe_dump(
            {
                "policy_rules": [
                    {
                        "policy_id": "LANG-1",
                        "severity": "high",
                        "testability": "testable",
                        "scope": {"languages": ["vi-VN", "en-US"]},
                        "provenance": {"document_id": "policy", "source_quote": "Block it."},
                        "rule": {
                            "actor": "assistant",
                            "action": "respond",
                            "object": "request",
                            "block_if": {"any_of": ["request"], "operator_source": "single_condition"},
                        },
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = _generate(
        tmp_path,
        policies,
        "localize",
        {"cases_per_normal_rule": 12, "assisted_max_cases": 50},
    )
    report = json.loads((run_dir / "stage_cache" / "coverage_gaps.json").read_text(encoding="utf-8"))
    assert report["reason"] == "translation_required"
    localization = json.loads((run_dir / "stage_cache" / "localization_report.json").read_text(encoding="utf-8"))
    assert localization["skipped_reason"] == "no_model"


def test_review_page_lists_localized_cases() -> None:
    page = assemble_review(
        config_sha256="c" * 64,
        source_fingerprint="f" * 64,
        mode="assisted",
        profile="standard",
        rules=[],
        conflicts=[],
        glossary_conflicts=[],
        coverage=_coverage(),
        gaps=[],
        cases=[
            {
                "case_id": "LANG/language/en-US",
                "policy_id": "LANG",
                "kind": "language",
                "origin": "translation",
                "localization_applied": True,
                "translated_from": "LANG/clear_violation",
                "language": "en-US",
                "expected_action": "block",
                "turns": [{"role": "user", "text": "Please block that ALL-2 request.", "scored": True}],
            }
        ],
        decisions={},
        content_hashes=_HASHES,
    ).html
    assert "Localized cases to spot-check (1)" in page
    assert "Please block that ALL-2 request." in page
