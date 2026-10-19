"""LLM paraphrase of scenario seeds into realistic user turns."""

import json
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.config import DEFAULT_GENERATION, load_resolved
from nemotron.steps.byob.runtime.benchmark_families.safety.paraphrase import (
    PARAPHRASE_TEMPERATURE,
    SYSTEM_PROMPT,
    apply_paraphrases,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from tests.steps.byob.safety_fixtures import FIVE_TOPIC, FIVE_TOPIC_REPLIES, ScriptedExtractor
from tests.steps.byob.test_safety_coverage import _rule
from tests.steps.byob.test_safety_rules import _config
from tests.steps.byob.test_safety_scenarios import _cases, _one

_ALL_OF = {"all_of": ["verb", "target"], "operator_source": "explicit"}


class FakeCompleter:
    identity = {
        "provider": "fake",
        "model": "paraphrase",
        "inference_parameters": {"temperature": 0.0, "max_tokens": 4096},
    }

    def __init__(self, replies: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.requested: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.waves: list[list[str]] = []
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
        del system_prompt, prompt, output_format, prefix
        self.inference_calls.append(inference_parameters)
        if self.fail:
            raise ConnectionError("endpoint refused the connection")
        answers: dict[str, Any] = {}
        wave: list[str] = []
        for row in rows:
            request_id = row["request_id"]
            self.requested.append(request_id)
            wave.append(request_id)
            payload = json.loads(row["payload"]) if isinstance(row.get("payload"), str) else {}
            self.payloads.append(payload)
            if request_id in self.replies:
                answers[request_id] = self.replies[request_id]
                continue
            count = int(payload.get("turn_count") or 1)
            turns = []
            for index in range(1, count + 1):
                turns.append({"role": "user", "text": f"User follow-up {request_id} #{index}."})
                if index < count:
                    turns.append({"role": "assistant", "text": f"Assistant reply {request_id} #{index}."})
            answers[request_id] = {"turns": turns}
        self.waves.append(wave)
        return answers


def test_paraphrase_rewrites_seed_turns_and_keeps_cell_meaning(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    extractor = FakeCompleter()

    updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    violation = _one(updated, "clear_violation")
    assert report["status"] == "applied"
    assert report["applied"] == len(build.cases)
    assert report["kept_seed"] == 0
    assert violation["origin"] == "paraphrase"
    assert violation["present_legs"] == ["verb", "target"]
    assert violation["expected_action"] == "block"
    assert violation["evidence"]["source_quote"] == "Block it."
    assert violation["turns"][0]["text"] != "Block it."
    assert violation["slot_id"] in violation["turns"][0]["text"]


def test_a_quote_copy_keeps_the_seed(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    violation = _one(build, "clear_violation")
    extractor = FakeCompleter({violation["slot_id"]: {"turns": [{"text": "Block it."}]}})

    updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    kept = _one(updated, "clear_violation")
    assert kept["origin"] == "seed"
    assert kept["turns"][0]["text"] == "Block it."
    assert any(row["slot_id"] == violation["slot_id"] and row["reason"] == "quote_copy" for row in report["failures"])
    assert report["status"] == "partial"


def test_multi_turn_paraphrase_writes_a_dialogue(tmp_path: Path) -> None:
    rule = _rule("MT", block=_ALL_OF, quote="Follow the multi-turn policy.", signals={"multi_turn_turns": 2})
    _plan, build = _cases([rule])
    seed = _one(build, "multi_turn")

    updated, report = apply_paraphrases(build, FakeCompleter(), tmp_path, enabled=True)

    case = _one(updated, "multi_turn")
    assert report["status"] == "applied"
    assert case["origin"] == "paraphrase"
    assert [turn["role"] for turn in case["turns"]] == ["user", "assistant", "user"]
    assert [turn["scored"] for turn in case["turns"]] == [False, False, True]
    assert "expected_action" not in case["turns"][0]
    assert "expected_action" not in case["turns"][1]
    assert case["turns"][-1]["expected_action"] == seed["expected_action"]
    assert case["turns"][0]["text"] != case["turns"][2]["text"]
    assert all(turn["text"] != "Follow the multi-turn policy." for turn in case["turns"])
    assert len(seed["turns"]) == 2
    assert {turn["role"] for turn in seed["turns"]} == {"user"}


def test_user_only_multi_turn_keeps_the_seed(tmp_path: Path) -> None:
    rule = _rule("MT", block=_ALL_OF, quote="Follow the multi-turn policy.", signals={"multi_turn_turns": 2})
    _plan, build = _cases([rule])
    case = _one(build, "multi_turn")
    extractor = FakeCompleter(
        {
            case["slot_id"]: {
                "turns": [
                    {"role": "user", "text": "First user message."},
                    {"role": "user", "text": "Second user message."},
                ]
            }
        }
    )

    updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    kept = _one(updated, "multi_turn")
    assert kept["origin"] == "seed"
    assert {turn["role"] for turn in kept["turns"]} == {"user"}
    assert any(row["slot_id"] == case["slot_id"] and row["reason"] == "turn_count" for row in report["failures"])


def test_a_wrong_turn_count_keeps_the_seed(tmp_path: Path) -> None:
    rule = _rule("MT", block=_ALL_OF, quote="Follow the multi-turn policy.", signals={"multi_turn_turns": 2})
    _plan, build = _cases([rule])
    case = _one(build, "multi_turn")
    extractor = FakeCompleter({case["slot_id"]: {"turns": [{"text": "Only one user message."}]}})

    updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    kept = _one(updated, "multi_turn")
    assert kept["origin"] == "seed"
    assert len(kept["turns"]) == len(case["turns"])
    assert any(row["slot_id"] == case["slot_id"] and row["reason"] == "turn_count" for row in report["failures"])


def test_model_failure_keeps_every_seed(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])

    updated, report = apply_paraphrases(build, FakeCompleter(fail=True), tmp_path, enabled=True)

    assert report["status"] == "partial"
    assert report["skipped_reason"] == "model_unavailable"
    assert report["applied"] == 0
    assert {case["origin"] for case in updated.cases} == {"seed"}


@pytest.mark.parametrize("reply", [None, [], "not a mapping"])
def test_invalid_batch_reply_keeps_every_seed(tmp_path: Path, reply: Any) -> None:
    class InvalidBatchCompleter(FakeCompleter):
        def complete(self, rows: list[Any], **kwargs: Any) -> Any:
            del rows, kwargs
            return reply

    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])

    updated, report = apply_paraphrases(build, InvalidBatchCompleter(), tmp_path, enabled=True)

    assert report["status"] == "partial"
    assert report["skipped_reason"] == "invalid_batch_reply"
    assert report["applied"] == 0
    assert report["failures"] == [
        {"slot_id": None, "reason": "model returned a non-mapping batch reply"}
    ]
    assert {case["origin"] for case in updated.cases} == {"seed"}


def test_disabled_paraphrase_does_not_call_the_model(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    extractor = FakeCompleter()

    updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=False)

    assert report["skipped_reason"] == "disabled"
    assert extractor.requested == []
    assert {case["origin"] for case in updated.cases} == {"seed"}


def test_cached_paraphrase_is_not_requested_again(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    first = FakeCompleter()
    apply_paraphrases(build, first, tmp_path, enabled=True)
    second = FakeCompleter()

    updated, report = apply_paraphrases(build, second, tmp_path, enabled=True)

    assert second.requested == []
    assert report["cached"] == len(build.cases)
    assert report["requested"] == 0
    assert {case["origin"] for case in updated.cases} == {"paraphrase"}


def test_default_config_enables_paraphrase() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.safety.config import BYOB_ROOT

    resolved = load_resolved(BYOB_ROOT / "safety" / "config" / "default.yaml")
    assert DEFAULT_GENERATION["modules"]["paraphrase"] is True
    assert resolved.generation["modules"]["paraphrase"] is True


def test_paraphrase_prompt_requires_scored_legs_and_bans_greetings() -> None:
    prompt = SYSTEM_PROMPT.casefold()
    assert "present_legs" in prompt
    assert "avoid_repeating" in prompt
    assert "next to the policy" in prompt
    assert "greeting" in prompt
    assert "xin chào" not in prompt
    assert "hello" not in prompt


def test_paraphrase_does_not_use_temperature_zero(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    extractor = FakeCompleter()

    _updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    assert extractor.inference_calls
    assert report["inference_parameters"]["temperature"] == PARAPHRASE_TEMPERATURE
    assert report["inference_parameters"]["max_tokens"] == 4096
    for params in extractor.inference_calls:
        assert params["temperature"] == PARAPHRASE_TEMPERATURE
        assert params["temperature"] != 0
        assert params["max_tokens"] == 4096


def _scored_follow_up(slot_id: str, payload: dict[str, Any]) -> str:
    count = int(payload.get("turn_count") or 1)
    return f"User follow-up {slot_id} #{count}."


def test_same_rule_paraphrase_payload_includes_prior_scored_turns(tmp_path: Path) -> None:
    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    extractor = FakeCompleter()

    apply_paraphrases(build, extractor, tmp_path, enabled=True)

    assert extractor.payloads
    assert extractor.payloads[0]["avoid_repeating"] == []
    first_scored = _scored_follow_up(extractor.requested[0], extractor.payloads[0])
    later = extractor.payloads[1:]
    assert later
    assert first_scored in later[0]["avoid_repeating"]
    assert all(len(payload["avoid_repeating"]) <= 4 for payload in extractor.payloads)


def test_avoid_examples_are_same_rule_only(tmp_path: Path) -> None:
    first = _rule("ALL-A", block=_ALL_OF, quote="Block A.", span="doc#1#span-01")
    second = _rule("ALL-B", block=_ALL_OF, quote="Block B.", span="doc#1#span-02")
    _plan, build = _cases([first, second])
    extractor = FakeCompleter()

    apply_paraphrases(build, extractor, tmp_path, enabled=True)

    assert extractor.waves
    first_wave = extractor.waves[0]
    assert len(first_wave) == 2
    by_slot = {case["slot_id"]: case["policy_id"] for case in build.cases}
    wave_payloads = extractor.payloads[:2]
    assert all(payload["avoid_repeating"] == [] for payload in wave_payloads)
    scored = {
        slot_id: _scored_follow_up(slot_id, payload)
        for slot_id, payload in zip(first_wave, wave_payloads)
    }
    later = list(zip(extractor.requested[2:], extractor.payloads[2:]))
    assert later
    seen_policy: set[str] = set()
    for slot_id, payload in later:
        own_rule = by_slot[slot_id]
        other_examples = [scored[prior] for prior in first_wave if by_slot[prior] != own_rule]
        assert all(example not in payload["avoid_repeating"] for example in other_examples)
        if own_rule in seen_policy:
            continue
        seen_policy.add(own_rule)
        own_examples = [scored[prior] for prior in first_wave if by_slot[prior] == own_rule]
        assert own_examples
        assert all(example in payload["avoid_repeating"] for example in own_examples)


def test_paraphrase_keeps_a_nonzero_configured_temperature(tmp_path: Path) -> None:
    class WarmCompleter(FakeCompleter):
        identity = {
            "provider": "fake",
            "model": "paraphrase",
            "inference_parameters": {"temperature": 0.4},
        }

    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])
    extractor = WarmCompleter()

    _updated, report = apply_paraphrases(build, extractor, tmp_path, enabled=True)

    assert report["inference_parameters"]["temperature"] == 0.4
    assert extractor.inference_calls[0]["temperature"] == 0.4


def test_exact_same_rule_example_is_rejected(tmp_path: Path) -> None:
    class RepeatingCompleter(FakeCompleter):
        def complete(self, rows: list[Any], **kwargs: Any) -> dict[str, Any]:
            del kwargs
            answers: dict[str, Any] = {}
            for row in rows:
                payload = json.loads(row["payload"])
                count = int(payload.get("turn_count") or 1)
                turns = []
                for index in range(1, count):
                    turns.extend(
                        [
                            {"role": "user", "text": f"Context message {index}."},
                            {"role": "assistant", "text": f"Context reply {index}."},
                        ]
                    )
                turns.append({"role": "user", "text": "Repeat this exact request."})
                answers[row["request_id"]] = {"turns": turns}
            return answers

    rule = _rule("ALL-2", block=_ALL_OF, quote="Block it.")
    _plan, build = _cases([rule])

    updated, report = apply_paraphrases(build, RepeatingCompleter(), tmp_path, enabled=True)

    assert report["applied"] == 1
    assert report["kept_seed"] == len(build.cases) - 1
    assert sum(case["origin"] == "paraphrase" for case in updated.cases) == 1
    repeated = [row for row in report["failures"] if row["reason"] == "repeated_same_rule_example"]
    assert len(repeated) == len(build.cases) - 1


def test_generate_paraphrases_seed_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "make_extractor", lambda config, run_dir: ScriptedExtractor(FIVE_TOPIC_REPLIES))
    config = _config(tmp_path, {"five-topic.md": FIVE_TOPIC})
    prepare_safety(config)

    with pytest.raises(SafetyStatus) as caught:
        generate_safety(config)

    assert caught.value.status is RunStatus.REVIEW_REQUIRED
    run_dir = tmp_path / "out" / "customer-safety-v1"
    cache = run_dir / "stage_cache"
    lines = (cache / "scenario_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines]
    report = json.loads((cache / "paraphrase_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "applied"
    assert report["applied"] == len(cases)
    assert cases
    assert {case["origin"] for case in cases} == {"paraphrase"}
    for case in cases:
        assert case["turns"][0]["text"] != case["evidence"]["source_quote"]
    record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
    assert record["scenarios"]["paraphrase"]["applied"] == len(cases)
