"""LLM paraphrase of scenario seeds into realistic user turns."""

import json
from pathlib import Path
from typing import Any

import pytest

from nemotron.steps.byob.runtime.benchmark_families.safety import extract
from nemotron.steps.byob.runtime.benchmark_families.safety.config import DEFAULT_GENERATION, load_resolved
from nemotron.steps.byob.runtime.benchmark_families.safety.paraphrase import apply_paraphrases
from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import generate_safety, prepare_safety
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, SafetyStatus
from tests.steps.byob.safety_fixtures import FIVE_TOPIC, FIVE_TOPIC_REPLIES, ScriptedExtractor
from tests.steps.byob.test_safety_coverage import _rule
from tests.steps.byob.test_safety_rules import _config
from tests.steps.byob.test_safety_scenarios import _cases, _one

_ALL_OF = {"all_of": ["verb", "target"], "operator_source": "explicit"}


class FakeCompleter:
    identity = {"provider": "fake", "model": "paraphrase"}

    def __init__(self, replies: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.requested: list[str] = []

    def complete(
        self,
        rows: list[Any],
        *,
        system_prompt: str = "",
        prompt: str = "",
        output_format: Any = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        del system_prompt, prompt, output_format, prefix
        if self.fail:
            raise ConnectionError("endpoint refused the connection")
        answers: dict[str, Any] = {}
        for row in rows:
            request_id = row["request_id"]
            self.requested.append(request_id)
            if request_id in self.replies:
                answers[request_id] = self.replies[request_id]
                continue
            payload = json.loads(row["payload"])
            count = int(payload.get("turn_count") or 1)
            turns = []
            for index in range(1, count + 1):
                turns.append({"role": "user", "text": f"User follow-up {request_id} #{index}."})
                if index < count:
                    turns.append({"role": "assistant", "text": f"Assistant reply {request_id} #{index}."})
            answers[request_id] = {"turns": turns}
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
