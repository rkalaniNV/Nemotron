import json
import re
from types import SimpleNamespace

from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.schemas import Message
from long_context_sdg.seeds import enrich_seed
from long_context_sdg.tool_registry import ToolRegistry

from tests.fixtures import FakeRetriever, FakeUser, fake_models, make_config


def _run(tmp_path, *, depth=1, judge=False, working_judge=True):
    cfg = make_config(tmp_path, judge=judge, depth_weights={1: 1, 2: 0, 3: 0})
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15, "retrieval_depth": depth}, cfg)
    models = fake_models(judge=working_judge)
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))
    return EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")


def test_fifteen_turn_episode_is_valid_and_compacts(tmp_path):
    record = _run(tmp_path)
    assert record.status == "accepted", record.validation
    assert record.metadata["turn_budget"] == 15
    assert record.compaction_events
    assert all("compacted turns" not in str(message.get("content", "")).lower() for message in record.messages)
    assert all(
        (tc.get("function") or {}).get("name") != "context.compress"
        for message in record.messages
        for tc in message.get("tool_calls", [])
    )


def test_retrieval_deadline_turns_use_distinct_queries(tmp_path):
    record = _run(tmp_path, depth=3)
    assert record.status == "accepted", record.validation
    for event in record.policy_events:
        rows = [r for r in record.retrieval_transcript if r["turn"] == event["turn"]]
        assert len(rows) >= event["required_retrievals_this_turn"]
        assert len({r["query"] for r in rows}) >= event["required_retrievals_this_turn"]


def test_judge_timeout_quarantines_without_losing_raw_record(tmp_path):
    record = _run(tmp_path, judge=True, working_judge=False)
    assert record.status == "quarantine"
    assert record.messages and record.validation["ok"] is True
    assert record.judgment["pending"] is True


def test_successful_judge_gates_to_accepted(tmp_path):
    record = _run(tmp_path, judge=True, working_judge=True)
    assert record.status == "accepted"
    assert record.judgment["rating"] == "success"


class ToolBiasedAssistant:
    """Requests a tool whenever the advertised response schema permits it."""

    def __init__(self):
        self.tool_passes = 0
        self.retrieval_passes = 0
        self.final_passes = 0

    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT RETRIEVAL ACTION SCHEMA" in system:
            self.retrieval_passes += 1
            value = {
                "reasoning": {"think": "Form a focused retrieval query."},
                "query": f"required biased query for turn {turn}",
            }
        elif "ASSISTANT FINAL ACTION SCHEMA" in system:
            self.final_passes += 1
            value = {
                "reasoning": {"think": "Synthesize the retrieved evidence."},
                "content": f"Tool-free grounded final answer for turn {turn}.",
            }
        else:
            self.tool_passes += 1
            value = {
                "reasoning": {"think": "Use the available retrieval tool."},
                "content": "",
                "tool_calls": [
                    {
                        "name": "retrieve",
                        "arguments": {"query": f"biased query for turn {turn}"},
                    }
                ],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_post_tool_pass_uses_tool_free_schema(tmp_path):
    cfg = make_config(tmp_path, judge=False, depth_weights={1: 1, 2: 0, 3: 0})
    seed = enrich_seed(
        {"query": "A domain question", "turn_budget": 15, "retrieval_depth": 1},
        cfg,
    )
    models = fake_models()
    assistant = ToolBiasedAssistant()
    models["assistant"] = assistant
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert len(record.tool_call_attempts) <= record.episode_spec["max_tool_calls_per_conversation"]
    assert (
        sum(x["name"] == "retrieve" for x in record.tool_call_attempts) <= record.episode_spec["max_retrieval_calls"]
    )
    assert assistant.tool_passes > 0
    assert assistant.final_passes == 15


def test_runtime_records_only_policy_interventions_and_enforces_per_turn_budget(tmp_path):
    record = _run(tmp_path, depth=3)

    assert record.status == "accepted", record.validation
    assert not hasattr(record, "decision_trace")
    assert 0 < len(record.policy_events) < record.episode_spec["turn_budget"]
    assert all(event["reason"] == "retrieval_deadline" for event in record.policy_events)
    per_turn = {}
    for attempt in record.tool_call_attempts:
        per_turn[attempt["turn"]] = per_turn.get(attempt["turn"], 0) + 1
    assert max(per_turn.values(), default=0) <= record.episode_spec["max_tool_calls_per_turn"]
    assert record.metadata["successful_retrieval_calls"] >= record.episode_spec["required_retrieval_calls"]


class CorrectingUser(FakeUser):
    def __init__(self):
        self.calls = 0

    def completion(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(message=SimpleNamespace(content="not valid json", role="assistant"))
        return super().completion(messages, **kwargs)


def test_user_message_retries_after_invalid_structured_output(tmp_path):
    cfg = make_config(tmp_path)
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    user = CorrectingUser()
    models["user"] = user
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert user.calls == record.episode_spec["turn_budget"]


def test_precompression_view_keeps_complete_conversation(tmp_path):
    cfg = make_config(tmp_path)
    runner = EpisodeRunner(cfg)
    messages = [Message(role="system", content="system", turn=0)]
    for turn in range(1, 6):
        messages.extend(
            [
                Message(role="user", content=f"u{turn}", turn=turn),
                Message(role="assistant", content=f"a{turn}", turn=turn),
            ]
        )

    view = runner._view(messages, None, recent_turns=3)

    assert [row["content"] for row in view] == [
        "u1",
        "a1",
        "u2",
        "a2",
        "u3",
        "a3",
        "u4",
        "a4",
        "u5",
        "a5",
    ]
