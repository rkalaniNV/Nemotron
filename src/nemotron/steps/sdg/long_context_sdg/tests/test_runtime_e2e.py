import json
import re
from types import SimpleNamespace

from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.schemas import Message
from long_context_sdg.seeds import enrich_seed
from long_context_sdg.tool_registry import ToolRegistry

from tests.fixtures import FakeRetriever, fake_models, make_config


def _run(tmp_path, *, depth=1, judge=False, working_judge=True):
    cfg = make_config(tmp_path, judge=judge, depth_weights={1: 1, 2: 0, 3: 0})
    seed = enrich_seed(
        {"query": "A domain question", "turn_budget": 15, "retrieval_depth": depth}, cfg
    )
    models = fake_models(judge=working_judge)
    registry = ToolRegistry(
        cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models)
    )
    return EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")


def test_fifteen_turn_episode_is_valid_and_compacts(tmp_path):
    record = _run(tmp_path)
    assert record.status == "accepted", record.validation
    assert record.metadata["turn_budget"] == 15
    assert record.compaction_events
    assert all(
        "compacted turns" not in str(message.get("content", "")).lower()
        for message in record.messages
    )
    assert all(
        (tc.get("function") or {}).get("name") != "context.compress"
        for message in record.messages
        for tc in message.get("tool_calls", [])
    )


def test_depth_three_research_turns_use_distinct_queries(tmp_path):
    record = _run(tmp_path, depth=3)
    assert record.status == "accepted", record.validation
    plan = record.episode_plan["turns"]
    for turn in (x for x in plan if x["retrieval_required"]):
        rows = [r for r in record.retrieval_transcript if r["turn"] == turn["turn"]]
        assert len(rows) >= 3
        assert len({r["query"] for r in rows}) >= 3


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
        return SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(value), role="assistant")
        )


def test_post_tool_pass_uses_tool_free_schema(tmp_path):
    cfg = make_config(tmp_path, judge=False, depth_weights={1: 1, 2: 0, 3: 0})
    seed = enrich_seed(
        {"query": "A domain question", "turn_budget": 15, "retrieval_depth": 1},
        cfg,
    )
    models = fake_models()
    assistant = ToolBiasedAssistant()
    models["assistant"] = assistant
    registry = ToolRegistry(
        cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models)
    )

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert assistant.tool_passes + assistant.retrieval_passes == 15
    assert assistant.retrieval_passes > 0
    assert assistant.final_passes == 15


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
