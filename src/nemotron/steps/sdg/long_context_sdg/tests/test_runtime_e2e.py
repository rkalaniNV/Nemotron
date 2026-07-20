import json
import re
from types import SimpleNamespace

from long_context_sdg.executors.base import ExecutionServices
from long_context_sdg.runtime import EpisodeRunner
from long_context_sdg.schemas import Message
from long_context_sdg.seeds import enrich_seed
from long_context_sdg.tool_registry import ToolRegistry

from tests.fixtures import FakeRetriever, FakeUser, fake_models, make_config


def _run(tmp_path, *, judge=False, working_judge=True):
    cfg = make_config(tmp_path, judge=judge)
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
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


def test_retrieval_is_not_forced_when_assistant_can_answer_naturally(tmp_path):
    record = _run(tmp_path)
    assert record.status == "accepted", record.validation
    assert record.retrieval_transcript == []
    assert not hasattr(record, "policy_events")


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
        self.final_passes = 0

    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
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
                        "arguments": {"query": f"independent-facet-{turn}"},
                    }
                ],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_post_tool_pass_uses_tool_free_schema(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    seed = enrich_seed(
        {"query": "A domain question", "turn_budget": 15},
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
    record = _run(tmp_path)

    assert record.status == "accepted", record.validation
    assert not hasattr(record, "decision_trace")
    assert not hasattr(record, "policy_events")
    per_turn = {}
    for attempt in record.tool_call_attempts:
        per_turn[attempt["turn"]] = per_turn.get(attempt["turn"], 0) + 1
    assert max(per_turn.values(), default=0) <= record.episode_spec["max_tool_calls_per_turn"]
    assert record.metadata["successful_retrieval_calls"] == 0


class RedundantAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system or "too lexically similar" in directive:
            value = {
                "reasoning": {"think": "Reuse the evidence instead of searching again."},
                "content": f"Answer using existing evidence for turn {turn}.",
            }
        else:
            value = {
                "reasoning": {"think": "Try a lookup."},
                "content": "",
                "tool_calls": [{"name": "retrieve", "arguments": {"query": "same repeated query"}}],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_lexical_duplicate_query_is_rejected_and_existing_evidence_is_reused(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    cfg.episode.max_retrieval_calls = 4
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    models["assistant"] = RedundantAssistant()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert len(record.retrieval_transcript) == 1
    assert any("too lexically similar" in item["error"] for item in record.metadata["rejected_tool_calls"])


class RetryAfterFailureAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
            value = {"reasoning": {"think": "Answer."}, "content": f"Final answer {turn}."}
        elif turn > 1 or "too lexically similar" in directive:
            value = {
                "reasoning": {"think": "Use the evidence already available."},
                "content": f"Answer from available context {turn}.",
                "tool_calls": [],
            }
        else:
            value = {
                "reasoning": {"think": "Retry the unresolved lookup."},
                "content": "",
                "tool_calls": [{"name": "retrieve", "arguments": {"query": "retryable facet"}}],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


class FailsOnceRetriever(FakeRetriever):
    def __init__(self):
        self.calls = 0

    def query(self, text, *, top_k=None):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("transient retrieval failure")
        return super().query(text, top_k=top_k)


def test_failed_retrieval_can_retry_within_turn_without_consuming_success_cap(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    models["assistant"] = RetryAfterFailureAssistant()
    retriever = FailsOnceRetriever()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=retriever, models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    first_turn = [item for item in record.tool_call_attempts if item["turn"] == 1]
    assert [item["success"] for item in first_turn] == [False, True]
    assert len(record.retrieval_transcript) == 1


class LowGainAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
            value = {"reasoning": {"think": "Answer."}, "content": f"Final answer {turn}."}
        elif "Tool results are now available" in directive or "low-gain search chain" in directive or turn > 3:
            value = {
                "reasoning": {"think": "Stop searching and answer."},
                "content": f"Answer from existing evidence {turn}.",
                "tool_calls": [],
            }
        else:
            queries = {
                1: "baseline mechanism alpha overview",
                2: "unresolved beta evidence limitations",
                3: "unresolved beta evidence caveats",
            }
            value = {
                "reasoning": {"think": "Search a remaining facet."},
                "content": "",
                "tool_calls": [{"name": "retrieve", "arguments": {"query": queries[turn]}}],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


class FixedEvidenceRetriever:
    def query(self, text, *, top_k=None):
        return FakeRetriever().query("fixed evidence", top_k=top_k)


def test_related_search_chain_stops_after_configured_low_gain_allowance(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    models["assistant"] = LowGainAssistant()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FixedEvidenceRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert len(record.retrieval_transcript) == 2
    assert record.retrieval_transcript[-1]["low_gain"] is True
    assert any("low-gain search chain" in item["error"] for item in record.metadata["rejected_tool_calls"])


class LexicallyVariedLowGainAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
            value = {"reasoning": {"think": "Answer."}, "content": f"Final answer {turn}."}
        elif (
            "Tool results are now available" in directive
            or "paused after repeated observed low-gain" in directive
            or turn > 4
        ):
            value = {
                "reasoning": {"think": "Stop searching and answer."},
                "content": f"Answer from existing evidence {turn}.",
                "tool_calls": [],
            }
        else:
            queries = {
                1: "baseline mechanism alpha overview",
                2: "dying declaration evidentiary exception",
                3: "admissibility statements made before death",
                4: "hearsay treatment of terminal statements",
            }
            value = {
                "reasoning": {"think": "Try another wording."},
                "content": "",
                "tool_calls": [{"name": "retrieve", "arguments": {"query": queries[turn]}}],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_lexically_varied_queries_cannot_reset_observed_low_gain_chain(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    cfg.episode.max_retrieval_calls = 6
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    models["assistant"] = LexicallyVariedLowGainAssistant()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FixedEvidenceRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert len(record.retrieval_transcript) == 3
    assert [row.get("consecutive_low_gain") for row in record.retrieval_transcript] == [0, 1, 2]
    assert any(
        "paused after repeated observed low-gain" in item["error"]
        for item in record.metadata["rejected_tool_calls"]
    )


class RetrievalAndMemoryAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        directive = str(getattr(messages[-1], "content", ""))
        turn_match = re.search(r"Turn (\d+)", directive)
        turn = int(turn_match.group(1)) if turn_match else 1
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
            value = {"reasoning": {"think": "Answer."}, "content": f"Final answer {turn}."}
        elif turn == 1:
            value = {
                "reasoning": {"think": "Evidence and saved preferences are both relevant."},
                "content": "",
                "tool_calls": [
                    {"name": "retrieve", "arguments": {"query": "independent evidence facet"}},
                    {"name": "memory_read", "arguments": {"scope": "user"}},
                ],
            }
        else:
            value = {
                "reasoning": {"think": "Answer conversationally."},
                "content": f"Natural answer {turn}.",
                "tool_calls": [],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_retrieval_cap_does_not_block_independent_memory_tool_in_same_action(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    seed = enrich_seed(
        {
            "query": "A domain question",
            "turn_budget": 15,
            "memory_seed": {"preferred_language": "English"},
        },
        cfg,
    )
    models = fake_models()
    models["assistant"] = RetrievalAndMemoryAssistant()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    first_turn_tools = [item["name"] for item in record.tool_call_attempts if item["turn"] == 1]
    assert first_turn_tools == ["retrieve", "memory_read"]
    assert record.memory_events[0]["action"] == "read"


class ParallelRetrievalAssistant:
    def completion(self, messages, **kwargs):
        system = str(getattr(messages[0], "content", ""))
        if "ASSISTANT FINAL ACTION SCHEMA" in system:
            value = {"reasoning": {"think": "Answer from the returned result."}, "content": "Final answer."}
        else:
            value = {
                "reasoning": {"think": "Try two lookups."},
                "content": "",
                "tool_calls": [
                    {"name": "retrieve", "arguments": {"query": "independent facet alpha"}},
                    {"name": "retrieve", "arguments": {"query": "independent facet beta"}},
                ],
            }
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value), role="assistant"))


def test_only_one_retrieval_executes_per_turn_by_default(tmp_path):
    cfg = make_config(tmp_path, judge=False)
    cfg.episode.max_retrieval_calls = 1
    seed = enrich_seed({"query": "A domain question", "turn_budget": 15}, cfg)
    models = fake_models()
    models["assistant"] = ParallelRetrievalAssistant()
    registry = ToolRegistry(cfg.tools, ExecutionServices(retriever=FakeRetriever(), models=models))

    record = EpisodeRunner(cfg).run(models, seed, registry, run_id="run-test")

    assert record.status == "accepted", record.validation
    assert len(record.retrieval_transcript) == 1
    assert max(
        (sum(item["turn"] == turn and item["name"] == "retrieve" for item in record.tool_call_attempts)
         for turn in range(1, 16)),
        default=0,
    ) == 1


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
