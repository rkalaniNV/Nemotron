"""Live tool executor: retrieval + memory policy."""

from __future__ import annotations

import json

import pytest

from mtsdg.runtime import ConversationState, LiveToolExecutor, ToolCall, ToolError
from mtsdg.schemas import MODEL_TOOLS
from tests.fixtures import FakeRetriever


def _call(name, args, cid="x"):
    return ToolCall.from_openai(
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    )


def _exec():
    return LiveToolExecutor(FakeRetriever(), allowed_tools=MODEL_TOOLS), ConversationState(conversation_id="c")


def test_retrieve_vague_vs_specific():
    ex, state = _exec()
    r_vague = ex.execute(_call("retrieve", {"query": "making new states"}), state)
    assert all(c["chunk_id"].startswith("doc_10") for c in r_vague.payload)  # distractor
    r_spec = ex.execute(_call("retrieve", {"query": "Article 3 Parliament form a new State"}), state)
    assert any(c["chunk_id"].startswith("doc_00") for c in r_spec.payload)   # authority
    # retrieved chunks accumulate for grounding.
    assert set(state.retrieved).issuperset({"doc_00_p1_r1", "doc_10_p3_r1"})


def test_retrieve_requires_query():
    ex, state = _exec()
    with pytest.raises(ToolError):
        ex.execute(_call("retrieve", {"query": "  "}), state)


def test_memory_write_allowed_and_disallowed():
    ex, state = _exec()
    ok = ex.execute(_call("memory_write", {"key": "verbosity", "value": "detailed",
                                           "scope": "user", "reason": "asked"}), state)
    assert ok.payload["saved"] and state.memory["verbosity"] == "detailed"
    with pytest.raises(ToolError):
        ex.execute(_call("memory_write", {"key": "ssn", "value": "x", "scope": "user", "reason": "no"}), state)
    with pytest.raises(ToolError):
        ex.execute(_call("memory_write", {"key": "focus_area", "value": {"a": 1},
                                          "scope": "user", "reason": "no"}), state)


def test_memory_read_restricts_to_allowed():
    ex, state = _exec()
    state.memory.update({"verbosity": "concise", "leaked": "x"})
    r = ex.execute(_call("memory_read", {"scope": "user"}), state)
    assert r.payload == {"verbosity": "concise"}


def test_unknown_tool_rejected():
    ex, state = _exec()
    with pytest.raises(ToolError):
        ex.execute(_call("context.compress", {"x": 1}), state)
