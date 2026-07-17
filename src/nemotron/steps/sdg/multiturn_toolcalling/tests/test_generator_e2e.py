"""End-to-end live Job B via fake models + fake retriever (no DD engine, no key).

Exercises: user turns, the live retrieve -> rewrite -> answer loop, automatic
(metadata-only) compaction at the token threshold, bounded reasoning, and the
clean structured_messages projection with NO compress artifacts.
"""

from __future__ import annotations

import json

from mtsdg.generator import EpisodeRunner
from mtsdg.generator_config import EpisodeSimulatorConfig
from tests.fixtures import FakeRetriever, make_fake_models, make_query


def _run(turn_budget=8, threshold=1200):
    query = make_query(turn_budget=turn_budget)
    cfg = EpisodeSimulatorConfig(
        name="conversation", run_trajectory_judge=False, majority_vote_n=1,
        context_token_threshold=threshold,
    )
    return EpisodeRunner(cfg).run_episode(make_fake_models(), query, FakeRetriever()), query


def test_episode_produces_valid_trajectory():
    result, query = _run()
    msgs = json.loads(result["structured_messages"])
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert roles.count("user") >= query.turn_budget - 1
    assert "tool" in roles
    assert result["trajectory_status"] is True, result["trajectory_validation"]


def test_live_query_rewrite_loop():
    result, _ = _run()
    msgs = json.loads(result["structured_messages"])
    retrieves = [
        m for m in msgs
        if m["role"] == "assistant" and any(
            (tc.get("function", {}) or {}).get("name") == "retrieve" for tc in (m.get("tool_calls") or [])
        )
    ]
    assert len(retrieves) >= 2
    # A vague first retrieve surfaced a distractor; a rewrite surfaced authority.
    tool_results = [m for m in msgs if m["role"] == "tool" and m.get("name") == "retrieve"]
    surfaced = set()
    for m in tool_results:
        for c in json.loads(m["content"]):
            surfaced.add(c["chunk_id"])
    assert any(cid.startswith("doc_10") for cid in surfaced)   # distractor from vague query
    assert any(cid.startswith("doc_00") for cid in surfaced)   # authority from rewrite


def test_compaction_metadata_only_not_in_chat():
    result, _ = _run()
    meta = json.loads(result["episode_metadata"])
    assert len(meta["compaction_events"]) >= 1
    assert json.loads(result["compaction_events"])  # hidden provenance retained
    msgs = json.loads(result["structured_messages"])
    for m in msgs:
        assert m.get("name") != "context.compress"
        for tc in m.get("tool_calls") or []:
            assert (tc.get("function", {}) or {}).get("name") != "context.compress"
        assert "compacted context" not in (m.get("content") or "")


def test_reasoning_bounded_and_grounded():
    result, _ = _run()
    msgs = json.loads(result["structured_messages"])
    reasoned = [m for m in msgs if m["role"] == "assistant" and m.get("reasoning_content")]
    assert reasoned
    from mtsdg.tokens import count_tokens
    assert all(count_tokens(m["reasoning_content"]) <= 400 for m in reasoned)
