"""Shared test fixtures: a fake retriever, a fake model facade, and a query.

No API key / no network. The fake facade pattern-matches prompts to drive the
retrieve -> assess -> rewrite -> answer loop; the fake retriever returns weak
chunks for a vague query and authority chunks for a precise (rewritten) one, so
the query-rewrite behaviour is exercised deterministically.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, Dict, List

from mtsdg.schemas import PersonaSeed, QuerySeed, RetrievalChunk

# --------------------------------------------------------------------------- #
# Fake retriever
# --------------------------------------------------------------------------- #

_AUTH = [
    RetrievalChunk(chunk_id="doc_00_p1_r1", title="Constitution — doc_00, p.1",
                   content="Article 3. Parliament may by law form a new State by separation of "
                           "territory, unite States, or alter boundaries/names, on the President's "
                           "recommendation and after referring the Bill to the State legislature."),
    RetrievalChunk(chunk_id="doc_00_p1_r2", title="Constitution — doc_00, p.1",
                   content="Article 4. Laws under Articles 2 and 3 amend the First and Fourth "
                           "Schedules and are not deemed amendments under Article 368."),
]
_DIST = [
    RetrievalChunk(chunk_id="doc_10_p3_r1", title="Constitution — doc_10, p.3",
                   content="Article 249. Parliament may legislate on a State List matter in the "
                           "national interest if the Council of States so resolves."),
]


class FakeRetriever:
    """Returns authority chunks when the query is specific, distractors when vague."""

    def __init__(self, *a, **k):
        self.calls: List[str] = []

    def query(self, text: str, num_chunks: int = 3) -> List[RetrievalChunk]:
        self.calls.append(text)
        specific = any(t in text.lower() for t in ("article", "form a new state", "boundaries", "parliament may"))
        pool = _AUTH if specific else _DIST
        return pool[:num_chunks]

    def health(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #


def make_query(turn_budget: int = 8) -> QuerySeed:
    return QuerySeed(
        query_id="q-test",
        query="How can Parliament form a new State or alter State boundaries under the Constitution?",
        naive_query="making new states in India",
        domain="indian-constitution",
        turn_budget=turn_budget,
        persona=PersonaSeed(role="law student", expertise="intermediate", style="precise"),
        memory_seed={"preferred_language": "en", "verbosity": "concise"},
    )


# --------------------------------------------------------------------------- #
# Fake model facade
# --------------------------------------------------------------------------- #


def _compression_event() -> Dict[str, Any]:
    return {
        "summary_id": "ctx-tmp", "covers_turns": [1, 4],
        "user_stated_facts": [], "constraints": [], "authorities": [], "tool_outcomes": [],
        "open_questions": ["confirm ratification requirement"], "decisions": [],
        "memory_preferences": {"preferred_language": "en"}, "source_message_ids": [], "no_new_claims": True,
    }


class FakeFacade:
    def __init__(self, *a, **k):
        self.calls: List[str] = []

    def completion(self, chat_messages, **kwargs):
        text = "\n".join(_msg_text(m) for m in chat_messages)
        self.calls.append(text)
        last_role = None
        if chat_messages:
            last = chat_messages[-1]
            last_role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
        content = self._route(text, last_role)
        msg = SimpleNamespace(role="assistant", content=content, reasoning_content=None, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def acompletion(self, chat_messages, **kwargs):
        return self.completion(chat_messages, **kwargs)

    def _route(self, text: str, last_role) -> str:
        if "Compress the completed conversation prefix" in text:
            return json.dumps(_compression_event(), ensure_ascii=False)
        if "role-play a user talking to a research assistant" in text:
            return "How specifically does Parliament form a new State — what is the procedure?"
        if "Decide the next step for the LAST user turn" in text:
            return self._assistant(text, last_role)
        if "judge of a synthetic tool-calling training trajectory" in text:
            return json.dumps({"coherence": 5, "grounding": 4, "helpfulness": 4,
                               "tool_use": 5, "rating": "success", "explanation": "ok"})
        return "<explanation>ok</explanation>\n<rating>success</rating>"

    def _assistant(self, text: str, last_role) -> str:
        # Route on retrieved-chunk presence (robust to schema/example mentions of
        # the tool name): no results yet -> retrieve; distractor only -> rewrite;
        # authority present -> answer.
        has_auth = "doc_00" in text
        if last_role != "tool":
            return json.dumps({
                "reasoning": {"think": "Need evidence; retrieve first.",
                              "task_understanding": "new-state formation", "retrieval_assessment": "",
                              "evidence_selection": [], "claims": [], "answer_plan": ["retrieve"]},
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "retrieve", "arguments": json.dumps({"query": "making new states in India", "top_k": 3})}}],
            }, ensure_ascii=False)
        if not has_auth:
            return json.dumps({
                "reasoning": {"think": "Weak/off-topic results; rewrite with Article terms.",
                              "task_understanding": "Article 3 procedure",
                              "retrieval_assessment": "insufficient: got Article 249 (State List), not new-state formation; rewriting",
                              "evidence_selection": [], "claims": [], "answer_plan": ["rewrite", "retrieve"]},
                "content": "",
                "tool_calls": [{"id": "c2", "type": "function", "function": {
                    "name": "retrieve", "arguments": json.dumps(
                        {"query": "Article 3 Parliament may form a new State boundaries", "top_k": 3})}}],
            }, ensure_ascii=False)
        ids = re.findall(r'"chunk_id":\s*"([^"]+)"', text)
        auth = [c for c in ids if c.startswith("doc_00")]
        cite = (auth or ids or ["doc_00_p1_r1"])[-1]
        return json.dumps({
            "reasoning": {"think": f"Refined retrieval returned Article 3/4; answer from {cite}.",
                          "task_understanding": "new-State formation procedure",
                          "retrieval_assessment": "sufficient: Article 3 on point",
                          "evidence_selection": [{"chunk_id": cite, "purpose": "primary source"}],
                          "claims": [{"claim": "Parliament forms new States by law under Article 3",
                                      "supporting_chunk_ids": [cite]}],
                          "answer_plan": ["explain", "cite"]},
            "content": f"Under Article 3, Parliament may by law form a new State or alter boundaries, on "
                       f"the President's recommendation (see {cite}).",
            "tool_calls": [],
        }, ensure_ascii=False)


def _msg_text(m) -> str:
    if isinstance(m, dict):
        return str(m.get("content", ""))
    content = getattr(m, "content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def make_fake_models(*_a) -> Dict[str, Any]:
    facade = FakeFacade()
    return {a: facade for a in ("teacher", "user", "assistant", "judge", "compressor")}
