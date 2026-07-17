"""Pydantic schemas for every artifact in the generic multi-turn long-context SDG
pipeline.

Domain-agnostic. The pipeline turns a ``queries.jsonl`` seed file into 20-25 turn
tool-calling trajectories over three model tools — ``retrieve`` (the live
retriever), ``memory_read``, ``memory_write``. Context compaction is automatic and
internal: it fires when the running context crosses a token threshold (32k in
production; a smaller simulation threshold in shorter synthetic runs) and never
appears in the emitted chat as a tool call.

Design principle: be permissive where the teacher LLM's phrasing may vary, but
strict on the provenance-carrying fields (chunk IDs, source message IDs,
``no_new_claims``, allowed memory keys).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Allowed-memory policy (memory_read / memory_write)
# --------------------------------------------------------------------------- #

#: Only these preference/context keys may ever be written to durable memory.
#: Generic user preferences — never task facts, secrets, or conclusions.
ALLOWED_MEMORY_KEYS = frozenset(
    {
        "preferred_language",
        "verbosity",
        "expertise_level",
        "response_format",
        "preferred_units",
        "focus_area",
        "citation_style",
    }
)

#: Fields a ``context.compress`` call may ask to preserve.
PreserveField = Literal[
    "user_stated_facts",
    "constraints",
    "authorities",
    "tool_results",
    "open_questions",
    "memory_preferences",
    "decisions",
]

RETRIEVER_TOOL = "retrieve"
MODEL_TOOLS = [RETRIEVER_TOOL, "memory_read", "memory_write"]


# --------------------------------------------------------------------------- #
# Input: queries.jsonl seed row
# --------------------------------------------------------------------------- #


class PersonaSeed(BaseModel):
    """Lightweight persona driving the user agent (generic)."""

    role: str = "user"
    expertise: Literal["novice", "intermediate", "expert"] = "intermediate"
    style: str = Field("", description="Short note on voice/tone, e.g. 'terse, technical'.")


class QuerySeed(BaseModel):
    """One input row of ``queries.jsonl`` — the seed information-need for an episode.

    The pipeline expands this single query into a coherent 20-25 turn conversation
    (subtopics, follow-ups, clarifications) grounded in passages the assistant
    pulls from the live retriever as it drives the retrieve -> assess -> rewrite ->
    answer loop.
    """

    query_id: str
    query: str = Field(..., description="The seed user question / information-need (precise).")
    naive_query: str = Field(
        "", description="A vaguer first phrasing the user would try before refining. "
        "Drives the retrieve->rewrite loop; falls back to `query` if empty."
    )
    domain: str = Field("general", description="Free-text domain hint, e.g. 'kubernetes'.")
    corpus_hints: List[str] = Field(
        default_factory=list, description="Optional keywords to steer retrieval queries."
    )
    turn_budget: int = Field(22, ge=6, le=40, description="Target number of user turns (20-25).")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    persona: PersonaSeed = Field(default_factory=PersonaSeed)
    memory_seed: Dict[str, Any] = Field(
        default_factory=dict, description="Initial durable memory (allowed keys only)."
    )


# --------------------------------------------------------------------------- #
# Runtime tool-result shape (what ``retrieve`` returns to the model)
# --------------------------------------------------------------------------- #


class RetrievalChunk(BaseModel):
    """The lean result ``retrieve`` returns to the model. No hidden fields."""

    chunk_id: str
    title: str = ""
    content: str
    source: str = "synthetic-corpus"
    url: Optional[str] = None
    date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Context-compaction result schema (conversation-scoped working state)
# --------------------------------------------------------------------------- #


class UserStatedFact(BaseModel):
    fact: str
    source_message_ids: List[str] = Field(default_factory=list)


class AuthorityRef(BaseModel):
    chunk_id: str
    title: str = ""


class KeyFact(BaseModel):
    """A substantive fact/rule already established from retrieval, with its sources.

    This is what lets a later turn answer a settled point FROM the summary instead
    of re-retrieving — the summary carries the substance, not just chunk pointers.
    """

    fact: str
    supporting_chunk_ids: List[str] = Field(default_factory=list)


class ToolOutcome(BaseModel):
    tool_call_id: str
    result: str


class CompressionEvent(BaseModel):
    """Result of ``context.compress``. Conversation-scoped; never written to memory.

    A source-linked rolling summary of a completed prefix. Adds no new facts.
    """

    summary_id: str
    covers_turns: List[int] = Field(..., min_length=2, max_length=2)
    user_stated_facts: List[UserStatedFact] = Field(default_factory=list)
    # Substantive answers established so far (so later turns reuse them, not re-retrieve).
    key_facts: List[KeyFact] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    authorities: List[AuthorityRef] = Field(default_factory=list)
    tool_outcomes: List[ToolOutcome] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    memory_preferences: Dict[str, Any] = Field(default_factory=dict)
    source_message_ids: List[str] = Field(default_factory=list)
    no_new_claims: bool = True


# --------------------------------------------------------------------------- #
# Bounded reasoning_content (think tokens)
# --------------------------------------------------------------------------- #


class EvidenceSelection(BaseModel):
    chunk_id: str
    purpose: str = ""


class ClaimAndSupport(BaseModel):
    claim: str
    supporting_chunk_ids: List[str] = Field(default_factory=list)


class ReasoningContent(BaseModel):
    """Bounded, auditable think tokens.

    ``think`` is the natural-language reasoning trace (the trainable field). The
    remaining structured fields are a hidden auditable index so grounding gates
    still run: every claim in ``claims`` must cite a chunk actually returned by
    ``retrieve``, and the ``think`` trace must stay within the token budget.

    ``retrieval_assessment`` records whether the last retrieval sufficed and, if
    not, why a rewrite is needed — this is what teaches the query-rewrite skill.
    """

    think: str = Field("", description="Natural-language reasoning trace (trainable).")
    task_understanding: str = ""
    retrieval_assessment: str = Field(
        "", description="Judgement of retrieved chunks: sufficient, or rewrite needed and why."
    )
    evidence_selection: List[EvidenceSelection] = Field(default_factory=list)
    claims: List[ClaimAndSupport] = Field(default_factory=list)
    answer_plan: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Messages / turn blocks / assistant turns
# --------------------------------------------------------------------------- #


class Message(BaseModel):
    """An OpenAI-style chat message. Permissive to accommodate tool traffic."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = ""
    reasoning_content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # Bookkeeping (kept in provenance, stripped from final SFT messages).
    turn: Optional[int] = None
    message_id: Optional[str] = None
    reasoning_structured: Optional[Dict[str, Any]] = None

    def to_openai(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        if self.reasoning_content is not None:
            out["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


class AssistantTurn(BaseModel):
    """One assistant action: tool calls (resolved live by the tool executor) or a
    final answer, always with a bounded ``reasoning`` trace. Majority voting selects
    the consensus over ``n`` of these."""

    reasoning: ReasoningContent = Field(default_factory=ReasoningContent)
    content: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class TrajectoryJudgment(BaseModel):
    """Structured verdict from the trajectory judge."""

    coherence: int = Field(3, ge=1, le=5)
    grounding: int = Field(3, ge=1, le=5)
    helpfulness: int = Field(3, ge=1, le=5)
    tool_use: int = Field(3, ge=1, le=5)
    rating: Literal["success", "failure"] = "success"
    explanation: str = ""
