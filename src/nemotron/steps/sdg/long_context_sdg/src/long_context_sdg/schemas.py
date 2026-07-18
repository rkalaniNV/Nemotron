"""Public schemas for seeds, tools, messages, compaction, and canonical records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


class Persona(BaseModel):
    role: str = "user"
    expertise: str = "intermediate"
    style: str = "natural and curious"


class EpisodeSeed(BaseModel):
    query_id: str
    query: str = Field(min_length=1)
    naive_query: str = ""
    persona: Persona = Field(default_factory=Persona)
    instructions: str = ""
    turn_budget: int = Field(18, ge=6, le=40)
    retrieval_depth: int = Field(2, ge=1, le=3)
    memory_seed: dict[str, Any] = Field(default_factory=dict)


class TurnPlan(BaseModel):
    turn: int
    intent: str
    retrieval_required: bool = False
    retrieval_depth: int = 0


class EpisodePlan(BaseModel):
    query_id: str
    turns: list[TurnPlan]


class RetrievalChunk(BaseModel):
    chunk_id: str
    content: str
    title: str = ""
    source: str = ""
    score: float | None = None
    url: str | None = None
    date: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    payload: Any
    simulated: bool = False

    def to_message(self) -> dict[str, Any]:
        import json

        payload = self.payload
        if self.simulated:
            payload = {"_sdg_simulated": True, "value": payload}
        return {
            "role": "tool",
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }


class ReasoningContent(BaseModel):
    think: str = ""
    task_understanding: str = ""
    retrieval_assessment: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    answer_plan: list[str] = Field(default_factory=list)


class AssistantAction(BaseModel):
    reasoning: ReasoningContent = Field(default_factory=ReasoningContent)
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class AssistantRetrievalAction(BaseModel):
    """One model-authored query that the runtime turns into a retrieve call."""

    reasoning: ReasoningContent = Field(default_factory=ReasoningContent)
    query: str = Field(min_length=1)


class AssistantFinalAction(BaseModel):
    """Tool-free assistant response used after the evidence requirement is met."""

    reasoning: ReasoningContent = Field(default_factory=ReasoningContent)
    content: str = Field(min_length=1)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    turn: int | None = None
    message_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content:
            out["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


class UserFact(BaseModel):
    fact: str
    source_message_ids: list[str] = Field(default_factory=list)


class KeyFact(BaseModel):
    fact: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class CompressionEvent(BaseModel):
    summary_id: str
    covers_turns: list[int] = Field(min_length=2, max_length=2)
    user_facts: list[UserFact] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    no_new_claims: bool = True


class ValidationReport(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrajectoryJudgment(BaseModel):
    scores: dict[str, int] = Field(default_factory=dict)
    rating: Literal["success", "failure"] = "failure"
    explanation: str = ""


class CanonicalRecord(BaseModel):
    run_id: str
    config_fingerprint: str
    query_id: str
    status: Literal["accepted", "rejected", "quarantine", "generation_failed"]
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    episode_plan: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_transcript: list[dict[str, Any]] = Field(default_factory=list)
    memory_events: list[dict[str, Any]] = Field(default_factory=list)
    compaction_events: list[dict[str, Any]] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=dict)
    judgment: dict[str, Any] = Field(default_factory=dict)
