"""Executor protocol and shared runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from long_context_sdg.schemas import RetrievalChunk, ToolCall, ToolResult


class ToolExecutionError(Exception):
    pass


@dataclass
class ConversationState:
    conversation_id: str
    turn: int = 1
    memory: dict[str, Any] = field(default_factory=dict)
    retrieved: dict[str, RetrievalChunk] = field(default_factory=dict)
    retrieval_transcript: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    rejected_tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionServices:
    retriever: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    simulator_alias: str = "assistant"


@dataclass
class ExecutionContext:
    tool_name: str
    tool_schema: dict[str, Any]
    instructions: str


class ToolExecutor(Protocol):
    def execute(
        self,
        call: ToolCall,
        state: ConversationState,
        context: ExecutionContext,
    ) -> ToolResult: ...
