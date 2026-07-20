"""Live retrieval executor."""

from __future__ import annotations

from typing import Any

from long_context_sdg.schemas import ToolCall, ToolResult

from .base import (
    ConversationState,
    ExecutionContext,
    ExecutionServices,
    ToolExecutionError,
)


class RetrievalExecutor:
    def __init__(self, *, services: ExecutionServices, **_: Any):
        if services.retriever is None:
            raise ValueError("RetrievalExecutor requires a retriever service")
        self.retriever = services.retriever

    def execute(self, call: ToolCall, state: ConversationState, context: ExecutionContext) -> ToolResult:
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolExecutionError("retrieve requires a non-empty query")
        top_k = call.arguments.get("top_k")
        chunks = self.retriever.query(query, top_k=top_k)
        for chunk in chunks:
            state.retrieved[chunk.chunk_id] = chunk
        payload = [c.model_dump() for c in chunks]
        state.retrieval_transcript.append(
            {
                "turn": state.turn,
                "query": query.strip(),
                "tool_call_id": call.id,
                "chunk_ids": [c.chunk_id for c in chunks],
                "success": bool(chunks),
            }
        )
        return ToolResult(tool_call_id=call.id, name=call.name, payload=payload)
