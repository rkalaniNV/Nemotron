"""Explicitly marked LLM-simulated tool executor."""

from __future__ import annotations

import json
from typing import Any

from long_context_sdg.llm import call_llm
from long_context_sdg.schemas import ToolCall, ToolResult

from .base import ConversationState, ExecutionContext, ExecutionServices


class SimulatedExecutor:
    def __init__(self, *, services: ExecutionServices, **_: Any):
        self.services = services

    def execute(
        self, call: ToolCall, state: ConversationState, context: ExecutionContext
    ) -> ToolResult:
        prompt = (
            "Simulate the external tool response. Return only a JSON value.\n"
            f"Instructions: {context.instructions}\n"
            f"Tool schema: {json.dumps(context.tool_schema, ensure_ascii=False)}\n"
            f"Arguments: {json.dumps(call.arguments, ensure_ascii=False)}"
        )
        response = call_llm(
            self.services.models,
            self.services.simulator_alias,
            [
                {
                    "role": "system",
                    "content": "You simulate a tool backend; do not act as the assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.get("content", "")
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = {"result": text}
        return ToolResult(
            tool_call_id=call.id, name=call.name, payload=payload, simulated=True
        )
