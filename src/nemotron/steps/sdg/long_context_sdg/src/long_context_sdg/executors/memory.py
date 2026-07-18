"""Validated in-conversation memory executor."""

from __future__ import annotations

from typing import Any

from long_context_sdg.schemas import ALLOWED_MEMORY_KEYS, ToolCall, ToolResult

from .base import (
    ConversationState,
    ExecutionContext,
    ExecutionServices,
    ToolExecutionError,
)

_SCALARS = (str, int, float, bool)


class MemoryExecutor:
    def __init__(self, *, services: ExecutionServices, **_: Any):
        del services

    def execute(
        self, call: ToolCall, state: ConversationState, context: ExecutionContext
    ) -> ToolResult:
        if call.name == "memory_read":
            return self._read(call, state)
        if call.name == "memory_write":
            return self._write(call, state)
        raise ToolExecutionError(f"MemoryExecutor cannot execute `{call.name}`")

    def _read(self, call: ToolCall, state: ConversationState) -> ToolResult:
        scope = call.arguments.get("scope")
        if scope not in ("user", "conversation"):
            raise ToolExecutionError("memory_read scope must be user or conversation")
        keys = call.arguments.get("keys")
        if keys is not None and not isinstance(keys, list):
            raise ToolExecutionError("memory_read keys must be an array")
        selected = (
            ALLOWED_MEMORY_KEYS if keys is None else set(keys) & ALLOWED_MEMORY_KEYS
        )
        payload = {k: state.memory[k] for k in selected if k in state.memory}
        state.memory_events.append(
            {
                "turn": state.turn,
                "action": "read",
                "scope": scope,
                "keys": sorted(selected),
            }
        )
        return ToolResult(tool_call_id=call.id, name=call.name, payload=payload)

    def _write(self, call: ToolCall, state: ConversationState) -> ToolResult:
        args = call.arguments
        for required in ("key", "value", "scope", "reason"):
            if required not in args:
                raise ToolExecutionError(f"memory_write missing `{required}`")
        key = args["key"]
        if key not in ALLOWED_MEMORY_KEYS:
            raise ToolExecutionError(f"memory key `{key}` is not allowed")
        if args["scope"] not in ("user", "conversation"):
            raise ToolExecutionError("memory_write scope must be user or conversation")
        if not isinstance(args["value"], _SCALARS):
            raise ToolExecutionError("memory_write value must be a JSON scalar")
        state.memory[key] = args["value"]
        state.memory_events.append(
            {
                "turn": state.turn,
                "action": "write",
                "scope": args["scope"],
                "key": key,
                "reason": str(args["reason"]),
            }
        )
        return ToolResult(
            tool_call_id=call.id, name=call.name, payload={"saved": True, "key": key}
        )
