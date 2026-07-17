"""Live runtime tool executor.

Resolves the assistant's tool calls at generation time against the *real* services:

- ``retrieve`` -> the live NeMo Retriever (real Constitution passages). Results are
  whatever the retriever returns for the model's actual query, so the
  retrieve -> assess -> rewrite -> retrieve loop is genuine: a vague query gets
  weak chunks, a precise rewrite gets better ones.
- ``memory_read`` / ``memory_write`` -> a validated in-conversation key/value store
  (allow-listed preference keys, scalar values only).

There is no prescripted tool plan and no pre-built chunk catalog: the LLM drives
retrieval and the retriever answers live. Compaction is handled separately by the
generator (automatic at the token threshold) and is never a tool call.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mtsdg.retriever import RetrieverClient
from mtsdg.schemas import ALLOWED_MEMORY_KEYS, RetrievalChunk

_JSON_SCALARS = (str, int, float, bool)


class ToolError(Exception):
    """Raised when a tool call is malformed or violates policy."""


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str = ""

    @classmethod
    def from_openai(cls, tc: Dict[str, Any]) -> "ToolCall":
        fn = tc.get("function", {}) or {}
        raw = fn.get("arguments", {})
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolError(f"tool `{fn.get('name')}` arguments are not valid JSON: {exc}") from exc
        elif isinstance(raw, dict):
            args = raw
        else:
            raise ToolError(f"tool `{fn.get('name')}` arguments must be an object")
        return cls(name=fn.get("name", ""), arguments=args, id=tc.get("id", ""))


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    payload: Any

    def to_message(self) -> Dict[str, Any]:
        return {
            "role": "tool",
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(self.payload, ensure_ascii=False),
        }


@dataclass
class ConversationState:
    conversation_id: str
    turn: int = 1
    memory: Dict[str, Any] = field(default_factory=dict)
    # chunk_id -> RetrievalChunk for everything retrieved so far (the agent's
    # working document memory; survives compaction so answers stay grounded).
    retrieved: Dict[str, RetrievalChunk] = field(default_factory=dict)
    _transcript: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def transcript(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._transcript)

    def _record(self, call: "ToolCall", result: "ToolResult") -> None:
        self._transcript.append(
            copy.deepcopy(
                {"turn": self.turn, "tool": call.name, "arguments": call.arguments,
                 "tool_call_id": call.id, "payload": result.payload}
            )
        )


class LiveToolExecutor:
    """Executes model tool calls against the live retriever + validated memory."""

    def __init__(self, retriever: RetrieverClient, *, allowed_tools: List[str], default_top_k: int = 3):
        self.retriever = retriever
        self.allowed_tools = set(allowed_tools)
        self.default_top_k = default_top_k

    def execute(self, call: ToolCall, state: ConversationState) -> ToolResult:
        if call.name not in self.allowed_tools:
            raise ToolError(f"tool `{call.name}` is not allowed in this episode")
        if call.name == "retrieve":
            result = self._retrieve(call, state)
        elif call.name == "memory_read":
            result = self._memory_read(call, state)
        elif call.name == "memory_write":
            result = self._memory_write(call, state)
        else:
            raise ToolError(f"unknown tool `{call.name}`")
        state._record(call, result)
        return result

    def _retrieve(self, call: ToolCall, state: ConversationState) -> ToolResult:
        query = call.arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("retrieve requires a non-empty `query`")
        top_k = call.arguments.get("top_k")
        if not (isinstance(top_k, int) and 1 <= top_k <= 8):
            top_k = self.default_top_k
        chunks = self.retriever.query(query.strip(), num_chunks=top_k)
        for c in chunks:
            state.retrieved[c.chunk_id] = c
        return ToolResult(call.id, "retrieve", [c.model_dump() for c in chunks])

    def _memory_read(self, call: ToolCall, state: ConversationState) -> ToolResult:
        scope = call.arguments.get("scope")
        if scope not in ("user", "conversation"):
            raise ToolError("memory_read requires scope in {user, conversation}")
        keys = call.arguments.get("keys")
        if keys is None:
            payload = {k: v for k, v in state.memory.items() if k in ALLOWED_MEMORY_KEYS}
        else:
            payload = {k: state.memory[k] for k in keys if k in ALLOWED_MEMORY_KEYS and k in state.memory}
        return ToolResult(call.id, "memory_read", payload)

    def _memory_write(self, call: ToolCall, state: ConversationState) -> ToolResult:
        args = call.arguments
        for req in ("key", "value", "scope", "reason"):
            if req not in args:
                raise ToolError(f"memory_write missing required field `{req}`")
        key = args["key"]
        if key not in ALLOWED_MEMORY_KEYS:
            raise ToolError(f"memory_write key `{key}` is not allowed; allowed: {sorted(ALLOWED_MEMORY_KEYS)}")
        if args["scope"] not in ("user", "conversation"):
            raise ToolError("memory_write scope must be user or conversation")
        if not isinstance(args["value"], _JSON_SCALARS):
            raise ToolError(f"memory_write value for `{key}` must be a scalar")
        state.memory[key] = args["value"]
        return ToolResult(call.id, "memory_write", {"saved": True, "key": key})
