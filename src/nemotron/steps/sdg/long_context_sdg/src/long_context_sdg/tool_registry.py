"""Trusted import-path executor registry with JSON-schema validation."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from typing import Any

from jsonschema import ValidationError, validate

from .config import ToolConfig
from .executors.base import (
    ConversationState,
    ExecutionContext,
    ExecutionServices,
    ToolExecutionError,
)
from .schemas import ToolCall, ToolResult


def normalize_tool_call(raw: dict[str, Any], fallback_id: str) -> ToolCall:
    if not isinstance(raw, dict):
        raise ToolExecutionError("tool call must be an object")
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    name = fn.get("name") or raw.get("name") or raw.get("tool")
    arguments = fn.get("arguments", raw.get("arguments", {}))
    if not arguments and "arguments" not in fn and "arguments" not in raw:
        parameters = fn.get("parameters")
        if isinstance(parameters, dict) and not {
            "type",
            "properties",
            "required",
        }.intersection(parameters):
            arguments = parameters
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ToolExecutionError("tool arguments must be an object")
    if not isinstance(name, str) or not name:
        raise ToolExecutionError("tool call has no name")
    return ToolCall(id=str(raw.get("id") or fallback_id), name=name, arguments=arguments)


class ToolRegistry:
    def __init__(self, definitions: Iterable[ToolConfig], services: ExecutionServices):
        self._configs: dict[str, ToolConfig] = {}
        self._executors: dict[str, Any] = {}
        for definition in definitions:
            if definition.name in self._configs:
                raise ValueError(f"duplicate tool `{definition.name}`")
            cls = _import_class(definition.executor)
            executor = cls(services=services, **definition.executor_kwargs)
            if not callable(getattr(executor, "execute", None)):
                raise TypeError(f"executor `{definition.executor}` has no callable execute method")
            self._configs[definition.name] = definition
            self._executors[definition.name] = executor

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [d.tool_schema for d in self._configs.values()]

    def execute(
        self,
        call: ToolCall,
        state: ConversationState,
        instructions: str,
    ) -> ToolResult:
        definition = self._configs.get(call.name)
        if definition is None:
            raise ToolExecutionError(f"unknown tool `{call.name}`")
        parameters = (definition.tool_schema.get("function") or {}).get("parameters") or {"type": "object"}
        try:
            validate(instance=call.arguments, schema=parameters)
        except ValidationError as exc:
            raise ToolExecutionError(f"tool `{call.name}` arguments fail schema: {exc.message}") from exc
        context = ExecutionContext(
            tool_name=call.name,
            tool_schema=definition.tool_schema,
            instructions=instructions,
        )
        return self._executors[call.name].execute(call, state, context)


def _import_class(path: str):
    if ":" not in path:
        raise ValueError(f"executor import `{path}` must use module:Class syntax")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"executor class `{class_name}` not found in `{module_name}`") from exc
