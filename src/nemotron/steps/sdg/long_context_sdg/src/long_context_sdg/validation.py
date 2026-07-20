"""Deterministic structural, grounding, budget, and replayability validation."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from jsonschema import ValidationError, validate

from .schemas import EpisodeSpec, Message, RetrievalPolicyEvent, ValidationReport

_QUERY_WS = re.compile(r"\s+")
_QUERY_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_query(value: str) -> str:
    return _QUERY_WS.sub(" ", _QUERY_PUNCT.sub(" ", value.casefold())).strip()


def validate_trajectory(
    messages: list[Message],
    *,
    spec: EpisodeSpec,
    policy_events: list[RetrievalPolicyEvent],
    retrieval_transcript: list[dict[str, Any]],
    tool_call_attempts: list[dict[str, Any]],
    tool_schemas: Iterable[dict[str, Any]],
    require_final_answer_each_turn: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    schema_by_name = {str((schema.get("function") or {}).get("name")): schema for schema in tool_schemas}
    open_calls: dict[str, str] = {}
    satisfied: set[str] = set()
    last_turn = 0
    users_by_turn: set[int] = set()
    final_by_turn: set[int] = set()

    for index, msg in enumerate(messages):
        if msg.turn is not None:
            if msg.turn < last_turn:
                errors.append(f"message[{index}] has non-monotonic turn {msg.turn}")
            last_turn = max(last_turn, msg.turn)
        if msg.role == "user" and msg.turn:
            users_by_turn.add(msg.turn)
        if msg.role == "assistant":
            if not msg.tool_calls and (msg.content or "").strip() and msg.turn:
                final_by_turn.add(msg.turn)
            for tool_call in msg.tool_calls or []:
                tool_call_id = str(tool_call.get("id") or "")
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                if not tool_call_id:
                    errors.append(f"message[{index}] has a tool call with no id")
                    continue
                if tool_call_id in open_calls:
                    errors.append(f"duplicate tool call id `{tool_call_id}`")
                open_calls[tool_call_id] = name
                if name == "context.compress":
                    errors.append("context.compress leaked into emitted messages")
                schema = schema_by_name.get(name)
                if schema is None:
                    errors.append(f"message[{index}] calls unknown tool `{name}`")
                    continue
                try:
                    arguments = function.get("arguments", {})
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    validate(
                        arguments,
                        (schema.get("function") or {}).get("parameters") or {"type": "object"},
                    )
                except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                    errors.append(f"message[{index}] tool `{name}` has invalid arguments: {exc}")
        elif msg.role == "tool":
            tool_call_id = msg.tool_call_id or ""
            if tool_call_id not in open_calls:
                errors.append(f"message[{index}] tool result `{tool_call_id}` has no matching call")
            else:
                satisfied.add(tool_call_id)
                expected = open_calls[tool_call_id]
                if msg.name and msg.name != expected:
                    errors.append(f"message[{index}] tool name `{msg.name}` does not match `{expected}`")

    dangling = sorted(set(open_calls) - satisfied)
    if dangling:
        errors.append(f"tool calls without results: {dangling}")

    expected_turns = list(range(1, spec.turn_budget + 1))
    user_turns = sorted(users_by_turn)
    if user_turns != expected_turns:
        errors.append(f"user-message turns are {user_turns}; expected {expected_turns}")
    if require_final_answer_each_turn:
        final_turns = sorted(final_by_turn)
        if final_turns != expected_turns:
            errors.append(f"final-answer turns are {final_turns}; expected {expected_turns}")

    event_turns = [event.turn for event in policy_events]
    if event_turns != sorted(set(event_turns)):
        errors.append(f"retrieval policy event turns must be unique and ordered: {event_turns}")
    for event in policy_events:
        if event.turn not in expected_turns:
            errors.append(f"retrieval policy event references out-of-range turn {event.turn}")
            continue
        successes = [
            row
            for row in retrieval_transcript
            if row.get("turn") == event.turn and row.get("success") and row.get("chunk_ids")
        ]
        distinct = {normalize_query(str(row.get("query", ""))) for row in successes}
        distinct.discard("")
        required = event.required_retrievals_this_turn
        if len(successes) < required:
            errors.append(
                f"retrieval deadline at turn {event.turn} completed {len(successes)} successful retrieval(s); "
                f"required {required}"
            )
        if len(distinct) < required:
            errors.append(
                f"retrieval deadline at turn {event.turn} used {len(distinct)} distinct query/queries; "
                f"required {required}"
            )

    distinct_global = {
        normalize_query(str(row.get("query", "")))
        for row in retrieval_transcript
        if row.get("success") and row.get("chunk_ids")
    }
    distinct_global.discard("")
    if len(distinct_global) < spec.required_retrieval_calls:
        errors.append(
            f"conversation completed {len(distinct_global)} distinct successful retrieval(s); "
            f"required {spec.required_retrieval_calls}"
        )

    if len(tool_call_attempts) > spec.max_tool_calls_per_conversation:
        errors.append(
            f"conversation attempted {len(tool_call_attempts)} tool call(s); maximum is "
            f"{spec.max_tool_calls_per_conversation}"
        )
    attempts_by_turn = Counter(int(attempt.get("turn", 0)) for attempt in tool_call_attempts)
    for turn, count in sorted(attempts_by_turn.items()):
        if count > spec.max_tool_calls_per_turn:
            errors.append(f"turn {turn} attempted {count} tool call(s); maximum is {spec.max_tool_calls_per_turn}")
    retrieval_attempts = sum(attempt.get("name") == "retrieve" for attempt in tool_call_attempts)
    if retrieval_attempts > spec.max_retrieval_calls:
        errors.append(
            f"conversation attempted {retrieval_attempts} retrieval call(s); maximum is {spec.max_retrieval_calls}"
        )
    emitted_call_ids = set(open_calls)
    successful_attempt_ids = {
        str(attempt.get("tool_call_id") or "") for attempt in tool_call_attempts if attempt.get("success")
    }
    if emitted_call_ids != successful_attempt_ids:
        errors.append(
            "emitted tool-call IDs do not match successful execution attempts: "
            f"emitted={sorted(emitted_call_ids)}, successful={sorted(successful_attempt_ids)}"
        )

    if messages:
        final = messages[-1]
        if final.role != "assistant" or final.tool_calls or not (final.content or "").strip():
            errors.append("trajectory does not end on a tool-free assistant answer")
    else:
        errors.append("trajectory is empty")
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


def reconstruct_messages(record: dict[str, Any]) -> list[Message]:
    turns = (record.get("metadata") or {}).get("message_turns") or []
    output = []
    for index, raw in enumerate(record.get("messages") or []):
        value = dict(raw)
        if index < len(turns):
            value["turn"] = turns[index]
        output.append(Message.model_validate(value))
    return output
