"""Deterministic structural, grounding, and replayability validation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from jsonschema import ValidationError, validate

from .schemas import EpisodePlan, Message, ValidationReport

_QUERY_WS = re.compile(r"\s+")
_QUERY_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_query(value: str) -> str:
    return _QUERY_WS.sub(" ", _QUERY_PUNCT.sub(" ", value.casefold())).strip()


def validate_trajectory(
    messages: list[Message],
    *,
    plan: EpisodePlan,
    retrieval_transcript: list[dict[str, Any]],
    tool_schemas: Iterable[dict[str, Any]],
    require_final_answer_each_turn: bool = True,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    schema_by_name = {
        str((schema.get("function") or {}).get("name")): schema
        for schema in tool_schemas
    }
    open_calls: dict[str, str] = {}
    satisfied = set()
    last_turn = 0
    users_by_turn = set()
    final_by_turn = set()

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
            for tc in msg.tool_calls or []:
                tcid = str(tc.get("id") or "")
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                if not tcid:
                    errors.append(f"message[{index}] has a tool call with no id")
                    continue
                if tcid in open_calls:
                    errors.append(f"duplicate tool call id `{tcid}`")
                open_calls[tcid] = name
                if name == "context.compress":
                    errors.append("context.compress leaked into emitted messages")
                schema = schema_by_name.get(name)
                if schema is None:
                    errors.append(f"message[{index}] calls unknown tool `{name}`")
                    continue
                try:
                    args = fn.get("arguments", {})
                    args = json.loads(args) if isinstance(args, str) else args
                    validate(
                        args,
                        (schema.get("function") or {}).get("parameters")
                        or {"type": "object"},
                    )
                except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                    errors.append(
                        f"message[{index}] tool `{name}` has invalid arguments: {exc}"
                    )
        elif msg.role == "tool":
            tcid = msg.tool_call_id or ""
            if tcid not in open_calls:
                errors.append(
                    f"message[{index}] tool result `{tcid}` has no matching call"
                )
            else:
                satisfied.add(tcid)
                expected = open_calls[tcid]
                if msg.name and msg.name != expected:
                    errors.append(
                        f"message[{index}] tool name `{msg.name}` does not match `{expected}`"
                    )

    dangling = sorted(set(open_calls) - satisfied)
    if dangling:
        errors.append(f"tool calls without results: {dangling}")

    for turn in plan.turns:
        if turn.turn not in users_by_turn:
            errors.append(f"turn {turn.turn} has no user message")
        if require_final_answer_each_turn and turn.turn not in final_by_turn:
            errors.append(f"turn {turn.turn} has no final assistant answer")
        if turn.retrieval_required:
            successes = [
                row
                for row in retrieval_transcript
                if row.get("turn") == turn.turn
                and row.get("success")
                and row.get("chunk_ids")
            ]
            distinct = {normalize_query(str(row.get("query", ""))) for row in successes}
            distinct.discard("")
            if len(successes) < turn.retrieval_depth:
                errors.append(
                    f"turn {turn.turn} completed {len(successes)} successful retrieval(s); "
                    f"required {turn.retrieval_depth}"
                )
            if len(distinct) < turn.retrieval_depth:
                errors.append(
                    f"turn {turn.turn} used {len(distinct)} distinct retrieval query/queries; "
                    f"required {turn.retrieval_depth}"
                )

    if messages:
        final = messages[-1]
        if (
            final.role != "assistant"
            or final.tool_calls
            or not (final.content or "").strip()
        ):
            errors.append("trajectory does not end on a tool-free assistant answer")
    else:
        errors.append("trajectory is empty")
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


def reconstruct_messages(record: dict[str, Any]) -> list[Message]:
    turns = (record.get("metadata") or {}).get("message_turns") or []
    out = []
    for index, raw in enumerate(record.get("messages") or []):
        value = dict(raw)
        if index < len(turns):
            value["turn"] = turns[index]
        out.append(Message.model_validate(value))
    return out
