"""Deterministic structural, grounding, budget, and replayability validation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from jsonschema import ValidationError, validate

from .schemas import EpisodeSpec, Message, ValidationReport

_QUERY_WS = re.compile(r"\s+")
_QUERY_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_VISIBLE_CHUNK_ID = re.compile(r"\b(?:h-[0-9a-f]{12,}|chunk-[A-Za-z0-9_-]+)\b", re.IGNORECASE)
_EXPLICIT_CITATION = re.compile(r"\[\[([^\[\]\r\n]+)\]\]")


def normalize_query(value: str) -> str:
    return _QUERY_WS.sub(" ", _QUERY_PUNCT.sub(" ", value.casefold())).strip()


def text_similarity(left: str, right: str) -> float:
    """Bag-of-token cosine similarity that also catches reordered paraphrases."""
    left_counts = Counter(normalize_query(left).split())
    right_counts = Counter(normalize_query(right).split())
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(count * right_counts.get(token, 0) for token, count in left_counts.items())
    norm = math.sqrt(
        sum(count * count for count in left_counts.values())
        * sum(count * count for count in right_counts.values())
    )
    return dot / norm if norm else 0.0


def query_similarity(left: str, right: str) -> float:
    """Lexical similarity only; observed evidence gain is checked separately."""
    left_tokens = set(normalize_query(left).split())
    right_tokens = set(normalize_query(right).split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return max(jaccard, text_similarity(left, right))


def validate_trajectory(
    messages: list[Message],
    *,
    spec: EpisodeSpec,
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
    known_chunk_ids = {
        str(chunk_id)
        for row in retrieval_transcript
        if row.get("success")
        for chunk_id in (row.get("chunk_ids") or [])
    }

    for index, msg in enumerate(messages):
        if msg.turn is not None:
            if msg.turn < last_turn:
                errors.append(f"message[{index}] has non-monotonic turn {msg.turn}")
            last_turn = max(last_turn, msg.turn)
        if msg.role == "user" and msg.turn:
            users_by_turn.add(msg.turn)
        if msg.role == "assistant":
            visible_ids = set(_VISIBLE_CHUNK_ID.findall(msg.content or ""))
            visible_ids.update(value.strip() for value in _EXPLICIT_CITATION.findall(msg.content or ""))
            unknown_visible_ids = sorted(visible_ids - known_chunk_ids)
            if unknown_visible_ids:
                errors.append(
                    f"message[{index}] cites unknown retrieved chunk IDs: {unknown_visible_ids}"
                )
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

    successful_retrievals = [
        row for row in retrieval_transcript if row.get("success") and row.get("chunk_ids")
    ]
    for index, current in enumerate(successful_retrievals):
        for previous in successful_retrievals[:index]:
            similarity = query_similarity(str(previous.get("query", "")), str(current.get("query", "")))
            if similarity >= spec.query_lexical_similarity_threshold:
                errors.append(
                    "redundant retrieval queries exceed lexical similarity threshold: "
                    f"{similarity:.3f} >= {spec.query_lexical_similarity_threshold:.3f}"
                )
                break
    low_gain_chain = 0
    previous_low_gain_query = ""
    for row in successful_retrievals:
        if not row.get("low_gain"):
            low_gain_chain = 0
            previous_low_gain_query = ""
            continue
        current_query = str(row.get("query", ""))
        low_gain_chain += 1
        related = bool(previous_low_gain_query) and (
            query_similarity(current_query, previous_low_gain_query)
            >= spec.low_gain_followup_similarity_threshold
        )
        previous_low_gain_query = current_query
        if low_gain_chain > spec.max_low_gain_chain + 1:
            errors.append(
                f"consecutive observed low-gain retrieval chain reached {low_gain_chain}; hard maximum is "
                f"{spec.max_low_gain_chain + 1}"
            )
            break
        if low_gain_chain > spec.max_low_gain_chain and related:
            errors.append(
                "a related retrieval executed after the configured low-gain chain allowance"
            )
            break

    if len(tool_call_attempts) > spec.max_tool_calls_per_conversation:
        errors.append(
            f"conversation attempted {len(tool_call_attempts)} tool call(s); maximum is "
            f"{spec.max_tool_calls_per_conversation}"
        )
    attempts_by_turn = Counter(int(attempt.get("turn", 0)) for attempt in tool_call_attempts)
    for turn, count in sorted(attempts_by_turn.items()):
        if count > spec.max_tool_calls_per_turn:
            errors.append(f"turn {turn} attempted {count} tool call(s); maximum is {spec.max_tool_calls_per_turn}")
    successful_retrieval_count = len(successful_retrievals)
    if successful_retrieval_count > spec.max_retrieval_calls:
        errors.append(
            f"conversation completed {successful_retrieval_count} retrieval call(s); maximum is "
            f"{spec.max_retrieval_calls}"
        )
    retrievals_by_turn = Counter(
        int(row.get("turn", 0))
        for row in successful_retrievals
    )
    for turn, count in sorted(retrievals_by_turn.items()):
        if count > spec.max_retrieval_calls_per_turn:
            errors.append(
                f"turn {turn} completed {count} retrieval call(s); maximum is "
                f"{spec.max_retrieval_calls_per_turn}"
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
