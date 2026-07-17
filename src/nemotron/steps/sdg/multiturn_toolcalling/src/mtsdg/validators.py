"""Deterministic validation gates.

These run before any LLM judge and cannot be repaired by one. Semantic checks
(coherence, helpfulness) live in the trajectory judge.

- :func:`validate_compression_event` — compression provenance checks.
- :func:`validate_trajectory` — structural checks over assembled structured_messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from mtsdg.reasoning import validate_reasoning_content
from mtsdg.schemas import ALLOWED_MEMORY_KEYS, CompressionEvent, ReasoningContent
from mtsdg.tokens import count_tokens
from mtsdg.tools.contracts import TOOL_SCHEMAS


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    @classmethod
    def combine(cls, reports: List["ValidationReport"]) -> "ValidationReport":
        errs: List[str] = []
        warns: List[str] = []
        for r in reports:
            errs += r.errors
            warns += r.warnings
        return cls(ok=not errs, errors=errs, warnings=warns)


# --------------------------------------------------------------------------- #
# compression provenance checks
# --------------------------------------------------------------------------- #


def validate_compression_event(
    event: CompressionEvent,
    *,
    from_turn: int,
    checkpoint_turn: int,
    prefix_message_ids: Set[str],
    prefix_chunk_ids: Set[str],
    prior_summary_chunk_ids: Optional[Set[str]] = None,
) -> ValidationReport:
    """Validate one ``context.compress`` result against its completed prefix."""
    errors: List[str] = []
    known_chunks = set(prefix_chunk_ids) | set(prior_summary_chunk_ids or set())

    if event.covers_turns and event.covers_turns[-1] != checkpoint_turn:
        errors.append(
            f"covers_turns ends at {event.covers_turns[-1]}, expected checkpoint {checkpoint_turn}."
        )
    if event.covers_turns and event.covers_turns[0] != from_turn:
        errors.append(
            f"covers_turns starts at {event.covers_turns[0]}, expected {from_turn}."
        )

    def _check_ids(ids: List[str], where: str) -> None:
        for mid in ids:
            if mid not in prefix_message_ids:
                errors.append(f"{where} references message `{mid}` not in the completed prefix.")

    _check_ids(event.source_message_ids, "source_message_ids")
    for i, f in enumerate(event.user_stated_facts):
        if not f.source_message_ids:
            errors.append(f"user_stated_facts[{i}] has no source_message_ids.")
        _check_ids(f.source_message_ids, f"user_stated_facts[{i}]")

    for a in event.authorities:
        if a.chunk_id not in known_chunks:
            errors.append(f"authority chunk `{a.chunk_id}` did not appear in prefix/prior summary.")

    if not event.no_new_claims:
        errors.append("no_new_claims is False — compression added new content.")

    for k in event.memory_preferences:
        if k not in ALLOWED_MEMORY_KEYS:
            errors.append(f"compression memory_preferences contains disallowed key `{k}`.")

    return ValidationReport(ok=not errors, errors=errors)


# --------------------------------------------------------------------------- #
# structural trajectory checks
# --------------------------------------------------------------------------- #


def validate_trajectory(
    messages: List[Dict[str, Any]],
    *,
    max_reasoning_tokens: int = 400,
) -> ValidationReport:
    """Deterministic structural checks over an assembled ``structured_messages`` list."""
    errors: List[str] = []
    warnings: List[str] = []

    open_tool_call_ids: Set[str] = set()
    satisfied_tool_call_ids: Set[str] = set()
    returned_chunks_so_far: Set[str] = set()
    last_turn = 0
    n_retrieve = 0
    n_compress = 0

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        turn = msg.get("turn")
        if isinstance(turn, int):
            if turn < last_turn:
                errors.append(f"message[{idx}] turn {turn} < previous {last_turn} (non-monotonic).")
            last_turn = max(last_turn, turn)

        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                tcid = tc.get("id", "")
                fn = tc.get("function", {}) or {}
                name = fn.get("name")
                if name not in TOOL_SCHEMAS:
                    errors.append(f"message[{idx}] calls unknown tool `{name}`.")
                    continue
                if name == "retrieve":
                    n_retrieve += 1
                if name == "context.compress":
                    # Compaction is automatic; it must never appear in the chat.
                    n_compress += 1
                    errors.append(f"message[{idx}] emits a context.compress tool call "
                                  "(compaction must be automatic, not a chat tool call).")
                open_tool_call_ids.add(tcid)
                _check_tool_args(name, fn.get("arguments"), idx, errors)
            rc_raw = msg.get("reasoning_content")
            if rc_raw:
                _check_reasoning(rc_raw, returned_chunks_so_far, idx, errors, warnings, max_reasoning_tokens)

        elif role == "tool":
            tcid = msg.get("tool_call_id", "")
            if tcid not in open_tool_call_ids:
                errors.append(
                    f"message[{idx}] tool result has tool_call_id `{tcid}` with no matching call."
                )
            else:
                satisfied_tool_call_ids.add(tcid)
            if msg.get("name") == "retrieve":
                for cid in _extract_chunk_ids(msg.get("content")):
                    returned_chunks_so_far.add(cid)

    dangling = open_tool_call_ids - satisfied_tool_call_ids
    if dangling:
        errors.append(f"tool calls without a tool result: {sorted(dangling)}.")

    # Coverage warnings (design conversation requirements).
    if n_retrieve < 2:
        warnings.append(f"only {n_retrieve} retrieve call(s); expected >= 2 (with a rewrite).")

    return ValidationReport(ok=not errors, errors=errors, warnings=warnings)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _check_tool_args(name: str, raw_args: Any, idx: int, errors: List[str]) -> None:
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            errors.append(f"message[{idx}] tool `{name}` arguments are not valid JSON.")
            return
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        errors.append(f"message[{idx}] tool `{name}` arguments must be an object.")
        return

    required = TOOL_SCHEMAS[name]["parameters"].get("required", [])
    for r in required:
        if r not in args:
            errors.append(f"message[{idx}] tool `{name}` missing required arg `{r}`.")

    if name == "memory_write" and "key" in args and args["key"] not in ALLOWED_MEMORY_KEYS:
        errors.append(f"message[{idx}] memory_write uses disallowed key `{args['key']}`.")
    if name == "retrieve":
        q = args.get("query", "")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"message[{idx}] retrieve has empty query.")


def _check_reasoning(
    rc_raw: Any, returned: Set[str], idx: int, errors: List[str], warnings: List[str],
    max_reasoning_tokens: int = 400,
) -> None:
    if isinstance(rc_raw, str):
        n = count_tokens(rc_raw)
        if n > max_reasoning_tokens:
            errors.append(f"message[{idx}] reasoning_content is {n} tokens (> {max_reasoning_tokens}).")
        return
    try:
        rc = rc_raw if isinstance(rc_raw, ReasoningContent) else ReasoningContent.model_validate(rc_raw)
    except Exception as exc:
        errors.append(f"message[{idx}] reasoning_content fails schema: {exc}")
        return
    report = validate_reasoning_content(rc, returned, max_tokens=max_reasoning_tokens)
    for e in report.errors:
        errors.append(f"message[{idx}] reasoning_content: {e}")


def _extract_chunk_ids(content: Any) -> List[str]:
    if not content:
        return []
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError):
        return []
    ids: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "chunk_id" in item:
                ids.append(item["chunk_id"])
    return ids
