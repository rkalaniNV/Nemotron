"""Generate and validate context-compaction events.

Built per compaction trigger, *after* the prefix is materialized. Each record
contains only the raw messages through the completed turn plus the prior summary —
never later turns. Teacher-invented provenance (message IDs / chunk IDs not in the
prefix) is dropped rather than failing the whole event, keeping only genuinely
source-linked content.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from mtsdg.core.llm import call_structured
from mtsdg.prompts import CONTEXT_COMPRESSION_PROMPT
from mtsdg.schemas import CompressionEvent
from mtsdg.validators import ValidationReport, validate_compression_event

DEFAULT_COMPRESSION_TOKEN_BUDGET = 400


def _prefix_ids(messages: List[Dict[str, Any]]) -> Set[str]:
    return {m["message_id"] for m in messages if m.get("message_id")}


def _prefix_chunks(messages: List[Dict[str, Any]]) -> Set[str]:
    """Chunk IDs that appeared in the prefix (retrieve results)."""
    chunk_ids: Set[str] = set()
    for m in messages:
        if m.get("role") != "tool" or m.get("name") != "retrieve":
            continue
        content = m.get("content")
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("chunk_id"):
                    chunk_ids.add(item["chunk_id"])
    return chunk_ids


def generate_compression_event(
    models: Dict[str, Any],
    alias: str,
    *,
    completed_messages: List[Dict[str, Any]],
    from_turn: int,
    checkpoint_turn: int,
    summary_id: str,
    prior_summary: Optional[CompressionEvent] = None,
    token_budget: int = DEFAULT_COMPRESSION_TOKEN_BUDGET,
) -> Tuple[CompressionEvent, ValidationReport]:
    """Generate one compression event and run deterministic provenance checks."""
    prompt = CONTEXT_COMPRESSION_PROMPT.format(
        compression_token_budget=token_budget,
        prior_summary=(prior_summary.model_dump_json() if prior_summary else "null"),
        completed_messages=json.dumps(
            [_public_message(m) for m in completed_messages], ensure_ascii=False
        ),
        compression_schema=json.dumps(CompressionEvent.model_json_schema(), ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": "You are a precise conversation summarizer. Return only JSON."},
        {"role": "user", "content": prompt},
    ]
    event = call_structured(models, alias, messages, CompressionEvent, response_format=None)

    # Force covers_turns / summary_id to the contract so downstream can key on it.
    event.covers_turns = [from_turn, checkpoint_turn]
    event.summary_id = summary_id

    prefix_ids = _prefix_ids(completed_messages)
    prefix_chunks = _prefix_chunks(completed_messages)
    prior_chunks = {a.chunk_id for a in prior_summary.authorities} if prior_summary else set()
    known_chunks = prefix_chunks | prior_chunks

    # Normalize: drop invented provenance, keep only source-linked content.
    event.source_message_ids = [m for m in event.source_message_ids if m in prefix_ids]
    kept_facts = []
    for f in event.user_stated_facts:
        f.source_message_ids = [m for m in f.source_message_ids if m in prefix_ids]
        if f.source_message_ids:
            kept_facts.append(f)
    event.user_stated_facts = kept_facts
    event.authorities = [a for a in event.authorities if a.chunk_id in known_chunks]
    # key_facts carry substance; drop only citations to chunks never retrieved (the
    # fact text is kept so later turns can reuse it).
    for kf in event.key_facts:
        kf.supporting_chunk_ids = [c for c in kf.supporting_chunk_ids if c in known_chunks]

    report = validate_compression_event(
        event,
        from_turn=from_turn,
        checkpoint_turn=checkpoint_turn,
        prefix_message_ids=prefix_ids,
        prefix_chunk_ids=prefix_chunks,
        prior_summary_chunk_ids=prior_chunks,
    )
    return event, report


def _public_message(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in m.items() if k in ("role", "content", "name", "tool_call_id", "message_id")}
