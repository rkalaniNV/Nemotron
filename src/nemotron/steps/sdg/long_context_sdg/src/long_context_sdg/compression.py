"""Source-linked context compaction used only in the model-facing view."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .llm import call_structured
from .schemas import CompressionEvent, Message, ValidationReport


def render_summary(event: CompressionEvent) -> str:
    lines = [
        f"[Compacted turns {event.covers_turns[0]}-{event.covers_turns[1]}: {event.summary_id}]"
    ]
    if event.user_facts:
        lines.append("User facts: " + "; ".join(x.fact for x in event.user_facts))
    if event.key_facts:
        lines.append("Established facts:")
        for fact in event.key_facts:
            citations = (
                f" [{', '.join(fact.supporting_chunk_ids)}]"
                if fact.supporting_chunk_ids
                else ""
            )
            lines.append(f"- {fact.fact}{citations}")
    if event.constraints:
        lines.append("Constraints: " + "; ".join(event.constraints))
    if event.open_questions:
        lines.append("Open questions: " + "; ".join(event.open_questions))
    return "\n".join(lines)


def validate_compression(
    event: CompressionEvent,
    *,
    from_turn: int,
    to_turn: int,
    message_ids: Iterable[str],
    chunk_ids: Iterable[str],
) -> ValidationReport:
    errors = []
    known_messages = set(message_ids)
    known_chunks = set(chunk_ids)
    if event.covers_turns != [from_turn, to_turn]:
        errors.append(f"covers_turns must be [{from_turn}, {to_turn}]")
    if not event.no_new_claims:
        errors.append("compression no_new_claims must be true")
    unknown_messages = sorted(set(event.source_message_ids) - known_messages)
    if unknown_messages:
        errors.append(f"compression references unknown messages: {unknown_messages}")
    cited = {cid for fact in event.key_facts for cid in fact.supporting_chunk_ids}
    unknown_chunks = sorted(cited - known_chunks)
    if unknown_chunks:
        errors.append(f"compression references unknown chunks: {unknown_chunks}")
    for fact in event.user_facts:
        if not fact.source_message_ids:
            errors.append("compressed user fact has no source message")
    return ValidationReport(ok=not errors, errors=errors)


def generate_compression(
    models: dict[str, object],
    messages: list[Message],
    *,
    from_turn: int,
    to_turn: int,
    summary_id: str,
    known_chunk_ids: Iterable[str],
    prior: CompressionEvent | None,
    instructions: str,
    token_budget: int,
) -> CompressionEvent:
    public = [m.to_openai() | {"message_id": m.message_id} for m in messages]
    prompt = (
        "Compress only the completed conversation prefix. Add no facts. Preserve source IDs and chunk IDs.\n"
        f"Effective instructions: {instructions}\n"
        f"Token budget: {token_budget}\n"
        f"Required summary_id: {summary_id}\n"
        f"Required covers_turns: [{from_turn}, {to_turn}]\n"
        f"Prior summary: {prior.model_dump_json() if prior else 'null'}\n"
        f"Messages: {json.dumps(public, ensure_ascii=False)}\n"
        f"Schema: {json.dumps(CompressionEvent.model_json_schema(), ensure_ascii=False)}"
    )
    event = call_structured(
        models,
        "compressor",
        [
            {
                "role": "system",
                "content": "You create source-linked rolling context summaries. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        CompressionEvent,
    )
    event.summary_id = summary_id
    event.covers_turns = [from_turn, to_turn]
    report = validate_compression(
        event,
        from_turn=from_turn,
        to_turn=to_turn,
        message_ids=[m.message_id for m in messages if m.message_id],
        chunk_ids=known_chunk_ids,
    )
    if not report.ok:
        raise ValueError("; ".join(report.errors))
    return event
