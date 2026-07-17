"""Deterministic trajectory assembler and ``structured_messages`` projection.

Concatenates the turn messages (user, assistant, and the tool-call / tool-result
messages produced live by the tool executor) into the final conversation, then
projects it to the ``structured_messages`` output shape used for tool-calling SFT.

The emitted conversation is the clean user/assistant/retrieve/memory sequence:
context compaction is automatic and internal (see :mod:`mtsdg.generator`), so no
``context.compress`` call or rolling summary ever appears in the chat. The summary
is used only to condition later turns during generation and kept as hidden
provenance.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from mtsdg.schemas import CompressionEvent, Message


def _assign_ids(messages: List[Message], start_index: int = 0) -> int:
    idx = start_index
    for m in messages:
        if not m.message_id:
            m.message_id = f"m-{idx:02d}"
        idx += 1
    return idx


def assemble_blocks(
    blocks: List[List[Message]],
    *,
    system_message: Optional[Message] = None,
) -> List[Message]:
    """Concatenate accepted blocks into one ordered message list with stable IDs."""
    assembled: List[Message] = []
    if system_message is not None:
        assembled.append(system_message)
    for block in blocks:
        assembled.extend(block)
    _assign_ids(assembled)
    return assembled


def project_structured_messages(
    assembled: List[Message], *, keep_reasoning: bool = True
) -> List[Dict[str, Any]]:
    """Project assembled messages into the OpenAI-style ``structured_messages`` list.

    Drops private bookkeeping (``turn``, ``message_id``) via ``Message.to_openai``.
    """
    out: List[Dict[str, Any]] = []
    for m in copy.deepcopy(assembled):
        rendered = m.to_openai()
        if not keep_reasoning:
            rendered.pop("reasoning_content", None)
        out.append(rendered)
    return out


def render_summary(ev: CompressionEvent) -> str:
    """A compact, source-linked textual rendering of a compaction summary.

    Used to condition later turns during generation (never emitted into the chat).
    """
    lines = [
        f"[Compressed conversation summary {ev.summary_id} covering turns {ev.covers_turns}]"
    ]
    if ev.user_stated_facts:
        lines.append("User-stated facts: " + "; ".join(f.fact for f in ev.user_stated_facts))
    # Established facts carry the substance so later turns can answer settled points
    # from here WITHOUT re-retrieving.
    if ev.key_facts:
        lines.append("Established facts (reuse these; do not re-retrieve them):")
        for kf in ev.key_facts:
            cites = f" [{', '.join(kf.supporting_chunk_ids)}]" if kf.supporting_chunk_ids else ""
            lines.append(f"  - {kf.fact}{cites}")
    if ev.constraints:
        lines.append("Constraints: " + "; ".join(ev.constraints))
    if ev.authorities:
        lines.append(
            "Sources already retrieved: " + "; ".join(f"{a.chunk_id} ({a.title})" for a in ev.authorities)
        )
    if ev.decisions:
        lines.append("Decisions: " + "; ".join(ev.decisions))
    if ev.open_questions:
        lines.append("Open questions: " + "; ".join(ev.open_questions))
    return "\n".join(lines)
