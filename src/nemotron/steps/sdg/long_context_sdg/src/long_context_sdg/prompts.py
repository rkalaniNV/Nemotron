"""Prompt builders shared by generation, compaction, and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    EpisodeSeed,
    UserTurn,
)


def assistant_system(
    seed: EpisodeSeed,
    tools: list[dict],
    known_chunk_ids: Iterable[str] = (),
    retrieval_history: Iterable[dict] = (),
) -> str:
    known = sorted(set(known_chunk_ids))
    history = [
        {
            "turn": item.get("turn"),
            "query": item.get("query"),
            "low_gain": bool(item.get("low_gain")),
            "chunk_ids": item.get("chunk_ids", []),
        }
        for item in retrieval_history
    ]
    return (
        "You are the assistant in a synthetic but realistic long-running conversation. "
        "Use tools when evidence or saved preferences are needed. Never invent tool results or citations. "
        "Return one JSON object matching the AssistantAction schema. A tool action may contain tool_calls; "
        "a final answer must contain content and no tool_calls. Keep reasoning bounded and cite only chunk IDs "
        "already returned by retrieve. When citing visible evidence, write the full ID as [[exact-chunk-id]]. "
        "Decide naturally whether a tool would materially improve the response, "
        "given the current user message and evidence already available. If retrieving again, pursue a genuinely "
        "new unresolved question rather than paraphrasing an earlier search.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"EXACT RETRIEVED CHUNK IDS ALLOWED IN CITATIONS:\n{json.dumps(known)}\n\n"
        f"RETRIEVAL HISTORY (reuse evidence; do not repeat these searches):\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        f"TOOLS:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        f"ASSISTANT ACTION SCHEMA:\n{json.dumps(AssistantAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_final_system(seed: EpisodeSeed, known_chunk_ids: Iterable[str] = ()) -> str:
    """Constrain the post-tool pass to an answer that cannot request more tools."""
    known = sorted(set(known_chunk_ids))
    return (
        "You are the assistant in a synthetic but realistic long-running conversation. "
        "No further tool call is permitted in this turn. Synthesize the final answer from available evidence. "
        "Do not request, mention, or emit any tool call. Never invent citations or claims. "
        "Support every factual domain claim with an inline citation formatted as [[full-exact-retrieved-ID]]. "
        "Never shorten, renumber, or invent an ID. If the available evidence does not support a claim, "
        "explicitly say so rather than pretending another search occurred. "
        "Return one JSON object matching the AssistantFinalAction schema and cite only chunk IDs already "
        "returned by retrieve.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"EXACT RETRIEVED CHUNK IDS ALLOWED IN CITATIONS:\n{json.dumps(known)}\n\n"
        f"ASSISTANT FINAL ACTION SCHEMA:\n"
        f"{json.dumps(AssistantFinalAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_turn_directive(turn: int) -> str:
    return (
        f"Turn {turn}. Respond naturally to the user's actual message. Decide whether a configured tool would "
        "materially improve this response based on the unresolved need and the evidence already available."
    )


def user_system(seed: EpisodeSeed) -> str:
    persona = seed.persona
    return (
        "Simulate only the user. Continue naturally from the conversation, stay in persona, and do not emit "
        "tool calls or describe being a simulator.\n"
        f"Role: {persona.role}; expertise: {persona.expertise}; style: {persona.style}.\n"
        f"Persona description: {persona.description or 'not specified'}.\n"
        f"Target language: {persona.language or 'follow the effective instructions'}.\n"
        f"Topic: {seed.query}\nEffective instructions: {seed.instructions}"
    )


def user_turn_prompt(*, turn: int, turns_remaining: int) -> str:
    return (
        f"Write the user's next natural message for turn {turn}; {turns_remaining} turn(s), including this one, "
        "remain in the requested episode length. Continue from what was actually said and stay in persona. "
        "Ask a plausible follow-up, clarification, challenge, application, or closing question only when it follows "
        "from the conversation. Do not manufacture a topic shift for diversity, repeat a resolved request, or ask "
        "for tool use merely because tools exist. Preserve the target language. "
        "Return JSON only.\n\n"
        f"OUTPUT SCHEMA:\n{json.dumps(UserTurn.model_json_schema(), ensure_ascii=False)}"
    )
