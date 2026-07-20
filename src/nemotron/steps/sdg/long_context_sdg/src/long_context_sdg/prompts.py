"""Prompt builders shared by generation, compaction, and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    AssistantRetrievalAction,
    EpisodeSeed,
    RetrievalPolicyEvent,
    UserTurn,
)


def assistant_system(
    seed: EpisodeSeed,
    tools: list[dict],
    known_chunk_ids: Iterable[str] = (),
) -> str:
    known = sorted(set(known_chunk_ids))
    return (
        "You are the assistant in a synthetic but realistic long-running conversation. "
        "Use tools when evidence or saved preferences are needed. Never invent tool results or citations. "
        "Return one JSON object matching the AssistantAction schema. A tool action may contain tool_calls; "
        "a final answer must contain content and no tool_calls. Keep reasoning bounded and cite only chunk IDs "
        "already returned by retrieve.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"EXACT RETRIEVED CHUNK IDS ALLOWED IN CITATIONS:\n{json.dumps(known)}\n\n"
        f"TOOLS:\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        f"ASSISTANT ACTION SCHEMA:\n{json.dumps(AssistantAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_final_system(seed: EpisodeSeed, known_chunk_ids: Iterable[str] = ()) -> str:
    """Constrain the post-tool pass to an answer that cannot request more tools."""
    known = sorted(set(known_chunk_ids))
    return (
        "You are the assistant in a synthetic but realistic long-running conversation. "
        "The required tool results are already present in the conversation. Synthesize the final answer now. "
        "Do not request, mention, or emit any tool call. Never invent citations or claims. "
        "Support every factual domain claim with an inline citation using a full, exact retrieved chunk ID. "
        "Never shorten, renumber, or invent an ID. If the available evidence does not support a claim, "
        "explicitly say so. "
        "Return one JSON object matching the AssistantFinalAction schema and cite only chunk IDs already "
        "returned by retrieve.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"EXACT RETRIEVED CHUNK IDS ALLOWED IN CITATIONS:\n{json.dumps(known)}\n\n"
        f"ASSISTANT FINAL ACTION SCHEMA:\n"
        f"{json.dumps(AssistantFinalAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_retrieval_system(seed: EpisodeSeed) -> str:
    """Require a model-authored retrieval query without optional answer/tool fields."""
    return (
        "You are preparing one required evidence lookup for a long-running conversation. "
        "Write one specific retrieval query that advances the current user request. Do not answer the user "
        "and do not emit a tool call; the runtime will execute your query as the retrieve tool. "
        "When earlier queries are present, make this query meaningfully distinct. Return one JSON object "
        "matching the AssistantRetrievalAction schema.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"ASSISTANT RETRIEVAL ACTION SCHEMA:\n"
        f"{json.dumps(AssistantRetrievalAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_turn_directive(
    turn: int,
    policy_event: RetrievalPolicyEvent | None,
    completed_retrievals: int,
) -> str:
    if policy_event and completed_retrievals < policy_event.required_retrievals_this_turn:
        requirement = (
            "The conversation is at its retrieval-budget deadline. Before answering, complete "
            f"{policy_event.required_retrievals_this_turn} successful retrieval call(s) with distinct normalized "
            f"queries. Completed so far: {completed_retrievals}."
        )
    elif policy_event:
        requirement = (
            "The retrieval deadline is satisfied. Synthesize a substantive final answer now without another tool call."
        )
    else:
        requirement = (
            "Respond naturally to the user's actual message. Retrieval is optional; use it only when new evidence "
            "is needed, and reuse established evidence when it is sufficient."
        )
    return f"Turn {turn}. {requirement}"


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
