"""Prompt builders shared by generation, compaction, and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    AssistantRetrievalAction,
    EpisodeSeed,
    TurnPlan,
)

INTENT_GUIDANCE = {
    "research": "Gather new evidence before answering.",
    "rewrite": "Reformulate the information need and retrieve better evidence before answering.",
    "clarify": "Ask a focused clarification when a missing detail would materially change the answer.",
    "user_context": "Elicit only relevant, non-sensitive user context or preferences before proceeding.",
    "scope": "Establish the task boundaries, constraints, or desired level of detail.",
    "orientation": "Give a concise high-level map before going deeper.",
    "direct_answer": "Answer directly from established information when additional tools are unnecessary.",
    "misconception_check": "Check and gently correct a premise before building on it.",
    "example_first": "Start with a concrete example, then connect it to the general explanation.",
    "deepen": "Explore a previously introduced point in greater depth.",
    "compare": "Contrast relevant alternatives using consistent criteria.",
    "synthesize": "Combine established evidence and prior discussion into a coherent conclusion.",
    "recall": "Use relevant details already established earlier in the conversation.",
    "apply_scenario": "Apply established information to the user's scenario.",
    "challenge_assumption": "Test an assumption constructively and explain its consequences.",
    "summarize": "Summarize the established discussion without introducing unsupported claims.",
}


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


def assistant_final_system(
    seed: EpisodeSeed, known_chunk_ids: Iterable[str] = ()
) -> str:
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
        "You are planning one required evidence lookup for a long-running conversation. "
        "Write one specific retrieval query that advances the current user request. Do not answer the user "
        "and do not emit a tool call; the runtime will execute your query as the retrieve tool. "
        "When earlier queries are present, make this query meaningfully distinct. Return one JSON object "
        "matching the AssistantRetrievalAction schema.\n\n"
        f"EFFECTIVE INSTRUCTIONS:\n{seed.instructions}\n\n"
        f"ASSISTANT RETRIEVAL ACTION SCHEMA:\n"
        f"{json.dumps(AssistantRetrievalAction.model_json_schema(), ensure_ascii=False)}"
    )


def assistant_turn_directive(plan: TurnPlan, completed_retrievals: int) -> str:
    if plan.retrieval_required and completed_retrievals < plan.retrieval_depth:
        requirement = (
            f"This is a research/rewrite turn. Before answering, complete {plan.retrieval_depth} successful "
            f"retrieval call(s) with distinct normalized queries. Completed so far: {completed_retrievals}."
        )
    elif plan.retrieval_required:
        requirement = (
            "The required retrieval depth is satisfied. Synthesize a substantive final answer now without "
            "another tool call."
        )
    else:
        requirement = "Retrieval is optional on this turn; use established evidence when sufficient."
    guidance = INTENT_GUIDANCE.get(
        plan.intent, f"Follow the semantic intent expressed by `{plan.intent}`."
    )
    return (
        f"Turn {plan.turn} intent: {plan.intent}. Intent guidance: {guidance} "
        f"{requirement}"
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
