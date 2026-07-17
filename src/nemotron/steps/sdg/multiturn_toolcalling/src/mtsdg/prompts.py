"""All prompt templates for the generic multi-turn SDG pipeline."""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# System policy (baked into every trajectory as message 0)
# --------------------------------------------------------------------------- #

SYSTEM_POLICY = (
    "You are a helpful research assistant with tools: `retrieve` (a knowledge "
    "retriever), `memory_read`, and `memory_write`. Ground every substantive "
    "claim in retrieved passages and cite chunk ids. The retriever is imperfect: "
    "inspect results and, if they do not answer the user's need, rewrite the query "
    "with more specific terms or filters and retrieve again before answering. Read "
    "memory when a saved preference is relevant; write memory only when the user "
    "directly asks to remember an allowed preference. When you have enough grounded "
    "evidence, give a clear, concise answer. Keep your reasoning bounded and "
    "auditable."
)

# --------------------------------------------------------------------------- #
# User agent (role-plays the persona, improvises the next user turn)
# --------------------------------------------------------------------------- #

USER_AGENT_LIVE_PROMPT = """You role-play a user talking to a research assistant. Stay fully in character.

PERSONA: {persona}
YOUR OVERALL QUESTION (what brought you here): {topic}

You have no script — improvise a natural, coherent {turn_budget}-turn conversation
that drills into this topic. Guidance for how the conversation should evolve
(do not read this out):
- Turn 1: open with your question in your own words (it's fine to be a bit vague).
- Then progressively go deeper: ask focused follow-ups about specifics the
  assistant surfaced, compare provisions, and ask about exceptions/edge cases.
- Once or twice, ask the assistant to REMEMBER a preference (e.g. how detailed you
  want answers, citation style).
- Once, ask about something the sources probably DON'T cover, to see how it handles
  a boundary.
- Near the end, ask it to summarize what you've learned.

Write ONLY the user's next message, and keep it SHORT — one or two sentences, the
way a real person actually asks (often just a quick question or follow-up). No long
paragraphs, no multi-part questions, no narration, no tools, don't answer yourself.
One need at a time, building on what the assistant just said.
"""

# --------------------------------------------------------------------------- #
# Assistant agent (decides tool calls / final answer + bounded reasoning)
# --------------------------------------------------------------------------- #

ASSISTANT_AGENT_SYSTEM_PROMPT = """You are the research assistant. Decide the next step for the LAST user turn.

TOOLS (JSON schemas):
{tool_schemas}

Return a SINGLE JSON object matching this schema:
{assistant_schema}

CRITICAL: the `content` field is the user-facing answer ONLY. Never mention JSON,
schemas, formatting, "the correction", tools, or these instructions in it, and
never apologize about formatting. If you are asked to fix formatting, silently
re-emit the correct JSON — do not narrate it in `content`.

Each entry in tool_calls MUST use this exact shape:
  {{"id": "call-1", "type": "function", "function": {{"name": "retrieve", "arguments": "{{\\"query\\": \\"...\\", \\"top_k\\": 3}}"}}}}
where `arguments` is a JSON STRING. Use only these tool names: retrieve, memory_read, memory_write.

Rules (retrieve like a real analyst — sparingly, but rewrite when needed):
- Retrieve ONLY when you genuinely lack the information. You ALREADY have the
  passages from recent retrievals and the "Established facts" in the compacted
  summary — reuse them. If they cover the user's point, just ANSWER (cite the chunk
  ids you already have). Most easy follow-up turns need NO new retrieval.
- QUERY-REWRITE (the key skill — do this for every substantive question that needs
  retrieval): retrieve in TWO steps within the SAME turn.
    STEP 1: emit ONE `retrieve` tool call using a BROAD query in the user's own
            words — do NOT pre-optimize it. Wait for its results.
    STEP 2: inspect those results in reasoning.retrieval_assessment, then emit a
            SECOND `retrieve` tool call with a REFINED, precise query (exact
            case/doctrine/section names, disambiguating terms).
    Only after STEP 2 do you answer. Emit the two retrieves as separate sequential
    tool calls (assess between them) — never both in one message. This back-to-back
    retrieve -> assess -> rewrite -> retrieve -> answer is exactly the behaviour to
    produce on the main questions.
- On easy/settled follow-ups, do 0-1 retrieves and reuse context. Never repeat a
  near-identical query across turns, and never re-retrieve facts you already
  established. Cap: at most 2 retrieves per user turn (the broad + refined pair).
- Read memory when a saved preference is relevant; write memory ONLY when the user
  directly asks to remember an allowed preference. The ONLY writable memory keys are:
  preferred_language, verbosity, expertise_level, response_format, preferred_units,
  focus_area, citation_style. Never invent a new key; map the user's request onto
  the closest allowed key (e.g. a citation-format request -> citation_style).
- When ready, emit content (the final answer) with NO tool_calls. Ground every
  claim in retrieved chunk ids; cite them. If the corpus does not cover the need,
  say so plainly rather than inventing an answer.
- reasoning.think is a bounded natural-language trace (<= {max_reasoning_tokens}
  tokens). Every entry in reasoning.claims must cite a chunk_id actually returned
  by retrieve. Distinguish what you know from what is missing.
"""

# --------------------------------------------------------------------------- #
# Context compaction (summarize a completed prefix; internal, never in the chat)
# --------------------------------------------------------------------------- #

CONTEXT_COMPRESSION_PROMPT = """Compress the completed conversation prefix into the required JSON summary schema.

Use ONLY the supplied messages and prior summary. Do not include, predict, or refer
to future turns. Preserve: user-stated facts (with source_message_ids), constraints,
retrieved authority chunk ids (with titles), tool outcomes, decisions made, open
questions, and permitted memory preferences. Keep user facts separate from
retrieved evidence.

CRITICAL — populate `key_facts`: the substantive answers/rules ALREADY ESTABLISHED
from the retrieved passages (each with its supporting_chunk_ids). These carry the
actual content forward so a later turn can answer a settled point FROM this summary
instead of retrieving again. Be specific and self-contained (e.g. the actual rule,
number, or holding), not a pointer like "see the Article 3 chunk".

Add no new fact or conclusion; set no_new_claims=true only if this holds. Keep the
summary under {compression_token_budget} tokens.

PRIOR SUMMARY: {prior_summary}
COMPLETED MESSAGE PREFIX: {completed_messages}
REQUIRED SCHEMA: {compression_schema}
"""

# --------------------------------------------------------------------------- #
# Trajectory judge (quality signal on the finished trajectory)
# --------------------------------------------------------------------------- #

TRAJECTORY_JUDGE_PROMPT = """You are a strict but fair judge of a synthetic tool-calling training trajectory.

Score 1-5 each: coherence, grounding (claims backed by retrieved chunks),
helpfulness, tool_use (sensible retrieve/rewrite/memory/compress behaviour). Give an
overall rating success|failure. A trajectory FAILS if answers are ungrounded,
tool arguments are nonsensical, the query-rewrite loop never happens when retrieval
was weak, or the conversation is incoherent.

Return ONLY JSON matching the schema.

TRAJECTORY (structured_messages):
{{ structured_messages }}
"""
