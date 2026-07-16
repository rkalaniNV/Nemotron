"""Generic inner prompts for the deep-research simulation.

DOMAIN-AGNOSTIC BY DESIGN. Nothing here mentions law, the Constitution, or any
corpus. Domain specifics arrive at runtime through the persona, theme, tool
schemas, and retrieved chunks — supplied by the outer config. Keep it that way:
put domain facts in the YAML/seeds, not here.

All templates use ``str.format()`` placeholders.
"""

# ── User agent (roleplay) — asks queries, answers clarifications ─────────────
USER_AGENT_SYSTEM_PROMPT = """You are roleplaying as a person talking to an AI research assistant that can search a knowledge base and call tools.

<INSTRUCTIONS>
- Using your persona and the theme, ask ONE realistic question that requires the assistant to research across multiple sources to answer well.
- Draw on your persona's background, but do NOT state that you are roleplaying.
- Do NOT reveal tool names or try to solve the task yourself.
- If the assistant asks you a clarifying question, answer it naturally and specifically from a plausible real-world situation. Do not refuse; invent reasonable specifics if needed.
- Keep messages natural and conversational — no stage directions or meta commentary.
</INSTRUCTIONS>

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

<THEME>
{theme}
</THEME>

<QUESTION_STYLE>
Shape your question according to these directives:
{directives}
</QUESTION_STYLE>

<TOPIC_SEED>
The question should be answerable using information related to the following material (do not quote it verbatim; ask as a real person would):
{seed_context}
</TOPIC_SEED>"""

USER_FOLLOWUP_PROMPT = """The assistant has answered your previous question. As the same persona, ask a natural follow-up that builds on the conversation and requires further research into a related area.

<CONVERSATION_SO_FAR>
{conversation}
</CONVERSATION_SO_FAR>

<NEXT_TOPIC_SEED>
{seed_context}
</NEXT_TOPIC_SEED>"""

# ── Assistant (the deep-research agent) ──────────────────────────────────────
ASSISTANT_SYSTEM_PROMPT = """You are a careful research assistant with access to tools, including a knowledge-base search.

You operate in two phases:
1. DISCUSSION: If the request is ambiguous or missing facts you need, ask the user a concise clarifying question first. Do not research yet.
2. RESEARCH: Once the task is clear, state a short research plan, then work autonomously — search, read results, reason about what is still missing, and search again — until you can answer. Do not ask the user anything during research; if a fact is genuinely unknown, state your assumption and proceed.

Rules:
- Reason step by step before each tool call.
- Ground every claim in retrieved evidence; cite the source identifiers you used.
- Do not fabricate sources, identifiers, or quotations.
- When you have enough evidence, give a complete, well-organized final answer."""

RESEARCH_PLAN_PROMPT = """The task is now clear. Before researching, write a SHORT research plan: a numbered checklist of the specific sub-questions you must resolve to answer the user. This checklist defines what "sufficient" means for this task. Output only the plan.

<TASK>
{user_query}
</TASK>"""

FINDING_DISTILL_PROMPT = """Summarize, in one or two sentences, ONLY the facts in the retrieved content that are relevant to the research task. Copy identifiers, numbers, and defined terms verbatim; do not paraphrase them. If nothing is relevant, say "nothing relevant".

<TASK>
{user_query}
</TASK>
<RETRIEVED_CONTENT>
{content}
</RETRIEVED_CONTENT>"""

# ── Sufficiency / gap analysis (drives depth up) ─────────────────────────────
SUFFICIENCY_PROMPT = """You are checking whether enough evidence has been gathered to fully answer the user's task.

<TASK>
{user_query}
</TASK>
<RESEARCH_PLAN>
{plan}
</RESEARCH_PLAN>
<FINDINGS_SO_FAR>
{findings}
</FINDINGS_SO_FAR>

Decide if the findings cover every item in the plan well enough to answer.
Respond strictly as JSON, no other text:
{{"sufficient": true|false, "missing": ["<unresolved sub-question>", ...], "next_query_hint": "<what to search next, or empty>"}}"""

# ── API response simulator (for NON-retrieval / auxiliary tools only) ────────
API_RESPONSE_SIM_SYSTEM_PROMPT = """You simulate the JSON response of a tool/API backend for a given tool call.
- Return realistic data consistent with the tool specification and the call arguments.
- Output strictly valid JSON only, no commentary.
- If the call is invalid or missing required parameters, return {"error": "Invalid tool call: <reason>"}."""

API_RESPONSE_SIM_TURN_PROMPT = """<TOOL_SPECIFICATION>
{tool_spec}
</TOOL_SPECIFICATION>
<TASK_CONTEXT>
{user_query}
</TASK_CONTEXT>
<TOOL_CALL>
{tool_call}
</TOOL_CALL>
Return only the JSON response."""

# ── Judges (reused rubric style from the reference pipeline) ─────────────────
_JUDGE_RESPONSE_FORMAT = """
Respond strictly in this format, nothing else:
<explanation>
[justification]
</explanation>
<rating>
[success or failure]
</rating>"""

QUERY_GATE_JUDGE_PROMPT = """You evaluate whether a user's question is a good seed for a multi-source research conversation with an assistant that has the tools below.

<TOOLS>
{tools}
</TOOLS>
<USER_QUESTION>
{user_query}
</USER_QUESTION>

Success if the question is natural, answerable with the tools, and genuinely benefits from researching more than one source. Failure if trivial, unanswerable with these tools, incoherent, or self-answering.
""" + _JUDGE_RESPONSE_FORMAT

TRAJECTORY_JUDGE_PROMPT = """You evaluate an assistant's entire deep-research trajectory (not single turns).

<TOOLS>
{tools}
</TOOLS>
<CONVERSATION>
{conversation}
</CONVERSATION>

A trajectory SUCCEEDS if, across the conversation: tool calls are relevant and well-formed; the assistant gathers missing information before answering; it clarifies genuinely ambiguous requests up front; reasoning is coherent and references prior findings; and the final answer is grounded in retrieved evidence and cites its sources. It FAILS on: repeated/irrelevant tool calls, answering before sufficient evidence, fabricated sources, forgetting prior findings, or incoherence. The simulated user must also stay in character as a user throughout.
""" + _JUDGE_RESPONSE_FORMAT

# System prompt for the judge (was previously empty).
JUDGE_SYSTEM_PROMPT = ("You are a meticulous evaluator of AI research trajectories. Judge only what "
                       "the transcript shows; be strict about grounding and coherence.")

# Rubric judge — used by the standalone judge stage (evaluate.py). Scores the
# SUBJECTIVE dimensions; objective grounding/validity are checked separately.
TRAJECTORY_RUBRIC_PROMPT = """Score this AI research trajectory on each dimension from 1 (poor) to 5 (excellent).

<TOOLS>
{tools}
</TOOLS>
<CONVERSATION>
{conversation}
</CONVERSATION>

Dimensions:
- faithfulness: the final answer's claims are supported by the retrieved tool results (no fabrication).
- coherence: reasoning is logical, references prior findings, and the conversation flows naturally.
- completeness: the final answer fully addresses what the user asked across the whole conversation.
- tool_use: tool calls are relevant, well-formed, and non-redundant.
- user_realism: the simulated user stays in character throughout.

Respond strictly as JSON, no other text:
{{"faithfulness": 1-5, "coherence": 1-5, "completeness": 1-5, "tool_use": 1-5, "user_realism": 1-5, "notes": "<one line>"}}"""


# ── In-loop coaching nudges (injected as user turns during research) ──────────
SEARCH_NUDGE = "You must search the knowledge base before answering. Call the search tool now."

# built as: INSUFFICIENT_NUDGE [+ INSUFFICIENT_NUDGE_HINT] + INSUFFICIENT_NUDGE_TAIL
INSUFFICIENT_NUDGE = "You have not yet gathered enough evidence to fully answer. "
INSUFFICIENT_NUDGE_HINT = "In particular, search next for: {hint} "
INSUFFICIENT_NUDGE_TAIL = "Identify what is still missing and search for it."

# shapes the forced final answer only (this turn is NOT stored in the trajectory)
FINAL_ANSWER_NUDGE = ("Based on the evidence you have gathered, give your final, well-organized "
                      "answer now, citing the source identifiers. Do not call any tools.")

# Re-voice the (persona-agnostic) Stage-2 seed query as the opening user turn, so
# each conversation reflects its sampled persona without regenerating questions.
PERSONA_QUERY_PROMPT = """You are the person described below. Re-ask the question in your own natural voice — same information need, phrased the way YOU would actually ask it given your background and situation. Keep it a single question. Do NOT add new facts, do NOT answer it, no meta commentary.

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>
<THEME>
{theme}
</THEME>
<QUESTION>
{query}
</QUESTION>

Return only the rephrased question."""

# system prompt for the simulated user answering a clarifying question (in character)
CLARIFY_ANSWER_SYSTEM = """You are the person described below, talking to a research assistant. Answer its clarifying question naturally and specifically, staying in character and inventing reasonable real-world details from your own situation if needed. Do not refuse and do not break character.

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>"""

# ── Follow-up query-kind directives (Stage 4 planner) ─────────────────────────
KIND_DIRECTIVES = {
    "half_baked": "Ask a vague, underspecified follow-up that a good assistant should clarify first.",
    "simple": "Ask a short, single-fact follow-up.",
    "crisp": "Ask a precise, well-scoped follow-up naming exactly what you want.",
    "complex_multistep": "Ask a follow-up whose answer needs combining several sources (multi-hop).",
}

# ── Fallback query directives (only the persona-invents-the-query path) ───────
ARCHETYPE_HINTS = {
    "definitional": "Ask what a specific concept or term means.",
    "procedural": "Ask about the steps, grounds, or process for something.",
    "comparative": "Ask to compare or distinguish two related items or concepts.",
    "temporal": "Ask whether something still holds or how it changed over time.",
    "hypothetical_fact_pattern": "Describe a concrete real-world situation and ask what applies.",
    "edge_case": "Ask about an unusual, boundary, or exception scenario.",
}
OUTCOME_HINTS = {
    "answerable": "The material should fully support an answer.",
    "partial": "The material should support only a partial answer; expect some limits.",
    "unanswerable": "The request should NOT be fully satisfiable from the material — a correct assistant would decline part of it.",
    "conflicting": "The situation may involve items that appear to tension with each other.",
}
AMBIGUITY_HINTS = {
    "low": "Give all needed details up front; the assistant should not need to clarify.",
    "medium": "Leave one key detail implicit so a good assistant may ask to clarify.",
    "high": "Be underspecified so the assistant must clarify before researching.",
}

# ── Question generation (Stage 2, offline) ────────────────────────────────────
QGEN_SYSTEM = """You write realistic user questions that will be answered by an AI research assistant with access to a knowledge-base search tool over a specific document collection.
- Ground every question in the provided source text; a diligent researcher must be able to answer it from that collection.
- Do NOT quote the source verbatim; phrase questions the way a real person would ask them.
- Return ONLY a JSON array, no prose."""

QGEN_USER = """<SOURCE_DOCUMENT id="{doc_id}">
{shard_text}
</SOURCE_DOCUMENT>

<RELATED_SECTIONS_IN_THIS_COLLECTION>
{neighbors}
</RELATED_SECTIONS_IN_THIS_COLLECTION>

Write exactly {n} distinct user questions grounded in the SOURCE_DOCUMENT above, spanning this difficulty spectrum (assign each question one level):
{levels}

For "complex_multistep", prefer questions whose answer requires synthesising the source with the related sections listed above (genuine multi-hop).

Return a JSON array of objects, each:
{{"query": "<the question a real person would type>",
  "level": "<one of: {level_names}>",
  "target_sections": ["<ids of sections this question is grounded in, from the source or related list>"],
  "rationale": "<one line: why answering needs research>"}}"""
