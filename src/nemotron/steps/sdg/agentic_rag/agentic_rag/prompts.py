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
