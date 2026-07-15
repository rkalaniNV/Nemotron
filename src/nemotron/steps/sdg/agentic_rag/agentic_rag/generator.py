"""DeepResearchSimulatorGenerator — the phased deep-research simulation loop.

One row -> one grounded multi-hop trajectory. The flow (all config-driven):

  DISCUSSION  -> RESEARCH PLAN -> autonomous TOOL LOOP -> ANSWER -> follow-up

Extensions layered on the reference multi-agent loop:
  (1) real retriever for RAG tools           -> tools.ToolEnvironment
  (2) sufficiency / gap loop (drives depth)  -> _check_sufficiency + min_hops
  (3) sliding window + scratchpad            -> context.build_assistant_view
  (4) salvage-on-failure (protects yield)    -> _finalize truncates to last-good

Requires the Data Designer runtime; imports are guarded so the rest of the
package stays importable/testable without DD installed.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell, ColumnGeneratorWithModelRegistry,
)

from .config import DeepResearchSimulatorConfig
from .context import ResearchMemory, build_assistant_view, build_judge_view
from .judges import run_inline_judge
from .llm import call_llm, call_llm_with_majority_vote
from .messages import (
    format_conversation_history_for_prompt, format_theme_for_prompt, format_tools_for_prompt,
    parse_theme,
)
from .persona import format_persona_for_prompt
from .prompts import (
    ASSISTANT_SYSTEM_PROMPT, FINDING_DISTILL_PROMPT, QUERY_GATE_JUDGE_PROMPT,
    RESEARCH_PLAN_PROMPT, SUFFICIENCY_PROMPT, TRAJECTORY_JUDGE_PROMPT,
    USER_AGENT_SYSTEM_PROMPT, USER_FOLLOWUP_PROMPT,
)
from .tools import ToolEnvironment, sample_tools
from .verifiers import ToolCallVerifier

MODEL_USER, MODEL_ASSISTANT, MODEL_JUDGE = "user_model", "assistant_model", "judge_model"
MODEL_ALIASES = [MODEL_USER, MODEL_ASSISTANT, "api_response_model", MODEL_JUDGE]


class DeepResearchSimulatorGenerator(
    ColumnGeneratorCellByCell[DeepResearchSimulatorConfig],
    ColumnGeneratorWithModelRegistry[DeepResearchSimulatorConfig],
):
    """Runs the phased deep-research RAG simulation for a single row."""

    # ── async entry point: DD runs the client in async mode ──────────────────
    async def agenerate(self, data: dict) -> dict:
        """Capture DD's event loop, then run the sync loop in a worker thread.

        The captured loop lets each LLM call bridge `acompletion` back to the
        main loop (see llm.DD_EVENT_LOOP). asyncio.to_thread copies the context,
        so the ContextVar set here is visible inside `generate`.
        """
        import asyncio
        from .llm import DD_EVENT_LOOP
        DD_EVENT_LOOP.set(asyncio.get_running_loop())
        return await asyncio.to_thread(self.generate, dict(data))

    # ── NDD entry point (sync loop; runs in a worker thread under agenerate) ──
    def generate(self, data: dict) -> dict:
        cfg = self.config
        models = {a: self.get_model(a) for a in MODEL_ALIASES}
        rng = random.Random(hash(str(data.get(cfg.bundle_column))) & 0xFFFFFFFF)

        bundle = _as_obj(data[cfg.bundle_column])
        persona = _as_obj(data[cfg.persona_column])
        theme = parse_theme(data[cfg.theme_column])
        all_tools = _as_obj(data[cfg.tools_column])
        gold_sections = [str(s) for s in bundle.get("member_sections", [])]
        seed_context = bundle.get("anchor_text", "") or bundle.get("seed_context", "")

        corpus = self._load_corpus(cfg)
        env = ToolEnvironment(cfg, corpus, gold_sections, rng=rng)
        tools = sample_tools(all_tools, cfg.max_tools, cfg.retrieval_tools, rng)

        # diversity samplers steer generation (config-driven column names)
        archetype = data.get(cfg.archetype_column, "")
        outcome = data.get(cfg.outcome_column, "")
        ambiguity = data.get(cfg.ambiguity_column, "")
        depth = data.get(cfg.depth_column, "")
        directives = _query_directives(archetype, outcome, ambiguity)
        eff_min_hops = _depth_to_hops(depth, cfg)

        # Stage 1: user query (shaped by the sampled archetype/outcome/ambiguity)
        user_query = self._gen_user_query(models, persona, theme, tools, seed_context, directives)
        data["user_query"] = user_query

        # Stage 2: gate
        if cfg.gate_query:
            _, _, ok = run_inline_judge(models, MODEL_JUDGE, QUERY_GATE_JUDGE_PROMPT.format(
                tools=format_tools_for_prompt(tools), user_query=user_query))
            if not ok:
                data = _skip(data, "query_gate_failed")
                data[cfg.name] = data["conversation_messages"]
                return data

        # Stage 3: phased conversation
        conversation: Dict[str, Any] = {"messages": [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}],
                                        "tools": tools, "metadata": {"gold_rank_log": [], "phases": []}}
        conversation["messages"].append({"role": "user", "content": user_query})
        assistant_history = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                             {"role": "user", "content": user_query}]

        status = True
        hops_total = 0
        for turn in range(cfg.max_turns):
            if turn > 0:
                follow = self._gen_followup(models, persona, theme, conversation, seed_context, directives)
                if not follow:
                    break
                conversation["messages"].append({"role": "user", "content": follow})
                assistant_history.append({"role": "user", "content": follow})

            memory = ResearchMemory()
            # DISCUSSION -> PLAN. High-ambiguity rows should exercise clarification;
            # low-ambiguity rows should proceed without over-clarifying.
            if cfg.allow_discussion and str(ambiguity).lower() != "low":
                self._discussion_phase(models, conversation, assistant_history, env, tools)
            if cfg.require_research_plan:
                memory.plan = self._research_plan(models, assistant_history, user_query)
                conversation["metadata"]["phases"].append({"turn": turn, "plan": memory.plan})

            # autonomous TOOL LOOP (depth floor set by the sampled depth_target)
            ok, hops = self._tool_loop(models, conversation, assistant_history, env, tools,
                                       user_query, memory, turn, eff_min_hops)
            hops_total += hops
            if not ok:
                status = False
                break

        return self._finalize(data, cfg, models, conversation, env, status, hops_total)

    # ── Stage helpers ────────────────────────────────────────────────────────
    def _gen_user_query(self, models, persona, theme, tools, seed_context, directives) -> str:
        sys_prompt = USER_AGENT_SYSTEM_PROMPT.format(
            persona=format_persona_for_prompt(persona),
            theme=format_theme_for_prompt(theme),
            directives=directives,
            seed_context=seed_context[:1500])
        resp = call_llm(models, MODEL_USER, [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Hi! I'm a research assistant with tools. What can I help you with?"}])
        return resp.get("content", "") if isinstance(resp, dict) else ""

    def _gen_followup(self, models, persona, theme, conversation, seed_context, directives) -> str:
        prompt = USER_FOLLOWUP_PROMPT.format(
            conversation=format_conversation_history_for_prompt(conversation["messages"]),
            seed_context=seed_context[:1000])
        resp = call_llm(models, MODEL_USER, [
            {"role": "system", "content": USER_AGENT_SYSTEM_PROMPT.format(
                persona=format_persona_for_prompt(persona),
                theme=format_theme_for_prompt(theme), directives=directives,
                seed_context=seed_context[:1000])},
            {"role": "user", "content": prompt}])
        return resp.get("content", "") if isinstance(resp, dict) else ""

    def _discussion_phase(self, models, conversation, assistant_history, env, tools) -> None:
        """Let the assistant ask clarifying questions; the user answers. Bounded."""
        cfg = self.config
        for _ in range(cfg.max_discussion_exchanges):
            view = build_assistant_view(assistant_history, None, window_k=cfg.context_window_k,
                                        compaction_mode=cfg.compaction_mode, use_scratchpad=False)
            turn = call_llm_with_majority_vote(models, MODEL_ASSISTANT, view, tools, n=cfg.majority_vote_n)
            # a clarifying question has no tool_calls and ends with a question
            if turn.get("tool_calls") or not _looks_like_question(turn.get("content", "")):
                return  # ready to research; do NOT consume this turn here
            conversation["messages"].append({"role": "assistant", "content": turn.get("content", "")})
            assistant_history.append({"role": "assistant", "content": turn.get("content", "")})
            answer = call_llm(models, MODEL_USER, _clarify_answer_msgs(conversation))
            ans = answer.get("content", "") if isinstance(answer, dict) else ""
            conversation["messages"].append({"role": "user", "content": ans})
            assistant_history.append({"role": "user", "content": ans})

    def _research_plan(self, models, assistant_history, user_query) -> str:
        resp = call_llm(models, MODEL_ASSISTANT,
                        assistant_history + [{"role": "user", "content": RESEARCH_PLAN_PROMPT.format(user_query=user_query)}])
        plan = resp.get("content", "") if isinstance(resp, dict) else ""
        assistant_history.append({"role": "assistant", "content": f"Research plan:\n{plan}"})
        return plan

    # ── the autonomous tool loop (extensions 1-3) ────────────────────────────
    def _tool_loop(self, models, conversation, assistant_history, env, tools,
                   user_query, memory, turn, min_hops) -> Tuple[bool, int]:
        cfg = self.config
        verifier = ToolCallVerifier()
        hops = 0
        for step in range(cfg.max_steps):
            view = build_assistant_view(assistant_history, memory, window_k=cfg.context_window_k,
                                        compaction_mode=cfg.compaction_mode, use_scratchpad=cfg.use_scratchpad)
            turn_msg = call_llm_with_majority_vote(models, MODEL_ASSISTANT, view, tools, n=cfg.majority_vote_n)
            a_msg = {"role": "assistant", "reasoning_content": turn_msg.get("reasoning_content", ""),
                     "content": turn_msg.get("content", ""), "tool_calls": turn_msg.get("tool_calls", [])}
            conversation["messages"].append(a_msg)
            assistant_history.append(a_msg)

            if a_msg["tool_calls"]:
                if not self._execute_tools(models, conversation, assistant_history, env, tools,
                                           user_query, memory, verifier, hops):
                    return False, hops
                hops += 1
                continue

            # no tool_calls -> candidate ANSWER; sufficiency decides (extension 2)
            if cfg.enforce_sufficiency and self._insufficient(models, user_query, memory, hops, min_hops):
                nudge = ("You have not yet gathered enough evidence to fully answer. "
                         "Identify what is still missing and search for it.")
                conversation["messages"].append({"role": "user", "content": nudge})
                assistant_history.append({"role": "user", "content": nudge})
                continue
            return True, hops  # accepted final answer

        # ran out of steps with tools still pending
        return (not conversation["messages"][-1].get("tool_calls")), hops

    def _execute_tools(self, models, conversation, assistant_history, env, tools,
                       user_query, memory, verifier, hop) -> bool:
        cfg = self.config
        last = conversation["messages"][-1]
        all_ok, err_msgs, _, _, correct = verifier.verify(last.get("tool_calls", []), tools)

        # correction loop for schema errors
        attempts = 0
        while not all_ok and err_msgs and attempts < cfg.max_correction_attempts:
            conversation["messages"].extend(err_msgs)
            assistant_history.extend(err_msgs)
            retry = call_llm_with_majority_vote(models, MODEL_ASSISTANT, assistant_history, tools, n=cfg.majority_vote_n)
            r_msg = {"role": "assistant", "content": retry.get("content", ""),
                     "tool_calls": retry.get("tool_calls", [])}
            conversation["messages"].append(r_msg)
            assistant_history.append(r_msg)
            all_ok, err_msgs, _, _, correct = verifier.verify(r_msg.get("tool_calls", []), tools)
            attempts += 1
        if not correct:
            return False

        for tc in correct:
            response, meta = env.respond(tc, models, user_query, tools)
            tool_msg = {"role": "tool", "content": response, "tool_call_id": tc["id"],
                        "_chunk_id": meta.get("_chunk_id")}
            conversation["messages"].append(tool_msg)
            assistant_history.append(tool_msg)
            if meta.get("gold_rank") is not None or "gold_rank" in meta:
                conversation["metadata"]["gold_rank_log"].append(
                    {"hop": hop, "tool": meta.get("tool"), "gold_rank": meta.get("gold_rank"),
                     "injected": meta.get("injected", False)})
            if self.config.use_scratchpad:
                finding = self._distill(models, user_query, response)
                memory.add(hop, meta.get("tool", tc["function"]["name"]),
                           meta.get("query", ""), finding, meta.get("returned", []))
        return True

    def _distill(self, models, user_query, content) -> str:
        resp = call_llm(models, MODEL_ASSISTANT, [
            {"role": "user", "content": FINDING_DISTILL_PROMPT.format(user_query=user_query, content=content[:2000])}])
        return (resp.get("content", "") if isinstance(resp, dict) else "").strip()

    def _insufficient(self, models, user_query, memory, hops, min_hops) -> bool:
        if hops < min_hops:
            return True  # enforce depth floor regardless of judge
        resp = call_llm(models, MODEL_JUDGE, [{"role": "user", "content": SUFFICIENCY_PROMPT.format(
            user_query=user_query, plan=memory.plan or "(none)", findings=memory.render())}])
        text = resp.get("content", "") if isinstance(resp, dict) else ""
        try:
            verdict = json.loads(_extract_json(text))
        except Exception:
            return False  # fail open: don't loop forever on a parse error
        if self.config.sufficiency_mode == "soft":
            return False
        return not bool(verdict.get("sufficient", True))

    # ── finalize (extension 4: salvage) ──────────────────────────────────────
    def _finalize(self, data, cfg, models, conversation, env, status, hops) -> dict:
        messages = conversation["messages"]
        salvaged = False
        if not status:
            cut = _last_good_boundary(messages)
            good_hops = _count_hops(messages[:cut])
            if good_hops >= cfg.salvage_min_hops:
                messages = messages[:cut]
                salvaged = True
                status = True

        if status:
            _, rating, ok = run_inline_judge(models, MODEL_JUDGE, TRAJECTORY_JUDGE_PROMPT.format(
                tools=format_tools_for_prompt(conversation["tools"]),
                conversation=format_conversation_history_for_prompt(
                    build_judge_view(messages, window_k=cfg.context_window_k, compaction_mode=cfg.compaction_mode))))
            data["trajectory_judgment"] = json.dumps({"rating": rating})
            status = ok
        else:
            data["trajectory_judgment"] = json.dumps({"rating": "failure", "reason": "unsalvageable"})

        stored = messages if cfg.store_full_trace else build_judge_view(
            messages, window_k=cfg.context_window_k, compaction_mode=cfg.compaction_mode)
        messages_json = json.dumps([_strip_internal(m) for m in stored], ensure_ascii=False, default=str)
        data.update({
            "conversation_messages": messages_json,
            "conversation_metadata": json.dumps(conversation["metadata"], ensure_ascii=False, default=str),
            "conversation_status": bool(status),
            "gold_rank_log": json.dumps(conversation["metadata"]["gold_rank_log"]),
            "hops_taken": hops,
            "salvaged": salvaged,
        })
        # the plugin's own (named) column holds the trajectory messages
        data[self.config.name] = messages_json
        return data

    # ── corpus loading ───────────────────────────────────────────────────────
    @staticmethod
    def _load_corpus(cfg) -> List[Dict[str, Any]]:
        if not cfg.corpus_path:
            return []
        with open(cfg.corpus_path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]


# ── module helpers ───────────────────────────────────────────────────────────
_ARCHETYPE_HINTS = {
    "definitional": "Ask what a specific concept or term means.",
    "procedural": "Ask about the steps, grounds, or process for something.",
    "comparative": "Ask to compare or distinguish two related provisions or concepts.",
    "temporal": "Ask whether something still holds or how it changed over time.",
    "hypothetical_fact_pattern": "Describe a concrete real-world situation and ask what applies.",
    "edge_case": "Ask about an unusual, boundary, or exception scenario.",
}
_OUTCOME_HINTS = {
    "answerable": "The material should fully support an answer.",
    "partial": "The material should support only a partial answer; expect some limits.",
    "unanswerable": "The request should NOT be fully satisfiable from the material — a correct assistant would decline part of it.",
    "conflicting": "The situation may involve provisions that appear to tension with each other.",
}
_AMBIGUITY_HINTS = {
    "low": "Give all needed details up front; the assistant should not need to clarify.",
    "medium": "Leave one key detail implicit so a good assistant may ask to clarify.",
    "high": "Be underspecified so the assistant must clarify before researching.",
}


def _query_directives(archetype, outcome, ambiguity) -> str:
    parts = []
    for val, table in ((archetype, _ARCHETYPE_HINTS), (outcome, _OUTCOME_HINTS), (ambiguity, _AMBIGUITY_HINTS)):
        hint = table.get(str(val).lower())
        if hint:
            parts.append(f"- {hint}")
    return "\n".join(parts) or "- Ask a natural, well-scoped question."


def _depth_to_hops(depth, cfg) -> int:
    """Map the sampled depth_target to an effective min-hop floor (bounded by max_steps)."""
    floor = {"shallow": 1, "moderate": 2, "deep": max(3, cfg.min_hops)}.get(str(depth).lower(), cfg.min_hops)
    return min(floor, cfg.max_steps)


def _as_obj(v):
    return v if isinstance(v, (dict, list)) else json.loads(v)


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    return t.endswith("?") or t.lower().startswith(("could you", "can you", "to clarify", "which", "what", "do you"))


def _clarify_answer_msgs(conversation):
    return [{"role": "system", "content": "You are the user. Answer the assistant's clarifying question "
             "naturally and specifically; invent reasonable real-world details if needed."},
            {"role": "user", "content": format_conversation_history_for_prompt(conversation["messages"])}]


def _extract_json(text: str) -> str:
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a >= 0 and b > a else "{}"


def _last_good_boundary(messages: List[Dict[str, Any]]) -> int:
    """Index to truncate at: keep through the last assistant answer that followed a tool response."""
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("content"):
            return i + 1
    return cut


def _count_hops(messages: List[Dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "tool")


def _strip_internal(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def _skip(data, reason) -> dict:
    data.update({"conversation_messages": "[]",
                 "conversation_metadata": json.dumps({"skipped": True, "reason": reason}),
                 "conversation_status": False, "gold_rank_log": "[]",
                 "hops_taken": 0, "salvaged": False,
                 "trajectory_judgment": json.dumps({"rating": "failure", "reason": reason})})
    return data
