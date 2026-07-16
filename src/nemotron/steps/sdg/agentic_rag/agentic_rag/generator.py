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

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from .planner import plan_conversation, turn_eff
from .prompts import (
    ASSISTANT_SYSTEM_PROMPT, FINDING_DISTILL_PROMPT, QUERY_GATE_JUDGE_PROMPT,
    RESEARCH_PLAN_PROMPT, SUFFICIENCY_PROMPT, TRAJECTORY_JUDGE_PROMPT,
    USER_AGENT_SYSTEM_PROMPT, USER_FOLLOWUP_PROMPT, KIND_DIRECTIVES,
    SEARCH_NUDGE, INSUFFICIENT_NUDGE, INSUFFICIENT_NUDGE_HINT, INSUFFICIENT_NUDGE_TAIL,
    FINAL_ANSWER_NUDGE, CLARIFY_ANSWER_SYSTEM, ARCHETYPE_HINTS, OUTCOME_HINTS, AMBIGUITY_HINTS,
    PERSONA_QUERY_PROMPT,
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
        # route model calls through direct OpenAI clients (correct tool-call
        # parsing) when configured; falls back to DD facades otherwise.
        from .llm import configure_direct_clients
        configure_direct_clients(cfg.model_clients)
        models = {a: self.get_model(a) for a in MODEL_ALIASES}

        # per-row inputs (new Stage-2 seed columns, with legacy-bundle fallback)
        bundle = _as_obj(data.get(cfg.bundle_column) or "{}")
        persona = _as_obj(data[cfg.persona_column])
        theme = parse_theme(data[cfg.theme_column])
        all_tools = _as_obj(data[cfg.tools_column])
        cluster_id = str(data.get(cfg.cluster_id_column, "") or bundle.get("cluster_id", ""))
        seed_query = str(data.get(cfg.query_column, "") or "").strip()
        # the seed query's difficulty kind (Stage 2) shapes the opening turn
        seed_kind = str(data.get(cfg.query_level_column, "") or "crisp").strip() or "crisp"

        gold_sections = _read_gold(data.get(cfg.gold_column)) or [str(s) for s in bundle.get("member_sections", [])]
        gold_docs = _read_gold(data.get(cfg.gold_doc_column))
        seed_context = bundle.get("anchor_text", "") or bundle.get("seed_context", "") or seed_query

        # reproducible per-row RNG (stable across processes, unlike builtin hash())
        seed_key = f"{cluster_id}|{seed_query}|{json.dumps(bundle, sort_keys=True, default=str)}"
        rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))

        index_dir = self._resolve_index_dir(cfg, cluster_id)
        corpus = self._load_corpus(cfg, index_dir)
        env = ToolEnvironment(cfg, corpus, gold_sections, rng=rng, index_dir=index_dir,
                              gold_doc_ids=gold_docs)
        tools = sample_tools(all_tools, cfg.max_tools, cfg.retrieval_tools, rng)

        # diversity samplers still shape a persona-invented query (fallback path)
        directives = _query_directives(data.get(cfg.archetype_column, ""),
                                       data.get(cfg.outcome_column, ""),
                                       data.get(cfg.ambiguity_column, ""))

        # Stage 4: sample a per-conversation PLAN — always multi-turn, with each
        # turn's shape (depth/clarify) driven by its query kind. Every row differs.
        plan = plan_conversation(cfg.conversation_plan, rng, seed_kind=seed_kind, min_turns=cfg.min_turns)
        data["conversation_plan"] = json.dumps(plan.kinds(), ensure_ascii=False)
        data["cluster_id"] = cluster_id

        # Stage 3: the user's opening turn — use the Stage-2 query if present,
        # otherwise fall back to inventing one from the persona/seed context.
        user_query = seed_query or self._gen_user_query(
            models, persona, theme, tools, seed_context, directives)
        # re-voice the persona-agnostic seed query in this row's persona (same
        # information need, the persona's phrasing) so persona shapes the opening.
        if seed_query and cfg.persona_voice:
            user_query = self._persona_voice(models, persona, theme, seed_query) or user_query
        data["user_query"] = user_query

        if cfg.gate_query:
            _, _, ok = run_inline_judge(models, MODEL_JUDGE, QUERY_GATE_JUDGE_PROMPT.format(
                tools=format_tools_for_prompt(tools), user_query=user_query))
            if not ok:
                data = _skip(data, "query_gate_failed")
                data[cfg.name] = data["conversation_messages"]
                return data

        conversation: Dict[str, Any] = {"messages": [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}],
                                        "tools": tools,
                                        "metadata": {"gold_rank_log": [], "phases": [],
                                                     "cluster_id": cluster_id, "plan": plan.kinds()}}
        conversation["messages"].append({"role": "user", "content": user_query})
        assistant_history = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                             {"role": "user", "content": user_query}]

        status = True
        hops_total = 0
        # execute the plan: one pass per turn, each shaped by its sampled query kind
        for turn, spec in enumerate(plan.turns):
            cur_query = user_query
            if turn > 0:
                follow = self._gen_followup(models, persona, theme, conversation, seed_context,
                                            KIND_DIRECTIVES.get(spec.kind, ""))
                if not follow:
                    break
                cur_query = follow
                conversation["messages"].append({"role": "user", "content": follow})
                assistant_history.append({"role": "user", "content": follow})

            eff = turn_eff(spec)
            memory = ResearchMemory()
            if eff["allow_discussion"]:      # a clarify-kind turn opens with clarification
                self._discussion_phase(models, persona, conversation, assistant_history, env, tools)
            if eff["require_plan"] and eff["max_steps"] > 1:
                memory.plan = self._research_plan(models, assistant_history, cur_query)
                conversation["metadata"]["phases"].append({"turn": turn, "kind": spec.kind, "plan": memory.plan})

            ok, hops = self._tool_loop(models, persona, conversation, assistant_history, env, tools,
                                       cur_query, memory, turn, eff)
            hops_total += hops
            if not ok:
                status = False
                break

        return self._finalize(data, cfg, models, conversation, env, status, hops_total)

    # ── per-cluster index / corpus resolution ─────────────────────────────────
    @staticmethod
    def _resolve_index_dir(cfg, cluster_id: str) -> Optional[str]:
        """Per-cluster streaming layout: <index_base_dir>/<cluster_id>/index.
        Falls back to the single cfg.index_dir when no base dir / cluster."""
        if cfg.index_base_dir and cluster_id:
            return str(Path(cfg.index_base_dir) / cluster_id / "index")
        return cfg.index_dir

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

    def _persona_voice(self, models, persona, theme, query) -> str:
        """Re-ask the seed query in the row's persona voice (same information need)."""
        resp = call_llm(models, MODEL_USER, [
            {"role": "user", "content": PERSONA_QUERY_PROMPT.format(
                persona=format_persona_for_prompt(persona),
                theme=format_theme_for_prompt(theme), query=query)}])
        return (resp.get("content", "") if isinstance(resp, dict) else "").strip()

    def _discussion_phase(self, models, persona, conversation, assistant_history, env, tools) -> None:
        """Let the assistant ask clarifying questions; the user (in persona) answers."""
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
            answer = call_llm(models, MODEL_USER, _clarify_answer_msgs(persona, conversation))
            ans = answer.get("content", "") if isinstance(answer, dict) else ""
            conversation["messages"].append({"role": "user", "content": ans})
            assistant_history.append({"role": "user", "content": ans})

    def _research_plan(self, models, assistant_history, user_query) -> str:
        resp = call_llm(models, MODEL_ASSISTANT,
                        assistant_history + [{"role": "user", "content": RESEARCH_PLAN_PROMPT.format(user_query=user_query)}])
        plan = resp.get("content", "") if isinstance(resp, dict) else ""
        assistant_history.append({"role": "assistant", "content": f"Research plan:\n{plan}"})
        return plan

    # ── the autonomous tool loop (Stages 3-5) ────────────────────────────────
    def _tool_loop(self, models, persona, conversation, assistant_history, env, tools,
                   user_query, memory, turn, eff) -> Tuple[bool, int]:
        cfg = self.config
        verifier = ToolCallVerifier()
        hops = 0
        clarify_used = 0
        for step in range(eff["max_steps"]):
            view = build_assistant_view(
                assistant_history, memory, window_k=cfg.context_window_k,
                compaction_mode=cfg.compaction_mode,
                compression_token_limit=cfg.compression_token_limit,
                preserve_last_user_turn=cfg.preserve_last_user_turn,
                use_scratchpad=cfg.use_scratchpad)
            # force a tool call until the depth floor is met so the agent grounds
            # its answer (some models otherwise answer from memory without searching)
            tc_choice = "required" if (cfg.force_first_tool and hops < eff["min_hops"]) else "auto"
            turn_msg = call_llm_with_majority_vote(models, MODEL_ASSISTANT, view, tools,
                                                   n=cfg.majority_vote_n, tool_choice=tc_choice)
            tool_calls = turn_msg.get("tool_calls", []) or []
            if cfg.max_tool_calls_per_turn and len(tool_calls) > cfg.max_tool_calls_per_turn:
                tool_calls = tool_calls[:cfg.max_tool_calls_per_turn]  # endpoint compat: one at a time
            a_msg = {"role": "assistant", "reasoning_content": turn_msg.get("reasoning_content", ""),
                     "content": turn_msg.get("content", ""), "tool_calls": tool_calls}
            conversation["messages"].append(a_msg)
            assistant_history.append(a_msg)

            if a_msg["tool_calls"]:
                if not self._execute_tools(models, conversation, assistant_history, env, tools,
                                           user_query, memory, verifier, hops):
                    return False, hops
                hops += 1
                continue

            # forced grounding: if the model answered/clarified without searching
            # while still below the hop floor (some models ignore tool_choice=
            # "required"), push it to search before it may answer.
            if cfg.force_first_tool and hops < eff["min_hops"]:
                conversation["messages"].append({"role": "user", "content": SEARCH_NUDGE})
                assistant_history.append({"role": "user", "content": SEARCH_NUDGE})
                continue

            # no tool_calls: the assistant either asked a clarifying question or
            # produced a candidate answer. If it's a question and clarification is
            # allowed, let the USER answer it (coherent multi-turn) instead of
            # nudging it to search — otherwise the loop dead-nudges and never
            # actually researches.
            content = a_msg.get("content", "") or ""
            if (_looks_like_question(content) and eff["allow_discussion"]
                    and clarify_used < cfg.max_discussion_exchanges):
                clarify_used += 1
                ans = call_llm(models, MODEL_USER, _clarify_answer_msgs(persona, conversation))
                ans_text = ans.get("content", "") if isinstance(ans, dict) else ""
                conversation["messages"].append({"role": "user", "content": ans_text})
                assistant_history.append({"role": "user", "content": ans_text})
                continue

            # candidate ANSWER; sufficiency decides (Stage 5 depth)
            if eff["enforce_sufficiency"]:
                insufficient, hint = self._insufficient(models, user_query, memory, hops, eff["min_hops"])
                if insufficient:
                    nudge = INSUFFICIENT_NUDGE
                    if hint:
                        nudge += INSUFFICIENT_NUDGE_HINT.format(hint=hint)
                    nudge += INSUFFICIENT_NUDGE_TAIL
                    conversation["messages"].append({"role": "user", "content": nudge})
                    assistant_history.append({"role": "user", "content": nudge})
                    continue
            return True, hops  # accepted final answer

        # ran out of steps: force a final grounded answer so the trajectory never
        # ends on a dangling tool call / tool response.
        self._force_final_answer(models, conversation, assistant_history, memory)
        return True, hops

    def _force_final_answer(self, models, conversation, assistant_history, memory) -> None:
        cfg = self.config
        view = build_assistant_view(
            assistant_history, memory, window_k=cfg.context_window_k,
            compaction_mode=cfg.compaction_mode,
            compression_token_limit=cfg.compression_token_limit,
            preserve_last_user_turn=cfg.preserve_last_user_turn, use_scratchpad=cfg.use_scratchpad)
        # nudge only shapes the view; it is NOT stored in the trajectory
        view = view + [{"role": "user", "content": FINAL_ANSWER_NUDGE}]
        resp = call_llm(models, MODEL_ASSISTANT, view)  # no tools -> a text answer
        answer = {"role": "assistant",
                  "reasoning_content": resp.get("reasoning_content", "") if isinstance(resp, dict) else "",
                  "content": (resp.get("content", "") if isinstance(resp, dict) else "").strip()
                  or "Based on the retrieved evidence above, here is my answer."}
        conversation["messages"].append(answer)
        assistant_history.append(answer)

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

        # Stage 5: a turn may emit several tool calls — execute them concurrently
        # (LLM-simulated tools are I/O-bound). Each call gets its own child RNG so
        # concurrency stays thread-safe and reproducible; results are re-ordered
        # to match the tool_calls so tool_call_id pairing is preserved.
        # ThreadPoolExecutor workers don't inherit contextvars, so re-set DD's
        # event loop in each worker to keep the async LLM bridge working.
        from .llm import DD_EVENT_LOOP
        loop = DD_EVENT_LOOP.get()

        def _run(tc, seed):
            if loop is not None:
                DD_EVENT_LOOP.set(loop)
            return env.respond(tc, models, user_query, tools, rng=random.Random(seed))

        seeds = [env.rng.getrandbits(32) for _ in correct]
        if cfg.parallel_tools and len(correct) > 1:
            with ThreadPoolExecutor(max_workers=min(len(correct), 8)) as ex:
                responses = list(ex.map(_run, correct, seeds))
        else:
            responses = [_run(tc, s) for tc, s in zip(correct, seeds)]

        for tc, (response, meta) in zip(correct, responses):
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

    def _insufficient(self, models, user_query, memory, hops, min_hops) -> Tuple[bool, str]:
        """Return (is_insufficient, next_query_hint). The hint (from the judge's
        gap analysis) is fed back into the nudge so the next search is targeted."""
        if hops < min_hops:
            return True, ""  # enforce depth floor regardless of judge
        resp = call_llm(models, MODEL_JUDGE, [{"role": "user", "content": SUFFICIENCY_PROMPT.format(
            user_query=user_query, plan=memory.plan or "(none)", findings=memory.render())}])
        text = resp.get("content", "") if isinstance(resp, dict) else ""
        try:
            verdict = json.loads(_extract_json(text))
        except Exception:
            return False, ""  # fail open: don't loop forever on a parse error
        if self.config.sufficiency_mode == "soft":
            return False, ""
        hint = verdict.get("next_query_hint") or ("; ".join(verdict.get("missing", []) or []))
        return (not bool(verdict.get("sufficient", True))), str(hint or "")

    # ── finalize (extension 4: salvage) ──────────────────────────────────────
    def _finalize(self, data, cfg, models, conversation, env, status, hops) -> dict:
        messages = conversation["messages"]
        # pristine, pre-judge / pre-salvage trajectory (dumped raw; differs from the
        # kept, judged, possibly-truncated version below).
        data["conversation_messages_raw"] = json.dumps(
            [_strip_internal(m) for m in messages], ensure_ascii=False, default=str)
        salvaged = False
        if not status:
            cut = _last_good_boundary(messages)
            good_hops = _count_hops(messages[:cut])
            if good_hops >= cfg.salvage_min_hops:
                messages = messages[:cut]
                salvaged = True
                status = True

        if status and cfg.inline_judge:
            _, rating, ok = run_inline_judge(models, MODEL_JUDGE, TRAJECTORY_JUDGE_PROMPT.format(
                tools=format_tools_for_prompt(conversation["tools"]),
                conversation=format_conversation_history_for_prompt(
                    build_judge_view(messages, window_k=cfg.context_window_k, compaction_mode=cfg.compaction_mode,
                                     compression_token_limit=cfg.compression_token_limit))))
            data["trajectory_judgment"] = json.dumps({"rating": rating})
            status = ok
        elif status:
            # inline judge OFF -> keep every completed trajectory; judging is a separate stage
            data["trajectory_judgment"] = json.dumps({"rating": "ungraded"})
        else:
            data["trajectory_judgment"] = json.dumps({"rating": "failure", "reason": "unsalvageable"})

        stored = messages if cfg.store_full_trace else build_judge_view(
            messages, window_k=cfg.context_window_k, compaction_mode=cfg.compaction_mode,
            compression_token_limit=cfg.compression_token_limit)
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

    # ── corpus loading (cached per source across rows) ───────────────────────
    def _load_corpus(self, cfg, index_dir: Optional[str]) -> List[Dict[str, Any]]:
        if not hasattr(self, "_corpus_cache"):
            self._corpus_cache: Dict[str, List[Dict[str, Any]]] = {}
        src = None
        if index_dir and (Path(index_dir) / "corpus.jsonl").exists():
            src = str(Path(index_dir) / "corpus.jsonl")   # per-cluster corpus
        elif cfg.corpus_path:
            src = cfg.corpus_path
        if not src:
            return []
        if src not in self._corpus_cache:
            with open(src, encoding="utf-8") as f:
                self._corpus_cache[src] = [json.loads(l) for l in f if l.strip()]
        return self._corpus_cache[src]


# ── module helpers ───────────────────────────────────────────────────────────
def _query_directives(archetype, outcome, ambiguity) -> str:
    parts = []
    for val, table in ((archetype, ARCHETYPE_HINTS), (outcome, OUTCOME_HINTS), (ambiguity, AMBIGUITY_HINTS)):
        hint = table.get(str(val).lower())
        if hint:
            parts.append(f"- {hint}")
    return "\n".join(parts) or "- Ask a natural, well-scoped question."


def _read_gold(val) -> List[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(s) for s in val]
    try:
        parsed = json.loads(val)
        return [str(s) for s in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _as_obj(v):
    return v if isinstance(v, (dict, list)) else json.loads(v)


def _looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    return t.endswith("?") or t.lower().startswith(("could you", "can you", "to clarify", "which", "what", "do you"))


def _clarify_answer_msgs(persona, conversation):
    return [{"role": "system", "content": CLARIFY_ANSWER_SYSTEM.format(
                persona=format_persona_for_prompt(persona))},
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
