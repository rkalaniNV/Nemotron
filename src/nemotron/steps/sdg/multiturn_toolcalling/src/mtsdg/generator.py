"""Live separate-agent trajectory generation (no prescripted plan).

For one ``queries.jsonl`` row the LLMs take over and produce the whole trajectory:

- **User Agent** opens with the question (a deliberately natural/vague first
  phrasing) and improvises coherent follow-ups over 20-25 turns.
- **Assistant Agent** decides tool calls and the final answer with a bounded think
  trace (majority-voted over N samples). It runs the genuine retrieve -> assess ->
  (rewrite -> retrieve) -> answer loop against the **live retriever**: a vague query
  returns weak chunks, a precise rewrite returns better ones.
- The **LiveToolExecutor** resolves tool calls against the real retriever + a
  validated memory store. No prescripted plan, no pre-built chunk catalog.

**Automatic context compaction (not a tool call).** A :class:`ContextMeter`
tracks the active context; at the token threshold (32k in production, a smaller
simulation value here) the pipeline silently compacts the prefix into a rolling
summary. Later turns are generated from [summary + recent raw turns] — so they
depend on the compacted context — but neither a ``context.compress`` call nor the
summary appears in the emitted chat. Compaction is recorded only in metadata.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from mtsdg.assembler import assemble_blocks, project_structured_messages, render_summary
from mtsdg.compression import generate_compression_event
from mtsdg.core.llm import (
    call_llm,
    call_structured_n,
    majority_vote_tool_calls,
    run_inline_judge,
)
from mtsdg.generator_config import EpisodeSimulatorConfig
from mtsdg.prompts import (
    ASSISTANT_AGENT_SYSTEM_PROMPT,
    SYSTEM_POLICY,
    USER_AGENT_LIVE_PROMPT,
)
from mtsdg.reasoning import ReasoningContent, reasoning_to_text, validate_reasoning_content
from mtsdg.retriever import RetrieverClient
from mtsdg.runtime import ConversationState, LiveToolExecutor, ToolCall, ToolError
from mtsdg.schemas import (
    MODEL_TOOLS,
    AssistantTurn,
    CompressionEvent,
    Message,
    PersonaSeed,
    QuerySeed,
)
from mtsdg.tokens import ContextMeter
from mtsdg.tools.contracts import TOOL_SCHEMAS
from mtsdg.validators import validate_trajectory

_EMPTY_ANSWER_FALLBACK = (
    "I don't have enough grounded evidence on that from the retrieved sources yet. "
    "Let me know if you'd like me to look at a related angle."
)

#: Serializes checkpoint appends across concurrent episode threads.
_CKPT_LOCK = threading.Lock()


def write_checkpoint(path: str, result: Dict[str, Any]) -> None:
    """Append one completed episode to the checkpoint JSONL the moment it finishes.

    Thread-safe. Best-effort: a checkpoint failure never breaks generation. Runs
    inside the DD generator so partial results survive a crash, kill, or a run that
    never reaches the final ``preview()`` return.
    """
    if not path:
        return
    try:
        meta = json.loads(result.get("episode_metadata") or "{}")
        record = {
            "query_id": meta.get("query_id"),
            "trajectory_status": result.get("trajectory_status"),
            "n_messages": meta.get("n_messages"),
            "n_retrieved_chunks": meta.get("n_retrieved_chunks"),
            "compaction_events": meta.get("compaction_events"),
            "compaction_triggers": meta.get("compaction_triggers"),
            "messages": json.loads(result.get("structured_messages") or "[]"),
            "trajectory_validation": json.loads(result.get("trajectory_validation") or "{}"),
            "trajectory_judgment": json.loads(result.get("trajectory_judgment") or "{}"),
        }
        line = json.dumps(record, ensure_ascii=False)
        with _CKPT_LOCK:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
    except Exception:
        pass  # never let checkpointing break the run


class EpisodeRunner:
    """Plain (DD-independent) orchestrator for one live episode.

    Testable without the DD engine: construct with a config and call
    :meth:`run_episode` with a models dict, a QuerySeed, and a retriever.
    """

    def __init__(self, config: EpisodeSimulatorConfig):
        self.config = config

    # -- orchestration ------------------------------------------------------ #

    def run_episode(
        self, models: Dict[str, Any], query: QuerySeed, retriever: RetrieverClient
    ) -> Dict[str, Any]:
        cfg = self.config
        executor = LiveToolExecutor(retriever, allowed_tools=MODEL_TOOLS, default_top_k=cfg.retrieve_top_k)
        state = ConversationState(conversation_id=query.query_id, memory=dict(query.memory_seed))
        meter = ContextMeter(threshold=cfg.context_token_threshold, min_turns_between=cfg.min_turns_between_compression)

        mid = _MessageIdCounter(start=0)
        system_msg = Message(role="system", content=SYSTEM_POLICY)
        all_messages: List[Message] = [system_msg]
        mid.stamp(all_messages)
        meter.add(system_msg)

        compactions: Dict[str, CompressionEvent] = {}
        prior_summary: Optional[CompressionEvent] = None
        reasoning_errors: List[str] = []
        reasoning_warnings: List[str] = []
        tool_warnings: List[str] = []
        user_ratings: List[Dict[str, Any]] = []

        for turn in range(1, query.turn_budget + 1):
            state.turn = turn
            turn_msgs = self._generate_turn(
                models, query, executor, state, turn, prior_summary,
                all_messages, reasoning_errors, reasoning_warnings, tool_warnings, user_ratings,
            )
            mid.stamp(turn_msgs)
            all_messages.extend(turn_msgs)
            meter.add_all(turn_msgs)

            if turn < query.turn_budget and meter.should_compress(turn):
                event = self._compact(models, query, all_messages, turn, prior_summary, len(compactions))
                if event is not None:
                    compactions[event.summary_id] = event
                    prior_summary = event
                    meter.reset_after_compression(turn, render_summary(event))
                    meter.add_all(_recent_messages(all_messages, cfg.recent_raw_turns))

        assembled = assemble_blocks([all_messages[1:]], system_message=all_messages[0])
        structured_messages = project_structured_messages(assembled)

        report = validate_trajectory(
            [m.to_openai() | _bookkeeping(m) for m in assembled],
            max_reasoning_tokens=cfg.max_reasoning_tokens,
        )
        report.errors.extend(reasoning_errors)
        report.warnings.extend(reasoning_warnings)
        # Dropped/invalid tool calls (e.g. a disallowed memory key) are soft: the
        # call never enters the emitted trajectory, so the corpus stays clean and
        # the trajectory is not failed for them.
        report.warnings.extend(tool_warnings)
        report.ok = report.ok and not reasoning_errors

        judgment: Dict[str, Any] = {"skipped": True}
        if cfg.run_trajectory_judge and report.ok:
            judgment = self._judge_trajectory(models, structured_messages)

        return {
            "structured_messages": json.dumps(structured_messages, ensure_ascii=False),
            "episode_metadata": json.dumps(
                {
                    "query_id": query.query_id,
                    "domain": query.domain,
                    "topic": query.query,
                    "turn_count": query.turn_budget,
                    "context_token_threshold": meter.threshold,
                    "compaction_events": [
                        {"summary_id": e.summary_id, "covers_turns": e.covers_turns}
                        for e in compactions.values()
                    ],
                    "compaction_triggers": meter.history,
                    "n_messages": len(structured_messages),
                    "n_retrieved_chunks": len(state.retrieved),
                    "user_turn_ratings": user_ratings,
                    "transcript": state.transcript,
                    "reasoning_provenance": [
                        {"message_id": m.message_id, "turn": m.turn, "structured": m.reasoning_structured}
                        for m in assembled if m.reasoning_structured
                    ],
                },
                ensure_ascii=False, default=str,
            ),
            "compaction_events": json.dumps(
                {k: v.model_dump() for k, v in compactions.items()}, ensure_ascii=False
            ),
            "trajectory_status": report.ok,
            "trajectory_validation": json.dumps(
                {"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, ensure_ascii=False
            ),
            "trajectory_judgment": json.dumps(judgment, ensure_ascii=False),
        }

    # -- one turn ----------------------------------------------------------- #

    def _generate_turn(
        self, models, query, executor, state, turn, prior_summary,
        all_messages, reasoning_errors, reasoning_warnings, tool_warnings, user_ratings,
    ) -> List[Message]:
        cfg = self.config
        out: List[Message] = []

        view = _conversation_view(all_messages, out, prior_summary, cfg.recent_raw_turns)
        if turn == 1 and (query.naive_query or query.query):
            # Seed the conversation with the user's own (deliberately natural,
            # possibly vague) opening so the assistant's first retrieve is realistic.
            user_text = query.naive_query or query.query
        else:
            user_text = self._user_turn(models, query, view)
        if cfg.run_inline_judge:
            user_ratings.append(self._judge_user_turn(models, view, user_text, turn))
        out.append(Message(role="user", content=user_text, turn=turn))

        produced_final = False
        for _step in range(cfg.max_steps):
            view = _conversation_view(all_messages, out, prior_summary, cfg.recent_raw_turns)
            at = self._assistant_turn(models, state, view)
            kept, tool_msgs = self._materialize_tools(at.tool_calls, executor, state, tool_warnings, turn)
            if kept:
                msg = Message(role="assistant", content="", tool_calls=kept, turn=turn)
                self._apply_reasoning(msg, at.reasoning, state, turn, reasoning_errors, reasoning_warnings, strict=False)
                out.append(msg)
                out.extend(tool_msgs)
                continue
            content = (at.content or "").strip()
            # A model whose structured output needed a JSON repair sometimes leaks
            # meta/apology text ("...conform to the JSON schema...") into `content`.
            # Treat that like an empty answer and force a real grounded one.
            if not content or _is_meta_answer(content):
                forced = view + [{"role": "user", "content":
                    "Please answer the user's question now, concisely, using the retrieved sources "
                    "(cite chunk ids). Do not mention JSON, schemas, or formatting."}]
                at2 = self._assistant_turn(models, state, forced)
                c2 = (at2.content or "").strip()
                if c2 and not at2.tool_calls and not _is_meta_answer(c2):
                    at, content = at2, c2
                elif _is_meta_answer(content):
                    content = ""  # drop the leaked meta text
            if not content:
                content = _EMPTY_ANSWER_FALLBACK
            msg = Message(role="assistant", content=content, turn=turn)
            self._apply_reasoning(msg, at.reasoning, state, turn, reasoning_errors, reasoning_warnings, strict=True)
            out.append(msg)
            produced_final = True
            break
        if not produced_final:
            out.append(Message(role="assistant", turn=turn, content=_EMPTY_ANSWER_FALLBACK))
        return out

    # -- agents ------------------------------------------------------------- #

    def _user_turn(self, models, query: QuerySeed, view: List[Dict[str, Any]]) -> str:
        system = USER_AGENT_LIVE_PROMPT.format(
            persona=_format_persona(query.persona), topic=query.query, turn_budget=query.turn_budget,
        )
        inverted: List[Dict[str, Any]] = []
        for m in view:
            role = m.get("role")
            if role == "user":
                inverted.append({"role": "assistant", "content": m.get("content", "")})
            elif role == "assistant" and m.get("content"):
                inverted.append({"role": "user", "content": m.get("content", "")})
        if not inverted or inverted[-1].get("role") != "user":
            inverted.append({"role": "user", "content": "(continue the conversation)"})
        msgs = [{"role": "system", "content": system}] + inverted
        resp = call_llm(models, self.config.user_alias, msgs)
        return (resp.get("content", "") if isinstance(resp, dict) else "").strip()

    def _assistant_turn(self, models, state: ConversationState, view: List[Dict[str, Any]]) -> AssistantTurn:
        cfg = self.config
        system = ASSISTANT_AGENT_SYSTEM_PROMPT.format(
            tool_schemas=json.dumps({n: TOOL_SCHEMAS[n] for n in MODEL_TOOLS}, ensure_ascii=False),
            max_reasoning_tokens=cfg.max_reasoning_tokens,
            assistant_schema=json.dumps(AssistantTurn.model_json_schema(), ensure_ascii=False),
        )
        # No per-turn dump of all retrieved chunks: after compaction the agent
        # works from the compacted summary + recent raw turns (which already carry
        # the recent retrieve results). If it needs an older passage's specifics it
        # RE-RETRIEVES — matching the real compaction semantics and keeping context
        # bounded.
        msgs = [{"role": "system", "content": system}] + view
        # response_format is omitted: the gpt-5.5 proxy 400s on json_object; the
        # schema is already in the prompt and the model returns valid JSON.
        candidates = call_structured_n(
            models, cfg.assistant_alias, msgs, AssistantTurn, cfg.majority_vote_n, response_format=None
        )
        if candidates:
            return majority_vote_tool_calls(candidates)

        plain = call_llm(models, cfg.assistant_alias, msgs + [{
            "role": "user",
            "content": "Answer the user's last question concisely using the retrieved sources "
                       "above (cite chunk ids). Write only the answer, not JSON.",
        }])
        text = (plain.get("content", "") if isinstance(plain, dict) else "").strip()
        if text:
            return AssistantTurn(
                content=text, tool_calls=[],
                reasoning=ReasoningContent(task_understanding="answer from retrieved sources", think=text[:300]),
            )
        return AssistantTurn(content=_EMPTY_ANSWER_FALLBACK, reasoning=ReasoningContent(think="sources limited"))

    def _materialize_tools(self, tool_calls, executor, state, tool_warnings, turn) -> Tuple[List, List[Message]]:
        kept: List[Dict[str, Any]] = []
        tool_msgs: List[Message] = []
        for i, raw in enumerate(tool_calls or []):
            tc = _normalize_tool_call(raw, f"{turn}-{i}")
            if tc is None:
                continue
            name = tc["function"]["name"]
            if name == "context.compress":
                continue  # compaction is automatic; the model never calls it
            try:
                result = executor.execute(ToolCall.from_openai(tc), state)
            except ToolError as exc:
                # Drop the invalid call (kept out of the trajectory) and record a
                # soft warning — do not fail the whole trajectory.
                tool_warnings.append(f"turn {turn} tool `{name}`: {exc}")
                continue
            kept.append(tc)
            tool_msgs.append(Message(**result.to_message(), turn=turn))
        return kept, tool_msgs

    def _apply_reasoning(self, msg, reasoning, state, turn, errors, warnings, *, strict) -> None:
        returned = set(state.retrieved.keys())
        if strict:
            v = validate_reasoning_content(reasoning, returned, max_tokens=self.config.max_reasoning_tokens)
            if not v.ok:
                errors.extend(f"turn {turn} reasoning: {e}" for e in v.errors)
            warnings.extend(f"turn {turn} reasoning: {w}" for w in v.warnings)
        msg.reasoning_structured = reasoning.model_dump()
        think = reasoning.think.strip()
        msg.reasoning_content = (think or reasoning_to_text(reasoning)) if strict else (think or None)

    # -- judges ------------------------------------------------------------- #

    def _judge_user_turn(self, models, view, user_text, turn) -> Dict[str, Any]:
        prompt = (
            "Judge whether this simulated user message is realistic, in-character, and an "
            "answerable information request.\n\nCONVERSATION:\n"
            + json.dumps(view, ensure_ascii=False)
            + f"\n\nUSER MESSAGE:\n{user_text}\n\n"
            "Respond with <explanation>...</explanation> and <rating>success|failure</rating>."
        )
        expl, rating, ok = run_inline_judge(models, self.config.judge_alias, prompt)
        return {"turn": turn, "rating": rating, "success": ok, "explanation": expl}

    def _judge_trajectory(self, models, structured_messages) -> Dict[str, Any]:
        from mtsdg.core.llm import call_structured
        from mtsdg.prompts import TRAJECTORY_JUDGE_PROMPT
        from mtsdg.schemas import TrajectoryJudgment

        prompt = TRAJECTORY_JUDGE_PROMPT.replace(
            "{{ structured_messages }}", json.dumps(structured_messages, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": "You are a strict but fair SDG trajectory judge. Return only JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            verdict = call_structured(models, self.config.judge_alias, messages, TrajectoryJudgment, response_format=None)
        except Exception as exc:
            return {"skipped": True, "reason": f"judge error: {exc}"}
        out = verdict.model_dump()
        out["success"] = verdict.rating == "success"
        return out

    # -- automatic compaction (internal; no chat artifacts) ----------------- #

    def _compact(self, models, query, all_messages, checkpoint, prior_summary, n_prior) -> Optional[CompressionEvent]:
        cfg = self.config
        summary_id = f"ctx-{n_prior + 1:03d}"
        from_turn = (prior_summary.covers_turns[1] + 1) if prior_summary else 1
        completed = [m.to_openai() | {"message_id": m.message_id} for m in all_messages]
        try:
            event, report = generate_compression_event(
                models, cfg.compressor_alias,
                completed_messages=completed, from_turn=from_turn, checkpoint_turn=checkpoint,
                summary_id=summary_id, prior_summary=prior_summary, token_budget=cfg.compression_token_budget,
            )
        except Exception:
            report, event = None, None
        if event is None or (report is not None and not report.ok):
            event = CompressionEvent(summary_id=summary_id, covers_turns=[from_turn, checkpoint], no_new_claims=True)
        return event


# --------------------------------------------------------------------------- #
# Data Designer column-generator wrapper
# --------------------------------------------------------------------------- #


class EpisodeSimulatorGenerator(
    ColumnGeneratorCellByCell[EpisodeSimulatorConfig],
    ColumnGeneratorWithModelRegistry[EpisodeSimulatorConfig],
):
    """DD column generator: resolves model aliases, builds the retriever client,
    and delegates to EpisodeRunner."""

    def generate(self, data: dict) -> dict:
        cfg = self.config
        aliases = [cfg.user_alias, cfg.assistant_alias, cfg.judge_alias, cfg.compressor_alias]
        # Use a direct synchronous HTTP client with a bounded read timeout, so a
        # backend that accepts a request but never responds fails fast instead of
        # blocking the run. Falls back to DD's model registry if LLM_API_KEY isn't
        # set (e.g. the mock/engine tests).
        if os.environ.get("MTSDG_DIRECT_HTTP", "1") != "0" and os.environ.get("LLM_API_KEY"):
            from mtsdg.model_configs import direct_facades_from_env
            models = direct_facades_from_env(aliases)
        else:
            models = {alias: self.get_model(alias) for alias in set(aliases)}
        try:
            query = _parse_query(data[cfg.episode_input_column])
            retriever = RetrieverClient(cfg.retriever_url)
        except Exception as exc:
            result = _skip_result(f"input parse error: {exc}")
        else:
            try:
                result = EpisodeRunner(cfg).run_episode(models, query, retriever)
            except Exception as exc:
                result = _skip_result(f"generation error: {exc}")
        # Checkpoint this episode immediately (survives crash/kill/long runs).
        write_checkpoint(cfg.checkpoint_path, result)
        data.update(result)
        data[cfg.name] = result["structured_messages"]
        return data


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class _MessageIdCounter:
    def __init__(self, start: int = 0):
        self.i = start

    def stamp(self, messages: List[Message]) -> None:
        for m in messages:
            if not m.message_id:
                m.message_id = f"m-{self.i:02d}"
            self.i += 1


def _conversation_view(all_messages, current_out, prior_summary, recent_turns) -> List[Dict[str, Any]]:
    view: List[Dict[str, Any]] = []
    if prior_summary is not None:
        view.append({"role": "system", "content": "[compacted context] " + render_summary(prior_summary)})
    for m in _recent_messages(all_messages, recent_turns):
        if m.role != "system":
            view.append(m.to_openai())
    for m in current_out:
        view.append(m.to_openai())
    return view


def _recent_messages(messages, n_turns) -> List[Message]:
    turns = sorted({m.turn for m in messages if m.turn is not None})
    keep = set(turns[-n_turns:]) if turns else set()
    return [m for m in messages if m.turn in keep]


def _format_persona(p: PersonaSeed) -> str:
    return f"role={p.role}, expertise={p.expertise}, style={p.style or 'neutral'}"


_META_MARKERS = (
    "json schema", "json object", "required schema", "conform to", "the required format",
    "reformat", "formatting error", "match the schema", "valid json", "moving forward",
    "i will ensure my responses", "apolog",
)


def _is_meta_answer(text: str) -> bool:
    """True if the answer text is meta-commentary about JSON/formatting (a leaked
    schema-repair apology) rather than a real user-facing answer."""
    if not text:
        return False
    t = text.lower()
    return any(m in t for m in _META_MARKERS)


def _normalize_tool_call(raw: Any, idx: str) -> Optional[Dict[str, Any]]:
    """Coerce a model-emitted tool call into canonical OpenAI shape.

    Teacher models vary: some emit the nested ``{"function":{"name","arguments"}}``
    form, others a flat ``{"name"/"tool", "arguments"/"parameters"}``. Normalize
    both so the executor and the exported messages are consistent; arguments are
    always a JSON string.
    """
    if not isinstance(raw, dict):
        return None
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else None
    name = (fn or {}).get("name") or raw.get("name") or raw.get("tool") or raw.get("tool_name")
    if not name or not isinstance(name, str):
        return None
    args = (fn or {}).get("arguments")
    if args is None:
        args = raw.get("arguments") or raw.get("args") or raw.get("parameters") or {}
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": raw.get("id") or f"call-{idx}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _bookkeeping(m: Message) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if m.turn is not None:
        out["turn"] = m.turn
    if m.name:
        out["name"] = m.name
    return out


def _parse_query(value: Any) -> QuerySeed:
    if isinstance(value, QuerySeed):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    return QuerySeed.model_validate(value)


def _skip_result(reason: str) -> Dict[str, Any]:
    return {
        "structured_messages": "[]",
        "episode_metadata": json.dumps({"skipped": True, "reason": reason}),
        "compaction_events": "{}",
        "trajectory_status": False,
        "trajectory_validation": json.dumps({"ok": False, "errors": [reason], "warnings": []}),
        "trajectory_judgment": json.dumps({"skipped": True, "reason": reason}),
    }
