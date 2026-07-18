"""DD-independent long-context episode runtime."""

from __future__ import annotations

import json
from typing import Any

from .compression import generate_compression, render_summary
from .config import PipelineConfig
from .executors.base import ConversationState
from .llm import call_llm, call_structured
from .planning import plan_episode
from .prompts import (
    assistant_final_system,
    assistant_retrieval_system,
    assistant_system,
    assistant_turn_directive,
    user_system,
)
from .reasoning import validate_reasoning
from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    AssistantRetrievalAction,
    CanonicalRecord,
    CompressionEvent,
    EpisodePlan,
    EpisodeSeed,
    Message,
    TrajectoryJudgment,
)
from .tokens import ContextMeter
from .tool_registry import ToolRegistry, normalize_tool_call
from .validation import normalize_query, validate_trajectory


class EpisodeGenerationError(RuntimeError):
    pass


class EpisodeRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        models: dict[str, Any],
        seed: EpisodeSeed,
        registry: ToolRegistry,
        *,
        run_id: str,
    ) -> CanonicalRecord:
        plan = plan_episode(seed, self.config.planning, self.config.run.seed)
        try:
            return self._run(models, seed, plan, registry, run_id=run_id)
        except Exception as exc:
            return CanonicalRecord(
                run_id=run_id,
                config_fingerprint=self.config.fingerprint(),
                query_id=seed.query_id,
                status="generation_failed",
                tools=registry.schemas,
                episode_plan=plan.model_dump(),
                metadata={
                    "query": seed.query,
                    "instructions": seed.instructions,
                    "turn_budget": seed.turn_budget,
                    "retrieval_depth": seed.retrieval_depth,
                },
                validation={"ok": False, "errors": [str(exc)], "warnings": []},
            )

    def _run(
        self,
        models: dict[str, Any],
        seed: EpisodeSeed,
        plan: EpisodePlan,
        registry: ToolRegistry,
        *,
        run_id: str,
    ) -> CanonicalRecord:
        cfg = self.config
        state = ConversationState(
            conversation_id=seed.query_id,
            memory=dict(seed.memory_seed),
        )
        system = Message(
            role="system", content=assistant_system(seed, registry.schemas), turn=0
        )
        messages = [system]
        self._stamp(messages)
        meter = ContextMeter(
            cfg.context.compression_threshold, cfg.context.min_turns_between_compression
        )
        meter.add_all(messages)
        prior_summary: CompressionEvent | None = None
        compactions: list[CompressionEvent] = []
        reasoning_errors: list[str] = []
        warnings: list[str] = []

        for turn_plan in plan.turns:
            state.turn = turn_plan.turn
            before = len(messages)
            user_text = (
                seed.naive_query
                if turn_plan.turn == 1
                else self._next_user(
                    models,
                    seed,
                    turn_plan.intent,
                    self._view(messages, prior_summary, cfg.context.recent_raw_turns),
                )
            )
            if not user_text.strip():
                raise EpisodeGenerationError(
                    f"user model returned an empty message at turn {turn_plan.turn}"
                )
            messages.append(
                Message(role="user", content=user_text.strip(), turn=turn_plan.turn)
            )
            self._assistant_turn(
                models,
                seed,
                turn_plan,
                messages,
                prior_summary,
                registry,
                state,
                reasoning_errors,
                warnings,
            )
            self._stamp(messages)
            meter.add_all(messages[before:])

            if turn_plan.turn < seed.turn_budget and meter.should_compress(
                turn_plan.turn
            ):
                try:
                    from_turn = (
                        (prior_summary.covers_turns[1] + 1) if prior_summary else 1
                    )
                    event = generate_compression(
                        models,
                        messages,
                        from_turn=from_turn,
                        to_turn=turn_plan.turn,
                        summary_id=f"ctx-{len(compactions) + 1:03d}",
                        known_chunk_ids=state.retrieved,
                        prior=prior_summary,
                        instructions=seed.instructions,
                        token_budget=cfg.context.compression_token_budget,
                    )
                except Exception as exc:
                    warnings.append(
                        f"compression at turn {turn_plan.turn} failed: {exc}"
                    )
                    if meter.active_tokens >= cfg.context.model_token_limit:
                        raise EpisodeGenerationError(
                            f"compression failed at {meter.active_tokens} active tokens, at or above model limit"
                        ) from exc
                else:
                    compactions.append(event)
                    prior_summary = event
                    recent = self._recent(messages, cfg.context.recent_raw_turns)
                    meter.reset(turn_plan.turn, render_summary(event), recent)

        report = validate_trajectory(
            messages,
            plan=plan,
            retrieval_transcript=state.retrieval_transcript,
            tool_schemas=registry.schemas,
            require_final_answer_each_turn=cfg.validation.require_final_answer_each_turn,
        )
        report.errors.extend(reasoning_errors)
        report.warnings.extend(warnings)
        report.warnings.extend(
            f"dropped tool call at turn {x.get('turn')}: {x.get('error')}"
            for x in state.rejected_tool_calls
        )
        report.ok = not report.errors

        status = "accepted" if report.ok else "rejected"
        judgment: dict[str, Any] = {"enabled": cfg.judge.enabled, "skipped": True}
        if report.ok and cfg.judge.enabled:
            try:
                verdict = self._judge(models, seed, messages, registry.schemas)
                judgment = verdict.model_dump()
                missing = sorted(set(cfg.judge.dimensions) - set(verdict.scores))
                below = {
                    k: verdict.scores.get(k, 0)
                    for k in cfg.judge.dimensions
                    if verdict.scores.get(k, 0) < cfg.judge.min_score
                }
                if missing or below or verdict.rating != "success":
                    status = "rejected"
                    judgment["gate_errors"] = {
                        "missing_dimensions": missing,
                        "below_threshold": below,
                    }
            except Exception as exc:
                status = "quarantine"
                judgment = {"enabled": True, "pending": True, "error": str(exc)}

        projected = [m.to_openai() for m in messages]
        return CanonicalRecord(
            run_id=run_id,
            config_fingerprint=cfg.fingerprint(),
            query_id=seed.query_id,
            status=status,
            messages=projected,
            tools=registry.schemas,
            episode_plan=plan.model_dump(),
            metadata={
                "query": seed.query,
                "instructions": seed.instructions,
                "turn_budget": seed.turn_budget,
                "retrieval_depth": seed.retrieval_depth,
                "n_messages": len(messages),
                "n_retrieved_chunks": len(state.retrieved),
                "context_history": meter.history,
                "message_turns": [m.turn for m in messages],
                "rejected_tool_calls": state.rejected_tool_calls,
            },
            retrieval_transcript=state.retrieval_transcript,
            memory_events=state.memory_events,
            compaction_events=[x.model_dump() for x in compactions],
            validation=report.model_dump(),
            judgment=judgment,
        )

    def _assistant_turn(
        self,
        models,
        seed,
        turn_plan,
        messages,
        prior_summary,
        registry,
        state,
        reasoning_errors,
        warnings,
    ) -> None:
        correction = ""
        force_final = False
        for step in range(self.config.planning.max_steps_per_turn):
            completed = self._completed_retrievals(
                state.retrieval_transcript, turn_plan.turn
            )
            view = self._view(
                messages, prior_summary, self.config.context.recent_raw_turns
            )
            directive = assistant_turn_directive(turn_plan, completed)
            if correction:
                directive += "\n" + correction
            final_only = force_final or (
                turn_plan.retrieval_required and completed >= turn_plan.retrieval_depth
            )
            needs_retrieval = (
                turn_plan.retrieval_required and completed < turn_plan.retrieval_depth
            )
            if needs_retrieval:
                retrieval = call_structured(
                    models,
                    "assistant",
                    [
                        {
                            "role": "system",
                            "content": assistant_retrieval_system(seed),
                        },
                        *view,
                        {"role": "user", "content": directive},
                    ],
                    AssistantRetrievalAction,
                )
                reasoning_report = validate_reasoning(
                    retrieval.reasoning,
                    state.retrieved,
                    max_tokens=self.config.context.max_reasoning_tokens,
                )
                reasoning_errors.extend(
                    f"turn {turn_plan.turn}: {x}" for x in reasoning_report.errors
                )
                warnings.extend(
                    f"turn {turn_plan.turn}: {x}" for x in reasoning_report.warnings
                )
                raw = {
                    "name": "retrieve",
                    "arguments": {"query": retrieval.query.strip()},
                }
                try:
                    call = normalize_tool_call(raw, f"call-{turn_plan.turn}-{step}-0")
                    result = registry.execute(call, state, seed.instructions)
                except Exception as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn_plan.turn,
                            "step": step,
                            "raw": raw,
                            "error": str(exc),
                        }
                    )
                    correction = "The prior retrieval query failed. Return a different, valid, nonempty query."
                    continue
                messages.append(
                    Message(
                        role="assistant",
                        content="",
                        reasoning_content=retrieval.reasoning.think or None,
                        tool_calls=[call.to_openai()],
                        turn=turn_plan.turn,
                    )
                )
                messages.append(
                    Message.model_validate(
                        result.to_message() | {"turn": turn_plan.turn}
                    )
                )
                completed = self._completed_retrievals(
                    state.retrieval_transcript, turn_plan.turn
                )
                correction = (
                    f"Complete {turn_plan.retrieval_depth - completed} additional successful retrieve "
                    "call(s) with distinct rewritten queries before answering."
                    if completed < turn_plan.retrieval_depth
                    else "The required evidence is available. Synthesize the final answer."
                )
                continue
            if final_only:
                final = call_structured(
                    models,
                    "assistant",
                    [
                        {
                            "role": "system",
                            "content": assistant_final_system(
                                seed, state.retrieved.keys()
                            ),
                        },
                        *view,
                        {"role": "user", "content": directive},
                    ],
                    AssistantFinalAction,
                )
                messages.append(
                    Message(
                        role="assistant",
                        content=final.content.strip(),
                        reasoning_content=final.reasoning.think or None,
                        turn=turn_plan.turn,
                    )
                )
                return
            action = call_structured(
                models,
                "assistant",
                [
                    {
                        "role": "system",
                        "content": assistant_system(
                            seed, registry.schemas, state.retrieved.keys()
                        ),
                    },
                    *view,
                    {"role": "user", "content": directive},
                ],
                AssistantAction,
            )

            reasoning_report = validate_reasoning(
                action.reasoning,
                state.retrieved,
                max_tokens=self.config.context.max_reasoning_tokens,
            )
            if reasoning_report.errors:
                correction = (
                    "Your reasoning cited an unavailable chunk ID. Retry using only these exact full IDs: "
                    + json.dumps(sorted(state.retrieved))
                )
                warnings.extend(
                    f"turn {turn_plan.turn}: corrected invalid reasoning citation: {x}"
                    for x in reasoning_report.errors
                )
                continue
            warnings.extend(
                f"turn {turn_plan.turn}: {x}" for x in reasoning_report.warnings
            )

            calls = []
            results = []
            for index, raw in enumerate(action.tool_calls):
                try:
                    call = normalize_tool_call(
                        raw, f"call-{turn_plan.turn}-{step}-{index}"
                    )
                    result = registry.execute(call, state, seed.instructions)
                except Exception as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn_plan.turn,
                            "step": step,
                            "raw": raw,
                            "error": str(exc),
                        }
                    )
                    continue
                calls.append(call.to_openai())
                results.append(result)
            if calls:
                messages.append(
                    Message(
                        role="assistant",
                        content="",
                        reasoning_content=action.reasoning.think or None,
                        tool_calls=calls,
                        turn=turn_plan.turn,
                    )
                )
                messages.extend(
                    Message.model_validate(
                        result.to_message() | {"turn": turn_plan.turn}
                    )
                    for result in results
                )
                completed = self._completed_retrievals(
                    state.retrieval_transcript, turn_plan.turn
                )
                if (
                    turn_plan.retrieval_required
                    and completed < turn_plan.retrieval_depth
                ):
                    correction = (
                        f"Complete {turn_plan.retrieval_depth - completed} additional successful retrieve "
                        "call(s) with distinct rewritten queries before answering."
                    )
                else:
                    force_final = True
                    correction = (
                        "Tool results are now available. Synthesize a substantive final answer without "
                        "another tool call."
                    )
                continue

            completed = self._completed_retrievals(
                state.retrieval_transcript, turn_plan.turn
            )
            if turn_plan.retrieval_required and completed < turn_plan.retrieval_depth:
                correction = (
                    f"Do not answer yet. Complete {turn_plan.retrieval_depth - completed} additional successful "
                    "retrieve call(s) using distinct rewritten queries."
                )
                continue
            if action.content.strip():
                messages.append(
                    Message(
                        role="assistant",
                        content=action.content.strip(),
                        reasoning_content=action.reasoning.think or None,
                        turn=turn_plan.turn,
                    )
                )
                return
            correction = "Provide a substantive final answer now, without a tool call."
        turn_rejections = [
            item
            for item in state.rejected_tool_calls
            if item.get("turn") == turn_plan.turn
        ]
        raise EpisodeGenerationError(
            f"turn {turn_plan.turn} did not produce a valid final answer within "
            f"{self.config.planning.max_steps_per_turn} steps; completed retrievals="
            f"{self._completed_retrievals(state.retrieval_transcript, turn_plan.turn)}/"
            f"{turn_plan.retrieval_depth}; rejected calls="
            f"{json.dumps(turn_rejections, ensure_ascii=False)}"
        )

    def _next_user(self, models, seed, intent: str, view: list[dict[str, Any]]) -> str:
        response = call_llm(
            models,
            "user",
            [
                {"role": "system", "content": user_system(seed)},
                *view,
                {
                    "role": "user",
                    "content": f"Write the next user message. Planned intent: {intent}. Output only that message.",
                },
            ],
        )
        return str(response.get("content", "")).strip()

    def _judge(self, models, seed, messages, tools) -> TrajectoryJudgment:
        dimensions = self.config.judge.dimensions
        prompt = (
            "Judge this synthetic long-context trajectory. Return JSON only. Score each requested dimension 1-5. "
            "Rating is success only when the trajectory follows the effective instructions and is suitable for SFT.\n"
            f"Effective instructions: {seed.instructions}\n"
            f"Dimensions: {json.dumps(dimensions)}\n"
            f"Tools: {json.dumps(tools, ensure_ascii=False)}\n"
            f"Messages: {json.dumps([m.to_openai() for m in messages], ensure_ascii=False)}\n"
            f"Schema: {json.dumps(TrajectoryJudgment.model_json_schema(), ensure_ascii=False)}"
        )
        return call_structured(
            models,
            "judge",
            [
                {
                    "role": "system",
                    "content": "You are a strict synthetic-data quality judge.",
                },
                {"role": "user", "content": prompt},
            ],
            TrajectoryJudgment,
        )

    @staticmethod
    def _completed_retrievals(transcript: list[dict[str, Any]], turn: int) -> int:
        queries = {
            normalize_query(str(row.get("query", "")))
            for row in transcript
            if row.get("turn") == turn and row.get("success") and row.get("chunk_ids")
        }
        queries.discard("")
        return len(queries)

    @staticmethod
    def _recent(messages: list[Message], n_turns: int) -> list[Message]:
        turns = sorted({m.turn for m in messages if m.turn and m.turn > 0})
        keep = set(turns[-n_turns:])
        return [m for m in messages if m.turn in keep]

    def _view(
        self,
        messages: list[Message],
        summary: CompressionEvent | None,
        recent_turns: int,
    ) -> list[dict[str, Any]]:
        if summary is None:
            return [m.to_openai() for m in messages if m.turn and m.turn > 0]
        return [
            {"role": "system", "content": render_summary(summary)},
            *(m.to_openai() for m in self._recent(messages, recent_turns)),
        ]

    @staticmethod
    def _stamp(messages: list[Message]) -> None:
        for index, message in enumerate(messages):
            if not message.message_id:
                message.message_id = f"m-{index:04d}"
