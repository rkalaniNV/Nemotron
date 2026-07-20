"""DD-independent long-context episode runtime."""

from __future__ import annotations

import json
from typing import Any

from .compression import generate_compression, render_summary
from .config import PipelineConfig
from .episode_control import (
    build_episode_spec,
    retrieval_deadline_event,
)
from .executors.base import ConversationState
from .llm import call_structured
from .prompts import (
    assistant_final_system,
    assistant_retrieval_system,
    assistant_system,
    assistant_turn_directive,
    user_system,
    user_turn_prompt,
)
from .reasoning import validate_reasoning
from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    AssistantRetrievalAction,
    CanonicalRecord,
    CompressionEvent,
    EpisodeSeed,
    EpisodeSpec,
    Message,
    RetrievalPolicyEvent,
    ToolCall,
    ToolResult,
    TrajectoryJudgment,
    UserTurn,
)
from .tokens import ContextMeter
from .tool_registry import ToolRegistry, normalize_tool_call
from .validation import normalize_query, validate_trajectory


class EpisodeGenerationError(RuntimeError):
    pass


class ToolBudgetExceededError(EpisodeGenerationError):
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
        spec = build_episode_spec(seed, self.config.episode, self.config.run.seed)
        policy_events: list[RetrievalPolicyEvent] = []
        state = ConversationState(
            conversation_id=seed.query_id,
            memory=dict(seed.memory_seed),
        )
        try:
            return self._run(
                models,
                seed,
                spec,
                policy_events,
                state,
                registry,
                run_id=run_id,
            )
        except Exception as exc:
            return CanonicalRecord(
                run_id=run_id,
                config_fingerprint=self.config.fingerprint(),
                query_id=seed.query_id,
                status="generation_failed",
                tools=registry.schemas,
                episode_spec=spec.model_dump(),
                policy_events=[event.model_dump() for event in policy_events],
                tool_call_attempts=state.tool_call_attempts,
                metadata={
                    "query": seed.query,
                    "persona": seed.persona.model_dump(),
                    "query_provenance": seed.query_provenance.model_dump() if seed.query_provenance else None,
                    "instructions": seed.instructions,
                    "turn_budget": seed.turn_budget,
                    "retrieval_depth": seed.retrieval_depth,
                    "required_retrieval_calls": spec.required_retrieval_calls,
                },
                retrieval_transcript=state.retrieval_transcript,
                memory_events=state.memory_events,
                validation={"ok": False, "errors": [str(exc)], "warnings": []},
            )

    def _run(
        self,
        models: dict[str, Any],
        seed: EpisodeSeed,
        spec: EpisodeSpec,
        policy_events: list[RetrievalPolicyEvent],
        state: ConversationState,
        registry: ToolRegistry,
        *,
        run_id: str,
    ) -> CanonicalRecord:
        cfg = self.config
        system = Message(role="system", content=assistant_system(seed, registry.schemas), turn=0)
        messages = [system]
        self._stamp(messages)
        meter = ContextMeter(cfg.context.compression_threshold, cfg.context.min_turns_between_compression)
        meter.add_all(messages)
        prior_summary: CompressionEvent | None = None
        compactions: list[CompressionEvent] = []
        reasoning_errors: list[str] = []
        warnings: list[str] = []

        for turn in range(1, spec.turn_budget + 1):
            state.turn = turn
            before = len(messages)
            successful_retrievals = self._completed_retrievals(state.retrieval_transcript)
            retrieval_attempts = self._retrieval_attempt_count(state)
            tool_calls = len(state.tool_call_attempts)
            if turn == 1:
                user_text = seed.naive_query
            else:
                proposal = self._next_user(
                    models,
                    seed,
                    self._view(messages, prior_summary, cfg.context.recent_raw_turns),
                    turn=turn,
                    spec=spec,
                )
                user_text = proposal.content
            policy_event = retrieval_deadline_event(
                spec,
                seed,
                turn=turn,
                successful_retrievals=successful_retrievals,
                retrieval_attempts=retrieval_attempts,
                tool_calls=tool_calls,
            )
            if policy_event is not None:
                policy_events.append(policy_event)
            if not user_text.strip():
                raise EpisodeGenerationError(f"user model returned an empty message at turn {turn}")
            messages.append(Message(role="user", content=user_text.strip(), turn=turn))
            self._assistant_turn(
                models,
                seed,
                spec,
                policy_event,
                messages,
                prior_summary,
                registry,
                state,
                reasoning_errors,
                warnings,
            )
            self._stamp(messages)
            meter.add_all(messages[before:])

            if turn < spec.turn_budget and meter.should_compress(turn):
                try:
                    from_turn = (prior_summary.covers_turns[1] + 1) if prior_summary else 1
                    event = generate_compression(
                        models,
                        messages,
                        from_turn=from_turn,
                        to_turn=turn,
                        summary_id=f"ctx-{len(compactions) + 1:03d}",
                        known_chunk_ids=state.retrieved,
                        prior=prior_summary,
                        instructions=seed.instructions,
                        token_budget=cfg.context.compression_token_budget,
                    )
                except Exception as exc:
                    warnings.append(f"compression at turn {turn} failed: {exc}")
                    if meter.active_tokens >= cfg.context.model_token_limit:
                        raise EpisodeGenerationError(
                            f"compression failed at {meter.active_tokens} active tokens, at or above model limit"
                        ) from exc
                else:
                    compactions.append(event)
                    prior_summary = event
                    recent = self._recent(messages, cfg.context.recent_raw_turns)
                    meter.reset(turn, render_summary(event), recent)

        report = validate_trajectory(
            messages,
            spec=spec,
            policy_events=policy_events,
            retrieval_transcript=state.retrieval_transcript,
            tool_call_attempts=state.tool_call_attempts,
            tool_schemas=registry.schemas,
            require_final_answer_each_turn=cfg.validation.require_final_answer_each_turn,
        )
        report.errors.extend(reasoning_errors)
        report.warnings.extend(warnings)
        report.warnings.extend(
            f"dropped tool call at turn {x.get('turn')}: {x.get('error')}" for x in state.rejected_tool_calls
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
            episode_spec=spec.model_dump(),
            policy_events=[event.model_dump() for event in policy_events],
            tool_call_attempts=state.tool_call_attempts,
            metadata={
                "query": seed.query,
                "persona": seed.persona.model_dump(),
                "query_provenance": seed.query_provenance.model_dump() if seed.query_provenance else None,
                "instructions": seed.instructions,
                "turn_budget": seed.turn_budget,
                "retrieval_depth": seed.retrieval_depth,
                "required_retrieval_calls": spec.required_retrieval_calls,
                "successful_retrieval_calls": self._completed_retrievals(state.retrieval_transcript),
                "tool_call_count": len(state.tool_call_attempts),
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
        spec,
        policy_event,
        messages,
        prior_summary,
        registry,
        state,
        reasoning_errors,
        warnings,
    ) -> None:
        correction = ""
        force_final = False
        turn = state.turn
        required_depth = policy_event.required_retrievals_this_turn if policy_event else 0
        for step in range(self.config.episode.max_steps_per_turn):
            completed = self._completed_retrievals(state.retrieval_transcript, turn)
            view = self._view(messages, prior_summary, self.config.context.recent_raw_turns)
            directive = assistant_turn_directive(turn, policy_event, completed)
            if correction:
                directive += "\n" + correction
            needs_retrieval = completed < required_depth
            has_tool_capacity = (
                self._tool_calls_this_turn(state, turn) < spec.max_tool_calls_per_turn
                and len(state.tool_call_attempts) < spec.max_tool_calls_per_conversation
            )
            final_only = (
                force_final
                or (policy_event is not None and completed >= required_depth)
                or (not needs_retrieval and not has_tool_capacity)
            )
            if needs_retrieval:
                if not self._has_retrieval_capacity(spec, state, turn):
                    raise EpisodeGenerationError(
                        f"turn {turn} cannot satisfy its retrieval deadline within the remaining tool budgets"
                    )
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
                reasoning_errors.extend(f"turn {turn}: {x}" for x in reasoning_report.errors)
                warnings.extend(f"turn {turn}: {x}" for x in reasoning_report.warnings)
                raw = {
                    "name": "retrieve",
                    "arguments": {"query": retrieval.query.strip()},
                }
                try:
                    call = normalize_tool_call(raw, f"call-{turn}-{step}-0")
                    result = self._execute_tool(call, registry, state, seed, spec, step=step)
                except ToolBudgetExceededError:
                    raise
                except Exception as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn,
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
                        turn=turn,
                    )
                )
                messages.append(Message.model_validate(result.to_message() | {"turn": turn}))
                completed = self._completed_retrievals(state.retrieval_transcript, turn)
                correction = (
                    f"Complete {required_depth - completed} additional "
                    "successful retrieve call(s) with distinct rewritten queries "
                    "before answering."
                    if completed < required_depth
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
                            "content": assistant_final_system(seed, state.retrieved.keys()),
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
                        turn=turn,
                    )
                )
                return
            action = call_structured(
                models,
                "assistant",
                [
                    {
                        "role": "system",
                        "content": assistant_system(seed, registry.schemas, state.retrieved.keys()),
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
                    "Your reasoning cited an unavailable chunk ID. Retry using only "
                    "these exact full IDs: " + json.dumps(sorted(state.retrieved))
                )
                warnings.extend(
                    f"turn {turn}: corrected invalid reasoning citation: {x}" for x in reasoning_report.errors
                )
                continue
            warnings.extend(f"turn {turn}: {x}" for x in reasoning_report.warnings)

            calls = []
            results = []
            for index, raw in enumerate(action.tool_calls):
                try:
                    call = normalize_tool_call(raw, f"call-{turn}-{step}-{index}")
                    result = self._execute_tool(call, registry, state, seed, spec, step=step)
                except ToolBudgetExceededError as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn,
                            "step": step,
                            "raw": raw,
                            "error": str(exc),
                        }
                    )
                    force_final = True
                    correction = (
                        "The tool budget is exhausted. Provide a substantive answer now without another tool call."
                    )
                    break
                except Exception as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn,
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
                        turn=turn,
                    )
                )
                messages.extend(Message.model_validate(result.to_message() | {"turn": turn}) for result in results)
                completed = self._completed_retrievals(state.retrieval_transcript, turn)
                if completed < required_depth:
                    correction = (
                        f"Complete {required_depth - completed} additional "
                        "successful retrieve call(s) with distinct rewritten queries "
                        "before answering."
                    )
                else:
                    force_final = True
                    correction = (
                        "Tool results are now available. Synthesize a substantive "
                        "final answer without another tool call."
                    )
                continue

            completed = self._completed_retrievals(state.retrieval_transcript, turn)
            if completed < required_depth:
                correction = (
                    f"Do not answer yet. Complete "
                    f"{required_depth - completed} additional successful "
                    "retrieve call(s) using distinct rewritten queries."
                )
                continue
            if action.content.strip():
                messages.append(
                    Message(
                        role="assistant",
                        content=action.content.strip(),
                        reasoning_content=action.reasoning.think or None,
                        turn=turn,
                    )
                )
                return
            correction = "Provide a substantive final answer now, without a tool call."
        turn_rejections = [item for item in state.rejected_tool_calls if item.get("turn") == turn]
        raise EpisodeGenerationError(
            f"turn {turn} did not produce a valid final answer within "
            f"{self.config.episode.max_steps_per_turn} steps; completed retrievals="
            f"{self._completed_retrievals(state.retrieval_transcript, turn)}/"
            f"{required_depth}; rejected calls="
            f"{json.dumps(turn_rejections, ensure_ascii=False)}"
        )

    def _next_user(
        self,
        models,
        seed,
        view: list[dict[str, Any]],
        *,
        turn: int,
        spec: EpisodeSpec,
    ) -> UserTurn:
        prompt = user_turn_prompt(
            turn=turn,
            turns_remaining=spec.turn_budget - turn + 1,
        )
        conversation = [
            {"role": "system", "content": user_system(seed)},
            *view,
            {"role": "user", "content": prompt},
        ]
        for _ in range(3):
            try:
                proposal = call_structured(
                    models,
                    "user",
                    conversation,
                    UserTurn,
                    attempts=1,
                )
            except Exception:
                proposal = None
            if proposal is not None and proposal.content.strip():
                return proposal.model_copy(update={"content": proposal.content.strip()})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "Invalid output. Return one nonempty, natural user message in the required JSON schema."
                    ),
                }
            )
        raise EpisodeGenerationError(f"user model did not produce a valid message at turn {turn}")

    def _execute_tool(
        self,
        call: ToolCall,
        registry: ToolRegistry,
        state: ConversationState,
        seed: EpisodeSeed,
        spec: EpisodeSpec,
        *,
        step: int,
    ) -> ToolResult:
        if self._tool_calls_this_turn(state, state.turn) >= spec.max_tool_calls_per_turn:
            raise ToolBudgetExceededError(
                f"turn {state.turn} reached max_tool_calls_per_turn={spec.max_tool_calls_per_turn}"
            )
        if len(state.tool_call_attempts) >= spec.max_tool_calls_per_conversation:
            raise ToolBudgetExceededError(
                f"conversation reached max_tool_calls_per_conversation={spec.max_tool_calls_per_conversation}"
            )
        if call.name == "retrieve" and self._retrieval_attempt_count(state) >= spec.max_retrieval_calls:
            raise ToolBudgetExceededError(f"conversation reached max_retrieval_calls={spec.max_retrieval_calls}")
        remaining_required = max(
            0,
            spec.required_retrieval_calls - self._completed_retrievals(state.retrieval_transcript),
        )
        remaining_after_call = spec.max_tool_calls_per_conversation - len(state.tool_call_attempts) - 1
        if call.name != "retrieve" and remaining_after_call < remaining_required:
            raise ToolBudgetExceededError("tool-call slot is reserved for the remaining retrieval target")
        attempt = {
            "turn": state.turn,
            "step": step,
            "tool_call_id": call.id,
            "name": call.name,
            "success": False,
        }
        state.tool_call_attempts.append(attempt)
        try:
            result = registry.execute(call, state, seed.instructions)
        except Exception as exc:
            attempt["error"] = str(exc)
            raise
        attempt["success"] = True
        return result

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
    def _completed_retrievals(transcript: list[dict[str, Any]], turn: int | None = None) -> int:
        queries = {
            normalize_query(str(row.get("query", "")))
            for row in transcript
            if (turn is None or row.get("turn") == turn) and row.get("success") and row.get("chunk_ids")
        }
        queries.discard("")
        return len(queries)

    @staticmethod
    def _retrieval_attempt_count(state: ConversationState) -> int:
        return sum(attempt.get("name") == "retrieve" for attempt in state.tool_call_attempts)

    @staticmethod
    def _tool_calls_this_turn(state: ConversationState, turn: int) -> int:
        return sum(attempt.get("turn") == turn for attempt in state.tool_call_attempts)

    def _has_retrieval_capacity(self, spec: EpisodeSpec, state: ConversationState, turn: int) -> bool:
        return (
            self._tool_calls_this_turn(state, turn) < spec.max_tool_calls_per_turn
            and len(state.tool_call_attempts) < spec.max_tool_calls_per_conversation
            and self._retrieval_attempt_count(state) < spec.max_retrieval_calls
        )

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
