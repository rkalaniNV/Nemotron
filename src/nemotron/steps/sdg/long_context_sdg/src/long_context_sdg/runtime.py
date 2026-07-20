"""DD-independent long-context episode runtime."""

from __future__ import annotations

import json
from typing import Any

from .compression import generate_compression, render_summary
from .config import PipelineConfig
from .episode_control import build_episode_spec
from .executors.base import ConversationState
from .llm import call_structured
from .prompts import (
    assistant_final_system,
    assistant_system,
    assistant_turn_directive,
    user_system,
    user_turn_prompt,
)
from .reasoning import validate_reasoning
from .schemas import (
    AssistantAction,
    AssistantFinalAction,
    CanonicalRecord,
    CompressionEvent,
    EpisodeSeed,
    EpisodeSpec,
    Message,
    ToolCall,
    ToolResult,
    TrajectoryJudgment,
    UserTurn,
)
from .tokens import ContextMeter
from .tool_registry import ToolRegistry, normalize_tool_call
from .validation import query_similarity, text_similarity, validate_trajectory


class EpisodeGenerationError(RuntimeError):
    pass


class ToolBudgetExceededError(EpisodeGenerationError):
    pass


class RedundantRetrievalError(EpisodeGenerationError):
    pass


class LowGainRetrievalLimitError(EpisodeGenerationError):
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
        state = ConversationState(
            conversation_id=seed.query_id,
            memory=dict(seed.memory_seed),
        )
        try:
            return self._run(
                models,
                seed,
                spec,
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
                tool_call_attempts=state.tool_call_attempts,
                metadata={
                    "query": seed.query,
                    "persona": seed.persona.model_dump(),
                    "query_provenance": seed.query_provenance.model_dump() if seed.query_provenance else None,
                    "instructions": seed.instructions,
                    "turn_budget": seed.turn_budget,
                    "max_retrieval_calls": spec.max_retrieval_calls,
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
            if not user_text.strip():
                raise EpisodeGenerationError(f"user model returned an empty message at turn {turn}")
            messages.append(Message(role="user", content=user_text.strip(), turn=turn))
            self._assistant_turn(
                models,
                seed,
                spec,
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
            tool_call_attempts=state.tool_call_attempts,
            metadata={
                "query": seed.query,
                "persona": seed.persona.model_dump(),
                "query_provenance": seed.query_provenance.model_dump() if seed.query_provenance else None,
                "instructions": seed.instructions,
                "turn_budget": seed.turn_budget,
                "successful_retrieval_calls": self._completed_retrievals(state.retrieval_transcript),
                "low_gain_retrieval_calls": sum(
                    bool(row.get("low_gain")) for row in state.retrieval_transcript
                ),
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
        messages,
        prior_summary,
        registry,
        state,
        reasoning_errors,
        warnings,
    ) -> None:
        correction = ""
        force_final = False
        budget_correction_used = False
        turn = state.turn
        for step in range(self.config.episode.max_steps_per_turn):
            view = self._view(messages, prior_summary, self.config.context.recent_raw_turns)
            directive = assistant_turn_directive(turn)
            if correction:
                directive += "\n" + correction
            has_tool_capacity = (
                self._tool_calls_this_turn(state, turn) < spec.max_tool_calls_per_turn
                and len(state.tool_call_attempts) < spec.max_tool_calls_per_conversation
            )
            final_only = force_final or not has_tool_capacity
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
                        "content": assistant_system(
                            seed,
                            registry.schemas,
                            state.retrieved.keys(),
                            state.retrieval_transcript,
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
            execution_errors = []
            budget_errors = []
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
                    execution_errors.append(str(exc))
                    budget_errors.append(str(exc))
                    continue
                except Exception as exc:
                    state.rejected_tool_calls.append(
                        {
                            "turn": turn,
                            "step": step,
                            "raw": raw,
                            "error": str(exc),
                        }
                    )
                    execution_errors.append(str(exc))
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
                correction = (
                    "Tool results are now available. Answer if the current evidence is sufficient. If it is not, "
                    "use another configured tool only when it materially advances this user request; respect any "
                    "per-turn retrieval limit reported above."
                )
                continue

            if execution_errors:
                if budget_errors:
                    if budget_correction_used:
                        force_final = True
                    budget_correction_used = True
                correction = (
                    "The proposed tool request was rejected: "
                    + "; ".join(execution_errors)
                    + ". Answer from existing evidence, or use a different configured tool only if it is still "
                    "available and materially needed."
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
            f"{self.config.episode.max_steps_per_turn} steps; rejected calls="
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
        if (
            call.name == "retrieve"
            and self._completed_retrievals(state.retrieval_transcript) >= spec.max_retrieval_calls
        ):
            raise ToolBudgetExceededError(f"conversation reached max_retrieval_calls={spec.max_retrieval_calls}")
        if (
            call.name == "retrieve"
            and self._completed_retrievals(state.retrieval_transcript, state.turn)
            >= spec.max_retrieval_calls_per_turn
        ):
            raise ToolBudgetExceededError(
                f"turn {state.turn} reached max_retrieval_calls_per_turn={spec.max_retrieval_calls_per_turn}"
            )
        best_query_similarity = 0.0
        if call.name == "retrieve":
            query = str(call.arguments.get("query", "")).strip()
            for row in state.retrieval_transcript:
                if not row.get("success") or not row.get("chunk_ids"):
                    continue
                best_query_similarity = max(
                    best_query_similarity,
                    query_similarity(query, str(row.get("query", ""))),
                )
            if best_query_similarity >= spec.query_lexical_similarity_threshold:
                raise RedundantRetrievalError(
                    "retrieval query is too lexically similar to an earlier successful search "
                    f"({best_query_similarity:.3f} >= {spec.query_lexical_similarity_threshold:.3f})"
                )
            consecutive_low_gain = 0
            latest_low_gain_query = ""
            for row in reversed(state.retrieval_transcript):
                if not row.get("success") or not row.get("chunk_ids"):
                    continue
                if not row.get("low_gain"):
                    break
                if not latest_low_gain_query:
                    latest_low_gain_query = str(row.get("query", ""))
                consecutive_low_gain += 1
            if consecutive_low_gain > spec.max_low_gain_chain:
                raise LowGainRetrievalLimitError(
                    "retrieval is paused after repeated observed low-gain results; answer from existing "
                    "evidence without another retrieval in this episode"
                )
            if consecutive_low_gain >= spec.max_low_gain_chain and latest_low_gain_query:
                followup_similarity = query_similarity(query, latest_low_gain_query)
                if followup_similarity >= spec.low_gain_followup_similarity_threshold:
                    raise LowGainRetrievalLimitError(
                        "retrieval would continue a low-gain search chain "
                        f"({followup_similarity:.3f} >= "
                        f"{spec.low_gain_followup_similarity_threshold:.3f}); reuse evidence or pursue a "
                        "genuinely different unresolved facet"
                    )
        prior_chunks = dict(state.retrieved)
        transcript_size = len(state.retrieval_transcript)
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
        if call.name == "retrieve" and len(state.retrieval_transcript) > transcript_size:
            row = state.retrieval_transcript[-1]
            chunk_ids = set(row.get("chunk_ids") or [])
            new_chunk_fraction = len(chunk_ids - set(prior_chunks)) / len(chunk_ids) if chunk_ids else 0.0
            payload = result.payload if isinstance(result.payload, list) else []
            returned_texts = [
                str(item.get("content") or item.get("text") or "")
                for item in payload
                if isinstance(item, dict)
            ]
            prior_texts = [chunk.content for chunk in prior_chunks.values() if chunk.content]
            similarities = [
                max((text_similarity(text, previous) for previous in prior_texts), default=0.0)
                for text in returned_texts
                if text
            ]
            evidence_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            low_gain = bool(prior_chunks) and (
                new_chunk_fraction < spec.min_new_chunk_fraction
                or evidence_similarity >= spec.evidence_lexical_similarity_threshold
            )
            prior_low_gain_chain = 0
            if low_gain:
                for previous in reversed(state.retrieval_transcript[:-1]):
                    if not previous.get("success") or not previous.get("chunk_ids"):
                        continue
                    if not previous.get("low_gain"):
                        break
                    prior_low_gain_chain += 1
            quality = {
                "max_prior_query_similarity": round(best_query_similarity, 6),
                "new_chunk_fraction": round(new_chunk_fraction, 6),
                "evidence_similarity": round(evidence_similarity, 6),
                "low_gain": low_gain,
                "consecutive_low_gain": prior_low_gain_chain + 1 if low_gain else 0,
            }
            row.update(quality)
            attempt["retrieval_quality"] = quality
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
        return sum(
            bool(row.get("success") and row.get("chunk_ids"))
            for row in transcript
            if turn is None or row.get("turn") == turn
        )

    @staticmethod
    def _tool_calls_this_turn(state: ConversationState, turn: int) -> int:
        return sum(attempt.get("turn") == turn for attempt in state.tool_call_attempts)

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
