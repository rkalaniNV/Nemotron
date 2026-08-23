"""Project executable evidence into the shared normalized trace scoring view."""

from __future__ import annotations

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    EXECUTABLE_NON_CANDIDATE_STOPS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_parser import (
    ParsedCall,
    ParsedTrace,
    ParsedTurn,
)


def parse_executable_trace(
    episode: ExecutableEpisode,
    task: ExecutableTaskSpec,
) -> ParsedTrace:
    """Read canonical turns and calls without re-parsing provider or oracle bytes."""

    turns = tuple(
        ParsedTurn(
            turn_index=observed.turn_index,
            kind=(
                "tool_calls"
                if task.script.turn(observed.turn_index).expects_tool_calls
                else "text"
            ),
            call_status=observed.call_status,
            advanced=observed.advanced,
            finish_reason=observed.finish_reason,
            assistant_content=observed.assistant_content,
            calls=tuple(
                ParsedCall(
                    turn_index=observed.turn_index,
                    position_in_turn=position,
                    provider_index=call.index,
                    id=call.id,
                    type=call.type,
                    function_name=call.function_name,
                    arguments_status=call.arguments_status,
                    parsed_arguments=call.parsed_arguments,
                )
                for position, call in enumerate(observed.tool_calls)
            ),
            detail=observed.detail,
        )
        for observed in episode.observed
    )
    return ParsedTrace(
        task_id=episode.task_id,
        candidate_alias=episode.candidate_alias,
        canonical_model_identity=episode.canonical_model_identity,
        plan_identity=episode.plan_identity,
        source_verification_identity=episode.source_verification_identity,
        script_hash=episode.script_hash,
        episode_hash=episode.episode_hash,
        status=episode.status,
        non_candidate_stop=episode.status in EXECUTABLE_NON_CANDIDATE_STOPS,
        scripted_turns=len(task.script.turns),
        turns=turns,
        unsent_turn_indexes=tuple(range(len(turns), len(task.script.turns))),
    )


__all__ = ["EXECUTABLE_NON_CANDIDATE_STOPS", "parse_executable_trace"]
