"""Per-sample constraints and sparse retrieval-deadline control."""

from __future__ import annotations

import hashlib
import random

from .config import EpisodePolicyConfig
from .schemas import EpisodeSeed, EpisodeSpec, RetrievalPolicyEvent


def _rng(run_seed: int, query_id: str, purpose: str) -> random.Random:
    digest = hashlib.sha256(f"{run_seed}|{query_id}|{purpose}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_episode_spec(
    seed: EpisodeSeed,
    policy: EpisodePolicyConfig,
    run_seed: int,
) -> EpisodeSpec:
    """Sample stable episode constraints without scripting semantic turns."""
    per_turn_capacity = min(seed.retrieval_depth, policy.max_tool_calls_per_turn)
    max_retrieval_calls = min(
        policy.retrieval_calls.max,
        policy.max_tool_calls_per_conversation,
        seed.turn_budget * per_turn_capacity,
    )
    if max_retrieval_calls < policy.retrieval_calls.min:
        raise ValueError(
            f"episode retrieval minimum {policy.retrieval_calls.min} is impossible "
            f"for query `{seed.query_id}`; maximum feasible is {max_retrieval_calls}"
        )
    rng = _rng(run_seed, seed.query_id, "episode-spec")
    required = rng.randint(policy.retrieval_calls.min, max_retrieval_calls)
    return EpisodeSpec(
        query_id=seed.query_id,
        turn_budget=seed.turn_budget,
        required_retrieval_calls=required,
        max_retrieval_calls=max_retrieval_calls,
        max_tool_calls_per_turn=policy.max_tool_calls_per_turn,
        max_tool_calls_per_conversation=policy.max_tool_calls_per_conversation,
    )


def retrieval_deadline_event(
    spec: EpisodeSpec,
    seed: EpisodeSeed,
    *,
    turn: int,
    successful_retrievals: int,
    retrieval_attempts: int,
    tool_calls: int,
) -> RetrievalPolicyEvent | None:
    """Intervene only when deferring retrieval would make the target impossible."""
    remaining_required = max(0, spec.required_retrieval_calls - successful_retrievals)
    if remaining_required == 0:
        return None

    attempt_capacity = min(
        spec.max_retrieval_calls - retrieval_attempts,
        spec.max_tool_calls_per_conversation - tool_calls,
    )
    turns_remaining = spec.turn_budget - turn + 1
    per_turn_capacity = min(
        seed.retrieval_depth,
        spec.max_tool_calls_per_turn,
        max(0, attempt_capacity),
    )
    total_turn_capacity = turns_remaining * per_turn_capacity
    total_capacity = min(attempt_capacity, total_turn_capacity)
    if remaining_required > total_capacity:
        raise ValueError(
            f"retrieval target is no longer feasible at turn {turn}: "
            f"{remaining_required} successful call(s) remain but only "
            f"{total_capacity} deadline slot(s) are available"
        )

    future_capacity = min(
        attempt_capacity,
        max(0, turns_remaining - 1) * per_turn_capacity,
    )
    required_now = max(0, remaining_required - future_capacity)
    if required_now == 0:
        return None
    return RetrievalPolicyEvent(
        turn=turn,
        required_retrievals_this_turn=required_now,
        successful_retrievals_before=successful_retrievals,
        retrieval_attempts_before=retrieval_attempts,
        tool_calls_before=tool_calls,
        turns_remaining=turns_remaining,
    )
