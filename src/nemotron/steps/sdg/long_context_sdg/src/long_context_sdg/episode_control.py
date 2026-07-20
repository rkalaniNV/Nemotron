"""Stable per-sample safety caps without retrieval quotas or turn plans."""

from __future__ import annotations

from .config import EpisodePolicyConfig
from .schemas import EpisodeSeed, EpisodeSpec


def build_episode_spec(
    seed: EpisodeSeed,
    policy: EpisodePolicyConfig,
    run_seed: int,
) -> EpisodeSpec:
    """Build deterministic caps; the assistant retrieves only for an evidence gap."""
    del run_seed
    max_retrieval_calls = min(
        policy.max_retrieval_calls,
        policy.max_tool_calls_per_conversation,
        seed.turn_budget * policy.max_retrieval_calls_per_turn,
    )
    novelty = policy.retrieval_novelty
    return EpisodeSpec(
        query_id=seed.query_id,
        turn_budget=seed.turn_budget,
        max_retrieval_calls=max_retrieval_calls,
        max_retrieval_calls_per_turn=policy.max_retrieval_calls_per_turn,
        max_tool_calls_per_turn=policy.max_tool_calls_per_turn,
        max_tool_calls_per_conversation=policy.max_tool_calls_per_conversation,
        query_lexical_similarity_threshold=novelty.query_lexical_similarity_threshold,
        evidence_lexical_similarity_threshold=novelty.evidence_lexical_similarity_threshold,
        min_new_chunk_fraction=novelty.min_new_chunk_fraction,
        max_low_gain_chain=novelty.max_low_gain_chain,
        low_gain_followup_similarity_threshold=novelty.low_gain_followup_similarity_threshold,
    )
