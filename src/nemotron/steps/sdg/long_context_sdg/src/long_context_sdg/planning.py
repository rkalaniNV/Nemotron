"""Deterministic long-conversation planning."""

from __future__ import annotations

import hashlib
import random

from .config import PlanningConfig
from .schemas import EpisodePlan, EpisodeSeed, TurnPlan

RESEARCH_INTENTS = frozenset({"research", "rewrite"})


def plan_episode(seed: EpisodeSeed, cfg: PlanningConfig, run_seed: int) -> EpisodePlan:
    digest = hashlib.sha256(f"{run_seed}|{seed.query_id}|plan".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    first_labels = list(cfg.first_turn_intents)
    first_weights = [cfg.first_turn_intents[x] for x in first_labels]
    labels = list(cfg.intents)
    weights = [cfg.intents[x] for x in labels]
    intents = rng.choices(first_labels, weights=first_weights, k=1)
    intents.extend(rng.choices(labels, weights=weights, k=seed.turn_budget - 1))
    if cfg.ensure_retrieval_turn and not any(
        intent in RESEARCH_INTENTS for intent in intents
    ):
        fallback_index = rng.randrange(1, seed.turn_budget)
        intents[fallback_index] = "research"
    turns = []
    for turn, intent in enumerate(intents, 1):
        required = intent in RESEARCH_INTENTS
        turns.append(
            TurnPlan(
                turn=turn,
                intent=intent,
                retrieval_required=required,
                retrieval_depth=seed.retrieval_depth if required else 0,
            )
        )
    return EpisodePlan(query_id=seed.query_id, turns=turns)
