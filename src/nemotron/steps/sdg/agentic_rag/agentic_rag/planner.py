"""Conversation planner — samples a DIFFERENT multi-turn shape for every row,
driven by the KIND of each query.

Every conversation is multi-turn (>= a configured minimum). The planner samples:
  1. how many turns (length diversity), and
  2. the SEQUENCE of query kinds — turn 0 is the seed query's own difficulty
     level; each follow-up's kind is drawn from a configured distribution.

The query kind then *shapes* the turn: a vague/``half_baked`` query becomes a
clarification turn, a ``complex_multistep`` query becomes a deep multi-step
research turn, a ``simple`` one a quick lookup, and so on. So the structure
follows the queries and follow-ups — not a fixed template — and varies per row.

Kinds, their depth/flags, the follow-up-kind distribution, and the turn-count
range are all declared in the outer config (``conversation_plan``). The defaults
here are only a runnable fallback, not the policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# query kind -> how it shapes a turn (depth floor, ceiling slack, clarify)
_DEFAULT_KIND_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "half_baked":        {"min_hops": 2, "max_hops": 5, "clarify": True},
    "simple":            {"min_hops": 1, "max_hops": 2},
    "crisp":             {"min_hops": 3, "max_hops": 6},
    "complex_multistep": {"min_hops": 6, "max_hops": 14},
}
# distribution of follow-up query kinds (drives type + length diversity)
_DEFAULT_FOLLOWUP_KINDS: List[Dict[str, Any]] = [
    {"kind": "complex_multistep", "weight": 3},
    {"kind": "crisp", "weight": 3},
    {"kind": "simple", "weight": 2},
    {"kind": "half_baked", "weight": 2},
]
_DEFAULT_NUM_TURNS = {"min": 3, "max": 6}


@dataclass
class TurnSpec:
    """The shape of one turn, derived from its query kind."""
    kind: str                  # the query kind driving this turn
    min_hops: int              # forced retrieval floor (0 = may answer from context)
    max_steps: int             # inner-loop ceiling
    clarify: bool              # run a clarification exchange first
    enforce_sufficiency: bool  # keep searching until the gap-check passes
    require_plan: bool         # emit a short research plan


@dataclass
class ConversationPlan:
    turns: List[TurnSpec] = field(default_factory=list)

    def kinds(self) -> List[str]:
        return [t.kind for t in self.turns]

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _rint(rng: random.Random, lo: Any, hi: Any, dlo: int, dhi: int) -> int:
    lo = int(lo if lo is not None else dlo)
    hi = int(hi if hi is not None else dhi)
    return rng.randint(min(lo, hi), max(lo, hi))


def _spec_for_kind(kind: str, kind_archetypes: Dict[str, Any], rng: random.Random) -> TurnSpec:
    a = kind_archetypes.get(kind) or kind_archetypes.get("crisp") or {"min_hops": 2, "max_hops": 4}
    hops = _rint(rng, a.get("min_hops"), a.get("max_hops"), 1, 4)
    return TurnSpec(
        kind=kind,
        min_hops=hops,
        max_steps=1 if hops == 0 else hops + int(a.get("answer_slack", 2)),
        clarify=bool(a.get("clarify", False)),
        enforce_sufficiency=bool(a.get("enforce_sufficiency", hops >= 2)),
        require_plan=bool(a.get("require_plan", hops >= 3)),
    )


def plan_conversation(spec: Dict[str, Any], rng: random.Random, *,
                      seed_kind: str, min_turns: int = 3) -> ConversationPlan:
    """Sample a conversation: turn 0 follows the seed query's kind; each follow-up
    turn draws a kind from the configured distribution, and each kind shapes its
    turn's depth/clarify."""
    spec = spec or {}
    nt = spec.get("num_turns") or _DEFAULT_NUM_TURNS
    n = max(min_turns, _rint(rng, nt.get("min"), nt.get("max"), min_turns, min_turns + 3))

    kind_archetypes = spec.get("kind_archetypes") or _DEFAULT_KIND_ARCHETYPES
    fu = spec.get("follow_up_kinds") or _DEFAULT_FOLLOWUP_KINDS
    fu_kinds = [f.get("kind", "crisp") for f in fu]
    fu_weights = [max(0.0, float(f.get("weight", 1))) for f in fu]

    seed_kind = seed_kind if seed_kind in kind_archetypes else "crisp"
    kinds = [seed_kind] + [rng.choices(fu_kinds, weights=fu_weights, k=1)[0] for _ in range(1, n)]
    return ConversationPlan(turns=[_spec_for_kind(k, kind_archetypes, rng) for k in kinds])


def turn_eff(spec: TurnSpec) -> Dict[str, Any]:
    """Adapt a TurnSpec into the per-turn control dict the loop consumes."""
    return {
        "min_hops": spec.min_hops,
        "max_steps": spec.max_steps,
        "allow_discussion": spec.clarify,
        "enforce_sufficiency": spec.enforce_sufficiency,
        "require_plan": spec.require_plan,
    }
