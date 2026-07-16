"""Planner — decide each conversation's shape.

Each row gets a per-conversation plan: how many turns, and for each turn its
depth (min_hops / max_steps) and whether clarification is allowed. Turn 0's kind
is the seed kind if the row carries one, otherwise it is SAMPLED from a weighted
distribution (so opening depth varies without any query-classification pass);
follow-up turns are likewise drawn from a weighted distribution so conversations
differ. All defaults are overridable via the config's ``conversation_plan`` dict.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

# kind -> turn archetype. min_hops floors depth; clarify allows a discussion turn.
_DEFAULT_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "factual":     {"min_hops": 1, "max_steps": 3, "clarify": False},
    "comparative": {"min_hops": 2, "max_steps": 5, "clarify": False},
    "multi_hop":   {"min_hops": 3, "max_steps": 7, "clarify": False},
    "exploratory": {"min_hops": 2, "max_steps": 6, "clarify": True},
    "ambiguous":   {"min_hops": 2, "max_steps": 5, "clarify": True},
}
# seed-kind distribution: sampled for turn 0 when the row carries no kind.
_DEFAULT_SEED_KINDS: List[Dict[str, Any]] = [
    {"kind": "factual", "weight": 3},
    {"kind": "comparative", "weight": 2},
    {"kind": "multi_hop", "weight": 3},
    {"kind": "exploratory", "weight": 2},
    {"kind": "ambiguous", "weight": 1},
]
# follow-up labels -> which base archetype they reuse, with sampling weights.
_DEFAULT_FOLLOWUPS: List[Dict[str, Any]] = [
    {"kind": "deepen",  "base": "multi_hop",   "weight": 3},
    {"kind": "compare", "base": "comparative", "weight": 2},
    {"kind": "clarify", "base": "ambiguous",   "weight": 1},
    {"kind": "related", "base": "factual",     "weight": 2},
]
_DEFAULT_NUM_TURNS = {"min": 2, "max": 4}


@dataclass
class TurnSpec:
    kind: str
    min_hops: int
    max_steps: int
    clarify: bool


@dataclass
class ConversationPlan:
    turns: List[TurnSpec] = field(default_factory=list)

    def kinds(self) -> List[str]:
        return [t.kind for t in self.turns]

    @property
    def n_turns(self) -> int:
        return len(self.turns)


def _archetypes(cfg_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {**_DEFAULT_ARCHETYPES, **(cfg_plan.get("kind_archetypes") or {})}


def _spec(kind: str, arch: Dict[str, Dict[str, Any]]) -> TurnSpec:
    a = arch.get(kind, _DEFAULT_ARCHETYPES["factual"])
    return TurnSpec(kind=kind, min_hops=int(a.get("min_hops", 1)),
                    max_steps=int(a.get("max_steps", 4)), clarify=bool(a.get("clarify", False)))


def plan_conversation(cfg_plan: Dict[str, Any], rng: random.Random, *, seed_kind: str = "",
                      min_turns: int = 2) -> ConversationPlan:
    cfg_plan = cfg_plan or {}
    arch = _archetypes(cfg_plan)
    nt = cfg_plan.get("num_turns") or _DEFAULT_NUM_TURNS
    lo = max(int(nt.get("min", 2)), int(min_turns))
    hi = max(lo, int(nt.get("max", lo)))
    n_turns = rng.randint(lo, hi)

    followups = cfg_plan.get("follow_up_kinds") or _DEFAULT_FOLLOWUPS
    labels = [f["kind"] for f in followups]
    weights = [float(f.get("weight", 1)) for f in followups]
    base_of = {f["kind"]: f.get("base", f["kind"]) for f in followups}

    # turn 0: honor a provided seed kind, else sample one (replaces classification)
    if not seed_kind or seed_kind not in arch:
        seeds = cfg_plan.get("seed_kinds") or _DEFAULT_SEED_KINDS
        seed_kind = rng.choices([s["kind"] for s in seeds],
                                weights=[float(s.get("weight", 1)) for s in seeds], k=1)[0]
    turns = [_spec(seed_kind, arch)]
    for _ in range(n_turns - 1):
        label = rng.choices(labels, weights=weights, k=1)[0]
        spec = _spec(base_of.get(label, "factual"), arch)
        spec.kind = label  # display the follow-up label, keep the base archetype's depth
        turns.append(spec)
    return ConversationPlan(turns=turns)
