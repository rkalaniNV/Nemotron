"""The coverage target, and the controlled sampler that spends it.

The (category x policy) matrix is the benchmark's shape. A3 hands the model the
semantics of a task and keeps the shape, so this module decides — before any prompt is
built — which cells exist, which are structurally impossible, and how many proposals
each feasible cell gets. The model is told which cell it is filling and is rejected if
it fills a different one.

Two improvements over the heuristic in `measurement/metrics.py`, which A0 already
flagged as circular:

  declared universe   a category's tools are declared here, not inferred from the
                      templates the category happens to already have. The inferred
                      version can only ever say that a category can host what it
                      already hosts, which makes every unwritten cell look impossible
                      in exactly the cases that matter.
  probed edges        `dependent_call` is feasible when some tool in the category
                      returns a value another tool in the category requires — an edge
                      `probe.dependency_edges` reads off the backend — not when the
                      category merely exposes two tools. Two unrelated read tools
                      cannot chain, and the old rule called that cell writable.

The universes below are the only authored input A3 adds. Categories are a pack concept
that the pack never states, so somebody has to say what `dispute` is about; saying it
in ten lines is the smallest honest version of that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

CATEGORY_TOOLS: dict[str, tuple[str, ...]] = {
    "balance_inquiry": ("get_account_balance", "get_card_limit"),
    "transaction_status": ("get_transaction_status", "list_recent_transactions"),
    "transfer": ("get_transfer_fee", "create_transfer", "get_transaction_status"),
    "qr_payment": ("get_vietqr_payment_status", "get_transaction_status"),
    "dispute": ("get_dispute_status", "create_dispute", "get_transaction_status"),
    # A request the pack answers with a refusal exposes read tools so the refusal is a
    # judgement rather than an absence of options.
    "out_of_scope": (),
}

POLICIES: tuple[str, ...] = (
    "single_turn",
    "missing_slot",
    "confirmation",
    "correction",
    "multi_tool",
    "dependent_call",
    "negative_path",
    "clarify_only",
    "irrelevant",
)

# Policies whose conversation calls no tool, so no tool universe can rule them out.
NO_TOOL_POLICIES = frozenset({"clarify_only", "irrelevant"})

POLICY_BRIEF: dict[str, str] = {
    "single_turn": (
        "The user states everything in one message and the assistant answers after its "
        "calls. There is no second user turn."
    ),
    "missing_slot": (
        "The user's opening message omits one value the tool requires. The assistant "
        "asks for it, the user supplies it, then the call runs. Mark the omitted slot "
        "visible_in_first_turn: false and leave it out of the opening sentence."
    ),
    "confirmation": (
        "The action changes state, so the assistant asks for approval and only calls "
        "after the user grants it. Requires a tool that declares "
        "x-requires-confirmation."
    ),
    "correction": (
        "The user approves, then changes one value before the call happens. Declare "
        "`corrects` with the slot being replaced and the source its replacement comes "
        "from; the replacement must be able to differ from the original."
    ),
    "multi_tool": (
        "One request needs two or more distinct tool calls. Set call_order: any when "
        "the calls are independent and may be issued in one batch."
    ),
    "dependent_call": (
        "A second call needs an argument that only the first call's result contains. "
        "Declare `depends_on` with the producing tool and the exact path into its "
        "result. required_tools must list the producer before the consumer."
    ),
    "negative_path": (
        "The call is expected to fail or be refused by the backend — an id that exists "
        "nowhere, or an amount the account cannot cover. The assertion must state what "
        "the backend refused, not that it succeeded."
    ),
    "clarify_only": (
        "The request is under-specified in a way the assistant cannot resolve, so it "
        "asks a clarifying question and calls nothing. required_tools must be empty, and "
        "the template must supply assistant_turn_templates.ask_for_slot. Do NOT declare a "
        "slot for the value the customer failed to give: nothing calls a tool, so there "
        "is nothing to bind it to. Declare only what the customer did say, typically one "
        "literal naming the thing they were vague about."
    ),
    "irrelevant": (
        "The request is outside what the tools can do, so the assistant declines and "
        "calls nothing. required_tools must be empty."
    ),
}


@dataclass(frozen=True)
class Cell:
    category: str
    policy: str
    feasible: bool
    reason: str
    universe: tuple[str, ...]
    target: int = 0


@dataclass
class CoverageSpec:
    cells: list[Cell]
    budget: int
    seed: int
    edges: list[dict[str, str]] = field(default_factory=list)

    @property
    def feasible(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.feasible]

    @property
    def assignments(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.target > 0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "seed": self.seed,
            "categories": sorted({cell.category for cell in self.cells}),
            "policies": list(POLICIES),
            "cells_total": len(self.cells),
            "cells_feasible": len(self.feasible),
            "cells_structural_empty": len(self.cells) - len(self.feasible),
            "proposals_requested": sum(cell.target for cell in self.cells),
            "universes": {name: list(tools) for name, tools in sorted(CATEGORY_TOOLS.items())},
            "dependency_edges": self.edges,
            "matrix": {
                cell.category: {
                    inner.policy: {
                        "feasible": inner.feasible,
                        "target": inner.target,
                        **({"reason": inner.reason} if inner.reason else {}),
                    }
                    for inner in self.cells
                    if inner.category == cell.category
                }
                for cell in self.cells
            },
        }


def cell_status(
    policy: str,
    universe: set[str] | tuple[str, ...],
    tools: dict[str, dict[str, Any]],
    edges: list[dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """Decide whether one (category, policy) cell can hold a task, and say why not.

    A cell declared structurally empty is a claim, not a shrug: it says a task of that
    shape cannot exist over this category's tools, so a coverage report should not
    count it as a gap. The claims below are checkable against tools.json and the probed
    edges, which is what separates them from the inferred heuristic they replace.
    """
    names = set(universe)
    if policy in NO_TOOL_POLICIES:
        return True, ""
    if not names:
        return False, "category exposes no tool, and every other policy must call one"

    confirming = {name for name in names if tools.get(name, {}).get("requires_confirmation")}
    parameterized = {name for name in names if tools.get(name, {}).get("required")}

    if policy == "confirmation":
        if not confirming:
            return False, "no tool in this category requires confirmation"
        return True, ""
    if policy == "multi_tool":
        if len(names) < 2:
            return False, "category exposes fewer than two tools"
        return True, ""
    if policy == "dependent_call":
        usable = [
            edge
            for edge in (edges or [])
            if edge["producer"] in names and edge["consumer"] in names
        ]
        if not usable:
            return False, "no tool in this category returns a value another one requires"
        return True, ""
    if policy in {"missing_slot", "correction", "negative_path"}:
        if not parameterized:
            return False, "no tool in this category takes a parameter to withhold, change, or fail on"
        return True, ""
    return True, ""


def build_spec(
    *,
    tools: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    budget: int,
    seed: int = 42,
) -> CoverageSpec:
    """Lay out the coverage target and spend `budget` proposals over it.

    Every feasible cell gets one proposal before any cell gets two: a benchmark that
    covers 44 cells once is worth more than one that covers 22 twice, and the leftover
    is spread by a seeded shuffle rather than by category order, so the surplus does not
    always land on whichever category sorts first.
    """
    cells: list[Cell] = []
    for category in sorted(CATEGORY_TOOLS):
        universe = CATEGORY_TOOLS[category]
        for policy in POLICIES:
            feasible, reason = cell_status(policy, universe, tools, edges)
            cells.append(
                Cell(
                    category=category,
                    policy=policy,
                    feasible=feasible,
                    reason=reason,
                    universe=universe,
                )
            )

    feasible_index = [index for index, cell in enumerate(cells) if cell.feasible]
    targets = {index: 1 for index in feasible_index}
    surplus = budget - len(feasible_index)
    if surplus > 0:
        order = list(feasible_index)
        random.Random(seed).shuffle(order)
        for step in range(surplus):
            targets[order[step % len(order)]] += 1

    spent = [
        Cell(
            category=cell.category,
            policy=cell.policy,
            feasible=cell.feasible,
            reason=cell.reason,
            universe=cell.universe,
            target=targets.get(index, 0),
        )
        for index, cell in enumerate(cells)
    ]
    return CoverageSpec(cells=spent, budget=budget, seed=seed, edges=edges)
