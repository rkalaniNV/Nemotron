"""Deterministic single-edit mutations of a backend source file.

Sites are found by walking the AST rather than by hand-listing line numbers, so the
operator set transfers to any pack's backend unchanged — which matters, because the
whole point of A6 is to replace an asserted constant (877 lines) with a measured one,
and a hand-listed inventory would only ever measure `banking_vn`.

Every mutant differs from the original by exactly one edit. That is what makes a
surviving mutant attributable: the line it touched is a line nothing checks.

Two families deserve their names explained:

  `guard_delete`      removes an `if <cond>: return _err(...)` validation guard. A
                      surviving guard-delete means the pack never sends the bad input
                      that guard exists to reject.
  `state_write_delete` removes an assignment into the state dict or an append to a
                      state list. A surviving state-write-delete means no assertion
                      reads that field — which is exactly the argument-level blindness
                      A4 measured, seen from the other side.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Iterator

CALL_LEVEL = "contract"
VALUE_LEVEL = "value"
STATE_LEVEL = "state"
GUARD_LEVEL = "guard"

_CMP_FLIP: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}

_BINOP_SWAP: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}


@dataclass(frozen=True)
class Site:
    """One place a mutation could be applied, with the edit it would make."""

    index: int
    operator: str
    family: str
    lineno: int
    before: str
    after: str

    @property
    def label(self) -> str:
        return f"{self.operator}@{self.lineno}"


@dataclass(frozen=True)
class Mutant:
    site: Site
    source: str


def _is_err_return(node: ast.AST) -> bool:
    """An `if ...: return _err(...)` body — the pack's validation-guard idiom."""
    if not isinstance(node, ast.If) or len(node.body) != 1 or node.orelse:
        return False
    inner = node.body[0]
    return (
        isinstance(inner, ast.Return)
        and isinstance(inner.value, ast.Call)
        and isinstance(inner.value.func, ast.Name)
        and inner.value.func.id == "_err"
    )


def _is_state_write(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Subscript) for t in node.targets)
    if isinstance(node, ast.AugAssign):
        return isinstance(node.target, ast.Subscript)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        return isinstance(func, ast.Attribute) and func.attr in {"append", "insert", "update", "pop"}
    return False


def _snippet(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — a snippet is a label, never load-bearing
        return f"<{type(node).__name__}>"


@dataclass(frozen=True)
class _Candidate:
    """One collected mutation opportunity.

    `report` is the node the result table names; `target` is the node `apply` edits.
    They differ for statement-level operators, where the edit is made on the *parent*
    block but the interesting node to report is the statement being removed. Conflating
    the two silently produced zero `delete_state_write` and `invert_guard` mutants —
    `apply` was handed the statement, wrote into its own body, and the operator
    disappeared from the run without any error.
    """

    operator: str
    family: str
    report: ast.AST
    description: str
    target: ast.AST
    apply: Callable[[Any], None]


def _collect(tree: ast.AST) -> list[_Candidate]:
    found: list[_Candidate] = []

    for node in ast.walk(tree):
        # -- comparison boundaries: off-by-one and inverted conditions --------------
        if isinstance(node, ast.Compare):
            for position, op in enumerate(node.ops):
                flipped = _CMP_FLIP.get(type(op))
                if flipped is None:
                    continue

                def apply(n: Any, _pos: int = position, _new: type = flipped) -> None:
                    n.ops[_pos] = _new()

                found.append(
                    _Candidate(
                        "flip_comparison", VALUE_LEVEL, node,
                        f"{type(op).__name__} -> {flipped.__name__}", node, apply,
                    )
                )

        # -- arithmetic: fee and amount computation --------------------------------
        elif isinstance(node, ast.BinOp):
            swapped = _BINOP_SWAP.get(type(node.op))
            if swapped is not None:

                def apply(n: Any, _new: type = swapped) -> None:
                    n.op = _new()

                found.append(
                    _Candidate(
                        "swap_arithmetic", VALUE_LEVEL, node,
                        f"{type(node.op).__name__} -> {swapped.__name__}", node, apply,
                    )
                )

        # -- literals: tier bounds, seeds, defaults, flags -------------------------
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            def apply(n: Any) -> None:
                n.value = not n.value

            found.append(
                _Candidate("negate_bool_literal", VALUE_LEVEL, node,
                           f"{node.value} -> {not node.value}", node, apply)
            )

        elif isinstance(node, ast.Constant) and isinstance(node.value, int):
            def apply(n: Any) -> None:
                n.value = n.value + 1

            found.append(
                _Candidate("perturb_int_literal", VALUE_LEVEL, node,
                           f"{node.value} -> {node.value + 1}", node, apply)
            )

    # -- statement-level edits. `target` is the PARENT block, `report` the statement --
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for position, stmt in enumerate(body):
            if _is_err_return(stmt):

                def apply_delete(p: Any, _pos: int = position) -> None:
                    p.body[_pos] = ast.Pass()

                found.append(
                    _Candidate("delete_guard", GUARD_LEVEL, stmt,
                               f"remove guard: {_snippet(stmt.test)}", parent, apply_delete)
                )

                # Deleting a guard and inverting it are different failures: one lets bad
                # input through, the other rejects good input. Both must be probed.
                def apply_invert(p: Any, _pos: int = position) -> None:
                    node_if = p.body[_pos]
                    node_if.test = ast.UnaryOp(op=ast.Not(), operand=node_if.test)

                found.append(
                    _Candidate("invert_guard", GUARD_LEVEL, stmt,
                               f"invert guard: {_snippet(stmt.test)}", parent, apply_invert)
                )

            elif _is_state_write(stmt):

                def apply_state(p: Any, _pos: int = position) -> None:
                    p.body[_pos] = ast.Pass()

                found.append(
                    _Candidate("delete_state_write", STATE_LEVEL, stmt,
                               f"remove: {_snippet(stmt)}", parent, apply_state)
                )

    # -- projection edits: a key dropped from a returned dict -----------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict) and len(node.value.keys) > 1:
            for position, key in enumerate(node.value.keys):
                if not isinstance(key, ast.Constant):
                    continue

                def apply_drop(n: Any, _pos: int = position) -> None:
                    del n.value.keys[_pos]
                    del n.value.values[_pos]

                found.append(
                    _Candidate("drop_result_key", CALL_LEVEL, node,
                               f"drop key {key.value!r} from result", node, apply_drop)
                )

    return found


def build_mutants(source: str) -> list[Mutant]:
    """Every single-edit mutant of `source`, in a stable order.

    The tree is re-parsed for each mutant so edits cannot compose. Order is the walk
    order of a fresh parse, which is deterministic for a given file, so mutant indices
    are stable across runs and a result table can be diffed.
    """
    catalogue = _collect(ast.parse(source))
    baseline = ast.unparse(ast.parse(source))
    mutants: list[Mutant] = []

    for index, candidate in enumerate(catalogue):
        fresh = ast.parse(source)
        sites = _collect(fresh)
        if index >= len(sites):  # pragma: no cover — parse is deterministic
            continue
        site = sites[index]
        try:
            site.apply(site.target)
            mutated = ast.unparse(ast.fix_missing_locations(fresh))
        except Exception:  # noqa: BLE001 — an operator that cannot apply is not a mutant
            continue

        # `ast.unparse` normalises formatting, so an edit that produces byte-identical
        # output changed nothing observable and must not be counted as a survivor.
        if mutated == baseline:
            continue

        mutants.append(
            Mutant(
                site=Site(
                    index=index,
                    operator=candidate.operator,
                    family=candidate.family,
                    lineno=getattr(candidate.report, "lineno", 0),
                    before=_snippet(candidate.report)[:160],
                    after=candidate.description,
                ),
                source=mutated,
            )
        )
    return mutants
