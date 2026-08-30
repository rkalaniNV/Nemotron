"""Run the mutation gate against a set of assertions and score what survives.

The gate never touches the generator. It re-uses `ProcessWorker.run_episode` exactly
as `stages/executable_replay.py` does, including the documented `trace` override on
`run_assertion`, so a corrupted episode reaches the assertion through the same code
path a real one does. Only `assertions_path` changes between arms, which is what
makes the human/LLM comparison a comparison of assertions rather than of harnesses.

Scoring vocabulary, per (task, assertion, mutation) triple:

  detected        the assertion raised AssertionError — the corruption was caught
  false_accept    the assertion returned normally — the corruption was not a check
  crash           the assertion raised something else; the episode fails, but for the
                  wrong reason, and the message an author would debug is misleading
  baseline_invalid the assertion already fails on the *uncorrupted* gold episode, so
                  nothing about it can be measured; the triple is excluded

`baseline_invalid` is reported per arm as a gold-pass rate. Without it a generator
could win the false-acceptance table by emitting `raise AssertionError` everywhere.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bfcl_ablation.mutate.operators import (
    ADVISORY_OPERATORS,
    ARGUMENT_LEVEL,
    CALL_LEVEL,
    OPERATOR_CLASSES,
    REEXECUTE,
    STATE_LEVEL,
    STATE_RESET,
    TRACE,
    Mutation,
    PackContext,
    build_mutations,
)

DETECTED = "detected"
FALSE_ACCEPT = "false_accept"
CRASH = "crash"
BASELINE_INVALID = "baseline_invalid"
EPISODE_ERROR = "episode_error"

# The worker flattens every assertion failure into a detail string; an AssertionError
# keeps only `str(exc)` while anything else is prefixed with its type name. This is
# the only signal available to tell a caught corruption from a broken assertion, and
# it misfires if an assertion raises AssertionError("TypeError: ...") on purpose.
_CRASH_DETAIL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Exit|Iteration|Interrupt|Warning)(?::|$)")


@dataclass
class Runtime:
    """Episode limits, widened for the parent's own concurrency.

    Import and episode ceilings come from the pack config, but the gate runs many
    spawned workers at once and a worker starved of CPU during `import nemotron`
    would be scored as a pack failure. Per-tool and per-assertion deadlines are left
    at the pack's own values, since those bound pack code rather than the harness.
    """

    clock: str
    tool_timeout_s: float
    assertion_timeout_s: float
    reset_timeout_s: float
    import_timeout_s: float
    episode_timeout_s: float

    @classmethod
    def from_config(cls, oracle_runtime: Any) -> Runtime:
        return cls(
            clock=oracle_runtime.clock,
            tool_timeout_s=oracle_runtime.tool_timeout_s,
            assertion_timeout_s=oracle_runtime.assertion_timeout_s,
            reset_timeout_s=oracle_runtime.reset_timeout_s,
            import_timeout_s=max(oracle_runtime.import_timeout_s, 90.0),
            episode_timeout_s=max(oracle_runtime.episode_timeout_s, 300.0),
        )


@dataclass
class Target:
    """One arm of the comparison: whose assertions are under test."""

    name: str
    assertions_path: Path
    import_root: Path


@dataclass
class Trial:
    task_id: str
    template_id: str
    assertion: str
    operator: str
    op_class: str
    detail: str
    outcome: str
    message: str | None = None


@dataclass
class TaskPlan:
    """A replayed episode plus every corruption of it, computed once and shared.

    The mutation set is derived from the *human* baseline run and then reused for
    every arm. Deriving it per arm would let a weak arm be graded on an easier
    exam."""

    task: dict[str, Any]
    calls: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    state_changed: bool
    mutations: list[Mutation] = field(default_factory=list)


class Gate:
    def __init__(
        self,
        *,
        backend_path: Path,
        fixtures: dict[str, Any],
        runtime: Runtime,
        worker: str = "process",
        concurrency: int = 6,
    ) -> None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

        self._worker = ProcessWorker(default_timeout_s=runtime.episode_timeout_s, worker=worker)
        self._backend_path = backend_path
        self._fixtures = fixtures
        self._runtime = runtime
        self._concurrency = concurrency
        self.episodes_run = 0

    # -- episode plumbing -------------------------------------------------------

    def _episode(
        self,
        *,
        task: dict[str, Any],
        steps: list[dict[str, Any]],
        target: Target,
    ) -> list[Any] | str:
        """Run one episode; a transport/import failure comes back as a message."""
        runtime = self._runtime
        try:
            outputs = self._worker.run_episode(
                backend_path=self._backend_path,
                endpoint_config=None,
                fixtures=copy.deepcopy(self._fixtures),
                clock_iso=runtime.clock,
                seed=int(task.get("seed") or 0),
                task_id=str(task["task_id"]),
                steps=steps,
                assertions_path=target.assertions_path,
                import_root=target.import_root,
                import_timeout_s=runtime.import_timeout_s,
                reset_timeout_s=runtime.reset_timeout_s,
                tool_timeout_s=runtime.tool_timeout_s,
                assertion_timeout_s=runtime.assertion_timeout_s,
                episode_timeout_s=runtime.episode_timeout_s,
            )
        except Exception as error:  # noqa: BLE001 — a dead worker is a result, not a stop
            return f"{type(error).__name__}: {error}"
        self.episodes_run += 1
        return outputs

    def _parallel(self, jobs: list[Callable[[], Any]]) -> list[Any]:
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            return [future.result() for future in [pool.submit(job) for job in jobs]]

    @staticmethod
    def _call_steps(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "op": "call_tool",
                "name": call["function_name"],
                "arguments": call.get("arguments") or {},
                "turn_index": int(call.get("turn_index", 0)),
            }
            for call in calls
        ]

    def replay(
        self,
        *,
        task: dict[str, Any],
        calls: list[dict[str, Any]],
        names: list[str],
        target: Target,
    ) -> dict[str, bool]:
        """Run `names` over an episode built from `calls` — any calls, not only gold.

        A4 needed this for the gold trace and its corruptions only, so it stayed private.
        A5 points the same operation at what a target model produced: identical steps,
        a different call list. An episode that fails to run returns `{}` rather than a
        dict of `False`, so "the harness broke" cannot be read as "the model was wrong".
        """
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(self._call_steps(calls))
        steps.extend({"op": "run_assertion", "name": name, "task": task} for name in names)
        outputs = self._episode(task=task, steps=steps, target=target)
        if isinstance(outputs, str) or not names:
            return {}
        return {
            name: bool(item.get("passed"))
            for name, item in zip(names, outputs[-len(names) :])
        }

    # -- phase 1: replay the gold episode ---------------------------------------

    def plan(self, tasks: list[dict[str, Any]], traces: dict[str, list[dict[str, Any]]], *, target: Target,
             context: PackContext) -> tuple[dict[str, TaskPlan], list[str]]:
        """Replay every task once to capture its real trace and whether state moved."""
        errors: list[str] = []

        def job(task: dict[str, Any]) -> Any:
            calls = traces[str(task["task_id"])]
            steps = [{"op": "reset"}, {"op": "get_state"}]
            steps.extend(self._call_steps(calls))
            steps.append({"op": "get_state"})
            return self._episode(task=task, steps=steps, target=target)

        plans: dict[str, TaskPlan] = {}
        for task, outputs in zip(tasks, self._parallel([lambda t=t: job(t) for t in tasks])):
            task_id = str(task["task_id"])
            calls = traces[task_id]
            if isinstance(outputs, str):
                errors.append(f"{task_id}: {outputs}")
                continue
            results = outputs[2 : 2 + len(calls)]
            trace = [
                {
                    "tool": call["function_name"],
                    "arguments": call.get("arguments") or {},
                    "result": result,
                    "turn_index": int(call.get("turn_index", 0)),
                }
                for call, result in zip(calls, results)
            ]
            changed = json.dumps(outputs[1], sort_keys=True) != json.dumps(outputs[-1], sort_keys=True)
            plan = TaskPlan(task=task, calls=calls, trace=trace, state_changed=changed)
            plan.mutations = build_mutations(
                calls=calls, trace=trace, context=context, state_changed=changed
            )
            plans[task_id] = plan
        return plans, errors

    # -- phase 2: score one arm -------------------------------------------------

    def score(self, plans: dict[str, TaskPlan], *, target: Target) -> dict[str, Any]:
        trials: list[Trial] = []
        gold: list[dict[str, Any]] = []

        baselines = self._parallel(
            [lambda p=plan: self._baseline(p, target) for plan in plans.values()]
        )
        valid: dict[str, set[str]] = {}
        for plan, outcome in zip(plans.values(), baselines):
            task_id = str(plan.task["task_id"])
            valid[task_id] = {name for name, ok in outcome.items() if ok}
            gold.extend(
                {
                    "task_id": task_id,
                    "template_id": plan.task.get("template_id"),
                    "assertion": name,
                    "passed": ok,
                }
                for name, ok in sorted(outcome.items())
            )

        jobs: list[Callable[[], Any]] = []
        specs: list[tuple[TaskPlan, list[Mutation], str]] = []
        for plan in plans.values():
            task_id = str(plan.task["task_id"])
            names = sorted(valid[task_id])
            if not names:
                continue
            # Trace-mode corruptions leave state and steps alone, so every one of them
            # fits in a single episode after the real calls have run. Only re-executed
            # corruptions need an episode each.
            trace_only = [m for m in plan.mutations if m.mode == TRACE]
            if trace_only:
                specs.append((plan, trace_only, "trace_batch"))
                jobs.append(lambda p=plan, ms=trace_only, ns=names: self._trace_batch(p, ms, ns, target))
            for mutation in plan.mutations:
                if mutation.mode == STATE_RESET:
                    specs.append((plan, [mutation], "state_reset"))
                    jobs.append(lambda p=plan, m=mutation, ns=names: self._state_reset(p, m, ns, target))
                elif mutation.mode == REEXECUTE:
                    specs.append((plan, [mutation], "reexecute"))
                    jobs.append(lambda p=plan, m=mutation, ns=names: self._reexecute(p, m, ns, target))

        for (plan, mutations, _kind), outputs in zip(specs, self._parallel(jobs)):
            names = sorted(valid[str(plan.task["task_id"])])
            trials.extend(self._read(plan, mutations, names, outputs))

        return {
            "target": target.name,
            "gold": gold,
            "trials": [trial.__dict__ for trial in trials],
        }

    def _baseline(self, plan: TaskPlan, target: Target) -> dict[str, bool]:
        names = list(plan.task.get("success_assertions") or [])
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(self._call_steps(plan.calls))
        steps.extend({"op": "run_assertion", "name": name, "task": plan.task} for name in names)
        outputs = self._episode(task=plan.task, steps=steps, target=target)
        if isinstance(outputs, str):
            return dict.fromkeys(names, False)
        return {
            name: bool(item.get("passed"))
            for name, item in zip(names, outputs[-len(names) :] if names else [])
        }

    def _trace_batch(
        self, plan: TaskPlan, mutations: list[Mutation], names: list[str], target: Target
    ) -> list[Any] | str:
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(self._call_steps(plan.calls))
        for mutation in mutations:
            steps.extend(
                {
                    "op": "run_assertion",
                    "name": name,
                    "task": plan.task,
                    "trace": [dict(entry) for entry in (mutation.trace or ())],
                }
                for name in names
            )
        outputs = self._episode(task=plan.task, steps=steps, target=target)
        return outputs if isinstance(outputs, str) else outputs[-len(mutations) * len(names) :]

    def _state_reset(
        self, plan: TaskPlan, mutation: Mutation, names: list[str], target: Target
    ) -> list[Any] | str:
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(
            {
                "op": "run_assertion",
                "name": name,
                "task": plan.task,
                "trace": [dict(entry) for entry in (mutation.trace or ())],
            }
            for name in names
        )
        outputs = self._episode(task=plan.task, steps=steps, target=target)
        return outputs if isinstance(outputs, str) else outputs[-len(names) :]

    def _reexecute(
        self, plan: TaskPlan, mutation: Mutation, names: list[str], target: Target
    ) -> list[Any] | str:
        steps: list[dict[str, Any]] = [{"op": "reset"}]
        steps.extend(self._call_steps(list(mutation.calls or ())))
        steps.extend({"op": "run_assertion", "name": name, "task": plan.task} for name in names)
        outputs = self._episode(task=plan.task, steps=steps, target=target)
        return outputs if isinstance(outputs, str) else outputs[-len(names) :]

    def _read(
        self,
        plan: TaskPlan,
        mutations: list[Mutation],
        names: list[str],
        outputs: list[Any] | str,
    ) -> list[Trial]:
        task_id = str(plan.task["task_id"])
        template_id = str(plan.task.get("template_id"))
        trials: list[Trial] = []
        for index, mutation in enumerate(mutations):
            for offset, name in enumerate(names):
                if isinstance(outputs, str):
                    outcome, message = EPISODE_ERROR, outputs
                else:
                    item = outputs[index * len(names) + offset]
                    detail = item.get("detail")
                    if item.get("passed"):
                        outcome, message = FALSE_ACCEPT, None
                    elif detail and _CRASH_DETAIL.match(str(detail)):
                        outcome, message = CRASH, str(detail)
                    else:
                        outcome, message = DETECTED, str(detail) if detail else None
                trials.append(
                    Trial(
                        task_id=task_id,
                        template_id=template_id,
                        assertion=name,
                        operator=mutation.operator,
                        op_class=mutation.op_class,
                        detail=mutation.detail,
                        outcome=outcome,
                        message=message,
                    )
                )
        return trials


# -- aggregation ----------------------------------------------------------------

_SCORED = (DETECTED, FALSE_ACCEPT, CRASH)


def _tally(trials: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in (*_SCORED, EPISODE_ERROR)}
    for trial in trials:
        counts[trial["outcome"]] = counts.get(trial["outcome"], 0) + 1
    scored = sum(counts[name] for name in _SCORED)
    return {
        "trials": scored,
        "detected": counts[DETECTED],
        "false_accept": counts[FALSE_ACCEPT],
        "crash": counts[CRASH],
        "episode_error": counts[EPISODE_ERROR],
        "false_acceptance_rate": round(counts[FALSE_ACCEPT] / scored, 4) if scored else None,
        "detection_rate": round(counts[DETECTED] / scored, 4) if scored else None,
        "crash_rate": round(counts[CRASH] / scored, 4) if scored else None,
    }


def summarize(scored: dict[str, Any], plans: dict[str, TaskPlan]) -> dict[str, Any]:
    """Per-operator, per-class and per-assertion tables — deliberately never one number.

    An aggregate would average a call-level operator that every assertion catches
    against a state-level one that almost none of them do, and report a benchmark as
    healthy on the strength of its easiest question.
    """
    trials = scored["trials"]
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_assertion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_assertion_class: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_operator[trial["operator"]].append(trial)
        by_class[trial["op_class"]].append(trial)
        by_assertion[trial["assertion"]].append(trial)
        by_assertion_class[(trial["assertion"], trial["op_class"])].append(trial)

    applicable: dict[str, set[str]] = defaultdict(set)
    generated: dict[str, int] = defaultdict(int)
    for task_id, plan in plans.items():
        for mutation in plan.mutations:
            applicable[mutation.operator].add(task_id)
            generated[mutation.operator] += 1

    operators: dict[str, Any] = {}
    for operator in sorted(set(OPERATOR_CLASSES) | set(by_operator)):
        row = _tally(by_operator.get(operator, []))
        row["op_class"] = OPERATOR_CLASSES.get(operator, "unknown")
        row["semantics"] = "advisory" if operator in ADVISORY_OPERATORS else "strict"
        row["tasks_applicable"] = len(applicable.get(operator, ()))
        row["tasks_total"] = len(plans)
        row["mutations_generated"] = generated.get(operator, 0)
        operators[operator] = row

    gold = scored["gold"]
    gold_pass = sum(1 for row in gold if row["passed"])
    assertions: dict[str, Any] = {}
    for name in sorted(set(by_assertion) | {row["assertion"] for row in gold}):
        row = _tally(by_assertion.get(name, []))
        row["by_class"] = {
            op_class: _tally(by_assertion_class.get((name, op_class), []))
            for op_class in (CALL_LEVEL, ARGUMENT_LEVEL, STATE_LEVEL)
            if by_assertion_class.get((name, op_class))
        }
        row["by_class_strict"] = {
            op_class: _tally(
                [
                    entry
                    for entry in by_assertion_class.get((name, op_class), [])
                    if entry["operator"] not in ADVISORY_OPERATORS
                ]
            )
            for op_class in (CALL_LEVEL, ARGUMENT_LEVEL, STATE_LEVEL)
            if by_assertion_class.get((name, op_class))
        }
        instances = [entry for entry in gold if entry["assertion"] == name]
        row["gold_instances"] = len(instances)
        row["gold_passed"] = sum(1 for entry in instances if entry["passed"])
        assertions[name] = row

    return {
        "target": scored["target"],
        "gold": {
            "instances": len(gold),
            "passed": gold_pass,
            "pass_rate": round(gold_pass / len(gold), 4) if gold else None,
            "failing": sorted(
                {(row["assertion"], row["template_id"]) for row in gold if not row["passed"]}
            ),
        },
        # Detection on the advisory operators is the over-strictness readout. Those
        # corruptions are not defects, so an arm that rejects them is buying its
        # false-acceptance rate by refusing episodes nobody said were wrong.
        "advisory": _tally(
            [row for row in trials if row["operator"] in ADVISORY_OPERATORS]
        ),
        "by_class": {name: _tally(rows) for name, rows in sorted(by_class.items())},
        "by_class_strict": {
            name: _tally([row for row in rows if row["operator"] not in ADVISORY_OPERATORS])
            for name, rows in sorted(by_class.items())
        },
        "by_operator": operators,
        "by_assertion": assertions,
    }


def surviving_mutations(scored: dict[str, Any]) -> dict[str, list[str]]:
    """Per assertion, the corruptions it let through — the feedback the LLM arm gets."""
    survivors: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for trial in scored["trials"]:
        if trial["outcome"] != FALSE_ACCEPT:
            continue
        # An advisory corruption is one an assertion is *right* to accept, so feeding it
        # back as a survivor asks the author to reject correct behaviour. `summarize`
        # already excludes these from every rate; this loop did not, so 10 of the 50
        # survivors handed to the LLM arm were repeated idempotent reads.
        if trial["operator"] in ADVISORY_OPERATORS:
            continue
        key = (trial["assertion"], trial["operator"], trial["template_id"])
        if key in seen:
            continue
        seen.add(key)
        survivors[trial["assertion"]].append(f"{trial['operator']}: {trial['detail']}")
    return dict(survivors)
