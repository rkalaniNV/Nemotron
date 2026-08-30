"""The kill ladder: which layer of the pack, if any, notices a corrupted backend.

Layers run cheapest-first and a mutant stops at the first one that kills it, so the
expensive layers only ever see mutants the cheap ones missed:

  L0 import      the mutant imports and `list_tools()` still returns the tool set
  L1 validation  the pack's 23 validation cases, on `expect.result_class` / `error_code`
  L2 traces      A0's 33 expected traces replay to the same results and the same final
                 state as the unmutated backend
  L3 assertions  the pack's own `success_assertions` accept those episodes
  L4 oracle      the `run_oracle_validation` checks, and `derive_pack_tier`
  L5 pipeline    a full unmodified `common.run_arm` still publishes and still earns gold

L2 is differential against a baseline captured from the unmutated backend rather than
against a hand-written expectation, because no artifact in the pack states what
`get_account_balance` should *return* — which is precisely the gap A4 measured as 0.610
argument-level false acceptance. Comparing against the baseline lets L2 detect a wrong
value that L1 and L3 both let through, and the difference between those layers is the
result this arm exists to produce. It is therefore the one layer that is NOT a check the
pack ships; it is the reference the pack lacks.

Only `backend.py` changes. For L0-L4 the mutant is passed to `ProcessWorker` as
`backend_path` while `import_root` stays the real pack, so nothing else about the pack
is perturbed. L5 needs a directory, so it copies the pack and swaps one file.
"""

from __future__ import annotations

import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ordered by cost, cheapest first: a mutant stops at the first layer that kills it, so
# the expensive layers only ever see what the cheap ones missed. L0-L3 swap only
# `backend_path`; L4 and L5 need a pack directory, so they copy the pack and replace one
# file.
L0_IMPORT = "L0_import"
L1_VALIDATION = "L1_validation_cases"
L2_TRACES = "L2_expected_traces"
L3_ASSERTIONS = "L3_assertions"
L4_ORACLE = "L4_oracle_validation"
L5_PIPELINE = "L5_pipeline"

LAYERS = (L0_IMPORT, L1_VALIDATION, L2_TRACES, L3_ASSERTIONS, L4_ORACLE, L5_PIPELINE)

SURVIVED = "survived"


@dataclass
class Baseline:
    """What the unmutated backend does, captured once and compared against."""

    tools: list[str]
    validation: dict[str, dict[str, Any]]
    traces: dict[str, dict[str, Any]]
    assertions: dict[str, dict[str, bool]]


@dataclass
class Runtime:
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
            # Widened for the parent's own concurrency: a worker starved of CPU during
            # `import nemotron` would be scored as a killed mutant, which is a harness
            # artefact and not a property of the edit.
            import_timeout_s=max(oracle_runtime.import_timeout_s, 90.0),
            episode_timeout_s=max(oracle_runtime.episode_timeout_s, 300.0),
        )


def _classify(result: Any, protocol: dict[str, str]) -> tuple[str, Any]:
    """Reduce a tool result to the pair `validation_cases.yaml` asserts on.

    This delegates to the production classifier rather than reimplementing it. A local
    copy missed `awaiting_confirmation` entirely — the pack's confirmation protocol is a
    third result class, not a flavour of success — and scored 2 of 23 cases as failing on
    the *unmutated* backend, which would have shown up as two permanently-dead layers.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        _classify_result,
    )

    if not isinstance(result, dict):
        return "episode_error", repr(result)[:120]
    return _classify_result(result, protocol)


class Runner:
    def __init__(
        self,
        *,
        pack_dir: Path,
        fixtures: dict[str, Any],
        runtime: Runtime,
        protocol: dict[str, str],
        worker: str = "process",
        concurrency: int = 6,
    ) -> None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

        self._worker = ProcessWorker(default_timeout_s=runtime.episode_timeout_s, worker=worker)
        self._pack_dir = pack_dir
        self._fixtures = fixtures
        self._runtime = runtime
        self._protocol = protocol
        self._concurrency = concurrency
        self.episodes_run = 0

    def _episode(
        self, *, backend_path: Path, steps: list[dict[str, Any]], task_id: str, seed: int = 0
    ) -> list[Any] | str:
        rt = self._runtime
        try:
            outputs = self._worker.run_episode(
                backend_path=backend_path,
                endpoint_config=None,
                fixtures=copy.deepcopy(self._fixtures),
                clock_iso=rt.clock,
                seed=seed,
                task_id=task_id,
                steps=steps,
                assertions_path=self._pack_dir / "assertions.py",
                import_root=self._pack_dir,
                import_timeout_s=rt.import_timeout_s,
                reset_timeout_s=rt.reset_timeout_s,
                tool_timeout_s=rt.tool_timeout_s,
                assertion_timeout_s=rt.assertion_timeout_s,
                episode_timeout_s=rt.episode_timeout_s,
            )
        except Exception as error:  # noqa: BLE001 — a dead worker is a result
            return f"{type(error).__name__}: {error}"
        self.episodes_run += 1
        return outputs

    # -- L0 ---------------------------------------------------------------------

    def tools(self, backend_path: Path) -> list[str] | str:
        out = self._episode(
            backend_path=backend_path, steps=[{"op": "reset"}, {"op": "list_tools"}], task_id="a6-l0"
        )
        if isinstance(out, str):
            return out
        listed = out[-1]
        return list(listed) if isinstance(listed, list) else f"unexpected list_tools: {listed!r}"

    # -- L1 ---------------------------------------------------------------------

    def validation(self, backend_path: Path, cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Per case, the failures the pack's own Check 5 would record. Empty list = pass.

        The four failure reasons mirror `oracle_validation.py`'s inlined check exactly:
        `result_class_mismatch`, `error_code_mismatch`, `result_field_mismatch` and
        `state_changed`. An earlier version compared only the first two, which made L1
        strictly weaker than the check it claims to be — mutants the real Check 5 kills
        were surviving L1 and being attributed to a later layer. The layer histogram is
        the point of this arm, so an under-strength L1 is not a conservative
        approximation, it is a wrong answer.

        `state_unchanged` needs the state either side of the call, so each case runs
        reset / get_state / call_tool / get_state — four steps, not two.
        """
        steps: list[dict[str, Any]] = []
        for case in cases:
            steps.append({"op": "reset"})
            steps.append({"op": "get_state"})
            steps.append(
                {
                    "op": "call_tool",
                    "name": str(case["tool"]),
                    "arguments": dict(case.get("arguments") or {}),
                    "turn_index": 0,
                }
            )
            steps.append({"op": "get_state"})

        out = self._episode(backend_path=backend_path, steps=steps, task_id="a6-l1")
        if isinstance(out, str):
            return {str(c["id"]): [{"reason": "episode_error", "detail": out[:160]}] for c in cases}

        outcomes: dict[str, list[dict[str, Any]]] = {}
        for index, case in enumerate(cases):
            before, result, after = out[4 * index + 1], out[4 * index + 2], out[4 * index + 3]
            expect = case.get("expect") or {}
            failures: list[dict[str, Any]] = []

            if not isinstance(result, dict):
                outcomes[str(case["id"])] = [{"reason": "non_object_result"}]
                continue

            klass, code = _classify(result, self._protocol)
            if expect.get("result_class") and klass != expect["result_class"]:
                failures.append(
                    {"reason": "result_class_mismatch", "expected": expect["result_class"], "got": klass}
                )
            if "error_code" in expect and expect["error_code"] != code:
                failures.append(
                    {"reason": "error_code_mismatch", "expected": expect["error_code"], "got": code}
                )
            for field, expected_value in expect.items():
                if field in {"result_class", "error_code", "state_unchanged"}:
                    continue
                if result.get(field) != expected_value:
                    failures.append(
                        {
                            "reason": "result_field_mismatch",
                            "field": field,
                            "expected": expected_value,
                            "got": result.get(field),
                        }
                    )
            if expect.get("state_unchanged") and after != before:
                failures.append({"reason": "state_changed"})

            outcomes[str(case["id"])] = failures
        return outcomes

    # -- L2 / L3 ----------------------------------------------------------------

    def replay(
        self,
        backend_path: Path,
        tasks: list[dict[str, Any]],
        traces: dict[str, list[dict[str, Any]]],
        *,
        with_assertions: bool,
    ) -> dict[str, dict[str, Any]]:
        def job(task: dict[str, Any]) -> tuple[str, Any]:
            task_id = str(task["task_id"])
            calls = traces.get(task_id) or []
            names = list(task.get("success_assertions") or []) if with_assertions else []
            steps: list[dict[str, Any]] = [{"op": "reset"}]
            steps.extend(
                {
                    "op": "call_tool",
                    "name": call["function_name"],
                    "arguments": call.get("arguments") or {},
                    "turn_index": int(call.get("turn_index", 0)),
                }
                for call in calls
            )
            steps.append({"op": "get_state"})
            steps.extend({"op": "run_assertion", "name": n, "task": task} for n in names)
            out = self._episode(
                backend_path=backend_path, steps=steps, task_id=task_id, seed=int(task.get("seed") or 0)
            )
            return task_id, (out, len(calls), names)

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            for task_id, (out, n_calls, names) in pool.map(job, tasks):
                if isinstance(out, str):
                    results[task_id] = {"error": out}
                    continue
                call_results = out[1 : 1 + n_calls]
                state = out[1 + n_calls]
                row: dict[str, Any] = {
                    "results": json.loads(json.dumps(call_results, sort_keys=True, default=str)),
                    "state": json.loads(json.dumps(state, sort_keys=True, default=str)),
                }
                if names:
                    verdicts = out[2 + n_calls :]
                    row["assertions"] = {
                        name: bool(item.get("passed")) if isinstance(item, dict) else False
                        for name, item in zip(names, verdicts)
                    }
                results[task_id] = row
        return results


def build_baseline(
    runner: Runner,
    *,
    backend_path: Path,
    cases: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
) -> Baseline:
    tools = runner.tools(backend_path)
    replayed = runner.replay(backend_path, tasks, traces, with_assertions=True)
    return Baseline(
        tools=tools if isinstance(tools, list) else [],
        validation=runner.validation(backend_path, cases),
        traces={k: {"results": v.get("results"), "state": v.get("state")} for k, v in replayed.items()},
        assertions={k: v.get("assertions", {}) for k, v in replayed.items()},
    )
