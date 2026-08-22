"""Discover what each tool's result actually contains, by calling it.

`tools.json` declares arguments and says nothing about return shape, so nothing in the
pack states that `list_recent_transactions` yields a `transaction_id` another tool can
consume. Two things need that fact:

  feasibility   whether a (category, dependent_call) cell can hold a task at all, which
                is a question about producer/consumer edges and not, as
                `metrics._cell_feasible` currently assumes, about how many tools the
                category exposes
  proposal      the `depends_on.path` an LLM has to write, which is a real path into a
                real result rather than a plausible-looking one

Both are answered by executing one successful call per tool through the production
process worker and recording the paths its result carries. The backend is the authority
on its own return shape; asking it is cheaper and more honest than parsing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# Scalar leaves only: a dependent argument must resolve to a scalar (`expected_trace`
# refuses a path that lands on a container), so a container path is not a usable edge.
_SCALAR = (str, int, float, bool)


def _paths(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a tool result into `dotted.path -> leaf key`.

    Lists are entered at index 0 only. A dependent call reads one element, and the
    element a deterministic replay reads is the first one; enumerating the rest would
    invent paths whose availability depends on the fixture row.
    """
    found: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, _SCALAR) and item is not None:
                found[path] = str(key)
            else:
                found.update(_paths(item, path))
    elif isinstance(value, list) and value:
        found.update(_paths(value[0], f"{prefix}.0" if prefix else "0"))
    return found


def probe_tool_results(
    *,
    pack_dir: Path,
    cases: list[dict[str, Any]],
    clock_iso: str,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run one success case per tool and report the scalar paths of each result.

    Every call is preceded by its own reset, so a mutating tool cannot leave state that
    changes the next tool's answer and, with it, the edges derived from it.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

    fixtures = json.loads((pack_dir / "fixtures.json").read_text(encoding="utf-8"))
    success = [case for case in cases if str(case.get("id", "")).startswith("success_")]

    steps: list[dict[str, Any]] = []
    for case in success:
        steps.append({"op": "reset"})
        steps.append(
            {
                "op": "call_tool",
                "name": str(case["tool"]),
                "arguments": dict(case.get("arguments") or {}),
                "turn_index": 0,
            }
        )

    outputs = ProcessWorker(default_timeout_s=120.0, worker="process").run_episode(
        backend_path=pack_dir / "backend.py",
        endpoint_config=None,
        fixtures=fixtures,
        clock_iso=clock_iso,
        seed=seed,
        task_id="a3-result-probe",
        steps=steps,
        assertions_path=pack_dir / "assertions.py",
        import_root=pack_dir,
        import_timeout_s=30.0,
        reset_timeout_s=10.0,
        tool_timeout_s=10.0,
        assertion_timeout_s=10.0,
        episode_timeout_s=120.0,
    )

    results: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(success):
        result = outputs[2 * index + 1]
        tool = str(case["tool"])
        if not isinstance(result, dict) or "error" in result:
            results[tool] = {"probed": False, "detail": str(result)[:200], "paths": {}}
            continue
        results[tool] = {"probed": True, "paths": _paths(result), "sample": result}
    return results


def dependency_edges(
    results: dict[str, dict[str, Any]],
    tools: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Return every (producer, consumer, path) a dependent call could actually use.

    An edge exists when a producer's result carries a scalar whose leaf key is the name
    of a parameter the consumer requires. Matching on the parameter name is the same
    rule `milestones._call_milestones` uses to place a `from_result` marker, so an edge
    reported here is one the A1 compiler can express.
    """
    edges: list[dict[str, str]] = []
    for producer, probe in sorted(results.items()):
        for path, leaf in sorted((probe.get("paths") or {}).items()):
            for consumer, spec in sorted(tools.items()):
                if consumer == producer or leaf not in spec["required"]:
                    continue
                edges.append({"producer": producer, "consumer": consumer, "parameter": leaf, "path": path})
    return edges


def load_validation_cases(pack_dir: Path) -> list[dict[str, Any]]:
    """Read the rehydrated pack's own validation cases, which already call every tool."""
    raw = yaml.safe_load((pack_dir / "validation_cases.yaml").read_text(encoding="utf-8")) or []
    return list(raw) if isinstance(raw, list) else list(raw.get("cases") or [])
