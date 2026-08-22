"""Measure properties of a generated benchmark.

Nothing in the ablation ladder is interpretable without this: the pipeline reports
that generation succeeded, never what the resulting benchmark looks like. Every
metric here is computed from stage-cache artifacts, so it applies unchanged to any
arm and to any pack.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

# Metric definitions are versioned because later arms compare against A0's numbers.
# A silent change to how a metric is computed would turn a definition change into an
# apparent benchmark change. Bump this on any behavioural edit and record why in
# METRICS.md; an arm whose recorded version differs from A0's is not comparable.
METRIC_CONTRACT_VERSION = "1.0"

# The masking rule is the load-bearing definition: "distinct utterances" means nothing
# without it, and A0 vs A2 diversity is only a comparison if both used this exact rule.
SLOT_MASK_RULE = (
    "replace each bound slot value with {slot_name} by exact substring match, longest "
    "value first; no case folding, no diacritic folding, no punctuation stripping, no "
    "tokenisation"
)

# Policies that need no tool call, so no tool universe can rule them out.
_ALWAYS_FEASIBLE = frozenset({"clarify_only", "irrelevant"})


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------------
# Authoring friction
# --------------------------------------------------------------------------------


def authoring_friction(loc: dict[str, int], templates: list[dict[str, Any]]) -> dict[str, Any]:
    template_count = len(templates)
    template_loc = loc.get("task_templates.yaml", 0)
    return {
        "loc_by_file": loc,
        "loc_total": loc.get("TOTAL", 0),
        "template_count": template_count,
        "loc_per_template": round(template_loc / template_count, 1) if template_count else 0.0,
        "category_count": len({str(t.get("category")) for t in templates}),
        "policy_count": len({str(t.get("turn_policy")) for t in templates}),
    }


# --------------------------------------------------------------------------------
# Distribution: joint (category x policy), with structurally-empty cells marked
# --------------------------------------------------------------------------------


def _confirming_tools(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools:
        function = tool.get("function") or tool
        if tool.get("x-requires-confirmation") or function.get("x-requires-confirmation"):
            names.add(str(function.get("name")))
    return names


def _required_params(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    for tool in tools:
        function = tool.get("function") or tool
        parameters = function.get("parameters") or {}
        params[str(function.get("name"))] = [str(p) for p in (parameters.get("required") or [])]
    return params


def _cell_feasible(
    policy: str,
    universe: set[str],
    tools: list[dict[str, Any]],
    edges: list[dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """Decide whether a (category, policy) cell *could* hold a task.

    This separates "nobody wrote one" from "one cannot exist", which is the difference
    between a coverage gap worth filling and a target that would force meaningless
    tasks. The judgement is a documented heuristic over the category's tool universe,
    not ground truth — see the caveat emitted alongside the matrix.

    `edges` sharpens the weakest rule. Without it, `dependent_call` is called feasible
    whenever a category exposes two tools, which is wrong for two unrelated read tools
    that cannot chain. With it — the (producer, consumer) pairs `propose.probe` reads
    off the backend — the cell is feasible only when some tool actually returns a value
    another one requires.
    """
    if policy in _ALWAYS_FEASIBLE:
        return True, "needs no tool"
    if not universe:
        return False, "category exposes no tool"

    confirming = _confirming_tools(tools) & universe
    required = _required_params(tools)
    has_params = any(required.get(name) for name in universe)

    if policy == "confirmation":
        return (bool(confirming), "no tool in this category requires confirmation" if not confirming else "")
    if policy == "dependent_call":
        if edges is not None:
            chained = any(e["producer"] in universe and e["consumer"] in universe for e in edges)
            return chained, "" if chained else "no tool in this category returns a value another one requires"
        ok = len(universe) >= 2
        return ok, "" if ok else "category exposes fewer than two tools"
    if policy == "multi_tool":
        ok = len(universe) >= 2
        return ok, "" if ok else "category exposes fewer than two tools"
    if policy in {"missing_slot", "correction"}:
        return has_params, "" if has_params else "no tool in this category takes a parameter"
    if policy == "negative_path":
        return has_params, "" if has_params else "no parameter can carry a failing value"
    return True, ""


def distribution(
    tasks: list[dict[str, Any]],
    templates: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    declared_universe: dict[str, set[str]] | None = None,
    dependency_edges: list[dict[str, str]] | None = None,
    declared_policies: list[str] | None = None,
) -> dict[str, Any]:
    categories = sorted({str(t.get("category")) for t in templates} | set(declared_universe or {}))
    # A policy no surviving template carries would otherwise drop out of the matrix
    # entirely, turning a coverage failure into a missing column.
    policies = sorted({str(t.get("turn_policy")) for t in templates} | set(declared_policies or []))

    # The category's tool universe is the union of what its templates expose. Categories
    # are not declared anywhere else in the pack, so this is the only definition
    # available -- and it is circular for a category nobody wrote templates for. An arm
    # that declares its universes up front (A3) passes them in and escapes the circle.
    universe: dict[str, set[str]] = defaultdict(set)
    for template in templates:
        universe[str(template.get("category"))].update(str(x) for x in (template.get("tools_present") or []))
    if declared_universe is not None:
        universe = defaultdict(set, {k: set(v) for k, v in declared_universe.items()})

    task_cells = Counter((str(t.get("category")), str(t.get("turn_policy"))) for t in tasks)
    template_cells = Counter((str(t.get("category")), str(t.get("turn_policy"))) for t in templates)

    matrix: dict[str, dict[str, Any]] = {}
    empty_structural = 0
    empty_unwritten = 0
    for category in categories:
        row: dict[str, Any] = {}
        for policy in policies:
            count = task_cells.get((category, policy), 0)
            if count:
                row[policy] = {"tasks": count, "templates": template_cells.get((category, policy), 0)}
                continue
            feasible, reason = _cell_feasible(policy, universe[category], tools, dependency_edges)
            if feasible:
                empty_unwritten += 1
                row[policy] = {"tasks": 0, "templates": 0, "status": "empty_unwritten"}
            else:
                empty_structural += 1
                row[policy] = {"tasks": 0, "templates": 0, "status": "empty_structural", "reason": reason}
        matrix[category] = row

    policy_totals = Counter(str(t.get("turn_policy")) for t in tasks)
    total = sum(policy_totals.values()) or 1
    return {
        "categories": categories,
        "policies": policies,
        "matrix": matrix,
        "cells_total": len(categories) * len(policies),
        "cells_populated": len(categories) * len(policies) - empty_structural - empty_unwritten,
        "cells_empty_unwritten": empty_unwritten,
        "cells_empty_structural": empty_structural,
        "policy_task_counts": dict(sorted(policy_totals.items())),
        "policy_task_share": {k: round(v / total, 4) for k, v in sorted(policy_totals.items())},
        "category_task_counts": dict(sorted(Counter(str(t.get("category")) for t in tasks).items())),
        "difficulty_task_counts": dict(sorted(Counter(str(t.get("difficulty")) for t in tasks).items())),
        "universe_source": "declared" if declared_universe is not None else "inferred_from_templates",
        "caveat": (
            "Structural emptiness is inferred from the union of tools_present across the "
            "templates a category already has. A category whose templates all expose one "
            "tool is reported as unable to host multi_tool even if the domain could."
            if declared_universe is None
            else "Structural emptiness is judged against the arm's declared category tool "
            "universes and, for dependent_call, against producer/consumer edges probed "
            "from the backend."
        ),
    }


# --------------------------------------------------------------------------------
# Coverage: fixture entities, tools, slot values
# --------------------------------------------------------------------------------


def coverage(
    tasks: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
    tools: list[dict[str, Any]],
    primary_keys: dict[str, str],
) -> dict[str, Any]:
    bound: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for ref in task.get("fixture_refs") or []:
            collection, _, row_id = str(ref).partition(".")
            bound[collection].add(row_id)

    fixture_coverage: dict[str, Any] = {}
    for collection, rows in sorted(fixtures.items()):
        if not isinstance(rows, list):
            continue
        key = primary_keys.get(collection)
        if key is None:
            singular = collection[:-1] if collection.endswith("s") else collection
            fields = next((list(r) for r in rows if isinstance(r, dict)), [])
            key = next((c for c in (f"{singular}_id", "id") if c in fields), None)
        ids = [str(r.get(key)) for r in rows if isinstance(r, dict) and key and r.get(key) is not None]
        used = sorted(bound.get(collection, set()) & set(ids))
        never = sorted(set(ids) - set(used))
        fixture_coverage[collection] = {
            "primary_key": key,
            "rows": len(rows),
            "entities_bound": len(used),
            "entities_never_bound": never,
            "coverage": round(len(used) / len(ids), 4) if ids else None,
        }

    declared_tools = sorted(
        str((t.get("function") or t).get("name")) for t in tools
    )
    called = Counter()
    for trace in traces:
        for call in _loads(trace.get("expected_tool_calls"), []):
            called[str(call.get("function_name"))] += 1

    slot_values: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for name, value in _loads(task.get("slots_initial"), {}).items():
            slot_values[f"{task.get('template_id')}.{name}"].add(str(value))

    return {
        "fixtures": fixture_coverage,
        "fixture_entities_total": sum(v["rows"] for v in fixture_coverage.values()),
        "fixture_entities_bound": sum(v["entities_bound"] for v in fixture_coverage.values()),
        "tools_declared": declared_tools,
        "tool_call_counts": dict(sorted(called.items())),
        "tools_never_called": sorted(set(declared_tools) - set(called)),
        "distinct_slot_values": {k: len(v) for k, v in sorted(slot_values.items())},
    }


# --------------------------------------------------------------------------------
# Surface: how many genuinely different sentences the benchmark contains
# --------------------------------------------------------------------------------


def _mask(text: str, bindings: dict[str, Any]) -> str:
    """Replace each bound slot value with its slot name.

    Masking by the exact bound value rather than by a regex over id-shaped tokens is
    what makes the count trustworthy: a template that varies only its account id
    collapses to one sentence, which is precisely the property being measured.
    Longest values first, so a value that contains another is not partly rewritten.
    """
    masked = text
    for name, value in sorted(bindings.items(), key=lambda kv: -len(str(kv[1]))):
        rendered = str(value)
        if rendered and rendered in masked:
            masked = masked.replace(rendered, "{" + name + "}")
    return masked


def surface(
    tasks: list[dict[str, Any]],
    rendered: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(t.get("task_id")): t for t in tasks}
    per_category: dict[str, list[str]] = defaultdict(list)
    per_template: dict[str, list[str]] = defaultdict(list)
    all_first: list[str] = []
    all_masked: list[str] = []

    for row in rendered:
        task = by_id.get(str(row.get("task_id")))
        if task is None:
            continue
        turns = _loads(row.get("turns"), [])
        first = next((t for t in turns if t.get("kind") == "user"), None)
        if first is None:
            continue
        text = str(first.get("content") or "")
        bindings = dict(_loads(task.get("slots_initial"), {}))
        for update in _loads(task.get("slot_updates"), []):
            bindings.update(update.get("values") or {})
            bindings.update(update.get("aliases") or {})
        masked = _mask(text, bindings)
        all_first.append(text)
        all_masked.append(masked)
        per_category[str(task.get("category"))].append(masked)
        per_template[str(task.get("template_id"))].append(masked)

    def summarize(values: list[str]) -> dict[str, Any]:
        return {
            "tasks": len(values),
            "distinct_masked": len(set(values)),
            "ratio": round(len(set(values)) / len(values), 4) if values else None,
        }

    repeats = {text: n for text, n in Counter(all_masked).items() if n > 1}
    return {
        "overall": {
            "tasks": len(all_first),
            "distinct_raw": len(set(all_first)),
            "distinct_masked": len(set(all_masked)),
            "surfaces_per_template": round(len(set(all_masked)) / len(per_template), 3) if per_template else None,
        },
        "by_category": {k: summarize(v) for k, v in sorted(per_category.items())},
        "by_template": {k: summarize(v) for k, v in sorted(per_template.items())},
        "most_repeated_masked": [
            {"utterance": text, "tasks": n}
            for text, n in sorted(repeats.items(), key=lambda kv: -kv[1])[:10]
        ],
        "lexical_shortcut_probe": {
            "runnable": len(set(all_masked)) > len(per_template),
            "reason": (
                "A generalization-gap probe needs several phrasings per intent. This arm "
                f"produced {len(set(all_masked))} distinct masked surfaces across "
                f"{len(per_template)} templates; with one surface per template there is no "
                "held-out phrasing to test on. The probe becomes available at A2."
            ),
        },
    }


# --------------------------------------------------------------------------------
# Funnel: where tasks die between expansion and publication
# --------------------------------------------------------------------------------


def _reasons(rows: list[dict[str, Any]], flag: str, reason_fields: tuple[str, ...]) -> dict[str, int]:
    counter: Counter = Counter()
    for row in rows:
        if row.get(flag) is True:
            continue
        for field in reason_fields:
            value = row.get(field)
            if value:
                counter[f"{field}={value}"] += 1
                break
        else:
            counter["unspecified"] += 1
    return dict(counter)


def funnel(tables: dict[str, list[dict[str, Any]]], run_manifest: dict[str, Any] | None) -> dict[str, Any]:
    rendered = tables.get("rendered_conversations") or []
    traces = tables.get("expected_traces") or []
    schema = tables.get("schema_validated_traces") or []
    replay = tables.get("replay_validated_tasks") or []
    raw = tables.get("benchmark_raw") or []
    published = tables.get("benchmark") or []
    expanded = tables.get("task_instances") or []

    stages = [
        ("expand", len(expanded), {}),
        ("state_machine", len(tables.get("conversation_plans") or []), {}),
        ("render_accepted", sum(1 for r in rendered if r.get("accepted") is True),
         _reasons(rendered, "accepted", ("guard_violations",))),
        ("expected_trace_derived", sum(1 for r in traces if r.get("derived") is True),
         _reasons(traces, "derived", ("drop_reason",))),
        ("schema_valid", sum(1 for r in schema if r.get("valid") is True),
         _reasons(schema, "valid", ("reject_reason",))),
        ("replay_valid", sum(1 for r in replay if r.get("valid") is True),
         _reasons(replay, "valid", ("reason", "detail"))),
        ("benchmark_raw", len(raw), {}),
        ("published", len(published), {}),
    ]

    steps = []
    previous = len(expanded) or 1
    for name, count, reasons in stages:
        steps.append(
            {
                "stage": name,
                "rows": count,
                "survival_from_expand": round(count / (len(expanded) or 1), 4),
                "lost_here": max(previous - count, 0),
                "drop_reasons": reasons,
            }
        )
        previous = count

    gold_rows = sum(1 for r in published if r.get("gold_eligible") is True)
    return {
        "steps": steps,
        "expanded": len(expanded),
        "published": len(published),
        "publish_rate": round(len(published) / (len(expanded) or 1), 4),
        "gold_rows": gold_rows,
        "gold_rate": round(gold_rows / (len(published) or 1), 4),
        "tiers": dict(sorted(Counter(str(r.get("tier")) for r in published).items())),
        "run_gold_eligible": (run_manifest or {}).get("gold_eligible"),
        "generation_mode": (run_manifest or {}).get("generation_mode"),
    }


# --------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------


def measure(
    *,
    arm: str,
    tables: dict[str, list[dict[str, Any]]],
    pack_dir: Path,
    loc: dict[str, int],
    run_manifest: dict[str, Any] | None,
    normalized_templates: Path | None = None,
    declared_universe: dict[str, set[str]] | None = None,
    dependency_edges: list[dict[str, str]] | None = None,
    declared_policies: list[str] | None = None,
) -> dict[str, Any]:
    """Compute every A0 metric for one arm.

    Templates are read from the stage cache when available: an arm that derives fields
    at load time authors a different file than the pipeline actually consumed, and the
    distribution must describe what ran.
    """
    tasks = tables.get("task_instances") or []
    if normalized_templates and normalized_templates.exists():
        templates = yaml.safe_load(normalized_templates.read_text(encoding="utf-8")) or []
    else:
        templates = yaml.safe_load((pack_dir / "task_templates.yaml").read_text(encoding="utf-8")) or []

    tools = json.loads((pack_dir / "tools.json").read_text(encoding="utf-8"))
    fixtures = json.loads((pack_dir / "fixtures.json").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((pack_dir / "manifest.yaml").read_text(encoding="utf-8")) or {}

    return {
        "arm": arm,
        "pack": str(pack_dir),
        "metrics_version": METRIC_CONTRACT_VERSION,
        "definitions": {
            "slot_mask_rule": SLOT_MASK_RULE,
            "loc_counting": "every line of every authored pack file plus the run config, including blanks and comments",
            "distinct_masked_surface": "cardinality of the set of slot-masked opening user turns",
            "fixture_entity_bound": "a fixture row whose primary key appears in some task's fixture_refs",
            "publish_rate": "published rows divided by expanded task instances",
            "cell_empty_structural": "a (category, policy) cell no tool in the category's universe can satisfy",
        },
        "authoring_friction": authoring_friction(loc, templates),
        "distribution": distribution(
            tasks,
            templates,
            tools,
            declared_universe=declared_universe,
            dependency_edges=dependency_edges,
            declared_policies=declared_policies,
        ),
        "coverage": coverage(
            tasks,
            tables.get("expected_traces") or [],
            fixtures,
            tools,
            manifest.get("primary_keys") or {},
        ),
        "surface": surface(tasks, tables.get("rendered_conversations") or []),
        "funnel": funnel(tables, run_manifest),
    }
