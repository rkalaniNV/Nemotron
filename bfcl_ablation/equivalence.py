"""Prove that an arm changed authoring friction and nothing else.

A1's claim is not "low risk" but "no semantic change", and that is checkable rather
than arguable. `task_id` is content-addressed over the pack id, template id, bound
fixture references and slot bindings, so two arms agree on `set(task_id)` only if
they bound the same records to the same slots. `expected_tool_calls` is what the
benchmark actually asserts. Equal on both means the derivation moved no ground truth.

Where the arms are *expected* to differ — the wording of simulator replies, now drawn
from pack-wide canonical templates — the difference is reported rather than asserted
away, because the two equality checks above do not cover phrasing.
"""

from __future__ import annotations

import json
from typing import Any

from bfcl_ablation import common


def _loads(raw: Any, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _canonical_calls(calls: Any) -> str:
    """Canonicalize a trace so ordering of mapping keys cannot fake a difference."""
    normalized = []
    for call in _loads(calls, []) or []:
        arguments = call.get("arguments")
        if isinstance(arguments, list):  # parquet map type arrives as pairs
            arguments = dict(arguments)
        normalized.append(
            {
                "turn_index": call.get("turn_index"),
                "call_group": call.get("call_group"),
                "position_in_group": call.get("position_in_group"),
                "function_name": call.get("function_name"),
                "arguments": {str(k): str(v) for k, v in sorted((arguments or {}).items())},
            }
        )
    normalized.sort(key=lambda c: (c["turn_index"] or 0, c["call_group"] or 0, c["position_in_group"] or 0))
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _traces_by_task(tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        str(row["task_id"]): _canonical_calls(row.get("expected_tool_calls"))
        for row in (tables.get("expected_traces") or [])
    }


def _published_calls_by_task(tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        str(row["task_id"]): _canonical_calls(row.get("expected_tool_calls"))
        for row in (tables.get("benchmark") or [])
    }


def _plans_by_task(tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Canonicalize each task's conversation plan.

    `set(task_id)` proves both arms bound the same records to the same slots, and
    `expected_tool_calls` proves they call the same tools with the same arguments.
    Neither observes `assistant_milestones` — so a compiler that emitted `ask_confirm`
    where the author wrote `ask_for_slot`, leaving the tool calls untouched, would pass
    both and change the conversation the benchmark actually tests. That is precisely
    the risk A1's milestone compiler carries, and it needs its own check.

    The stage cache already stores the plan in a wording-free projection: `steps` keeps
    `kind`, `milestone_type`, `call_group` and `tools`, and drops `content_template`.
    No further normalization is needed, which is why this check is free.
    """
    plans: dict[str, str] = {}
    for row in tables.get("conversation_plans") or []:
        plans[str(row["task_id"])] = json.dumps(
            {
                "steps": _loads(row.get("steps"), []),
                "num_user_turns": row.get("num_user_turns"),
                "num_tool_calls": row.get("num_tool_calls"),
                "has_user_confirmation": row.get("has_user_confirmation"),
                "has_slot_correction": row.get("has_slot_correction"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    return plans


def _validation_coverage(pack_dir: Any) -> set[tuple[str, str]]:
    """The (tool, outcome class) pairs a pack's validation cases probe.

    A1 replaces ~20 hand-written cases with generated ones. Nothing in the other
    checks observes a validation case, so probe coverage could fall silently and show
    up only as fewer lines. Gold eligibility is a coarse backstop: it fails when a tool
    loses coverage entirely, not when an outcome class does.
    """
    import yaml

    path = pack_dir / "validation_cases.yaml"
    if not path.exists():
        return set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = raw.get("cases", []) if isinstance(raw, dict) else raw
    pairs: set[tuple[str, str]] = set()
    for case in cases:
        expect = case.get("expect") or {}
        outcome = str(expect.get("result_class"))
        # An error code or a business status is a different probe from a bare success,
        # so it belongs in the class rather than being flattened into it.
        if expect.get("error_code"):
            outcome = f"{outcome}/{expect['error_code']}"
        elif expect.get("status"):
            outcome = f"{outcome}/{expect['status']}"
        pairs.add((str(case.get("tool")), outcome))
    return pairs


def _surfaces_by_task(tables: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    surfaces: dict[str, list[str]] = {}
    for row in tables.get("rendered_conversations") or []:
        turns = _loads(row.get("turns"), [])
        surfaces[str(row["task_id"])] = [
            str(turn.get("content") or "") for turn in turns if turn.get("kind") == "user"
        ]
    return surfaces


def compare(
    baseline: common.ArmResult,
    candidate: common.ArmResult,
    *,
    baseline_tables: dict[str, list[dict[str, Any]]] | None = None,
    candidate_tables: dict[str, list[dict[str, Any]]] | None = None,
    opening_turn_may_change: bool = False,
) -> dict[str, Any]:
    """Compare two arms on everything that defines the benchmark.

    `opening_turn_may_change` selects which contract is being enforced. A1 keeps the
    request byte-identical and a reworded opening turn is a failure. A2 rewords the
    opening turn on purpose — that is the whole intervention — so holding it to A1's
    contract would report the arm as broken for doing its job. The other four checks
    are unconditional either way: no arm is allowed to move task identity, the expected
    calls, the conversation shape, or probe coverage.
    """
    left = baseline_tables if baseline_tables is not None else common.load_stage_tables(baseline)
    right = candidate_tables if candidate_tables is not None else common.load_stage_tables(candidate)

    left_ids = {str(t["task_id"]) for t in left.get("task_instances") or []}
    right_ids = {str(t["task_id"]) for t in right.get("task_instances") or []}
    only_left = sorted(left_ids - right_ids)
    only_right = sorted(right_ids - left_ids)

    def trace_mismatches(a: dict[str, str], b: dict[str, str]) -> list[dict[str, str]]:
        return [
            {"task_id": task_id, "baseline": a[task_id][:400], "candidate": b[task_id][:400]}
            for task_id in sorted(set(a) & set(b))
            if a[task_id] != b[task_id]
        ]

    stage_mismatch = trace_mismatches(_traces_by_task(left), _traces_by_task(right))
    published_mismatch = trace_mismatches(_published_calls_by_task(left), _published_calls_by_task(right))

    left_surface = _surfaces_by_task(left)
    right_surface = _surfaces_by_task(right)
    shared_surface = sorted(set(left_surface) & set(right_surface))
    surface_diffs = [
        {"task_id": task_id, "baseline": left_surface[task_id], "candidate": right_surface[task_id]}
        for task_id in shared_surface
        if left_surface[task_id] != right_surface[task_id]
    ]
    # The opening request is what the model is scored on answering. A1 may reword a
    # simulator reply; rewording the request itself would change the task.
    first_turn_diffs = [
        {"task_id": task_id, "baseline": left_surface[task_id][:1], "candidate": right_surface[task_id][:1]}
        for task_id in shared_surface
        if left_surface[task_id][:1] != right_surface[task_id][:1]
    ]

    left_plans = _plans_by_task(left)
    right_plans = _plans_by_task(right)
    plan_mismatch = [
        {"task_id": task_id, "baseline": left_plans[task_id][:400], "candidate": right_plans[task_id][:400]}
        for task_id in sorted(set(left_plans) & set(right_plans))
        if left_plans[task_id] != right_plans[task_id]
    ]

    left_cases = _validation_coverage(baseline.pack_dir)
    right_cases = _validation_coverage(candidate.pack_dir)
    coverage_lost = sorted(left_cases - right_cases)
    coverage_gained = sorted(right_cases - left_cases)

    left_manifest = common.read_json(baseline.run_manifest) or {}
    right_manifest = common.read_json(candidate.run_manifest) or {}

    task_ids_equal = not only_left and not only_right
    traces_equal = not stage_mismatch and not published_mismatch
    first_turns_equal = not first_turn_diffs
    # An arm that is allowed to reword still reports the count; it just is not failed on it.
    first_turns_gate = True if opening_turn_may_change else first_turns_equal
    plans_equal = not plan_mismatch
    coverage_held = not coverage_lost
    return {
        "baseline_arm": baseline.arm,
        "candidate_arm": candidate.arm,
        "task_ids": {
            "equal": task_ids_equal,
            "baseline_count": len(left_ids),
            "candidate_count": len(right_ids),
            "only_in_baseline": only_left,
            "only_in_candidate": only_right,
        },
        "conversation_plans": {
            "equal": plans_equal,
            "compared": len(set(left_plans) & set(right_plans)),
            "mismatches": plan_mismatch,
        },
        "validation_case_coverage": {
            "held": coverage_held,
            "baseline_pairs": len(left_cases),
            "candidate_pairs": len(right_cases),
            "lost": [list(pair) for pair in coverage_lost],
            "gained": [list(pair) for pair in coverage_gained],
        },
        "expected_tool_calls": {
            "equal": traces_equal,
            "compared": len(left_ids & right_ids),
            "stage_mismatches": stage_mismatch,
            "published_mismatches": published_mismatch,
        },
        "surface": {
            "identical": not surface_diffs,
            "first_turns_identical": not first_turn_diffs,
            "opening_turn_may_change": opening_turn_may_change,
            "tasks_with_changed_first_turn": len(first_turn_diffs),
            "first_turn_examples": first_turn_diffs[:5],
            "tasks_with_changed_wording": len(surface_diffs),
            "examples": surface_diffs[:5],
            "note": (
                "Simulator replies are drawn from pack-wide canonical templates at A1, so a "
                "changed first-turn surface would be a real regression while a changed later "
                "turn is the intended trade."
            ),
        },
        "publication": {
            "baseline_published": len(left.get("benchmark") or []),
            "candidate_published": len(right.get("benchmark") or []),
            "baseline_gold_eligible": left_manifest.get("gold_eligible"),
            "candidate_gold_eligible": right_manifest.get("gold_eligible"),
            "gold_preserved": bool(left_manifest.get("gold_eligible")) == bool(right_manifest.get("gold_eligible")),
        },
        "verdict": (
            "EQUIVALENT"
            if task_ids_equal and traces_equal and first_turns_gate and plans_equal and coverage_held
            else "DIVERGED"
        ),
    }


def render(result: dict[str, Any]) -> str:
    ids = result["task_ids"]
    calls = result["expected_tool_calls"]
    surface = result["surface"]
    pub = result["publication"]
    plans = result.get("conversation_plans", {"equal": True, "compared": 0, "mismatches": []})
    cases = result.get("validation_case_coverage", {"held": True, "lost": [], "baseline_pairs": 0, "candidate_pairs": 0})

    lines = [
        f"# Equivalence: `{result['baseline_arm']}` vs `{result['candidate_arm']}`",
        "",
        f"**{result['verdict']}**",
        "",
        "| check | result |",
        "| --- | --- |",
        f"| `set(task_id)` equal | {'YES' if ids['equal'] else 'NO'} "
        f"({ids['baseline_count']} vs {ids['candidate_count']}) |",
        f"| `expected_tool_calls` equal | {'YES' if calls['equal'] else 'NO'} "
        f"({calls['compared']} tasks compared) |",
        f"| `assistant_milestones` / plan equal | {'YES' if plans['equal'] else 'NO'} "
        f"({plans['compared']} tasks compared) |",
        f"| validation-case coverage held | {'YES' if cases['held'] else 'NO'} "
        f"({cases['baseline_pairs']} -> {cases['candidate_pairs']} (tool, outcome) pairs) |",
        f"| gold eligibility preserved | {'YES' if pub['gold_preserved'] else 'NO'} "
        f"({pub['baseline_gold_eligible']} -> {pub['candidate_gold_eligible']}) |",
        f"| published rows | {pub['baseline_published']} -> {pub['candidate_published']} |",
        f"| opening user turn identical | {'YES' if surface['first_turns_identical'] else 'NO'} "
        f"({surface['tasks_with_changed_first_turn']} changed) |",
        f"| all user turns identical | {'YES' if surface['identical'] else 'NO'} "
        f"({surface['tasks_with_changed_wording']} tasks reworded) |",
        "",
    ]

    if not ids["equal"]:
        lines += ["## task_id divergence", ""]
        for task_id in ids["only_in_baseline"][:10]:
            lines.append(f"- only in baseline: `{task_id}`")
        for task_id in ids["only_in_candidate"][:10]:
            lines.append(f"- only in candidate: `{task_id}`")
        lines.append("")

    for label, key in (("stage", "stage_mismatches"), ("published", "published_mismatches")):
        if calls[key]:
            lines += [f"## expected_tool_calls divergence ({label})", ""]
            for entry in calls[key][:5]:
                lines += [
                    f"`{entry['task_id']}`",
                    "",
                    f"- baseline:  `{entry['baseline']}`",
                    f"- candidate: `{entry['candidate']}`",
                    "",
                ]

    if plans["mismatches"]:
        lines += ["## Conversation-plan divergence", ""]
        for entry in plans["mismatches"][:5]:
            lines += [
                f"`{entry['task_id']}`",
                "",
                f"- baseline:  `{entry['baseline']}`",
                f"- candidate: `{entry['candidate']}`",
                "",
            ]

    if cases["lost"]:
        lines += ["## Validation-case coverage lost", ""]
        for tool, outcome in cases["lost"]:
            lines.append(f"- `{tool}` no longer probed for `{outcome}`")
        lines.append("")

    if surface["examples"]:
        lines += ["## Reworded turns (expected at A1)", "", f"> {surface['note']}", ""]
        for entry in surface["examples"]:
            lines.append(f"- `{entry['task_id']}`")
            lines.append(f"  - baseline:  {entry['baseline']}")
            lines.append(f"  - candidate: {entry['candidate']}")
        lines.append("")

    return "\n".join(lines)
