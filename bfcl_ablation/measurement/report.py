"""Render a measurement payload as Markdown.

The JSON is the record; this is what a person reads. Numbers that would mislead if
skimmed carry their caveat inline rather than in a footnote.
"""

from __future__ import annotations

from typing import Any

_CELL_MARK = {"empty_structural": "--", "empty_unwritten": "."}


def _matrix_table(dist: dict[str, Any]) -> list[str]:
    policies = dist["policies"]
    width = max([18] + [len(p) + 2 for p in policies])
    lines = [
        "| category | " + " | ".join(policies) + " |",
        "| --- | " + " | ".join("---:" for _ in policies) + " |",
    ]
    for category, row in dist["matrix"].items():
        cells = []
        for policy in policies:
            cell = row[policy]
            status = cell.get("status")
            cells.append(str(cell["tasks"]) if not status else _CELL_MARK.get(status, "?"))
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    del width
    return lines


def render(metrics: dict[str, Any]) -> str:
    friction = metrics["authoring_friction"]
    dist = metrics["distribution"]
    cov = metrics["coverage"]
    surf = metrics["surface"]
    fun = metrics["funnel"]

    out: list[str] = []
    add = out.append

    add(f"# BFCL ablation — arm `{metrics['arm']}`")
    add("")
    add(f"Pack: `{metrics['pack']}`")
    add("")

    add("## 1. Authoring friction")
    add("")
    add("| file | lines |")
    add("| --- | ---: |")
    for name, count in friction["loc_by_file"].items():
        if name == "TOTAL":
            continue
        add(f"| {name} | {count} |")
    add(f"| **TOTAL** | **{friction['loc_total']}** |")
    add("")
    add(
        f"{friction['template_count']} templates across {friction['category_count']} categories "
        f"and {friction['policy_count']} turn policies "
        f"({friction['loc_per_template']} template lines each)."
    )
    add("")

    add("## 2. Distribution — joint (category x policy)")
    add("")
    add("Cell = task count. `.` = feasible but unwritten. `--` = structurally empty.")
    add("")
    out.extend(_matrix_table(dist))
    add("")
    add(
        f"{dist['cells_populated']}/{dist['cells_total']} cells populated; "
        f"{dist['cells_empty_unwritten']} unwritten; {dist['cells_empty_structural']} structurally empty."
    )
    add("")
    add(f"> {dist['caveat']}")
    add("")
    add("| turn_policy | tasks | share |")
    add("| --- | ---: | ---: |")
    for policy, count in dist["policy_task_counts"].items():
        add(f"| {policy} | {count} | {dist['policy_task_share'][policy]:.1%} |")
    add("")

    add("## 3. Coverage")
    add("")
    add("| collection | key | rows | bound | coverage | never bound |")
    add("| --- | --- | ---: | ---: | ---: | --- |")
    for name, entry in cov["fixtures"].items():
        never = ", ".join(entry["entities_never_bound"][:6]) or "-"
        if len(entry["entities_never_bound"]) > 6:
            never += f" (+{len(entry['entities_never_bound']) - 6})"
        pct = f"{entry['coverage']:.0%}" if entry["coverage"] is not None else "n/a"
        add(f"| {name} | {entry['primary_key']} | {entry['rows']} | {entry['entities_bound']} | {pct} | {never} |")
    add("")
    add(
        f"Fixture entities bound: {cov['fixture_entities_bound']}/{cov['fixture_entities_total']}."
    )
    if cov["tools_never_called"]:
        add(f"Tools never called in any expected trace: `{'`, `'.join(cov['tools_never_called'])}`.")
    else:
        add("Every declared tool appears in at least one expected trace.")
    add("")

    add("## 4. Surface diversity")
    add("")
    overall = surf["overall"]
    add(
        f"{overall['tasks']} tasks -> {overall['distinct_raw']} distinct raw first turns -> "
        f"**{overall['distinct_masked']} distinct slot-masked** "
        f"({overall['surfaces_per_template']} per template)."
    )
    add("")
    add("Slot-masking substitutes each bound value with its slot name, so two tasks that ")
    add("differ only by account id collapse to one sentence.")
    add("")
    add("| category | tasks | distinct masked | ratio |")
    add("| --- | ---: | ---: | ---: |")
    for name, entry in surf["by_category"].items():
        ratio = f"{entry['ratio']:.2f}" if entry["ratio"] is not None else "n/a"
        add(f"| {name} | {entry['tasks']} | {entry['distinct_masked']} | {ratio} |")
    add("")
    if surf["most_repeated_masked"]:
        add("Most-repeated masked utterances:")
        add("")
        for entry in surf["most_repeated_masked"][:5]:
            add(f"- x{entry['tasks']} — `{entry['utterance']}`")
        add("")
    probe = surf["lexical_shortcut_probe"]
    add(f"**Lexical-shortcut probe:** {'runnable' if probe['runnable'] else 'not runnable'}. {probe['reason']}")
    add("")

    add("## 5. Publish funnel")
    add("")
    add("| stage | rows | survival | lost here |")
    add("| --- | ---: | ---: | ---: |")
    for step in fun["steps"]:
        add(f"| {step['stage']} | {step['rows']} | {step['survival_from_expand']:.1%} | {step['lost_here']} |")
    add("")
    for step in fun["steps"]:
        if step["drop_reasons"]:
            add(f"- `{step['stage']}` drops: {step['drop_reasons']}")
    add("")
    add(
        f"Publish rate {fun['publish_rate']:.1%} ({fun['published']}/{fun['expanded']}). "
        f"Gold rows {fun['gold_rows']}/{fun['published']} ({fun['gold_rate']:.1%}). "
        f"Run gold_eligible: `{fun['run_gold_eligible']}`, mode `{fun['generation_mode']}`."
    )
    add("")
    return "\n".join(out)
