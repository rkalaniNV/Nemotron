"""Human-readable rendering for A7; JSON remains the evidence record."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from bfcl_ablation.quality_gate.schema import AuditCheck


def _cell(value: Any, limit: int = 180) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        text = f"{value:.4f}"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render(
    metrics: dict[str, Any],
    checks: list[AuditCheck],
    label_coverage: dict[str, Any],
) -> str:
    out: list[str] = []
    add = out.append
    add("# BFCL ablation — A7 Independent Quality Gate")
    add("")
    add(f"**Publication decision: `{metrics['publication_decision']}`.**")
    add("")
    add(
        "A7 audits frozen A0–A6 evidence. It does not rerun the production pipeline, "
        "call an LLM, or turn missing human evidence into a pass."
    )
    add("")
    add("## Decision split")
    add("")
    for dimension, status in metrics["rollup"].items():
        add(f"- `{dimension}`: **{status}**")
    add("")
    add(
        "`study_validity` asks whether each experimental conclusion is supported. "
        "`release_readiness` asks whether the generated benchmark variants are safe to publish."
    )
    add("")

    add("## Headline warnings")
    add("")
    for warning in metrics["headline_warnings"]:
        add(f"- {warning}")
    add("")

    add("## Human-review coverage")
    add("")
    add(
        f"Complete items: **{label_coverage.get('items_complete', 0)}/"
        f"{label_coverage.get('items_required', 0)}**; "
        f"declared reviewers: {label_coverage.get('reviewers_declared', 0)}; "
        f"unadjudicated disagreements: {label_coverage.get('disagreements_unadjudicated', 0)}."
    )
    add("")
    if label_coverage.get("issues"):
        for issue in label_coverage["issues"]:
            add(f"- Label issue: {issue}")
        add("")
    prevalence = label_coverage.get("prevalence_sample") or {}
    controls = label_coverage.get("intent_shift_controls") or {}
    add(
        f"- Paraphrase prevalence sample: {prevalence.get('reviewed', 0)}/"
        f"{prevalence.get('required', 0)} reviewed; errors={prevalence.get('errors', 0)}."
    )
    add(
        f"- Intent-shift controls: {controls.get('reviewed', 0)}/"
        f"{controls.get('required', 0)} reviewed; misses={controls.get('misses', 0)}."
    )
    add("")

    grouped: dict[str, list[AuditCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.arm].append(check)
    arm_order = ["global", "a0", "a1", "a2", "a3", "a4", "a5", "a6"]
    add("## Checks by arm")
    add("")
    for arm in arm_order:
        arm_checks = grouped.get(arm) or []
        if not arm_checks:
            continue
        add(f"### {arm.upper() if arm != 'global' else 'Global'}")
        add("")
        add("| check | dimension | status | value | detail |")
        add("| --- | --- | --- | --- | --- |")
        for check in arm_checks:
            add(
                f"| `{check.check_id}` | {check.dimension} | **{check.status}** | "
                f"{_cell(check.value)} | {_cell(check.detail)} |"
            )
        add("")
        for check in arm_checks:
            if check.caveats:
                add(f"- `{check.check_id}` caveat: {'; '.join(check.caveats)}")
        if any(check.caveats for check in arm_checks):
            add("")

    blockers = [
        check
        for check in checks
        if check.gating and check.status in {"FAIL", "INCONCLUSIVE"}
    ]
    add("## What blocks a clean release")
    add("")
    if not blockers:
        add("No gating failure or missing evidence remains.")
    else:
        for check in blockers:
            add(f"- **{check.status}** `{check.check_id}` — {check.detail}")
    add("")

    add("## Artifact provenance")
    add("")
    add("| artifact | present | metrics version | sha256 |")
    add("| --- | --- | --- | --- |")
    for key, record in metrics["artifacts"].items():
        digest = str(record.get("sha256") or "-")
        add(
            f"| `{key}` | {record.get('present', False)} | "
            f"{record.get('metrics_version', '-')} | `{digest[:16]}` |"
        )
    add("")
    add(
        "Threshold policy: "
        f"`{metrics['threshold_policy']['path']}` "
        f"(`{metrics['threshold_policy']['sha256'][:16]}`), "
        f"contract `{metrics['gate_contract_version']}`."
    )
    add("")
    return "\n".join(out)
