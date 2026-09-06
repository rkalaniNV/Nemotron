"""Claim-level A0-A6 audits and conservative A7 rollups."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bfcl_ablation import common
from bfcl_ablation.measurement.metrics import METRIC_CONTRACT_VERSION
from bfcl_ablation.quality_gate.artifacts import reported_path_exists
from bfcl_ablation.quality_gate.schema import (
    GATE_CONTRACT_VERSION,
    AuditCheck,
    AuditDimension,
    AuditStatus,
    ThresholdPolicy,
    rollup,
)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial rate."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def _check(
    check_id: str,
    arm: str,
    dimension: AuditDimension,
    status: AuditStatus,
    claim: str,
    detail: str,
    *,
    gating: bool = True,
    value: Any = None,
    threshold: Any = None,
    numerator: int | None = None,
    denominator: int | None = None,
    ci95: tuple[float, float] | None = None,
    source_paths: list[str] | None = None,
    caveats: list[str] | None = None,
) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        arm=arm,
        dimension=dimension,
        status=status,
        claim=claim,
        detail=detail,
        gating=gating,
        value=value,
        threshold=threshold,
        numerator=numerator,
        denominator=denominator,
        ci95=ci95,
        source_paths=source_paths or [],
        caveats=caveats or [],
    )


def _rate_status(value: float, maximum: float) -> AuditStatus:
    return "PASS" if value <= maximum else "FAIL"


def _required_missing(inventory: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        key
        for key, record in inventory.items()
        if record.get("required") and (not record.get("present") or record.get("error"))
    )


def _global_checks(
    artifacts: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    missing = _required_missing(inventory)
    present = sum(
        bool(record.get("present") and not record.get("error"))
        for record in inventory.values()
        if record.get("required")
    )
    required = sum(bool(record.get("required")) for record in inventory.values())
    checks.append(
        _check(
            "G-ARTIFACTS",
            "global",
            "integrity",
            "PASS" if not missing else "INCONCLUSIVE",
            "All required A0-A6 artifacts are readable",
            f"{present}/{required} required artifacts are readable"
            + (f"; missing or invalid: {', '.join(missing)}" if missing else ""),
            numerator=present,
            denominator=required,
            source_paths=[record["path"] for record in inventory.values() if record.get("required")],
        )
    )
    for dimension in ("study_validity", "release_readiness"):
        checks.append(
            _check(
                f"G-EVIDENCE-{dimension.upper()}",
                "global",
                dimension,  # type: ignore[arg-type]
                "PASS" if not missing else "INCONCLUSIVE",
                "The evidence bundle is complete enough for the requested decision",
                "All required artifacts are present" if not missing else "Required artifacts are missing",
                source_paths=[inventory[key]["path"] for key in missing],
            )
        )

    versions: dict[str, Any] = {}
    arm_mismatches: list[str] = []
    for arm in range(7):
        key = f"a{arm}_metrics"
        data = artifacts.get(key)
        if not isinstance(data, dict):
            versions[f"a{arm}"] = None
            continue
        versions[f"a{arm}"] = data.get("metrics_version")
        if str(data.get("arm", "")).lower() != f"a{arm}":
            arm_mismatches.append(f"a{arm}:{data.get('arm')!r}")
    version_values = set(versions.values())
    version_ok = version_values == {METRIC_CONTRACT_VERSION}
    checks.append(
        _check(
            "G-METRIC-VERSION",
            "global",
            "integrity",
            "PASS" if version_ok else ("INCONCLUSIVE" if None in version_values else "FAIL"),
            "Every compared arm uses the same metric contract",
            f"Observed versions: {versions}",
            value=versions,
            threshold=METRIC_CONTRACT_VERSION,
            source_paths=[f"A{arm}/metrics.json:metrics_version" for arm in range(7)],
        )
    )
    checks.append(
        _check(
            "G-ARM-TAGS",
            "global",
            "integrity",
            "PASS" if not arm_mismatches else "FAIL",
            "Each metrics artifact identifies the arm encoded by its path",
            "All arm tags agree" if not arm_mismatches else f"Mismatches: {arm_mismatches}",
            source_paths=[f"A{arm}/metrics.json:arm" for arm in range(7)],
        )
    )
    checks.append(
        _check(
            "G-VERSION-ENFORCEMENT",
            "global",
            "study_validity",
            "CONDITIONAL",
            "Metric-version compatibility is enforced before cross-arm comparison",
            "A7 enforces it, but A0-A6 runners only record the version",
            gating=False,
            source_paths=["results/METRICS.md:metrics_version"],
        )
    )
    return checks


def _a0_checks(artifacts: dict[str, Any]) -> list[AuditCheck]:
    data = artifacts.get("a0_metrics")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    distribution = data.get("distribution") or {}
    funnel = data.get("funnel") or {}
    coverage = data.get("coverage") or {}
    surface = data.get("surface") or {}
    policy_total = sum(int(value) for value in (distribution.get("policy_task_counts") or {}).values())
    expanded = int(funnel.get("expanded") or 0)
    checks.append(
        _check(
            "A0-ACCOUNTING",
            "a0",
            "integrity",
            "PASS" if policy_total == expanded and expanded > 0 else "FAIL",
            "A0 policy counts reconcile with expanded tasks",
            f"Policy total {policy_total}; expanded {expanded}",
            value={"policy_total": policy_total, "expanded": expanded},
            source_paths=["A0/metrics.json:distribution.policy_task_counts", "A0/metrics.json:funnel.expanded"],
        )
    )
    tool_missing = list(coverage.get("tools_never_called") or [])
    checks.append(
        _check(
            "A0-TOOL-COVERAGE",
            "a0",
            "release_readiness",
            "PASS" if not tool_missing else "FAIL",
            "Every declared tool is exercised by an expected trace",
            "All declared tools are called" if not tool_missing else f"Never called: {tool_missing}",
            value=len(tool_missing),
            threshold=0,
            source_paths=["A0/metrics.json:coverage.tools_never_called"],
        )
    )
    overall = surface.get("overall") or {}
    templates = int((data.get("authoring_friction") or {}).get("template_count") or 0)
    distinct = int(overall.get("distinct_masked") or 0)
    checks.append(
        _check(
            "A0-SURFACE-BASELINE",
            "a0",
            "study_validity",
            "PASS" if distinct == templates and templates > 0 else "FAIL",
            "A0 has one slot-masked opening per template",
            f"{distinct} masked surfaces across {templates} templates",
            numerator=distinct,
            denominator=templates,
            source_paths=[
                "A0/metrics.json:surface.overall.distinct_masked",
                "A0/metrics.json:authoring_friction.template_count",
            ],
        )
    )
    checks.append(
        _check(
            "A0-PUBLISH-IS-THROUGHPUT",
            "a0",
            "study_validity",
            "CONDITIONAL",
            "Publish rate is interpreted only as pipeline throughput",
            f"Publish rate is {float(funnel.get('publish_rate') or 0):.1%}; it supplies no content-quality evidence",
            gating=False,
            value=funnel.get("publish_rate"),
            source_paths=["A0/metrics.json:funnel.publish_rate", "results/METRICS.md:Publish funnel"],
        )
    )
    bound = int(coverage.get("fixture_entities_bound") or 0)
    total = int(coverage.get("fixture_entities_total") or 0)
    checks.append(
        _check(
            "A0-FIXTURE-COVERAGE",
            "a0",
            "release_readiness",
            "CONDITIONAL",
            "Fixture coverage is reported with its reachability caveat",
            f"{bound}/{total} fixture entities are bound; the denominator includes backend-only rows",
            numerator=bound,
            denominator=total,
            source_paths=["A0/metrics.json:coverage.fixture_entities_bound"],
        )
    )

    sweep = artifacts.get("budget_sweep")
    if isinstance(sweep, list) and sweep:
        surfaces = {int(row["budget"]): int(row["distinct_surfaces"]) for row in sweep}
        tasks = {int(row["budget"]): int(row["tasks"]) for row in sweep}
        stable = len(set(surfaces.values())) == 1
        checks.append(
            _check(
                "A0-BUDGET-SWEEP",
                "a0",
                "study_validity",
                "PASS" if stable else "FAIL",
                "Increasing task budget does not itself increase wording diversity",
                f"Tasks by budget: {tasks}; distinct surfaces: {surfaces}",
                value={"tasks": tasks, "surfaces": surfaces},
                source_paths=["budget_sweep.json"],
            )
        )
    return checks


def _a1_checks(artifacts: dict[str, Any]) -> list[AuditCheck]:
    data = artifacts.get("a1_metrics")
    proof = artifacts.get("a1_equivalence")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    equivalence = data.get("equivalence") or {}
    gates = {
        "task_ids": bool((equivalence.get("task_ids") or {}).get("equal")),
        "expected_tool_calls": bool((equivalence.get("expected_tool_calls") or {}).get("equal")),
        "conversation_plans": bool((equivalence.get("conversation_plans") or {}).get("equal")),
        "validation_coverage": bool((equivalence.get("validation_case_coverage") or {}).get("held")),
        "opening_turn": bool((equivalence.get("surface") or {}).get("first_turns_identical")),
    }
    recomputed = "EQUIVALENT" if all(gates.values()) else "DIVERGED"
    recorded = equivalence.get("verdict")
    proof_recorded = proof.get("verdict") if isinstance(proof, dict) else None
    checks.append(
        _check(
            "A1-EQUIVALENCE",
            "a1",
            "study_validity",
            "PASS" if recorded == recomputed == proof_recorded else "FAIL",
            "A1's equivalence verdict follows all five declared gates",
            f"Gates {gates}; metrics={recorded!r}; proof={proof_recorded!r}",
            value=gates,
            threshold="all true",
            source_paths=["A1/metrics.json:equivalence", "A1/vs_a0_equivalence.json"],
        )
    )
    simplification = data.get("simplification") or {}
    baseline = int((simplification.get("loc_baseline") or {}).get("TOTAL") or 0)
    saved = int(simplification.get("loc_saved") or 0)
    candidate = int((simplification.get("loc_a1") or {}).get("TOTAL") or 0)
    loc_ok = baseline > 0 and baseline - candidate == saved
    checks.append(
        _check(
            "A1-LOC",
            "a1",
            "integrity",
            "PASS" if loc_ok else "FAIL",
            "A1 line savings reconcile with authored-pack totals",
            f"{baseline} - {candidate} = {saved}",
            value={"baseline": baseline, "candidate": candidate, "saved": saved},
            source_paths=["A1/metrics.json:simplification"],
        )
    )
    later = int((equivalence.get("surface") or {}).get("tasks_with_changed_wording") or 0)
    opening = int((equivalence.get("surface") or {}).get("tasks_with_changed_first_turn") or 0)
    checks.append(
        _check(
            "A1-DIALOGUE-SCOPE",
            "a1",
            "release_readiness",
            "CONDITIONAL" if opening == 0 and later > 0 else ("PASS" if opening == later == 0 else "FAIL"),
            "Equivalence does not overclaim byte-identical full conversations",
            f"{opening} opening turns and {later} complete conversations changed",
            value={"opening_changed": opening, "dialogues_changed": later},
            source_paths=["A1/metrics.json:equivalence.surface"],
        )
    )
    return checks


def _a2_checks(
    artifacts: dict[str, Any],
    policy: ThresholdPolicy,
    label_coverage: dict[str, Any],
) -> list[AuditCheck]:
    data = artifacts.get("a2_metrics")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    rungs = [
        {**rung, "_budget": budget_name}
        for budget_name, budget in (data.get("budgets") or {}).items()
        for rung in (budget.get("rungs") or [])
    ]
    structural_matches = all(
        (rung.get("verdict") == "FROZEN")
        == bool(rung.get("task_ids_equal") and rung.get("expected_tool_calls_equal"))
        for rung in rungs
    )
    checks.append(
        _check(
            "A2-FROZEN-SCOPE",
            "a2",
            "integrity",
            "PASS" if structural_matches and rungs else "FAIL",
            "The recorded FROZEN verdict matches its two structural predicates",
            f"{len(rungs)} rungs checked; FROZEN is not treated as semantic equivalence",
            value=len(rungs),
            source_paths=["A2/metrics.json:budgets.*.rungs"],
            caveats=["task IDs and expected calls do not depend on opening-turn wording"],
        )
    )
    checks.append(
        _check(
            "A2-FROZEN-NOT-SEMANTIC",
            "a2",
            "study_validity",
            "CONDITIONAL",
            "A2's FROZEN headline is limited to declared structural fields",
            "Semantic preservation requires independent human labels",
            source_paths=["bfcl_ablation/run_a2.py:FROZEN", "A2/metrics.json:intent_check"],
        )
    )
    generation = data.get("generation") or {}
    no_pool = list(generation.get("templates_without_pool") or [])
    checks.append(
        _check(
            "A2-POOL-COVERAGE",
            "a2",
            "integrity",
            "PASS" if not no_pool else "FAIL",
            "Every authored template has a paraphrase pool",
            "All templates have pools" if not no_pool else f"Missing pools: {no_pool}",
            value=len(no_pool),
            threshold=0,
            source_paths=["A2/metrics.json:generation.templates_without_pool"],
        )
    )
    max_rung = max(rungs, key=lambda row: int(row.get("distinct_masked_surfaces") or 0), default={})
    checks.append(
        _check(
            "A2-DIVERSITY",
            "a2",
            "study_validity",
            "PASS" if int(max_rung.get("distinct_masked_surfaces") or 0) > 17 else "FAIL",
            "A2 increases slot-masked surface diversity",
            (
                f"Maximum {max_rung.get('distinct_masked_surfaces', 0)} surfaces "
                f"at budget={max_rung.get('_budget')} N={max_rung.get('n')}"
            ),
            value=max_rung.get("distinct_masked_surfaces"),
            threshold=">17",
            source_paths=["A2/metrics.json:budgets.*.rungs.distinct_masked_surfaces"],
            caveats=["the maximum combines paraphrase fan-out with a larger task budget"],
        )
    )
    intent = data.get("intent_check") or {}
    recall = float((intent.get("shift_recall") or {}).get("rate") or 0.0)
    canonical_fa = float((intent.get("canonical_false_alarm") or {}).get("rate") or 0.0)
    substitution = intent.get("paraphrase_substitution") or {}
    substitution_rate = float(substitution.get("rate") or 0.0)
    diagnostic_ok = (
        recall >= policy.a2.min_shift_recall
        and canonical_fa <= policy.a2.max_canonical_false_alarm
        and substitution_rate <= policy.a2.max_llm_substitution_rate
    )
    checks.append(
        _check(
            "A2-LLM-CHECKER",
            "a2",
            "study_validity",
            "CONDITIONAL" if diagnostic_ok else "FAIL",
            "The same-family intent checker is diagnostic, not semantic ground truth",
            (
                f"shift recall={recall:.1%}, canonical false alarm={canonical_fa:.1%}, "
                f"substitution flags={substitution_rate:.1%}"
            ),
            value={
                "shift_recall": recall,
                "canonical_false_alarm": canonical_fa,
                "substitution_rate": substitution_rate,
            },
            threshold=policy.a2.model_dump(),
            source_paths=["A2/metrics.json:intent_check"],
            caveats=["generator and checker use the same model family"],
        )
    )
    retained = int(substitution.get("n") or 0)
    rejected = sum(int(value) for value in (generation.get("rejection_reasons") or {}).values())
    reported_rejection = float(generation.get("rejection_rate") or 0.0)
    denominator = retained + rejected
    reconciles = denominator > 0 and math.isclose(rejected / denominator, reported_rejection, abs_tol=5e-5)
    checks.append(
        _check(
            "A2-GENERATION-DENOMINATOR",
            "a2",
            "study_validity",
            "CONDITIONAL" if reconciles else "FAIL",
            "A2 rejection rate states which candidate population it divides by",
            (
                f"{rejected}/{denominator} examined candidates={reported_rejection:.2%}; "
                "this is not all candidates returned because quota overflow was not inspected"
            ),
            gating=False,
            numerator=rejected,
            denominator=denominator,
            source_paths=["A2/metrics.json:generation"],
        )
    )

    sample = label_coverage.get("prevalence_sample") or {}
    controls = label_coverage.get("intent_shift_controls") or {}
    reviewed = int(sample.get("reviewed") or 0)
    required = int(sample.get("required") or 0)
    errors = int(sample.get("errors") or 0)
    human_ci = wilson_interval(errors, reviewed)
    if required == 0 or reviewed < required:
        human_status: AuditStatus = "INCONCLUSIVE"
        human_detail = f"Human prevalence labels complete for {reviewed}/{required} required pairs"
    else:
        rate = errors / reviewed
        if rate > policy.human.max_semantic_error_rate:
            human_status = "FAIL"
        elif human_ci is None or human_ci[1] > policy.human.max_semantic_error_upper_ci95:
            human_status = "INCONCLUSIVE"
        else:
            human_status = "PASS"
        human_detail = f"Human semantic errors {errors}/{reviewed}; Wilson CI95={human_ci}"
    for dimension in ("study_validity", "release_readiness"):
        checks.append(
            _check(
                f"A2-HUMAN-SEMANTICS-{dimension.upper()}",
                "a2",
                dimension,  # type: ignore[arg-type]
                human_status,
                "Independent reviewers confirm wording-only semantic preservation",
                human_detail,
                value=(errors / reviewed) if reviewed else None,
                threshold={
                    "max_rate": policy.human.max_semantic_error_rate,
                    "max_upper_ci95": policy.human.max_semantic_error_upper_ci95,
                },
                numerator=errors,
                denominator=reviewed,
                ci95=human_ci,
                source_paths=["A7/human_labels.yaml"],
            )
        )
    control_reviewed = int(controls.get("reviewed") or 0)
    control_required = int(controls.get("required") or 0)
    control_misses = int(controls.get("misses") or 0)
    if control_required == 0 or control_reviewed < control_required:
        control_status: AuditStatus = "INCONCLUSIVE"
    else:
        control_status = _rate_status(
            control_misses / control_reviewed,
            policy.human.max_control_miss_rate,
        )
    checks.append(
        _check(
            "A2-HUMAN-CONTROLS",
            "a2",
            "study_validity",
            control_status,
            "Human review detects known intent-shift controls",
            f"Misses {control_misses}/{control_reviewed}; required controls {control_required}",
            value=(control_misses / control_reviewed) if control_reviewed else None,
            threshold=policy.human.max_control_miss_rate,
            numerator=control_misses,
            denominator=control_reviewed,
            ci95=wilson_interval(control_misses, control_reviewed),
            source_paths=["A7/human_labels.yaml"],
        )
    )
    return checks


def _a3_checks(artifacts: dict[str, Any], policy: ThresholdPolicy) -> list[AuditCheck]:
    data = artifacts.get("a3_metrics")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    proposal = data.get("proposal") or {}
    accepted = int(proposal.get("accepted") or 0)
    drops = sum(int(value) for value in (proposal.get("drop_buckets") or {}).values())
    requested = int(proposal.get("proposals_requested") or 0)
    checks.append(
        _check(
            "A3-PROPOSAL-ACCOUNTING",
            "a3",
            "integrity",
            "PASS" if accepted + drops == requested and requested > 0 else "FAIL",
            "Accepted and dropped A3 proposals reconcile with requested proposals",
            f"{accepted} accepted + {drops} dropped = {requested} requested",
            value={"accepted": accepted, "dropped": drops, "requested": requested},
            source_paths=["A3/metrics.json:proposal"],
        )
    )
    bias = data.get("selection_bias") or {}
    vacuous = bias.get("vacuous_gold") or {}
    unfalsifiable = float(vacuous.get("unfalsifiable_share_of_accepted") or 0.0)
    missing_tools = list((data.get("coverage") or {}).get("tools_never_called") or [])
    checks.append(
        _check(
            "A3-SKEW-FINDING",
            "a3",
            "study_validity",
            "PASS" if unfalsifiable > policy.a3.max_unfalsifiable_share and missing_tools else "CONDITIONAL",
            "The proposal-plus-validation stack is skewed toward weakly falsifiable tasks",
            f"Unfalsifiable share={unfalsifiable:.1%}; tools never called={len(missing_tools)}",
            value={"unfalsifiable_share": unfalsifiable, "tools_never_called": missing_tools},
            source_paths=["A3/metrics.json:selection_bias", "A3/metrics.json:coverage"],
            caveats=["this identifies stack-level survivorship bias, not an LLM-only causal effect"],
        )
    )
    release_fail = (
        unfalsifiable > policy.a3.max_unfalsifiable_share
        or len(missing_tools) > policy.a3.max_tools_never_called
    )
    checks.append(
        _check(
            "A3-CANDIDATE-COVERAGE",
            "a3",
            "release_readiness",
            "FAIL" if release_fail else "PASS",
            "The A3-generated candidate meets falsifiability and tool-coverage policy",
            f"Unfalsifiable share={unfalsifiable:.1%}; never-called tools={missing_tools}",
            value={"unfalsifiable_share": unfalsifiable, "tools_never_called": len(missing_tools)},
            threshold=policy.a3.model_dump(),
            source_paths=["A3/metrics.json:selection_bias.vacuous_gold", "A3/metrics.json:coverage"],
        )
    )
    pack_exists = reported_path_exists(data.get("pack"))
    checks.append(
        _check(
            "A3-PROVENANCE",
            "a3",
            "study_validity",
            "PASS" if pack_exists else "CONDITIONAL",
            "The exact generated pack behind A3 metrics remains available",
            f"Recorded pack {data.get('pack')!r} exists={pack_exists}",
            source_paths=["A3/metrics.json:pack"],
            caveats=["other a3f/a3g/a3h runs exist without a recorded selection rationale"],
        )
    )
    checks.append(
        _check(
            "A3-HUMAN-SEMANTICS",
            "a3",
            "study_validity",
            "INCONCLUSIVE",
            "Independent reviewers confirm no-tool policy labels and answerability",
            "The current artifact does not preserve a complete blinded A3 review queue",
            source_paths=["A3/metrics.json:selection_bias.vacuous_gold"],
        )
    )
    return checks


def _a4_trial_tally(rows: list[dict[str, Any]], op_class: str) -> tuple[int, int]:
    selected = [
        row
        for row in rows
        if row.get("op_class") == op_class
        and not (op_class == "call_level" and row.get("operator") == "duplicate_call_readonly")
    ]
    return sum(row.get("outcome") == "false_accept" for row in selected), len(selected)


def _a4_checks(artifacts: dict[str, Any], policy: ThresholdPolicy) -> list[AuditCheck]:
    data = artifacts.get("a4_metrics")
    trials = artifacts.get("a4_trials")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    human = (data.get("arms") or {}).get("human") or {}
    strict = human.get("by_class_strict") or {}
    trial_rows = trials.get("human") if isinstance(trials, dict) else None
    reconciled = True
    trial_values: dict[str, dict[str, int]] = {}
    if isinstance(trial_rows, list):
        for op_class in ("argument_level", "call_level", "state_level"):
            false_accept, total = _a4_trial_tally(trial_rows, op_class)
            recorded = strict.get(op_class) or {}
            trial_values[op_class] = {"false_accept": false_accept, "trials": total}
            if false_accept != int(recorded.get("false_accept") or 0) or total != int(recorded.get("trials") or 0):
                reconciled = False
    else:
        reconciled = False
    checks.append(
        _check(
            "A4-TRIAL-RECONCILIATION",
            "a4",
            "integrity",
            "PASS" if reconciled else "FAIL",
            "A4 strict class totals are reproducible from trial rows",
            f"Recomputed totals: {trial_values}",
            value=trial_values,
            source_paths=["A4/trials.json:human", "A4/metrics.json:arms.human.by_class_strict"],
        )
    )
    limits = {
        "argument_level": policy.a4.max_argument_false_acceptance,
        "call_level": policy.a4.max_call_false_acceptance,
        "state_level": policy.a4.max_state_false_acceptance,
    }
    observed: dict[str, float] = {}
    for op_class, maximum in limits.items():
        entry = strict.get(op_class) or {}
        value = float(entry.get("false_acceptance_rate") or 0.0)
        observed[op_class] = value
        checks.append(
            _check(
                f"A4-{op_class.upper()}-FAR",
                "a4",
                "release_readiness",
                _rate_status(value, maximum),
                f"Human assertions bound strict {op_class} false acceptance",
                f"False acceptance={value:.1%}; policy maximum={maximum:.1%}",
                value=value,
                threshold=maximum,
                numerator=int(entry.get("false_accept") or 0),
                denominator=int(entry.get("trials") or 0),
                ci95=wilson_interval(
                    int(entry.get("false_accept") or 0),
                    int(entry.get("trials") or 0),
                ),
                source_paths=[f"A4/metrics.json:arms.human.by_class_strict.{op_class}"],
            )
        )
    checks.append(
        _check(
            "A4-HUMAN-WEAKNESS-FINDING",
            "a4",
            "study_validity",
            "PASS" if any(observed[key] > limits[key] for key in limits) else "CONDITIONAL",
            "A4 supports the claim that human assertions miss material corruptions",
            f"Strict FAR by class: {observed}",
            value=observed,
            source_paths=["A4/metrics.json:arms.human.by_class_strict", "A4/trials.json:human"],
        )
    )
    gold = human.get("gold") or {}
    gold_rate = float(gold.get("pass_rate") or 0.0)
    checks.append(
        _check(
            "A4-GOLD-FLOOR",
            "a4",
            "release_readiness",
            "PASS" if gold_rate >= policy.a4.min_gold_pass_rate else "FAIL",
            "Assertions retain uncorrupted gold episodes",
            f"Gold pass rate={gold_rate:.1%}",
            value=gold_rate,
            threshold=policy.a4.min_gold_pass_rate,
            numerator=int(gold.get("passed") or 0),
            denominator=int(gold.get("instances") or 0),
            source_paths=["A4/metrics.json:arms.human.gold"],
        )
    )
    checks.append(
        _check(
            "A4-FEEDBACK-GENERALIZATION",
            "a4",
            "study_validity",
            "INCONCLUSIVE",
            "LLM-feedback assertion gains generalize to held-out mutations",
            "Feedback assertions were authored and scored on the same mutation plans",
            source_paths=["A4/metrics.json:arms.llm_feedback", "bfcl_ablation/run_a4.py"],
        )
    )
    return checks


def _recompute_a5(trials: list[dict[str, Any]], verdict: str) -> dict[str, Any]:
    paired: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in trials:
        arm = str(row.get("arm") or "")
        task_id = str(row.get("task_id") or "")
        if arm in {"a0", "a2"} and task_id:
            paired[task_id][arm] = bool(row.get(verdict))
    complete = {key: value for key, value in paired.items() if set(value) == {"a0", "a2"}}
    a0_correct = sum(value["a0"] for value in complete.values())
    a2_correct = sum(value["a2"] for value in complete.values())
    a0_only = sum(value["a0"] and not value["a2"] for value in complete.values())
    a2_only = sum(value["a2"] and not value["a0"] for value in complete.values())
    return {
        "n": len(complete),
        "a0_correct": a0_correct,
        "a2_correct": a2_correct,
        "a0_only": a0_only,
        "a2_only": a2_only,
        "discordant": a0_only + a2_only,
    }


def _a5_checks(
    artifacts: dict[str, Any],
    policy: ThresholdPolicy,
    label_coverage: dict[str, Any],
) -> list[AuditCheck]:
    data = artifacts.get("a5_metrics")
    trials = artifacts.get("a5_trials")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    trial_rows = trials if isinstance(trials, list) else []
    recomputed = {name: _recompute_a5(trial_rows, name) for name in ("ast_match", "assertion")}
    recorded = data.get("paired") or {}
    coherent = True
    for name, value in recomputed.items():
        entry = recorded.get(name) or {}
        contingency = entry.get("contingency") or {}
        coherent = coherent and (
            value["n"] == int(entry.get("n") or 0)
            and value["a0_only"] == int(contingency.get("a0_only") or 0)
            and value["a2_only"] == int(contingency.get("a2_only") or 0)
        )
    checks.append(
        _check(
            "A5-PAIR-RECONCILIATION",
            "a5",
            "integrity",
            "PASS" if coherent else "FAIL",
            "A5 paired counts are reproducible from trial rows",
            f"Recomputed: {recomputed}",
            value=recomputed,
            source_paths=["A5/trials.json", "A5/metrics.json:paired"],
        )
    )
    ast = recorded.get("ast_match") or {}
    assertion = recorded.get("assertion") or {}
    delta = float(ast.get("delta") or 0.0)
    discordant = int(ast.get("discordant") or 0)
    n = int(ast.get("n") or 0)
    observed = discordant > 0 and int(assertion.get("discordant") or 0) == 0
    checks.append(
        _check(
            "A5-BEHAVIORAL-FLIPS",
            "a5",
            "study_validity",
            "PASS" if observed else "CONDITIONAL",
            "A5 supports the existence of wording-sensitive call-set behavior",
            (
                f"AST discordant={discordant}/{n}; assertion discordant="
                f"{int(assertion.get('discordant') or 0)}/{int(assertion.get('n') or 0)}"
            ),
            numerator=discordant,
            denominator=n,
            source_paths=["A5/metrics.json:paired", "A5/trials.json"],
            caveats=["both observed flips belong to one template"],
        )
    )
    enough_power = n >= policy.a5.min_paired_tasks and discordant >= policy.a5.min_discordant_pairs
    checks.append(
        _check(
            "A5-EFFECT-ESTIMATE",
            "a5",
            "study_validity",
            "PASS" if enough_power else "INCONCLUSIVE",
            "A5 has enough paired evidence to estimate a wording effect",
            (
                f"n={n} (minimum {policy.a5.min_paired_tasks}); discordant={discordant} "
                f"(minimum {policy.a5.min_discordant_pairs}); McNemar p={ast.get('mcnemar_p')}"
            ),
            value={"n": n, "discordant": discordant, "mcnemar_p": ast.get("mcnemar_p")},
            threshold={
                "min_paired_tasks": policy.a5.min_paired_tasks,
                "min_discordant_pairs": policy.a5.min_discordant_pairs,
            },
            source_paths=["A5/metrics.json:paired.ast_match"],
        )
    )
    checks.append(
        _check(
            "A5-RELEASE-STABILITY",
            "a5",
            "release_readiness",
            "PASS" if abs(delta) <= policy.a5.max_absolute_score_delta and enough_power else (
                "FAIL" if abs(delta) > policy.a5.max_absolute_score_delta else "INCONCLUSIVE"
            ),
            "Canonical and paraphrased wording meet the release stability policy",
            f"Observed AST score delta={delta:+.1%}; allowed absolute delta={policy.a5.max_absolute_score_delta:.1%}",
            value=delta,
            threshold=policy.a5.max_absolute_score_delta,
            source_paths=["A5/metrics.json:paired.ast_match.delta"],
            caveats=["statistical significance is not required to treat observed release regressions as blockers"],
        )
    )
    checks.append(
        _check(
            "A5-EXTERNAL-VALIDITY",
            "a5",
            "study_validity",
            "INCONCLUSIVE",
            "Wording stability generalizes across model families and multiple wording sets",
            "The stored experiment uses one model family and one selected A2 wording",
            threshold={
                "min_model_families": policy.a5.min_model_families,
                "min_wordings": policy.a5.min_wordings,
            },
            source_paths=["A5/metrics.json:target_model", "A5/metrics.json:wordings"],
        )
    )
    human = label_coverage.get("model_disagreements") or {}
    human_reviewed = int(human.get("reviewed") or 0)
    human_required = int(human.get("required") or 0)
    human_rejected = int(human.get("rejected") or 0)
    checks.append(
        _check(
            "A5-HUMAN-DISAGREEMENT",
            "a5",
            "study_validity",
            (
                "INCONCLUSIVE"
                if human_required == 0 or human_reviewed < human_required
                else ("PASS" if human_rejected > 0 else "CONDITIONAL")
            ),
            "Humans adjudicate whether assertion-lenient extra calls violate the benchmark contract",
            f"Reviewed {human_reviewed}/{human_required}; rejected as unacceptable={human_rejected}",
            numerator=human_rejected,
            denominator=human_reviewed,
            source_paths=["A7/human_labels.yaml", "A5/metrics.json:verdict_disagreement"],
        )
    )
    return checks


def _a6_checks(
    artifacts: dict[str, Any],
    policy: ThresholdPolicy,
    label_coverage: dict[str, Any],
) -> list[AuditCheck]:
    data = artifacts.get("a6_metrics")
    trials = artifacts.get("a6_trials")
    triage = artifacts.get("a6_triage")
    if not isinstance(data, dict):
        return []
    checks: list[AuditCheck] = []
    trial_rows = trials if isinstance(trials, list) else []
    killed = Counter(str(row.get("killed_by")) for row in trial_rows)
    triage_counts = (triage.get("counts") or {}) if isinstance(triage, dict) else {}
    verdicts = (triage.get("verdicts") or []) if isinstance(triage, dict) else []
    survived = int((data.get("by_layer") or {}).get("survived") or 0)
    triaged = sum(int(triage_counts.get(key) or 0) for key in ("equivalent", "unreachable", "real_gap"))
    partition_ok = survived == triaged == len(verdicts) == killed.get("survived", 0)
    checks.append(
        _check(
            "A6-TRIAGE-PARTITION",
            "a6",
            "integrity",
            "PASS" if partition_ok else "FAIL",
            "Every raw A6 survivor has exactly one triage classification",
            (
                f"metrics survived={survived}; trial survivors={killed.get('survived', 0)}; "
                f"triage partition={triaged}; verdict rows={len(verdicts)}"
            ),
            value=dict(killed),
            source_paths=["A6/metrics.json:by_layer.survived", "A6/trials.json", "A6/triage.json"],
        )
    )
    l2 = killed.get("L2_expected_traces", 0)
    real_gaps = int(triage_counts.get("real_gap") or 0)
    observable = int(data.get("observable") or 0) + real_gaps
    lower_n = real_gaps
    upper_n = real_gaps + l2
    lower = lower_n / observable if observable else 0.0
    upper = upper_n / observable if observable else 0.0
    bounds = {
        "lower_count": lower_n,
        "upper_count": upper_n,
        "observable": observable,
        "lower_rate": round(lower, 4),
        "upper_rate": round(upper, 4),
    }
    checks.append(
        _check(
            "A6-BLIND-BOUNDS",
            "a6",
            "study_validity",
            "INCONCLUSIVE" if l2 else "PASS",
            "A6 measures all-shipped-layer blind rate without skipping L4/L5",
            (
                f"Current evidence bounds blind mutants at {lower_n}/{observable}–"
                f"{upper_n}/{observable} ({lower:.1%}–{upper:.1%}); "
                f"{l2} L2 rows lack L4/L5 outcomes"
            ),
            value=bounds,
            threshold=policy.a6.max_all_gate_blind_rate,
            numerator=upper_n,
            denominator=observable,
            source_paths=["A6/trials.json", "A6/triage.json", "bfcl_ablation/run_a6.py"],
            caveats=["raw metrics.blind_rate=45/103 is not an all-layer measurement"],
        )
    )
    if lower > policy.a6.max_all_gate_blind_rate:
        release_status: AuditStatus = "FAIL"
    elif upper <= policy.a6.max_all_gate_blind_rate and l2 == 0:
        release_status = "PASS"
    else:
        release_status = "INCONCLUSIVE"
    checks.append(
        _check(
            "A6-RELEASE-BLIND-RATE",
            "a6",
            "release_readiness",
            release_status,
            "Backend blind rate is below the release-policy maximum",
            (
                f"Allowed ≤{policy.a6.max_all_gate_blind_rate:.1%}; "
                f"observed interval {lower:.1%}–{upper:.1%}"
            ),
            value=bounds,
            threshold=policy.a6.max_all_gate_blind_rate,
            source_paths=["A6/trials.json", "A6/triage.json"],
        )
    )
    critical_real = sum(
        verdict.get("classification") == "real_gap"
        and str(verdict.get("severity") or "").lower() in {"high", "critical"}
        for verdict in verdicts
    )
    checks.append(
        _check(
            "A6-REAL-GAPS",
            "a6",
            "study_validity",
            "PASS" if real_gaps > 0 else "CONDITIONAL",
            "A6 supports the existence of concrete backend contract gaps",
            f"Triaged real gaps={real_gaps}; high/critical real gaps={critical_real}",
            value={"real_gaps": real_gaps, "critical_real_gaps": critical_real},
            source_paths=["A6/triage.json"],
        )
    )
    checks.append(
        _check(
            "A6-CRITICAL-GAPS",
            "a6",
            "release_readiness",
            "PASS" if critical_real <= policy.a6.max_critical_real_gaps else "FAIL",
            "No high-severity real gap remains in the reviewed A6 triage",
            f"High/critical real gaps={critical_real}",
            value=critical_real,
            threshold=policy.a6.max_critical_real_gaps,
            source_paths=["A6/triage.json"],
        )
    )
    human = label_coverage.get("mutant_triage") or {}
    human_reviewed = int(human.get("reviewed") or 0)
    human_required = int(human.get("required") or 0)
    human_agreements = int(human.get("agreements") or 0)
    checks.append(
        _check(
            "A6-HUMAN-TRIAGE",
            "a6",
            "study_validity",
            (
                "INCONCLUSIVE"
                if human_required == 0 or human_reviewed < human_required
                else ("PASS" if human_agreements == human_required else "FAIL")
            ),
            "Independent human review confirms the four reported real-gap classifications",
            (
                f"Reviewed {human_reviewed}/{human_required}; "
                f"classification agreements={human_agreements}"
            ),
            source_paths=["A7/human_labels.yaml", "A6/triage.json"],
        )
    )
    return checks


def _dimension_rollup(checks: list[AuditCheck], dimension: AuditDimension) -> AuditStatus:
    return rollup(checks, dimension)


def _arm_rollup(checks: list[AuditCheck], arm: str, dimension: AuditDimension) -> AuditStatus:
    return rollup([check for check in checks if check.arm == arm], dimension)


def run_quality_gate(
    *,
    artifacts: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    policy: ThresholdPolicy,
    threshold_provenance: dict[str, Any],
    label_coverage: dict[str, Any],
    labels_path: str | None,
) -> tuple[dict[str, Any], list[AuditCheck]]:
    """Evaluate frozen artifacts and return summary metrics plus detailed checks."""
    checks: list[AuditCheck] = []
    checks.extend(_global_checks(artifacts, inventory))
    checks.extend(_a0_checks(artifacts))
    checks.extend(_a1_checks(artifacts))
    checks.extend(_a2_checks(artifacts, policy, label_coverage))
    checks.extend(_a3_checks(artifacts, policy))
    checks.extend(_a4_checks(artifacts, policy))
    checks.extend(_a5_checks(artifacts, policy, label_coverage))
    checks.extend(_a6_checks(artifacts, policy, label_coverage))

    duplicate_ids = sorted(
        check_id for check_id, count in Counter(check.check_id for check in checks).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate A7 check IDs: {duplicate_ids}")

    dimensions: tuple[AuditDimension, ...] = (
        "integrity",
        "study_validity",
        "release_readiness",
    )
    overall = {dimension: _dimension_rollup(checks, dimension) for dimension in dimensions}
    arms = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"]
    by_arm = {
        arm: {
            dimension: _arm_rollup(checks, arm, dimension)
            for dimension in dimensions
            if any(check.arm == arm and check.dimension == dimension for check in checks)
        }
        for arm in arms
    }
    status_counts = Counter(check.status for check in checks)
    publication = {
        "PASS": "READY",
        "CONDITIONAL": "READY_WITH_CONDITIONS",
        "FAIL": "NOT_READY",
        "INCONCLUSIVE": "INCONCLUSIVE",
    }[overall["release_readiness"]]
    metrics = {
        "arm": "a7",
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "metrics_version": METRIC_CONTRACT_VERSION,
        "env": common.env_note(),
        "audited_arms": arms,
        "artifacts": inventory,
        "threshold_policy": threshold_provenance,
        "human_labels": {
            "path": labels_path,
            "schema_version": label_coverage.get("schema_version"),
            "items_required": label_coverage.get("items_required"),
            "items_complete": label_coverage.get("items_complete"),
            "complete": label_coverage.get("complete"),
            "issues": label_coverage.get("issues"),
        },
        "check_counts": dict(sorted(status_counts.items())),
        "rollup": overall,
        "by_arm": by_arm,
        "publication_decision": publication,
        "derived": {
            "a6_all_gate_blind_bounds": next(
                (check.value for check in checks if check.check_id == "A6-BLIND-BOUNDS"),
                None,
            ),
            "a5_assertion_accuracy": (artifacts.get("a5_metrics") or {}).get("paired", {}).get(
                "assertion", {}
            ).get("accuracy_a0"),
            "a2_human_semantic_ci95": next(
                (check.ci95 for check in checks if check.check_id == "A2-HUMAN-SEMANTICS-STUDY_VALIDITY"),
                None,
            ),
        },
        "headline_warnings": [
            "A0/A3 publish and gold rates are throughput, not content-quality evidence.",
            "A2 FROZEN checks task IDs and expected calls; it is not semantic equivalence.",
            "A4 must be read per strict operator class; aggregate false acceptance hides the weakness.",
            "A5 assertion agreement is 100%, but assertion accuracy is 32/33 on each wording.",
            "A6 raw blind_rate is not an all-layer result; current evidence bounds it at 3.7%-45.8%.",
        ],
        "definitions": {
            "study_validity": "whether an arm's stated conclusion is supported by its stored evidence",
            "release_readiness": "whether the produced benchmark or variant meets the selected publication policy",
            "INCONCLUSIVE": "required independent evidence is missing or the design cannot identify the claim",
            "CONDITIONAL": "the claim is supported only within an explicitly limited scope",
        },
    }
    return metrics, checks
