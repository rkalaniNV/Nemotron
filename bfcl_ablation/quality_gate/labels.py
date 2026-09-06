"""Build, load, and score the human-review queue used by A7."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation.quality_gate.schema import (
    HUMAN_LABEL_CONTRACT_VERSION,
    Adjudication,
    HumanReviewFile,
    HumanReviewItem,
    ReviewVerdict,
    ThresholdPolicy,
)
from bfcl_ablation.surface.intent_check import disagreement_kind


def load_review_file(path: Path) -> HumanReviewFile:
    """Load a strict human-review YAML file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return HumanReviewFile.model_validate(payload)


def write_review_file(path: Path, review: HumanReviewFile) -> None:
    """Write a deterministic annotator-facing YAML template."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = review.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def _surface_token(population: str, row: dict[str, Any]) -> str:
    identity = "\0".join(
        (
            population,
            str(row.get("template_id") or ""),
            str(int(row.get("variant_index") or 0)),
            str(row.get("text") or ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _surface_item(
    *,
    population: str,
    row: dict[str, Any],
    canonical: dict[str, str],
) -> HumanReviewItem:
    template_id = str(row.get("template_id") or "")
    token = _surface_token(population, row)
    return HumanReviewItem(
        item_id=f"surface-{token}",
        kind="paraphrase_pair",
        source_arm="a2",
        source_ref=f"surface:{token}",
        template_id=template_id,
        variant_index=None,
        reference_text=canonical[template_id],
        candidate_text=str(row.get("text") or ""),
        context={},
    )


def _surface_selections(
    artifacts: dict[str, Any],
    *,
    sample_per_template: int,
) -> tuple[dict[str, str], list[tuple[str, dict[str, Any], str]]]:
    a2 = artifacts.get("a2_metrics") or {}
    intent = a2.get("intent_check") or {}
    rows = intent.get("rows") or {}
    canonical_rows = rows.get("canonical") or []
    paraphrase_rows = rows.get("paraphrases") or []
    shift_rows = rows.get("shifts") or []
    canonical = {
        str(row.get("template_id")): str(row.get("text") or "")
        for row in canonical_rows
        if row.get("template_id")
    }

    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paraphrase_rows:
        template_id = str(row.get("template_id") or "")
        if template_id in canonical:
            by_template[template_id].append(row)

    selected: list[tuple[str, dict[str, Any], str]] = []
    selected_refs: set[str] = set()
    for template_id in sorted(by_template):
        candidates = sorted(
            by_template[template_id],
            key=lambda row: (int(row.get("variant_index") or 0), str(row.get("text") or "")),
        )
        for row in candidates[:sample_per_template]:
            ref = _surface_token("paraphrase", row)
            selected_refs.add(ref)
            selected.append(("paraphrase", row, "deterministic_prevalence"))

    for row in sorted(
        (row for row in paraphrase_rows if disagreement_kind(row) == "substituted"),
        key=lambda row: (str(row.get("template_id") or ""), int(row.get("variant_index") or 0)),
    ):
        ref = _surface_token("paraphrase", row)
        if ref in selected_refs:
            continue
        selected.append(("paraphrase", row, "checker_disagreement_diagnostic"))

    first_shift: dict[str, dict[str, Any]] = {}
    for row in shift_rows:
        template_id = str(row.get("template_id") or "")
        if template_id in canonical and template_id not in first_shift:
            first_shift[template_id] = row
    for template_id in sorted(first_shift):
        selected.append(("shift", first_shift[template_id], "intent_shift_control"))
    return canonical, selected


def build_review_expectations(
    artifacts: dict[str, Any],
    *,
    sample_per_template: int = 3,
) -> dict[str, dict[str, Any]]:
    """Return evaluator-only expected strata; these fields are never emitted to reviewers."""
    _, selections = _surface_selections(
        artifacts,
        sample_per_template=sample_per_template,
    )
    expectations = {
        f"surface-{_surface_token(population, row)}": {
            "sampling_stratum": role,
            "expected_intent_preserved": population != "shift",
        }
        for population, row, role in selections
    }
    triage = artifacts.get("a6_triage") or {}
    for verdict in triage.get("verdicts") or []:
        if verdict.get("classification") == "real_gap":
            expectations[f"a6-real-gap-m{int(verdict['index']):04d}"] = {
                "expected_mutant_classification": "real_gap"
            }
    return expectations


def build_review_queue(
    artifacts: dict[str, Any],
    *,
    sample_per_template: int = 3,
) -> HumanReviewFile:
    """Create a deterministic queue without exposing checker or oracle verdicts.

    The prevalence sample takes the first ``sample_per_template`` variants for each
    template, independent of checker output. Checker-substituted rows outside that
    sample are appended as diagnostics and excluded from prevalence estimates.
    One hidden intent-shift control per template measures whether reviewers can detect
    a known semantic change.
    """
    canonical, selections = _surface_selections(
        artifacts,
        sample_per_template=sample_per_template,
    )
    items = sorted(
        (
            _surface_item(population=population, row=row, canonical=canonical)
            for population, row, _ in selections
        ),
        key=lambda item: item.item_id,
    )

    a5 = artifacts.get("a5_metrics") or {}
    a5_trials = artifacts.get("a5_trials") or []
    trial_by_key = {
        (str(row.get("arm")), str(row.get("task_id"))): row
        for row in a5_trials
        if isinstance(row, dict)
    }
    for example in (a5.get("verdict_disagreement") or {}).get("examples_lenient") or []:
        task_id = str(example.get("task_id") or "")
        trial = trial_by_key.get(("a2", task_id), {})
        items.append(
            HumanReviewItem(
                item_id=f"a5-lenient-{task_id}",
                kind="model_disagreement",
                source_arm="a5",
                source_ref=f"model-output:{hashlib.sha256(task_id.encode()).hexdigest()[:16]}",
                template_id=str(trial.get("template_id") or "") or None,
                task_id=task_id,
                candidate_text=str(trial.get("opening_turn") or "")
                or "Review the observed call set in context.",
                context={"observed_calls": example.get("got") or []},
            )
        )

    triage = artifacts.get("a6_triage") or {}
    for verdict in triage.get("verdicts") or []:
        if verdict.get("classification") != "real_gap":
            continue
        index = int(verdict["index"])
        items.append(
            HumanReviewItem(
                item_id=f"a6-real-gap-m{index:04d}",
                kind="mutant_triage",
                source_arm="a6",
                source_ref=f"triage.verdicts:index={index}",
                context={
                    "mutant_index": index,
                    "lineno": verdict.get("lineno"),
                    "operator": verdict.get("operator"),
                },
            )
        )

    return HumanReviewFile(
        schema_version=HUMAN_LABEL_CONTRACT_VERSION,
        rubric_version="1.0",
        pack_id="banking_vn",
        language="vi",
        reviewers=[],
        items=items,
    )


def merge_review_labels(queue: HumanReviewFile, supplied: HumanReviewFile) -> tuple[HumanReviewFile, list[str]]:
    """Overlay labels onto the generated queue and report completeness drift."""
    issues: list[str] = []
    if supplied.pack_id != queue.pack_id:
        issues.append(f"pack_id mismatch: expected {queue.pack_id!r}, got {supplied.pack_id!r}")
    if supplied.language != queue.language:
        issues.append(f"language mismatch: expected {queue.language!r}, got {supplied.language!r}")
    expected = {item.item_id: item for item in queue.items}
    provided = {item.item_id: item for item in supplied.items}
    missing = sorted(set(expected) - set(provided))
    extra = sorted(set(provided) - set(expected))
    if missing:
        issues.append(f"missing {len(missing)} queue items")
    if extra:
        issues.append(f"contains {len(extra)} unknown queue items")

    merged = deepcopy(queue)
    merged.reviewers = deepcopy(supplied.reviewers)
    merged_by_id = {item.item_id: item for item in merged.items}
    immutable = (
        "kind",
        "source_arm",
        "source_ref",
        "template_id",
        "task_id",
        "variant_index",
        "reference_text",
        "candidate_text",
        "context",
    )
    for item_id in sorted(set(expected) & set(provided)):
        source = provided[item_id]
        target = merged_by_id[item_id]
        drift = [field for field in immutable if getattr(source, field) != getattr(target, field)]
        if drift:
            issues.append(f"item {item_id!r} changed immutable fields: {', '.join(drift)}")
            continue
        target.labels = deepcopy(source.labels)
        target.adjudication = deepcopy(source.adjudication)
    return HumanReviewFile.model_validate(merged.model_dump()), issues


def _effective_verdict(item: HumanReviewItem) -> ReviewVerdict | Adjudication | None:
    if item.adjudication is not None:
        return item.adjudication
    if not item.labels:
        return None
    first = item.labels[0]
    comparable = (
        "intent_preserved",
        "acceptable_for_benchmark",
        "required_tools",
        "turn_policy",
        "mutant_classification",
        "severity",
    )
    if any(any(getattr(label, key) != getattr(first, key) for key in comparable) for label in item.labels[1:]):
        return None
    return first


def label_coverage(
    review: HumanReviewFile,
    policy: ThresholdPolicy,
    issues: list[str],
    expectations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize completeness, agreement, and semantic/control outcomes."""
    expectations = expectations or {}
    required = int(policy.human.min_reviewers_per_item)
    complete = 0
    disagreements = 0
    prevalence_total = 0
    prevalence_errors = 0
    control_total = 0
    control_misses = 0
    critical_errors = 0
    expected_prevalence = 0
    expected_controls = 0
    kind_required: defaultdict[str, int] = defaultdict(int)
    kind_complete: defaultdict[str, int] = defaultdict(int)
    mutant_reviewed = 0
    mutant_agreements = 0
    model_disagreements_reviewed = 0
    model_disagreements_rejected = 0

    item_rows: list[dict[str, Any]] = []
    for item in review.items:
        kind_required[item.kind] += 1
        enough = len(item.labels) >= required
        effective = _effective_verdict(item) if enough else None
        expected = expectations.get(item.item_id) or {}
        stratum = expected.get("sampling_stratum") or item.context.get("sampling_stratum")
        expected_intent = expected.get("expected_intent_preserved")
        is_control = expected_intent is False or item.kind == "control"
        if item.kind == "paraphrase_pair" and stratum == "deterministic_prevalence":
            expected_prevalence += 1
        if is_control:
            expected_controls += 1
        if enough and effective is None:
            disagreements += 1
        if enough and effective is not None:
            complete += 1
            kind_complete[item.kind] += 1
            if item.kind == "paraphrase_pair" and stratum == "deterministic_prevalence":
                prevalence_total += 1
                bad = effective.intent_preserved is not True or not effective.acceptable_for_benchmark
                prevalence_errors += int(bad)
                critical_errors += int(bad and effective.severity == "critical")
            elif is_control:
                control_total += 1
                control_misses += int(effective.intent_preserved is not False)
            elif item.kind in {"model_disagreement", "mutant_triage", "task_semantics"}:
                critical_errors += int(
                    not effective.acceptable_for_benchmark and effective.severity == "critical"
                )
                if item.kind == "mutant_triage":
                    mutant_reviewed += 1
                    mutant_agreements += int(
                        effective.mutant_classification
                        == expected.get(
                            "expected_mutant_classification",
                            item.context.get("reported_classification"),
                        )
                    )
                elif item.kind == "model_disagreement":
                    model_disagreements_reviewed += 1
                    model_disagreements_rejected += int(not effective.acceptable_for_benchmark)
        item_rows.append(
            {
                "item_id": item.item_id,
                "kind": item.kind,
                "labels": len(item.labels),
                "adjudicated": item.adjudication is not None,
                "complete": enough and effective is not None,
                "sampling_stratum": stratum,
            }
        )

    return {
        "schema_version": review.schema_version,
        "reviewers_declared": len(review.reviewers),
        "min_reviewers_per_item": required,
        "items_required": len(review.items),
        "items_complete": complete,
        "complete": complete == len(review.items) and not issues,
        "disagreements_unadjudicated": disagreements,
        "by_kind": {
            kind: {
                "required": required_count,
                "complete": kind_complete.get(kind, 0),
            }
            for kind, required_count in sorted(kind_required.items())
        },
        "prevalence_sample": {
            "errors": prevalence_errors,
            "reviewed": prevalence_total,
            "required": expected_prevalence,
        },
        "intent_shift_controls": {
            "misses": control_misses,
            "reviewed": control_total,
            "required": expected_controls,
        },
        "critical_errors": critical_errors,
        "mutant_triage": {
            "reviewed": mutant_reviewed,
            "agreements": mutant_agreements,
            "required": kind_required.get("mutant_triage", 0),
        },
        "model_disagreements": {
            "reviewed": model_disagreements_reviewed,
            "rejected": model_disagreements_rejected,
            "required": kind_required.get("model_disagreement", 0),
        },
        "issues": list(issues),
        "items": item_rows,
    }
