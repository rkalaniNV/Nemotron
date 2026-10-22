# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One review packet, its HTML page, and the decision sheet that round-trips it.

The HTML page is the reading surface: each pending card shows the source quote,
the extracted meaning, and what approve, reject, or edit does. The CSV is the
action sheet. Eligible rules start pre-accepted. A rule that still needs a
decision, every unresolved conflict, and every material judge disagreement is a
card. Representative examples and cluster samples are capped. Disagreements,
undefined actions, missing evidence, and unresolved lists are not. A reviewer
records approve, reject, or edit in the CSV. Display columns on the CSV are
rebuilt on generate; only decision and note are read back. Conflict cards are
identified by content, not by position. The signed packet binds the decision
sheet and the HTML page. Reject, or edit with a note, withholds that rule and
any contrast family that contains it. Rejecting an audit card drops that
cluster from publication.
"""

import csv
import hashlib
import html
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.safety.artifacts import write_atomic
from nemotron.steps.byob.runtime.benchmark_families.safety.dataset import DatasetInputError, load_jsonl_objects
from nemotron.steps.byob.runtime.benchmark_families.safety.labeling import (
    AUDITED_FILE,
    PROFILE_REASONS,
    attach_flagged_items,
    spot_check_rules,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, abort

PACKET_NAME = "publication_review_packet.yaml"
REVIEW_PAGE = "review.html"
DECISIONS_NAME = "review_decisions.csv"
DECISION_COLUMNS = ("card_id", "kind", "policy_id", "rule_status", "decision", "note")
DISPLAY_COLUMNS = ("title", "quote", "expected_action", "why")
CSV_COLUMNS = (*DECISION_COLUMNS, *DISPLAY_COLUMNS)
EXAMPLE_CAP = 3
PACKET_VERSION = 1
_EXAMPLE_TEXT = 240
_COLUMNS = CSV_COLUMNS
_CONDITION_NAMES = ("block_if", "allow_if", "escalate_if", "require_confirmation_if")
_OPERATOR_NAMES = ("all_of", "any_of", "none_of")
_GAP_MEANING = {
    "rule_cap": (
        "This rule already used its per-rule case cap. Raise the matching per-rule "
        "budget named in Recommend, then re-run generate. The CSV cannot add this cell."
    ),
    "budget_cap": "The run hit assisted_max_cases. Raise that budget, then re-run generate.",
    "translation_required": (
        "A required target-language or code-switch cell was not generated. Stage 10 "
        "translates it after paraphrase; a missing model, an identity copy of the source, "
        "or a dropped identifier leaves this gap."
    ),
    "span_missing": "The source span for this cell is missing.",
    "max_turns": "This cell needs more turns than budget.max_turns allows.",
}
_DECISIONS = frozenset({"pending", "pre_accepted", "approve", "reject", "edit"})
_REVIEWER_DECISIONS = frozenset({"pending", "approve", "reject", "edit"})
_RESOLVED = frozenset({"pre_accepted", "approve", "reject"})
_EXCLUDED = frozenset({"quarantined", "rejected"})
_METADATA = ("kind", "policy_id", "rule_status")
_CONTENT_FILES = {
    "deduplicated_cases_sha256": "stage_cache/deduplicated_cases.jsonl",
    "split_manifest_sha256": "stage_cache/split_manifest.json",
    "eligible_rules_sha256": "stage_cache/policy_rules_eligible.yaml",
    "routed_rules_sha256": "stage_cache/policy_rules.jsonl",
    "conflict_report_sha256": "stage_cache/conflict_report.json",
    "coverage_gaps_sha256": "stage_cache/coverage_gaps.json",
    "contrast_families_sha256": "stage_cache/contrast_families.jsonl",
    "validation_report_sha256": "stage_cache/validation_report.json",
    "review_decisions_sha256": DECISIONS_NAME,
    "review_html_sha256": REVIEW_PAGE,
}
_DERIVED_HASHES = frozenset({"review_decisions_sha256", "review_html_sha256"})


@dataclass(frozen=True)
class ReviewBuild:
    """The three review files and the decisions that still block publication."""

    packet: dict[str, Any]
    yaml_text: str
    html: str
    csv_text: str
    pending: tuple[str, ...]
    withheld_policies: frozenset[str]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.yaml_text.encode("utf-8")).hexdigest()


def load_review(
    run_dir: Path,
    *,
    config_sha256: str,
    source_fingerprint: str,
    mode: str,
    profile: str,
    example_cap: int = EXAMPLE_CAP,
) -> ReviewBuild:
    """Read the stage outputs and the current decision sheet into one review build."""
    cache = run_dir / "stage_cache"
    coverage = _mapping(_read_json(cache / "coverage_gaps.json", run_dir), "coverage_gaps.json", run_dir)
    gaps = coverage.get("gaps")
    if not isinstance(gaps, list):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, "coverage_gaps.json must list gaps.")
    report = _mapping(_read_json(cache / "conflict_report.json", run_dir), "conflict_report.json", run_dir)
    conflicts = report.get("conflicts")
    glossary = report.get("glossary_conflicts")
    if not isinstance(conflicts, list) or not isinstance(glossary, list):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            "conflict_report.json must list conflicts and glossary_conflicts.",
        )
    try:
        decisions, metadata = load_decisions(run_dir / DECISIONS_NAME, run_dir)
        split_manifest = _mapping(
            _read_json(cache / "split_manifest.json", run_dir),
            "split_manifest.json",
            run_dir,
        )
        quality_gates = split_manifest.get("quality_gates")
        if not isinstance(quality_gates, Mapping):
            raise ValueError("split_manifest.json must contain quality_gates")
        surfaces = split_manifest.get("unrealized_surfaces", [])
        if not isinstance(surfaces, list) or any(not isinstance(row, Mapping) for row in surfaces):
            raise ValueError("split_manifest.json has damaged unrealized_surfaces")
        return assemble_review(
            config_sha256=config_sha256,
            source_fingerprint=source_fingerprint,
            mode=mode,
            profile=profile,
            rules=_read_jsonl(cache / "policy_rules.jsonl", run_dir),
            conflicts=conflicts,
            glossary_conflicts=glossary,
            coverage=coverage,
            gaps=gaps,
            cases=_read_jsonl(cache / "deduplicated_cases.jsonl", run_dir),
            decisions=decisions,
            decision_metadata=metadata,
            content_hashes=_file_hashes(run_dir),
            example_cap=example_cap,
            quality_gates=quality_gates,
            unrealized_surfaces=surfaces,
            audit_queue=_optional_jsonl(cache / AUDITED_FILE, run_dir),
            rejected_cases=_optional_jsonl(cache / "rejected_cases.jsonl", run_dir),
        )
    except (OSError, ValueError) as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"review packet cannot be built: {exc}") from exc


def assemble_review(
    *,
    config_sha256: str,
    source_fingerprint: str,
    mode: str,
    profile: str,
    rules: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    glossary_conflicts: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    gaps: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, tuple[str, str]],
    content_hashes: Mapping[str, str],
    example_cap: int = EXAMPLE_CAP,
    decision_metadata: Mapping[str, Mapping[str, str]] | None = None,
    quality_gates: Mapping[str, Any] | None = None,
    unrealized_surfaces: Sequence[Mapping[str, Any]] = (),
    audit_queue: Sequence[Mapping[str, Any]] = (),
    rejected_cases: Sequence[Mapping[str, Any]] = (),
) -> ReviewBuild:
    """Build the packet, the page, and the CSV from in-memory rows.

    ``content_hashes`` binds the dataset files. The decision-sheet and review-page
    hashes are added here after those files are rendered, so the signed packet
    covers the reviewer's view.
    """
    if isinstance(example_cap, bool) or not isinstance(example_cap, int) or example_cap < 0:
        raise ValueError("example_cap must be a non-negative integer")
    missing = [key for key in _CONTENT_FILES if key not in _DERIVED_HASHES and key not in content_hashes]
    if missing:
        raise ValueError(f"content hashes missing {', '.join(missing)}")
    audit_queue = attach_flagged_items(audit_queue, cases, _flagged_policy_ids(rules, decisions))
    cards = _cards(rules, conflicts, glossary_conflicts, decisions, audit_queue)
    card_ids = [card["card_id"] for card in cards]
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("review cards are not unique")
    unknown_decisions = sorted(set(decisions) - set(card_ids))
    if unknown_decisions:
        raise ValueError(f"review decisions name unknown cards: {', '.join(unknown_decisions)}")
    _check_decision_metadata(cards, decision_metadata)
    csv_text = _csv_text(cards)
    blocking = [dict(gap) for gap in gaps if isinstance(gap, Mapping) and gap.get("blocks_gold") is True]
    unresolved_conflicts = _unresolved_conflicts(conflicts, glossary_conflicts)
    packet = {
        "packet_version": PACKET_VERSION,
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "assurance_profile": profile,
        "mode": mode,
        "example_cap": example_cap,
        "coverage": {
            "certification": coverage.get("certification"),
            "reason": coverage.get("reason"),
            "required_cases": coverage.get("required_cases"),
            "scheduled_cases": coverage.get("scheduled_cases"),
            "missing_required": list(coverage.get("missing_required") or []),
            "capacity": dict(coverage.get("capacity") or {}),
            "quality_gate_preflight": dict(coverage.get("quality_gate_preflight") or {}),
        },
        "quality_gates": dict(quality_gates or {"enabled": False, "passed": True, "failures": []}),
        "blocking_gaps": sorted(blocking, key=_gap_key),
        "unrealized_surfaces": sorted(
            (dict(row) for row in unrealized_surfaces if isinstance(row, Mapping)),
            key=lambda row: str(row.get("case_id") or ""),
        ),
        "contrast_pairs": _contrast_pairs(cases),
        "adversarial_cases": _adversarial_cases(cases),
        "localized_cases": _localized_cases(cases),
        "symmetry_pairs": _symmetry_pairs(cases),
        "audit_queue": [dict(item) for item in audit_queue if isinstance(item, Mapping)],
        "spot_check_rules": spot_check_rules(rules),
        "undefined_actions": _reason_rows(rejected_cases, "novel_action"),
        "missing_evidence": _reason_rows(rejected_cases, "evidence_missing", "span_missing"),
        "gaps": [dict(gap) for gap in gaps if isinstance(gap, Mapping)],
        "unresolved_rules": _unresolved_rules(rules),
        "excluded_rules": _excluded_rules(rules),
        "unresolved_conflicts": unresolved_conflicts,
        "cards": cards,
        "examples": _examples(cases, example_cap),
        "content": {
            **{key: content_hashes[key] for key in _CONTENT_FILES if key not in _DERIVED_HASHES},
            "review_decisions_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        },
    }
    html = render_html(packet)
    packet["content"]["review_html_sha256"] = hashlib.sha256(html.encode("utf-8")).hexdigest()
    return ReviewBuild(
        packet=packet,
        yaml_text=yaml.safe_dump(packet, sort_keys=True, allow_unicode=True, width=4096),
        html=html,
        csv_text=csv_text,
        pending=tuple(card["card_id"] for card in cards if card_is_pending(card)),
        withheld_policies=withheld_policies(cards),
    )


def install_review(run_dir: Path, build: ReviewBuild) -> None:
    """Write the packet, the page, and the decision sheet together."""
    write_atomic(run_dir / PACKET_NAME, build.yaml_text)
    write_atomic(run_dir / REVIEW_PAGE, build.html)
    write_atomic(run_dir / DECISIONS_NAME, build.csv_text)


def load_decisions(path: Path, run_dir: Path) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, str]]]:
    """Read approve, reject, and edit rows. A missing sheet means no decisions yet."""
    if not path.is_file():
        return {}, {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{DECISIONS_NAME} cannot be read.") from exc
    if not text.strip():
        return {}, {}
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or not set(DECISION_COLUMNS) <= set(reader.fieldnames):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{DECISIONS_NAME} must have columns {', '.join(DECISION_COLUMNS)}.",
        )
    found: dict[str, tuple[str, str]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for row in reader:
        card_id = (row.get("card_id") or "").strip()
        if not card_id:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{DECISIONS_NAME} has a row without a card_id.")
        if card_id in found:
            raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{DECISIONS_NAME} repeats {card_id}.")
        decision = (row.get("decision") or "").strip().casefold()
        if decision not in _DECISIONS:
            raise abort(
                RunStatus.INVALID_RUN_ARTIFACT,
                run_dir,
                f"{card_id} has an unknown decision {row.get('decision')!r}.",
            )
        found[card_id] = (decision, (row.get("note") or "").strip())
        metadata[card_id] = {field: (row.get(field) or "").strip() for field in _METADATA}
    return found, metadata


def render_html(packet: Mapping[str, Any]) -> str:
    """Render the page so a reviewer can decide from the quote, not from an id table."""
    unresolved = packet.get("unresolved_rules") or []
    conflicts = packet.get("unresolved_conflicts") or []
    blocking = packet.get("blocking_gaps") or []
    excluded = packet.get("excluded_rules") or []
    examples = packet.get("examples") or []
    coverage = packet.get("coverage") if isinstance(packet.get("coverage"), Mapping) else {}
    capacity = coverage.get("capacity") if isinstance(coverage.get("capacity"), Mapping) else {}
    preflight = (
        coverage.get("quality_gate_preflight") if isinstance(coverage.get("quality_gate_preflight"), Mapping) else {}
    )
    quality_gates = packet.get("quality_gates") if isinstance(packet.get("quality_gates"), Mapping) else {}
    gate_failures = [row for row in quality_gates.get("failures") or [] if isinstance(row, Mapping)]
    gates_enabled = quality_gates.get("enabled") is True
    on_failure = quality_gates.get("on_failure") if quality_gates.get("on_failure") in {"silver", "fail"} else "silver"
    cards = [card for card in packet.get("cards") or [] if isinstance(card, Mapping)]
    pending_cards = [card for card in cards if card_is_pending(card)]
    kept_cards = [card for card in cards if card.get("decision") == "pre_accepted"]
    decided_cards = [card for card in cards if str(card.get("decision") or "") in {"approve", "reject", "edit"}]
    pre_accepted = len(kept_cards)
    pending = len(pending_cards)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        "<title>Safety review</title>",
        f"<style>{_PAGE_STYLE}</style></head><body>",
        "<h1>Safety review</h1>",
        _task_banner(pending, pre_accepted, len(excluded), len(blocking)),
        f"<p>Cards: {len(cards)}. Pending: {pending}.</p>",
        "<p>Record approve, reject, or edit in <code>review_decisions.csv</code>. "
        "An edit needs a note. Then sign the regenerated packet. Only <code>decision</code> and "
        "<code>note</code> are read back; the other CSV columns are rebuilt on generate.</p>",
        "<p>Representative examples are capped. Every unresolved rule, unresolved conflict, "
        "blocking coverage gap, and unmet minimum dataset quality gate is listed in full.</p>",
        f"<p>Pre-accepted rules: {pre_accepted}. They stay in the benchmark unless "
        "review_decisions.csv sets reject or edit.</p>",
        _decision_legend(),
        _pending_section(pending_cards),
        _kept_section(kept_cards),
        _decided_section(decided_cards),
        "<details><summary>Table view of the same pending items</summary>",
        f"<h3>Unresolved rules ({len(unresolved)})</h3>",
        _rule_table(unresolved),
        f"<h3>Unresolved conflicts ({len(conflicts)})</h3>",
        _conflict_table(conflicts),
        "</details>",
        f"<h2>Blocking coverage gaps ({len(blocking)})</h2>",
        "<p>These missing cells block Gold. They are not CSV decisions.</p>",
        _gap_table(blocking),
        _gap_recommendation(blocking),
        *_unrealized_section(packet.get("unrealized_surfaces") or []),
        *_contrast_section(packet.get("contrast_pairs") or []),
        *_adversarial_section(packet.get("adversarial_cases") or []),
        *_localized_section(packet.get("localized_cases") or []),
        *_symmetry_section(packet.get("symmetry_pairs") or []),
        *_audit_sections(packet),
        f"<h2>Minimum dataset quality gates ({len(gate_failures)} unmet)</h2>",
        _quality_gate_table(gate_failures, enabled=gates_enabled, on_failure=on_failure),
        "<details><summary>Generation capacity preflight</summary>",
        _capacity_preflight(capacity, preflight),
        "</details>",
        *_excluded_section(excluded),
        "<details><summary>Representative examples — capped samples from generated cases</summary>",
        "<p>Capped samples from rules that already generated cases. Pending rules have none.</p>",
        _example_sections(examples),
        "</details>",
        "</body></html>\n",
    ]
    return "\n".join(part for part in parts if part)


def _quality_gate_table(
    rows: Sequence[Mapping[str, Any]], *, enabled: bool = False, on_failure: str = "silver"
) -> str:
    if not enabled:
        return "<p>No minimum dataset quality gates are configured.</p>"
    if on_failure == "fail":
        consequence = "an unmet gate blocks publication after review."
    else:
        consequence = "an unmet gate publishes as Silver."
    if not rows:
        return f"<p>All configured minimum dataset gates passed. On failure, {consequence}</p>"
    body = [
        f"<p>On failure, {consequence}</p>",
        "<table><thead><tr><th>Gate</th><th>Slice</th><th>Required</th>"
        "<th>Actual</th><th>Missing</th></tr></thead><tbody>",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        body.append(
            "<tr><td>"
            + _text(row.get("gate"))
            + "</td><td>"
            + _text(row.get("slice"))
            + "</td><td>"
            + _text(row.get("required"))
            + "</td><td>"
            + _text(row.get("actual"))
            + "</td><td>"
            + _text(row.get("missing"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _capacity_preflight(capacity: Mapping[str, Any], preflight: Mapping[str, Any]) -> str:
    if not capacity:
        return "<p>No capacity preflight is available.</p>"
    possible = preflight.get("possible")
    result = "possible from grounded inventory" if possible is not False else "impossible from grounded inventory"
    return (
        "<p>This is an upper bound before validation, exact deduplication, and splitting.</p>"
        "<p><strong>"
        + _text(result)
        + "</strong></p><ul><li>Global case cap: "
        + _text(capacity.get("case_cap"))
        + "</li><li>Planned candidates: "
        + _text(capacity.get("planned_candidates"))
        + "</li><li>Final-case upper bound: "
        + _text(capacity.get("final_case_upper_bound"))
        + "</li><li>Additional distinct inventory cells: "
        + _text(capacity.get("additional_distinct_cells"))
        + "</li></ul>"
    )


def _excluded_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    parts = [
        f"<h2>Excluded rules ({len(rows)})</h2>",
        "<p>These rules are not in the CSV and cannot be approved from review.</p>",
    ]
    recommendation = _excluded_recommendation(rows)
    if recommendation:
        parts.append(recommendation)
    if rows:
        parts.extend(
            [
                "<details><summary>Show excluded rules</summary>",
                _excluded_table(rows),
                "</details>",
            ]
        )
    return parts


def _gap_recommendation(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    reasons = {str(row.get("reason") or "") for row in rows}
    actions = []
    if "rule_cap" in reasons:
        actions.append(_rule_cap_action(rows))
    if "budget_cap" in reasons:
        actions.append("raise <code>budget.assisted_max_cases</code> (or <code>preview_max_cases</code> in preview)")
    if "max_turns" in reasons:
        actions.append("raise <code>budget.max_turns</code> to at least 2")
    if "translation_required" in reasons:
        actions.append(
            "configure a generation model and re-run generate so Stage 10 can translate, "
            "or provide reviewed native translations"
        )
    if "span_missing" in reasons:
        actions.append("restore an exact source span and quote for the rule")
    if not actions:
        actions.append("inspect each reason above and correct the policy or generation budget")
    return _recommend(
        actions,
        "then run <code>generate</code> again. Do not resolve these gaps by editing the CSV.",
    )


def _rule_cap_action(rows: Sequence[Mapping[str, Any]]) -> str:
    severities = {str(row.get("severity") or "") for row in rows}
    knobs = []
    if severities & {"high", "medium", "low"} or "critical" not in severities:
        knobs.append("<code>budget.cases_per_normal_rule</code>")
    if "critical" in severities:
        knobs.append("<code>budget.cases_per_critical_rule</code>")
    joined = " and ".join(knobs) if len(knobs) == 2 else knobs[0]
    return f"raise {joined} so every required cell of those rules fits"


def _excluded_recommendation(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    blob = " ".join(_reason_blob(row) for row in rows).casefold()
    statuses = {str(row.get("status") or "") for row in rows}
    actions = []
    if "confidence" in blob and "below" in blob:
        actions.append(
            "rewrite those clauses more explicitly so extraction confidence can clear the quarantine threshold"
        )
    if "operator not stated" in blob:
        actions.append("state <code>all_of</code>, <code>any_of</code>, or <code>none_of</code> in the source clause")
    if "acting partners" in blob:
        actions.append("separate allow exceptions from the blocking clause so each rule has one acting pair")
    if "needs_clarification" in blob or "non_testable" in blob or "testability" in blob:
        actions.append("rewrite the clause as one atomic, testable condition with an exact quote")
    if "evidence" in blob or "quote_not_in_span" in blob or "span_missing" in blob:
        actions.append("restore an exact source quote inside its span")
    if "rejected" in statuses:
        actions.append("correct malformed or unsupported rule fields")
    if not actions:
        actions.append("clarify the source policy text using the exclusion reasons shown above")
    return _recommend(
        actions,
        "then run <code>prepare</code> and <code>generate</code> again. "
        "Excluded rules cannot be restored through <code>review_decisions.csv</code>.",
    )


def _reason_blob(row: Mapping[str, Any]) -> str:
    reasons = row.get("reasons")
    if isinstance(reasons, list):
        return " ".join(str(item) for item in reasons)
    return str(reasons or "")


def _recommend(actions: Sequence[str], suffix: str) -> str:
    return '<p class="recommend"><strong>Recommend:</strong> ' + "; ".join(actions) + ", " + suffix + "</p>"


_PAGE_STYLE = (
    "body{font:16px/1.45 system-ui,Segoe UI,sans-serif;max-width:52rem;margin:0 auto;"
    "padding:1.25rem;color:#1c1917}"
    "h1{font-size:1.45rem;margin:0 0 .6rem}"
    "h2{font-size:1.15rem;margin:1.6rem 0 .6rem}"
    "h3{font-size:1rem;margin:.2rem 0 .5rem}"
    ".banner{background:#1c1917;color:#fafaf9;padding:1rem 1.1rem;margin:0 0 1rem}"
    ".banner p{margin:.35rem 0 0}"
    ".legend{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:1rem 0}"
    ".legend div{border:1px solid #d6d3d1;padding:.75rem}"
    ".legend p{margin:0}"
    ".legend p + p{margin:.25rem 0 0}"
    ".item{border:1px solid #d6d3d1;border-left:4px solid #b45309;padding:1rem 1.05rem;margin:0 0 1rem}"
    ".item.conflict{border-left-color:#6d28d9}"
    ".why{margin:0 0 .6rem;font-weight:650}"
    "blockquote{margin:.35rem 0 .8rem;padding:.75rem .9rem;background:#f5f5f4}"
    ".chips span{display:inline-block;border:1px solid #d6d3d1;padding:.1rem .5rem;"
    "margin:0 .35rem .35rem 0;font-size:.8rem}"
    ".mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;color:#57534e}"
    ".recommend{background:#ecfccb;border-left:4px solid #4d7c0f;padding:.75rem .9rem}"
    "table{width:100%;border-collapse:collapse}"
    "th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e7e5e4;"
    "padding:.4rem .5rem;font-size:.9rem}"
    "details{margin:1rem 0}"
    "summary{cursor:pointer;font-weight:650}"
    "@media(max-width:40rem){.legend{grid-template-columns:1fr}}"
)


def _task_banner(pending: int, kept: int, excluded: int, blocking: int) -> str:
    if pending:
        headline = f"{pending} card{'s' if pending != 1 else ''} need a decision."
        next_step = (
            "Open each card below, compare the policy quote with the extracted meaning, "
            "then record approve, reject, or edit in review_decisions.csv."
        )
    else:
        headline = "No cards need a decision."
        next_step = (
            "You can sign the packet. Change a pre-accepted row to reject or edit only if that extraction is wrong."
        )
    extras = []
    if kept:
        extras.append(f"{kept} auto-kept")
    if excluded:
        extras.append(f"{excluded} excluded")
    if blocking:
        extras.append(f"{blocking} coverage gaps block Gold")
    extra = " · ".join(extras)
    return (
        '<div class="banner"><p><strong>'
        + _text(headline)
        + "</strong></p><p>"
        + _text(next_step)
        + "</p>"
        + (f"<p>{_text(extra)}</p>" if extra else "")
        + "</div>"
    )


def _decision_legend() -> str:
    return (
        "<section><h2>What you are deciding</h2>"
        "<p>Each pending card is one extracted policy rule, one unresolved conflict, "
        "or one clustered judge disagreement. Compare the source quote with the extracted "
        "meaning, or the conversation with the Python gold label. Approving a rule that is "
        "not eligible_for_generation does not generate new cases; it remains a documented "
        "Silver limitation.</p>"
        '<div class="legend">'
        "<div><p><strong>approve</strong></p><p>Keep this extraction. For a pending rule, "
        "that documents a limitation and does not generate new cases.</p></div>"
        "<div><p><strong>reject</strong></p><p>Remove this rule, or the rules named on a "
        "conflict, and any contrast family that contains them.</p></div>"
        "<div><p><strong>edit + note</strong></p><p>Same removal as reject for this run. "
        "The note records the intended correction. A note is required. On an audit card, "
        "reject or edit drops that cluster from publication.</p></div>"
        "</div>"
        "<p>Do not change <code>pre_accepted</code> rows unless the extraction is wrong. "
        "Coverage gaps, cluster samples, and spot-checks of auto-accepted rules are not "
        "CSV decisions.</p></section>"
    )


def _pending_section(cards: Sequence[Mapping[str, Any]]) -> str:
    heading = f"<h2>Decide these {len(cards)} cards</h2>" if cards else "<h2>Decide these cards</h2>"
    if not cards:
        return heading + "<p>None. No pending review items.</p>"
    parts = [
        heading,
        "<p>These rows are <code>pending</code> in review_decisions.csv. They block signing "
        "until each one is approve, reject, or edit with a note.</p>",
    ]
    total = len(cards)
    for index, card in enumerate(cards, start=1):
        parts.append(_decision_card(card, index, total))
    return "\n".join(parts)


def _kept_section(cards: Sequence[Mapping[str, Any]]) -> str:
    if not cards:
        return ""
    return (
        f"<details><summary>Auto-kept rules ({len(cards)}) — no action unless wrong</summary>"
        "<p>These eligible rules already generated cases. They stay in the benchmark unless "
        "review_decisions.csv sets reject or edit.</p>" + _compact_card_table(cards) + "</details>"
    )


def _decided_section(cards: Sequence[Mapping[str, Any]]) -> str:
    if not cards:
        return ""
    return f"<details><summary>Already decided ({len(cards)})</summary>" + _compact_card_table(cards) + "</details>"


def _decision_card(card: Mapping[str, Any], index: int, total: int) -> str:
    kind = str(card.get("kind") or "rule")
    css = "item conflict" if kind == "conflict" else "item"
    title = str(card.get("title") or card.get("policy_id") or card.get("card_id") or "")
    why = str(card.get("why") or card.get("summary") or "")
    quote = str(card.get("quote") or "")
    conditions = str(card.get("conditions") or "")
    chips = []
    for value in (
        kind,
        card.get("expected_action"),
        card.get("severity"),
        card.get("section"),
        card.get("decision"),
    ):
        if value:
            chips.append(f"<span>{_text(value)}</span>")
    if kind == "conflict":
        consequence = (
            "Approve keeps this conflict as a documented Silver limitation. Reject or edit withholds the named rules."
        )
    elif kind == "audit":
        consequence = (
            "Approve keeps the Python gold label. Reject or edit with a note drops this "
            "case and the other members of its cluster from publication."
        )
    else:
        consequence = (
            "This rule is not eligible for generation. Approve keeps it as a documented "
            "Silver limitation and does not generate new cases. Reject or edit removes it."
        )
    quote_block = f"<p>Policy quote</p><blockquote>{_text(quote)}</blockquote>" if quote else ""
    extracted = f"<p>Extracted as</p><p>{_text(conditions)}</p>" if conditions else ""
    return (
        f'<article class="{css}">'
        f'<p class="why">Decision {index} of {total}: {_text(why)}</p>'
        f"<h3>{_text(title)}</h3>"
        f'<p class="chips">{"".join(chips)}</p>' + quote_block + extracted + f"<p>{_text(consequence)}</p>"
        f'<p class="mono">CSV row: {_text(card.get("card_id"))}</p>'
        "</article>"
    )


def _compact_card_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [
        "<table><thead><tr><th>Title / quote</th><th>Action</th><th>Decision</th><th>CSV row</th></tr></thead><tbody>"
    ]
    for row in rows:
        title = str(row.get("title") or "")
        quote = str(row.get("quote") or "")
        label = title if title else quote
        detail = quote if title and quote and quote != title else ""
        body.append(
            "<tr><td>"
            + _text(label)
            + ("<br>" + _text(detail) if detail else "")
            + "</td><td>"
            + _text(row.get("expected_action"))
            + "</td><td>"
            + _text(row.get("decision"))
            + '</td><td class="mono">'
            + _text(row.get("card_id"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def withheld_policies(cards: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Policies removed by reject, or by edit with a non-empty note."""
    return frozenset(_withheld(cards))


def card_is_pending(card: Mapping[str, Any]) -> bool:
    """True when the card still needs approve, reject, or a noted edit."""
    decision = str(card.get("decision") or "")
    if decision in _RESOLVED:
        return False
    note = card.get("note")
    return not (decision == "edit" and isinstance(note, str) and note.strip())


def withheld_audit_case_ids(packet: Mapping[str, Any]) -> set[str]:
    """Cases dropped when a reviewer rejects or edits a clustered audit card."""
    cards = packet.get("cards") if isinstance(packet.get("cards"), list) else []
    dropped_cards = set()
    for card in cards:
        if not isinstance(card, Mapping) or str(card.get("kind") or "") != "audit":
            continue
        decision = str(card.get("decision") or "")
        note = card.get("note")
        note_text = note.strip() if isinstance(note, str) else ""
        if decision == "reject" or (decision == "edit" and note_text):
            dropped_cards.add(str(card.get("card_id") or ""))
    dropped: set[str] = set()
    queue = packet.get("audit_queue") if isinstance(packet.get("audit_queue"), list) else []
    for item in queue:
        if not isinstance(item, Mapping) or str(item.get("card_id") or "") not in dropped_cards:
            continue
        case_id = item.get("case_id")
        if isinstance(case_id, str) and case_id:
            dropped.add(case_id)
        members = item.get("member_ids")
        if isinstance(members, list):
            dropped.update(str(member) for member in members if isinstance(member, str) and member)
    dropped.discard("")
    return dropped


def withheld_case_ids(
    cases: Sequence[Mapping[str, Any]], families: Sequence[Mapping[str, Any]], policies: frozenset[str]
) -> set[str]:
    """Cases of a withheld policy, plus every member of a contrast family that loses one."""
    dropped = {
        str(case["case_id"])
        for case in cases
        if isinstance(case.get("case_id"), str) and case.get("policy_id") in policies
    }
    changed = True
    while changed:
        changed = False
        for family in families:
            if not isinstance(family, Mapping):
                continue
            members = family.get("members")
            if not isinstance(members, list):
                continue
            ids = [
                member["case_id"]
                for member in members
                if isinstance(member, Mapping) and isinstance(member.get("case_id"), str)
            ]
            family_policy = family.get("policy_id")
            hit = family_policy in policies or any(case_id in dropped for case_id in ids)
            if not hit:
                continue
            for case_id in ids:
                if case_id not in dropped:
                    dropped.add(case_id)
                    changed = True
    return dropped


def pending_detail(pending: Sequence[str]) -> str:
    """Name the cards that still need a decision, without hiding how many there are."""
    shown = ", ".join(pending[:5])
    extra = f" and {len(pending) - 5} more" if len(pending) > 5 else ""
    return (
        f"Review cards still pending: {shown}{extra}. "
        "Record approve, reject, or edit in review_decisions.csv, then sign the packet again."
    )


def content_matches(run_dir: Path, packet: Mapping[str, Any]) -> str:
    """Empty when the packet's file hashes still match the files. Otherwise the file name."""
    content = packet.get("content")
    if not isinstance(content, Mapping):
        return PACKET_NAME
    for key, relative in _CONTENT_FILES.items():
        expected = content.get(key)
        path = run_dir / relative
        try:
            actual = _sha256(path) if path.is_file() else ""
        except OSError:
            actual = ""
        if not isinstance(expected, str) or actual != expected:
            return path.name
    return ""


def _cards(
    rules: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    glossary_conflicts: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, tuple[str, str]],
    audit_queue: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    cards = [_rule_card(row, decisions) for row in rules if _status(row) not in _EXCLUDED and _status(row)]
    cards.extend(_conflict_card(row, decisions) for row in _sorted_conflicts(conflicts))
    cards.extend(_glossary_card(row, decisions) for row in _sorted_glossary(glossary_conflicts))
    cards.extend(
        _audit_card(item, decisions)
        for item in audit_queue
        if isinstance(item, Mapping) and item.get("needs_decision") is True
    )
    return sorted(cards, key=lambda card: card["card_id"])


def _rule_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    policy_id = _policy_id(row)
    status = _status(row)
    card_id = f"rule:{policy_id}"
    default = "pre_accepted" if status == "eligible_for_generation" else "pending"
    decision, note = _recorded_decision(card_id, default, decisions)
    why = (
        "Auto-kept eligible rule."
        if status == "eligible_for_generation"
        else "; ".join(_reasons(row)) or "Needs a reviewer decision."
    )
    summary = "Pre-accepted eligible rule." if status == "eligible_for_generation" else why
    return {
        "card_id": card_id,
        "kind": "rule",
        "policy_id": policy_id,
        "rule_status": status,
        "summary": summary,
        "decision": decision,
        "note": note,
        "title": _heading(row),
        "quote": _quote(row),
        "expected_action": _expected_action(row),
        "severity": _severity(row),
        "section": _section(row),
        "conditions": _conditions_text(row),
        "why": why,
    }


def _conflict_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    card_id = conflict_card_id(row)
    decision, note = _recorded_decision(card_id, "pending", decisions)
    policies = sorted(_policy_list(row.get("policy_ids")))
    detail = str(row.get("detail") or "")
    return {
        "card_id": card_id,
        "kind": "conflict",
        "policy_id": "|".join(policies),
        "rule_status": str(row.get("type") or "conflict"),
        "summary": detail,
        "decision": decision,
        "note": note,
        "title": str(row.get("type") or "conflict"),
        "quote": "",
        "expected_action": "",
        "severity": "",
        "section": "",
        "conditions": ", ".join(policies),
        "why": detail or "Unresolved conflict.",
    }


def _audit_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    card_id = str(row.get("card_id") or "")
    decision, note = _recorded_decision(card_id, "pending", decisions)
    policy_id = str(row.get("policy_id") or "")
    reason = str(row.get("reason") or "audit")
    size = row.get("cluster_size") if isinstance(row.get("cluster_size"), int) else 1
    python_action = str(row.get("python_action") or "")
    judge_action = str(row.get("judge_action") or "")
    why = (
        f"{reason.replace('_', ' ')}: Python gold {python_action or 'unlabeled'}"
        + (f", judge proposed {judge_action}" if judge_action else "")
        + f". Cluster size {size}."
    )
    return {
        "card_id": card_id,
        "kind": "audit",
        "policy_id": policy_id,
        "rule_status": reason,
        "summary": why,
        "decision": decision,
        "note": note,
        "title": str(row.get("case_id") or card_id),
        "quote": str(row.get("scored_text") or "")[:_EXAMPLE_TEXT],
        "expected_action": python_action,
        "severity": "",
        "section": "",
        "conditions": ", ".join(str(item) for item in row.get("policy_ids") or [] if isinstance(item, str)),
        "why": why,
    }


def _glossary_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    card_id = glossary_card_id(row)
    decision, note = _recorded_decision(card_id, "pending", decisions)
    key = str(row.get("key") or "")
    summary = f'"{key}" has conflicting definitions'
    return {
        "card_id": card_id,
        "kind": "conflict",
        "policy_id": "",
        "rule_status": "glossary",
        "summary": summary,
        "decision": decision,
        "note": note,
        "title": "glossary",
        "quote": key,
        "expected_action": "",
        "severity": "",
        "section": "",
        "conditions": "",
        "why": summary,
    }


def conflict_card_id(row: Mapping[str, Any]) -> str:
    """Stable id for one unresolved conflict, independent of list position."""
    return _stable_card_id("conflict", _conflict_key(row))


def glossary_card_id(row: Mapping[str, Any]) -> str:
    """Stable id for one glossary conflict, independent of list position."""
    return _stable_card_id("glossary", _glossary_key(row))


def _stable_card_id(kind: str, identity: tuple[Any, ...]) -> str:
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{kind}:{digest}"


def _recorded_decision(card_id: str, default: str, decisions: Mapping[str, tuple[str, str]]) -> tuple[str, str]:
    if card_id not in decisions:
        return default, ""
    decision, note = decisions[card_id]
    note = note.strip() if isinstance(note, str) else ""
    if decision == "pre_accepted":
        if default != "pre_accepted":
            raise ValueError(f"{card_id} cannot use pre_accepted; record approve, reject, or edit.")
        return "pre_accepted", note
    if decision not in _REVIEWER_DECISIONS:
        raise ValueError(f"{card_id} has an unknown decision {decision!r}.")
    return decision, note


def _check_decision_metadata(
    cards: Sequence[Mapping[str, str]], metadata: Mapping[str, Mapping[str, str]] | None
) -> None:
    if not metadata:
        return
    by_id = {card["card_id"]: card for card in cards}
    for card_id, meta in metadata.items():
        card = by_id.get(card_id)
        if card is None:
            continue
        for field in _METADATA:
            actual = meta.get(field, "")
            if actual != card[field]:
                raise ValueError(f"{card_id} {field} {actual!r} does not match the current review card.")


def _unresolved_rules(rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in rules:
        if _status(row) != "needs_decision":
            continue
        item = {
            "policy_id": _policy_id(row),
            "status": "needs_decision",
            "reasons": _reasons(row),
            "quote": _quote(row),
        }
        confidence = row.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            item["confidence"] = confidence
        rows.append(item)
    return sorted(rows, key=lambda item: item["policy_id"])


def _excluded_rules(rules: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = [
        {
            "policy_id": _policy_id(row),
            "status": _status(row),
            "title": _heading(row),
            "quote": _quote(row),
            "reasons": "; ".join(_reasons(row)),
        }
        for row in rules
        if _status(row) in _EXCLUDED
    ]
    return sorted(rows, key=lambda item: item["policy_id"])


def _unresolved_conflicts(
    conflicts: Sequence[Mapping[str, Any]], glossary_conflicts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _sorted_conflicts(conflicts):
        rows.append(
            {
                "card_id": conflict_card_id(row),
                "type": str(row.get("type") or "conflict"),
                "policy_ids": _policy_list(row.get("policy_ids")),
                "detail": str(row.get("detail") or ""),
                "case_context": [str(item) for item in row.get("case_context") or []],
            }
        )
    for row in _sorted_glossary(glossary_conflicts):
        rows.append(
            {
                "card_id": glossary_card_id(row),
                "type": "glossary",
                "policy_ids": [],
                "detail": f'"{row.get("key") or ""}" has conflicting definitions',
                "definitions": row.get("definitions") if isinstance(row.get("definitions"), list) else [],
            }
        )
    return rows


def _examples(cases: Sequence[Mapping[str, Any]], example_cap: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        policy_id = case.get("policy_id")
        if isinstance(policy_id, str):
            grouped.setdefault(policy_id, []).append(case)
    ordered = {
        policy_id: sorted(rows, key=lambda case: str(case.get("case_id"))) for policy_id, rows in grouped.items()
    }
    selected: dict[str, list[Mapping[str, Any]]] = {policy_id: [] for policy_id in grouped}
    policy_order = sorted(grouped, key=lambda policy_id: hashlib.sha256(policy_id.encode("utf-8")).hexdigest())
    depth = 0
    remaining = example_cap
    while remaining and any(depth < len(ordered[policy_id]) for policy_id in policy_order):
        for policy_id in policy_order:
            if remaining == 0:
                break
            if depth < len(ordered[policy_id]):
                selected[policy_id].append(ordered[policy_id][depth])
                remaining -= 1
        depth += 1
    sections = []
    for policy_id in sorted(grouped):
        shown = selected[policy_id]
        sections.append(
            {
                "policy_id": policy_id,
                "total": len(ordered[policy_id]),
                "shown": len(shown),
                "cases": [_example_case(case) for case in shown],
            }
        )
    return sections


def _contrast_pairs(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every paraphrased contrast allow beside the block text it was anchored to."""
    rows = []
    for case in cases:
        anchor = case.get("contrast_anchor")
        if not isinstance(anchor, Mapping) or case.get("origin") != "paraphrase":
            continue
        rows.append(
            {
                "case_id": case.get("case_id"),
                "policy_id": case.get("policy_id"),
                "anchored": anchor.get("anchored") is not False and bool(anchor.get("text")),
                "contrast_loose": case.get("contrast_loose") is True,
                "changed_dimension": anchor.get("changed_dimension"),
                "block_case_id": anchor.get("case_id"),
                "block_text": str(anchor.get("text") or "")[:_EXAMPLE_TEXT],
                "allow_text": _scored_text(case)[:_EXAMPLE_TEXT],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            not row.get("contrast_loose"),
            bool(row.get("anchored")),
            str(row.get("case_id") or ""),
        ),
    )


def _audit_sections(packet: Mapping[str, Any]) -> list[str]:
    queue = [item for item in packet.get("audit_queue") or [] if isinstance(item, Mapping)]
    spot = [item for item in packet.get("spot_check_rules") or [] if isinstance(item, str) and item]
    undefined = [item for item in packet.get("undefined_actions") or [] if isinstance(item, Mapping)]
    missing = [item for item in packet.get("missing_evidence") or [] if isinstance(item, Mapping)]
    pending = [
        item
        for item in queue
        if item.get("needs_decision") is True and str(item.get("reason") or "") not in PROFILE_REASONS
    ]
    required = [item for item in queue if str(item.get("reason") or "") in PROFILE_REASONS]
    samples = [
        item
        for item in queue
        if str(item.get("reason") or "") in {"cluster_representative", "review_sample"}
    ]
    parts: list[str] = []
    if pending:
        parts.extend(
            [
                f"<h2>Judge disagreements and novel actions ({len(pending)})</h2>",
                "<p>These rows are CSV cards. Python gold stays unless you reject or edit the card. "
                "<code>max_review_items</code> does not hide them.</p>",
                _audit_table(pending),
            ]
        )
    if required:
        parts.extend(
            [
                f"<h2>Profile-required audit slices ({len(required)})</h2>",
                "<p>Regulated adds every critical boundary. High Assurance adds critical, legal, "
                "political, and native-language Gold cases whose conversation language overlaps "
                "the rule scope. These lists are not truncated. Profile slices that still need a "
                "decision are CSV cards. Flagged or edited rules are listed here without a second "
                "card when the rule row already withholds them.</p>",
                _audit_table(required),
            ]
        )
    if spot:
        parts.extend(
            [
                f"<h2>Spot-check of auto-accepted rules ({len(spot)}, pre-marked approve)</h2>",
                "<p>At least five eligible rules or five percent, whichever is larger. They stay "
                "in the benchmark unless the matching <code>rule:</code> CSV row is reject or edit.</p>",
                "<p>" + ", ".join(_text(item) for item in spot) + "</p>",
            ]
        )
    if samples:
        parts.extend(
            [
                f"<details><summary>Cluster representatives and sample ({len(samples)})</summary>",
                "<p><code>budget.max_review_items</code> caps this section only. One representative "
                "per generation cluster, plus a hash-stable sample of the rest.</p>",
                _audit_table(samples),
                "</details>",
            ]
        )
    if undefined:
        parts.extend(
            [
                f"<h2>Undefined actions ({len(undefined)})</h2>",
                "<p>Custom actions outside the decision contract. They are not Gold rows. "
                "<code>max_review_items</code> does not hide them.</p>",
                _rejection_table(undefined),
            ]
        )
    if missing:
        parts.extend(
            [
                f"<h2>Missing evidence ({len(missing)})</h2>",
                "<p>Cases rejected because source evidence is missing or the span is absent. "
                "<code>max_review_items</code> does not hide them.</p>",
                _rejection_table(missing),
            ]
        )
    return parts


def _audit_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [
        "<table><thead><tr><th>Case</th><th>Reason</th><th>Policies</th><th>Cluster</th>"
        "<th>Python</th><th>Judge</th></tr></thead><tbody>"
    ]
    for row in rows:
        policies = ", ".join(str(item) for item in row.get("policy_ids") or [] if isinstance(item, str))
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("reason"))
            + "</td><td>"
            + _text(policies)
            + "</td><td>"
            + _text(row.get("cluster_size"))
            + "</td><td>"
            + _text(row.get("python_action"))
            + "</td><td>"
            + _text(row.get("judge_action"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _rejection_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = ["<table><thead><tr><th>Case</th><th>Policy</th><th>Reasons</th></tr></thead><tbody>"]
    for row in rows:
        reasons = row.get("reasons")
        reason_text = ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else str(reasons or "")
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("policy_id"))
            + "</td><td>"
            + _text(reason_text)
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _reason_rows(rows: Sequence[Mapping[str, Any]], *needles: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    wanted = set(needles)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reasons = row.get("reasons") if isinstance(row.get("reasons"), list) else []
        if wanted & {str(item) for item in reasons}:
            found.append(dict(row))
    found.sort(key=lambda row: str(row.get("case_id") or ""))
    return found


def _optional_jsonl(path: Path, run_dir: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _read_jsonl(path, run_dir)


def _scored_text(case: Mapping[str, Any]) -> str:
    turns = case.get("turns")
    if not isinstance(turns, list):
        return ""
    texts = [
        turn["text"]
        for turn in turns
        if isinstance(turn, Mapping) and turn.get("scored") is True and isinstance(turn.get("text"), str)
    ]
    return texts[-1] if texts else ""


def _unrealized_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    body = [
        f"<h2>Unrealized allow surfaces ({len(rows)})</h2>",
        "<p>These required allow cells still repeat the source quote, so they are not over-refusal "
        "probes. They were rejected and block Gold. They are not CSV decisions.</p>",
        "<table><thead><tr><th>Case</th><th>Policy</th><th>Kind</th></tr></thead><tbody>",
    ]
    for row in rows:
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("policy_id"))
            + "</td><td>"
            + _text(row.get("kind"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    body.append(
        _recommend(
            [
                "set <code>generation_model_config</code> so paraphrase can run",
                "check <code>stage_cache/paraphrase_report.json</code> for failed replies",
            ],
            "then run <code>generate</code> again.",
        )
    )
    return body


def _contrast_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    unanchored = sum(1 for row in rows if not row.get("anchored"))
    loose = sum(1 for row in rows if row.get("contrast_loose"))
    body = [
        f"<details><summary>Contrast pairs to spot-check ({len(rows)}, {unanchored} without anchor, "
        f"{loose} contrast_loose)</summary>",
        "<p>Each allow was written from its block sibling with one fact removed. Reject the rule "
        "in the CSV if an allow still states the removed fact. <code>contrast_loose</code> rows "
        "and rows without an anchor are listed first. A loose allow dropped the topic lock or "
        "kept a trigger span: it is not an over-refusal probe. Rows without an anchor were "
        "written without a realized block sibling and need the closest check.</p>",
        "<table><thead><tr><th>Allow case</th><th>Removed</th><th>Block text</th>"
        "<th>Allow text</th><th>Lock</th></tr></thead><tbody>",
    ]
    for row in rows:
        block_text = row.get("block_text") if row.get("anchored") else "(no anchor: block kept its seed)"
        lock = "contrast_loose" if row.get("contrast_loose") else "kept"
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("changed_dimension"))
            + "</td><td>"
            + _text(block_text)
            + "</td><td>"
            + _text(row.get("allow_text"))
            + "</td><td>"
            + _text(lock)
            + "</td></tr>"
        )
    body.extend(["</tbody></table>", "</details>"])
    return body


def _adversarial_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "case_id": case.get("case_id"),
            "policy_id": case.get("policy_id"),
            "attack_family": case.get("attack_family"),
            "attack_frame": case.get("attack_frame") if isinstance(case.get("attack_frame"), str) else None,
            "expected_action": case.get("expected_action"),
            "text": _scored_text(case),
        }
        for case in cases
        if case.get("kind") == "adversarial" and case.get("attack_applied") is True
    ]
    return sorted(rows, key=lambda row: str(row.get("case_id") or ""))


def _adversarial_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    body = [
        f"<details><summary>Adversarial cases to spot-check ({len(rows)})</summary>",
        "<p>Each case wraps a violation in an attack and keeps its expected action. Reject the rule "
        "in the CSV if the attack turned the request into something the policy allows, such as "
        "discussion, analysis, or fiction review.</p>",
        "<table><thead><tr><th>Case</th><th>Attack</th><th>Expected</th><th>Frame</th>"
        "<th>Scored text</th></tr></thead><tbody>",
    ]
    for row in rows:
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("attack_family"))
            + "</td><td>"
            + _text(row.get("expected_action"))
            + "</td><td>"
            + _text(row.get("attack_frame") or "(character transform)")
            + "</td><td>"
            + _text(row.get("text"))
            + "</td></tr>"
        )
    body.extend(["</tbody></table>", "</details>"])
    return body


def _localized_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "case_id": case.get("case_id"),
            "policy_id": case.get("policy_id"),
            "kind": case.get("kind"),
            "language": case.get("language"),
            "languages": case.get("languages"),
            "translated_from": case.get("translated_from"),
            "localization_variant": case.get("localization_variant"),
            "expected_action": case.get("expected_action"),
            "text": _scored_text(case),
        }
        for case in cases
        if case.get("origin") == "translation" and case.get("localization_applied") is True
    ]
    return sorted(rows, key=lambda row: str(row.get("case_id") or ""))


def _localized_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    body = [
        f"<details><summary>Localized cases to spot-check ({len(rows)})</summary>",
        "<p>Each case is a translation or code-switch of a realized sibling. Reject the rule "
        "in the CSV if the rewrite changed the policy meaning, dropped an identifier, or is "
        "still the source-language quote.</p>",
        "<table><thead><tr><th>Case</th><th>Kind</th><th>Language</th><th>From</th>"
        "<th>Expected</th><th>Scored text</th></tr></thead><tbody>",
    ]
    for row in rows:
        language = row.get("language") or ",".join(
            item for item in (row.get("languages") or []) if isinstance(item, str)
        )
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("kind"))
            + "</td><td>"
            + _text(language)
            + "</td><td>"
            + _text(row.get("translated_from"))
            + "</td><td>"
            + _text(row.get("expected_action"))
            + "</td><td>"
            + _text(row.get("text"))
            + "</td></tr>"
        )
    body.extend(["</tbody></table>", "</details>"])
    return body


def _symmetry_pairs(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(case.get("case_id") or ""): case for case in cases if isinstance(case, Mapping)}
    rows = []
    for case in cases:
        if not isinstance(case, Mapping) or case.get("swap_applied") is not True:
            continue
        sibling = by_id.get(str(case.get("swapped_from") or ""))
        rows.append(
            {
                "case_id": case.get("case_id"),
                "policy_id": case.get("policy_id"),
                "swap_axis": case.get("swap_axis"),
                "swap_from": case.get("swap_from"),
                "swap_to": case.get("swap_to"),
                "claim": case.get("symmetry_claim"),
                "evidence": case.get("symmetry_evidence"),
                "expected_action": case.get("expected_action"),
                "sibling_text": _scored_text(sibling) if sibling is not None else "",
                "text": _scored_text(case),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("case_id") or ""))


def _symmetry_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    body = [
        f"<details><summary>Symmetry pairs to spot-check ({len(rows)})</summary>",
        "<p>Each case keeps the sibling's expected action and changes only the named actor. "
        "Reject the rule in the CSV if the rewrite changed the request, dropped an identifier, "
        "or treated one actor more harshly than the other.</p>",
        "<table><thead><tr><th>Case</th><th>Axis</th><th>From</th><th>To</th>"
        "<th>Evidence</th><th>Expected</th><th>Sibling text</th><th>Swapped text</th></tr></thead><tbody>",
    ]
    for row in rows:
        body.append(
            "<tr><td>"
            + _text(row.get("case_id"))
            + "</td><td>"
            + _text(row.get("swap_axis"))
            + "</td><td>"
            + _text(row.get("swap_from"))
            + "</td><td>"
            + _text(row.get("swap_to"))
            + "</td><td>"
            + _text(row.get("evidence"))
            + "</td><td>"
            + _text(row.get("expected_action"))
            + "</td><td>"
            + _text(row.get("sibling_text"))
            + "</td><td>"
            + _text(row.get("text"))
            + "</td></tr>"
        )
    body.extend(["</tbody></table>", "</details>"])
    return body


def _example_case(case: Mapping[str, Any]) -> dict[str, Any]:
    turns = case.get("turns")
    text = ""
    if isinstance(turns, list) and turns and isinstance(turns[0], Mapping):
        raw = turns[0].get("text")
        if isinstance(raw, str):
            text = raw[:_EXAMPLE_TEXT]
    return {
        "case_id": case.get("case_id"),
        "kind": case.get("kind"),
        "expected_action": case.get("expected_action"),
        "text": text,
    }


def _sorted_conflicts(conflicts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for row in conflicts:
        if not isinstance(row, Mapping):
            raise ValueError("a conflict must be an object")
        resolution = row.get("resolution", "unresolved")
        if resolution == "unresolved":
            rows.append(row)
    return sorted(rows, key=_conflict_key)


def _sorted_glossary(conflicts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for row in conflicts:
        if not isinstance(row, Mapping):
            raise ValueError("a glossary conflict must be an object")
        rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("key") or ""))


def _conflict_key(row: Mapping[str, Any]) -> tuple:
    return (
        str(row.get("type") or ""),
        tuple(sorted(_policy_list(row.get("policy_ids")))),
        str(row.get("detail") or ""),
        tuple(str(item) for item in row.get("case_context") or []),
    )


def _glossary_key(row: Mapping[str, Any]) -> tuple:
    definitions = row.get("definitions") if isinstance(row.get("definitions"), list) else []
    payload = json.dumps(definitions, ensure_ascii=False, sort_keys=True, default=str)
    return (str(row.get("key") or ""), payload)


def _policy_id(row: Mapping[str, Any]) -> str:
    policy_id = row.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("a rule row needs a policy_id")
    return policy_id


def _status(row: Mapping[str, Any]) -> str:
    if not isinstance(row, Mapping):
        raise ValueError("a rule row must be an object")
    status = row.get("status")
    if not isinstance(status, str):
        raise ValueError(f"{row.get('policy_id')} needs a status")
    return status


def _reasons(row: Mapping[str, Any]) -> list[str]:
    reasons = row.get("reasons")
    if not isinstance(reasons, list):
        return []
    return [str(item) for item in reasons]


def _quote(row: Mapping[str, Any]) -> str:
    nested = _stored_rule(row)
    provenance = nested.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    quote = provenance.get("source_quote")
    return quote if isinstance(quote, str) else ""


def _stored_rule(row: Mapping[str, Any]) -> Mapping[str, Any]:
    rule = row.get("rule")
    return rule if isinstance(rule, Mapping) else {}


def _heading(row: Mapping[str, Any]) -> str:
    nested = _stored_rule(row)
    title = nested.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    quote = _quote(row)
    if quote:
        return quote if len(quote) <= 80 else quote[:77] + "..."
    return _policy_id(row)


def _expected_action(row: Mapping[str, Any]) -> str:
    nested = _stored_rule(row)
    body = nested.get("rule")
    if isinstance(body, Mapping):
        has_allow = isinstance(body.get("allow_if"), Mapping)
        has_block = isinstance(body.get("block_if"), Mapping)
        has_escalate = isinstance(body.get("escalate_if"), Mapping)
        if has_allow and not has_block and not has_escalate:
            return "allow"
        if has_escalate and not has_block:
            return "escalate"
    decision = nested.get("decision")
    if not isinstance(decision, Mapping):
        return ""
    action = decision.get("violation_action")
    return action if isinstance(action, str) else ""


def _severity(row: Mapping[str, Any]) -> str:
    severity = _stored_rule(row).get("severity")
    return severity if isinstance(severity, str) else ""


def _section(row: Mapping[str, Any]) -> str:
    provenance = _stored_rule(row).get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    section = provenance.get("section")
    return section if isinstance(section, str) else ""


def _conditions_text(row: Mapping[str, Any]) -> str:
    body = _stored_rule(row).get("rule")
    if not isinstance(body, Mapping):
        return ""
    parts: list[str] = []
    for name in _CONDITION_NAMES:
        group = body.get(name)
        if not isinstance(group, Mapping):
            continue
        for operator in _OPERATOR_NAMES:
            legs = group.get(operator)
            if isinstance(legs, (list, tuple)) and legs:
                parts.append(f"{name} {operator.replace('_', ' ')}: {', '.join(str(leg) for leg in legs)}")
    return "; ".join(parts)


def _policy_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _flagged_policy_ids(
    rules: Sequence[Mapping[str, Any]], decisions: Mapping[str, tuple[str, str]]
) -> list[str]:
    flagged: list[str] = []
    for row in rules:
        policy_id = _policy_id(row)
        if not policy_id:
            continue
        if f"rule:{policy_id}" not in decisions:
            continue
        decision, note = decisions[f"rule:{policy_id}"]
        note_text = note.strip() if isinstance(note, str) else ""
        if decision == "reject" or (decision == "edit" and note_text):
            flagged.append(policy_id)
    return flagged


def _withheld(cards: Sequence[Mapping[str, Any]]) -> set[str]:
    withheld: set[str] = set()
    for card in cards:
        if not isinstance(card, Mapping):
            continue
        decision = str(card.get("decision") or "")
        note = card.get("note")
        note_text = note.strip() if isinstance(note, str) else ""
        removes = decision == "reject" or (decision == "edit" and bool(note_text))
        if not removes:
            continue
        kind = str(card.get("kind") or "")
        policy_id = str(card.get("policy_id") or "")
        if kind == "rule" and policy_id:
            withheld.add(policy_id)
        elif kind == "conflict":
            withheld.update(part for part in policy_id.split("|") if part)
    return withheld


def _gap_key(gap: Mapping[str, Any]) -> tuple[str, str]:
    return str(gap.get("slot_id") or ""), str(gap.get("reason") or "")


def write_decisions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the decision sheet in the same columns and line endings generate reads back.

    Extra keys are ignored. An empty sheet is refused so a caller cannot replace a
    populated file with a header that has no cards.
    """
    if not rows:
        raise ValueError(f"{DECISIONS_NAME} has no cards to write.")
    cards = []
    for row in rows:
        card = {column: "" if row.get(column) is None else str(row.get(column)) for column in CSV_COLUMNS}
        if not card["card_id"].strip():
            raise ValueError(f"{DECISIONS_NAME} has a row without a card_id.")
        cards.append(card)
    write_atomic(path, _csv_text(cards))


def _csv_text(cards: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for card in cards:
        writer.writerow({column: card.get(column, "") for column in _COLUMNS})
    return buffer.getvalue()


def _rule_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = [
        "<table><thead><tr><th>Policy</th><th>Reasons</th><th>Quote</th></tr></thead><tbody>",
    ]
    for row in rows:
        reasons = ", ".join(str(item) for item in row.get("reasons") or [])
        body.append(
            "<tr><td>"
            + _text(row.get("policy_id"))
            + "</td><td>"
            + _text(reasons)
            + "</td><td>"
            + _text(row.get("quote"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _conflict_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = ["<table><thead><tr><th>Card</th><th>Type</th><th>Policies</th><th>Detail</th></tr></thead><tbody>"]
    for row in rows:
        policies = ", ".join(str(item) for item in row.get("policy_ids") or [])
        body.append(
            "<tr><td>"
            + _text(row.get("card_id"))
            + "</td><td>"
            + _text(row.get("type"))
            + "</td><td>"
            + _text(policies)
            + "</td><td>"
            + _text(row.get("detail"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _gap_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = [
        "<table><thead><tr><th>Missing cell</th><th>Kind</th><th>Reason</th><th>What to do</th></tr></thead><tbody>"
    ]
    for row in rows:
        reason = str(row.get("reason") or "")
        body.append(
            "<tr><td>"
            + _text(row.get("slot_id"))
            + "</td><td>"
            + _text(row.get("kind"))
            + "</td><td>"
            + _text(reason)
            + "</td><td>"
            + _text(_GAP_MEANING.get(reason, "This required cell was not scheduled."))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _excluded_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = ["<table><thead><tr><th>Title / quote</th><th>Status</th><th>Why excluded</th></tr></thead><tbody>"]
    for row in rows:
        title = str(row.get("title") or row.get("policy_id") or "")
        quote = str(row.get("quote") or "")
        policy_id = str(row.get("policy_id") or "")
        body.append(
            "<tr><td>"
            + _text(title)
            + (f"<br>{_text(quote)}" if quote and quote != title else "")
            + (f'<br><span class="mono">{_text(policy_id)}</span>' if policy_id else "")
            + "</td><td>"
            + _text(row.get("status"))
            + "</td><td>"
            + _text(row.get("reasons"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _example_sections(sections: Sequence[Mapping[str, Any]]) -> str:
    if not sections:
        return "<p>None.</p>"
    parts = []
    for section in sections:
        total = section.get("total")
        shown = section.get("shown")
        parts.append(f"<h3>{_text(section.get('policy_id'))}</h3>")
        parts.append(f"<p>Showing {shown} of {total} cases.</p>")
        parts.append("<ul>")
        for case in section.get("cases") or []:
            parts.append(
                "<li>"
                + _text(case.get("case_id"))
                + " "
                + _text(case.get("kind"))
                + " "
                + _text(case.get("expected_action"))
                + ": "
                + _text(case.get("text"))
                + "</li>"
            )
        parts.append("</ul>")
    return "\n".join(parts)


def _text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _read_json(path: Path, run_dir: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read.") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} is not valid JSON.") from exc


def _read_jsonl(path: Path, run_dir: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{path.name} cannot be read.") from exc
    try:
        return load_jsonl_objects(text, path.name)
    except DatasetInputError as exc:
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, str(exc)) from exc


def _mapping(value: Any, name: str, run_dir: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise abort(RunStatus.INVALID_RUN_ARTIFACT, run_dir, f"{name} must contain an object.")
    return value


def _file_hashes(run_dir: Path) -> dict[str, str]:
    return {key: _sha256(run_dir / relative) for key, relative in _CONTENT_FILES.items() if key not in _DERIVED_HASHES}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
