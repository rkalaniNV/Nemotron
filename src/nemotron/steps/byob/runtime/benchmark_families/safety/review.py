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

Eligible rules start pre-accepted. A rule that still needs a decision, and every
unresolved conflict, is a card. Representative examples are capped. The unresolved
lists are not. A reviewer records approve, reject, or edit in the CSV. Conflict
cards are identified by content, not by position. The signed packet binds the
decision sheet and the HTML page. Reject, or edit with a note, withholds that
rule and any contrast family that contains it.
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
from nemotron.steps.byob.runtime.benchmark_families.safety.status import RunStatus, abort

PACKET_NAME = "publication_review_packet.yaml"
REVIEW_PAGE = "review.html"
DECISIONS_NAME = "review_decisions.csv"
EXAMPLE_CAP = 3
PACKET_VERSION = 1
_EXAMPLE_TEXT = 240
_COLUMNS = ("card_id", "kind", "policy_id", "rule_status", "decision", "note")
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
) -> ReviewBuild:
    """Build the packet, the page, and the CSV from in-memory rows.

    ``content_hashes`` binds the dataset files. The decision-sheet and review-page
    hashes are added here after those files are rendered, so the signed packet
    covers the reviewer's view.
    """
    if isinstance(example_cap, bool) or not isinstance(example_cap, int) or example_cap < 1:
        raise ValueError("example_cap must be a positive integer")
    missing = [key for key in _CONTENT_FILES if key not in _DERIVED_HASHES and key not in content_hashes]
    if missing:
        raise ValueError(f"content hashes missing {', '.join(missing)}")
    cards = _cards(rules, conflicts, glossary_conflicts, decisions)
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
        },
        "blocking_gaps": sorted(blocking, key=_gap_key),
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


def load_decisions(
    path: Path, run_dir: Path
) -> tuple[dict[str, tuple[str, str]], dict[str, dict[str, str]]]:
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
    if reader.fieldnames is None or not set(_COLUMNS) <= set(reader.fieldnames):
        raise abort(
            RunStatus.INVALID_RUN_ARTIFACT,
            run_dir,
            f"{DECISIONS_NAME} must have columns {', '.join(_COLUMNS)}.",
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
    """Render the page from the packet so a capped example cannot hide an unresolved item."""
    unresolved = packet.get("unresolved_rules") or []
    conflicts = packet.get("unresolved_conflicts") or []
    blocking = packet.get("blocking_gaps") or []
    excluded = packet.get("excluded_rules") or []
    examples = packet.get("examples") or []
    cards = packet.get("cards") or []
    pre_accepted = sum(1 for card in cards if card.get("decision") == "pre_accepted")
    pending = sum(1 for card in cards if card_is_pending(card))
    estimated_minutes = max(1, (len(cards) + 1) // 2)
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        "<title>Safety review</title></head><body>",
        "<h1>Safety review</h1>",
        f"<p>Cards: {len(cards)}. Pending: {pending}. Estimated review time: {estimated_minutes} minutes.</p>",
        "<p>Record approve, reject, or edit in <code>review_decisions.csv</code>. "
        "An edit needs a note. Then sign the regenerated packet.</p>",
        "<p>Representative examples are capped. Every unresolved rule, unresolved conflict, "
        "and blocking coverage gap is listed in full.</p>",
        f"<p>Pre-accepted rules: {pre_accepted}. They stay in the benchmark unless "
        "review_decisions.csv sets reject or edit.</p>",
        "<h2>Review cards</h2>",
        _card_table(cards),
        f"<h2>Unresolved rules ({len(unresolved)})</h2>",
        _rule_table(unresolved),
        f"<h2>Unresolved conflicts ({len(conflicts)})</h2>",
        _conflict_table(conflicts),
        f"<h2>Blocking coverage gaps ({len(blocking)})</h2>",
        _gap_table(blocking),
        f"<h2>Excluded rules ({len(excluded)})</h2>",
        _excluded_table(excluded),
        "<h2>Representative examples</h2>",
        _example_sections(examples),
        "</body></html>\n",
    ]
    return "\n".join(parts)


def _card_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = [
        "<table><thead><tr><th>Card</th><th>Kind</th><th>Policy</th>"
        "<th>Status</th><th>Decision</th><th>Summary</th></tr></thead><tbody>"
    ]
    for row in rows:
        body.append(
            "<tr><td>"
            + _text(row.get("card_id"))
            + "</td><td>"
            + _text(row.get("kind"))
            + "</td><td>"
            + _text(row.get("policy_id"))
            + "</td><td>"
            + _text(row.get("rule_status"))
            + "</td><td>"
            + _text(row.get("decision"))
            + "</td><td>"
            + _text(row.get("summary"))
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
) -> list[dict[str, str]]:
    cards = [_rule_card(row, decisions) for row in rules if _status(row) not in _EXCLUDED and _status(row)]
    cards.extend(_conflict_card(row, decisions) for row in _sorted_conflicts(conflicts))
    cards.extend(_glossary_card(row, decisions) for row in _sorted_glossary(glossary_conflicts))
    return sorted(cards, key=lambda card: card["card_id"])


def _rule_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    policy_id = _policy_id(row)
    status = _status(row)
    card_id = f"rule:{policy_id}"
    default = "pre_accepted" if status == "eligible_for_generation" else "pending"
    decision, note = _recorded_decision(card_id, default, decisions)
    summary = "Pre-accepted eligible rule." if status == "eligible_for_generation" else "; ".join(_reasons(row))
    return {
        "card_id": card_id,
        "kind": "rule",
        "policy_id": policy_id,
        "rule_status": status,
        "summary": summary,
        "decision": decision,
        "note": note,
    }


def _conflict_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    card_id = conflict_card_id(row)
    decision, note = _recorded_decision(card_id, "pending", decisions)
    policies = sorted(_policy_list(row.get("policy_ids")))
    return {
        "card_id": card_id,
        "kind": "conflict",
        "policy_id": "|".join(policies),
        "rule_status": str(row.get("type") or "conflict"),
        "summary": str(row.get("detail") or ""),
        "decision": decision,
        "note": note,
    }


def _glossary_card(row: Mapping[str, Any], decisions: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    card_id = glossary_card_id(row)
    decision, note = _recorded_decision(card_id, "pending", decisions)
    key = str(row.get("key") or "")
    return {
        "card_id": card_id,
        "kind": "conflict",
        "policy_id": "",
        "rule_status": "glossary",
        "summary": f'"{key}" has conflicting definitions',
        "decision": decision,
        "note": note,
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


def _recorded_decision(
    card_id: str, default: str, decisions: Mapping[str, tuple[str, str]]
) -> tuple[str, str]:
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
        {"policy_id": _policy_id(row), "status": _status(row)} for row in rules if _status(row) in _EXCLUDED
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
    rule = row.get("rule")
    if not isinstance(rule, Mapping):
        return ""
    provenance = rule.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    quote = provenance.get("source_quote")
    return quote if isinstance(quote, str) else ""


def _policy_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


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


def _csv_text(cards: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for card in cards:
        writer.writerow({column: card[column] for column in _COLUMNS})
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
    body = ["<table><thead><tr><th>Slot</th><th>Policy</th><th>Kind</th><th>Reason</th></tr></thead><tbody>"]
    for row in rows:
        body.append(
            "<tr><td>"
            + _text(row.get("slot_id"))
            + "</td><td>"
            + _text(row.get("policy_id"))
            + "</td><td>"
            + _text(row.get("kind"))
            + "</td><td>"
            + _text(row.get("reason"))
            + "</td></tr>"
        )
    body.append("</tbody></table>")
    return "\n".join(body)


def _excluded_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>None.</p>"
    body = ["<ul>"]
    for row in rows:
        body.append("<li>" + _text(row.get("policy_id")) + " (" + _text(row.get("status")) + ")</li>")
    body.append("</ul>")
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
    return {
        key: _sha256(run_dir / relative)
        for key, relative in _CONTENT_FILES.items()
        if key not in _DERIVED_HASHES
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
