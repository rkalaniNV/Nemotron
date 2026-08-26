"""Deterministic review packets and the human approval gate for an MCP pack.

Review is not a boolean attached to whatever files happen to be present. The packet pins every
record the reviewer saw, names evidence that is still missing, and remains buildable even when
blocked so failures can be reviewed without being mistaken for approval. An approval covers one
exact packet digest and every risk by stable id; it cannot float forward to a changed catalog,
new warning, or different validation run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ConformanceAttestation,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    pack_fingerprint,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.mcp.authoring.provenance import IntakeProvenance
from nemotron.steps.byob.runtime.mcp.config import (
    load_mcp_oracle_config,
    load_unique_yaml_document,
    load_unique_yaml_mapping,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import load_evidence_bundle
from nemotron.steps.byob.runtime.pack_authoring.provenance import (
    DraftProvenance,
    ProvenanceError,
)

REVIEW_PACKET_VERSION = "bfcl-mcp-review-packet-v1"
REVIEW_APPROVAL_VERSION = "bfcl-mcp-review-approval-v1"
REQUIRED_CHECKLIST = frozenset(
    {
        "semantics",
        "control_mapping",
        "descriptions_and_snapshots",
        "held_out_policy",
        "assumptions",
        "validation_evidence",
    }
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReviewError(ValueError):
    """Raised when review inputs do not form one coherent, approvable record."""


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token!r} is not allowed")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def load_json_document(path: Path, label: str) -> Any:
    """Load strict JSON with no duplicate keys, NaN, or infinities."""
    source = path.resolve()
    try:
        return json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReviewError(f"cannot read {label} {source}: {exc}") from exc


def load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    """Load a strict JSON object."""
    source = path.resolve()
    raw = load_json_document(source, label)
    if not isinstance(raw, dict):
        raise ReviewError(f"{label} must be a JSON object: {source}")
    return raw


def _require_digest(document: Mapping[str, Any], field: str, label: str) -> str:
    claimed = document.get(field)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ReviewError(f"{label}.{field} must be sha256:<64 lowercase hex characters>")
    unsigned = {key: value for key, value in document.items() if key != field}
    observed = sha256_json(unsigned)
    if claimed != observed:
        raise ReviewError(
            f"{label} changed after {field} was computed: "
            f"claimed {claimed!r}, observed {observed!r}"
        )
    return claimed


def _risk_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{sha256_json(value).removeprefix('sha256:')[:16]}"


def _validation_failures(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for check in [*(report.get("checks") or []), *(report.get("extra_checks") or [])]:
        if not isinstance(check, Mapping):
            continue
        check_id = str(check.get("id", "unknown"))
        status = str(check.get("status", "missing"))
        declared = check.get("failures") or []
        if status != "pass" and not declared:
            failures.append(
                {
                    "check_id": check_id,
                    "check_name": check.get("name"),
                    "status": status,
                    "reason": "check_not_passed",
                }
            )
        for failure in declared if isinstance(declared, list) else []:
            failures.append(
                {
                    "check_id": check_id,
                    "check_name": check.get("name"),
                    "status": status,
                    "failure": failure,
                }
            )
    return failures


def _conformance_entry(report: Mapping[str, Any]) -> dict[str, Any] | None:
    for check in report.get("extra_checks") or []:
        if isinstance(check, dict) and check.get("id") == "A1":
            return dict(check)
    return None


def _metadata_risks(
    evidence: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for finding in evidence.get("review", {}).get("advisory", []):
        identifier = f"{finding.get('location')}:{finding.get('code')}"
        risks.append(
            {
                "id": identifier,
                "source": "evidence_advisory",
                "detail": finding,
            }
        )
    for warning in evidence.get("catalog", {}).get("warnings", []):
        risks.append(
            {
                "id": _risk_id("catalog_warning", warning),
                "source": "catalog_warning",
                "detail": warning,
            }
        )
    conformance = _conformance_entry(validation_report)
    if conformance is not None:
        verdict = conformance.get("conformance") or {}
        for finding in verdict.get("findings") or []:
            risks.append(
                {
                    "id": f"conformance_finding:{finding}",
                    "source": "endpoint_conformance",
                    "detail": finding,
                }
            )
        for cap in verdict.get("caps") or []:
            risks.append(
                {
                    "id": f"conformance_cap:{cap}",
                    "source": "endpoint_conformance",
                    "detail": cap,
                }
            )
    return sorted(risks, key=lambda item: item["id"])


def _blockers(
    *,
    draft: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    calls_complete: bool,
    state_deltas_complete: bool,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []

    def add(identifier: str, detail: str) -> None:
        blockers.append({"id": identifier, "detail": detail})

    if draft.get("blocked_on"):
        add("draft_unknowns", f"draft remains blocked on {sorted(draft['blocked_on'])}")
    if not draft.get("assertions_compiled"):
        add("assertions_not_compiled", "assertions.py was not compiled from reviewed specifications")
    if draft.get("compilation_refusals"):
        add(
            "assertion_compilation_refused",
            "; ".join(str(item) for item in draft["compilation_refusals"]),
        )
    if not validation_report.get("gold_eligible"):
        add("pack_not_gold", f"validation tier is {validation_report.get('tier')!r}")
    conformance = _conformance_entry(validation_report)
    effective_level = (
        (conformance.get("conformance") or {}).get("effective_level")
        if conformance is not None
        else None
    )
    if effective_level != "L2":
        add("endpoint_not_l2", f"effective conformance level is {effective_level!r}")
    if _validation_failures(validation_report):
        add("validation_failures", "one or more validation checks did not pass")
    if not calls_complete:
        add("observed_calls_missing", "validation did not publish the complete observed call log")
    if not state_deltas_complete:
        add(
            "state_deltas_missing",
            "validation did not publish complete before/after state deltas",
        )
    return sorted(blockers, key=lambda item: item["id"])


@dataclass(frozen=True)
class ReviewPacket:
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.document["packet_digest"])

    def verify_digest(self) -> None:
        _require_digest(self.document, "packet_digest", "review packet")


def _canonical_pack_review(pack_root: Path) -> tuple[dict[str, Any], str]:
    root = pack_root.resolve()
    paths = resolve_declared_pack_paths(
        OraclePackRef(manifest_path=root / "manifest.yaml"),
        (root,),
    )
    fingerprint = pack_fingerprint(paths)
    fixtures = (
        load_json_document(paths.fixtures_path, "pack fixtures")
        if paths.fixtures_path is not None
        else None
    )
    held_out = (
        load_unique_yaml_document(paths.held_out_path, "pack held-out policy")
        if paths.held_out_path is not None
        else None
    )
    endpoint = (
        load_unique_yaml_mapping(paths.endpoint_config_path, "pack endpoint config")
        if paths.endpoint_config_path is not None
        else None
    )
    document = {
        "fingerprint": f"sha256:{fingerprint}",
        "manifest": load_unique_yaml_mapping(paths.manifest_path, "pack manifest"),
        "tools": load_json_document(paths.tools_path, "pack tools"),
        "fixtures": fixtures,
        "task_templates": load_unique_yaml_document(
            paths.templates_path,
            "pack task templates",
        ),
        "validation_cases": load_unique_yaml_document(
            paths.validation_cases_path,
            "pack validation cases",
        ),
        "assertions": {
            "digest": "sha256:"
            + hashlib.sha256(paths.assertions_path.read_bytes()).hexdigest(),
            "source": paths.assertions_path.read_text(encoding="utf-8"),
        },
        "endpoint_config": endpoint,
        "held_out": held_out,
    }
    return document, fingerprint


def build_review_packet(
    evidence_path: Path,
    intake_provenance_path: Path,
    gateway_attestation_path: Path,
    draft_provenance_path: Path,
    validation_report_path: Path,
    mcp_config_path: Path,
    pack_root: Path,
    *,
    held_out_path: Path | None = None,
) -> ReviewPacket:
    """MCP-501: assemble every fact a domain reviewer must accept or reject."""
    evidence = load_evidence_bundle(evidence_path)
    intake_document = load_json_mapping(intake_provenance_path, "intake provenance")
    intake = IntakeProvenance(document=intake_document)
    intake.verify_digest()
    gateway_attestation = load_json_mapping(
        gateway_attestation_path,
        "gateway attestation",
    )
    ConformanceAttestation.from_mapping(
        gateway_attestation,
        source="gateway attestation",
    )
    draft_document = load_json_mapping(draft_provenance_path, "draft provenance")
    draft = DraftProvenance(document=draft_document)
    try:
        draft.verify_digest()
    except ProvenanceError as exc:
        raise ReviewError(str(exc)) from exc
    validation = load_json_mapping(validation_report_path, "oracle validation report")
    loaded_profile = load_mcp_oracle_config(mcp_config_path)
    canonical_pack, current_fingerprint = _canonical_pack_review(pack_root)
    observed_attestation_digest = attestation_digest(gateway_attestation)
    endpoint_document = canonical_pack.get("endpoint_config")
    if not isinstance(endpoint_document, Mapping):
        raise ReviewError("MCP canonical pack must use an endpoint oracle")
    if (
        (endpoint_document.get("attestation") or {}).get("expected_digest")
        != observed_attestation_digest
    ):
        raise ReviewError(
            "canonical endpoint config does not pin the reviewed gateway attestation"
        )
    discovered_content = evidence.document["identity"]["effective_content_digest"]
    if (endpoint_document.get("expected") or {}).get("content_digest") != discovered_content:
        raise ReviewError(
            "canonical endpoint config pins a different effective content digest than "
            f"discovery observed ({discovered_content})"
        )

    if draft_document.get("evidence", {}).get("bundle_digest") != evidence.digest:
        raise ReviewError("draft provenance does not cover this evidence bundle")
    if intake_document.get("evidence_bundle", {}).get("digest") != evidence.digest:
        raise ReviewError("intake provenance does not cover this evidence bundle")
    if (
        intake_document.get("gateway_attestation", {}).get("digest")
        != observed_attestation_digest
    ):
        raise ReviewError(
            "intake provenance does not cover this gateway attestation"
        )
    if draft_document.get("pack") != evidence.document.get("pack"):
        raise ReviewError("draft provenance and evidence name different packs")
    if validation.get("pack_id") != evidence.pack_id:
        raise ReviewError("validation report and evidence name different packs")
    if validation.get("pack_fingerprint") != current_fingerprint:
        raise ReviewError(
            "validation report does not cover the canonical pack under review"
        )
    expected_profile_digest = evidence.document["identity"]["source_config_digest"]
    observed_profile_digest = sha256_json(loaded_profile.raw_document)
    if observed_profile_digest != expected_profile_digest:
        raise ReviewError(
            "MCP profile differs from discovery: "
            f"expected {expected_profile_digest}, observed {observed_profile_digest}"
        )

    held_out: dict[str, Any]
    pack_held_out = canonical_pack["held_out"]
    if held_out_path is None and pack_held_out is None:
        held_out = {"status": "not_declared", "digest": None, "policy": None}
    else:
        policy = (
            load_unique_yaml_mapping(held_out_path.resolve(), "held-out policy")
            if held_out_path is not None
            else pack_held_out
        )
        if pack_held_out != policy:
            raise ReviewError(
                "held-out policy supplied for review differs from the canonical pack"
            )
        held_out = {
            "status": "declared",
            "digest": sha256_json(policy),
            "policy": policy,
        }

    observations = validation.get("mcp_observations")
    if not isinstance(observations, Mapping):
        observations = {}
    observed_calls = observations.get("calls")
    state_deltas = observations.get("state_deltas")
    calls_complete = observations.get("calls_complete") is True and isinstance(
        observed_calls, list
    )
    state_deltas_complete = observations.get("state_deltas_complete") is True and isinstance(
        state_deltas, list
    )

    failures = _validation_failures(validation)
    risks = _metadata_risks(evidence.document, validation)
    blockers = _blockers(
        draft=draft_document,
        validation_report=validation,
        calls_complete=calls_complete,
        state_deltas_complete=state_deltas_complete,
    )
    tools = [
        {
            "published_name": entry["published_name"],
            "source_name": entry["source_name"],
            "description": entry["description"],
            "parameters": entry["untrusted_schemas"]["parameters"],
            "output_schema": entry["untrusted_schemas"]["output_schema"],
            "annotations": entry["untrusted_schemas"]["annotations"],
            "mutates": entry["declared"]["mutates"],
            "mutation_source": entry["declared"]["mutation_source"],
            "requires_confirmation": entry["declared"]["requires_confirmation"],
            "raw_digest": entry["raw_digest"],
        }
        for entry in evidence.document["tools"]
    ]
    document: dict[str, Any] = {
        "schema_version": REVIEW_PACKET_VERSION,
        "status": "ready_for_approval" if not blockers else "blocked",
        "pack": dict(evidence.document["pack"]),
        "source_digests": {
            "evidence_bundle": evidence.digest,
            "intake_provenance": intake_document["record_digest"],
            "gateway_attestation": observed_attestation_digest,
            "draft_provenance": draft_document["record_digest"],
            "validation_report": sha256_json(validation),
            "mcp_config": observed_profile_digest,
            "held_out_policy": held_out["digest"],
            "canonical_pack": canonical_pack["fingerprint"],
        },
        "identity": dict(evidence.document["identity"]),
        "oracle": dict(evidence.document["oracle"]),
        "mode": evidence.document["mode"],
        "control_mapping": loaded_profile.value.control.model_dump(mode="json"),
        "fixture_policy": {
            **dict(evidence.document["fixtures"]),
            "held_out": held_out,
        },
        "tools": tools,
        "canonical_pack": canonical_pack,
        "exclusions": list(evidence.document["catalog"]["exclusions"]),
        "metadata_risks": risks,
        "assumptions": list(evidence.document["assumptions"]),
        "authoring": {
            "model": draft_document.get("model"),
            "model_calls": list(draft_document.get("calls") or []),
            "artifact_digests": dict(draft_document.get("artifact_digests") or {}),
            "blocked_on": list(draft_document.get("blocked_on") or []),
            "assertions_compiled": bool(draft_document.get("assertions_compiled")),
            "compilation_refusals": list(
                draft_document.get("compilation_refusals") or []
            ),
        },
        "validation": {
            "gold_eligible": bool(validation.get("gold_eligible")),
            "tier": validation.get("tier"),
            "pack_fingerprint": validation.get("pack_fingerprint"),
            "endpoint_metadata": validation.get("endpoint_metadata"),
            "conformance": _conformance_entry(validation),
            "failures": failures,
        },
        "observations": {
            "calls_complete": calls_complete,
            "calls": list(observed_calls) if isinstance(observed_calls, list) else [],
            "state_deltas_complete": state_deltas_complete,
            "state_deltas": list(state_deltas) if isinstance(state_deltas, list) else [],
        },
        "blockers": blockers,
    }
    document["packet_digest"] = sha256_json(document)
    return ReviewPacket(document=document)


def write_review_packet(packet: ReviewPacket, path: Path) -> Path:
    packet.verify_digest()
    return write_canonical_json(packet.document, path)


def load_review_packet(path: Path) -> ReviewPacket:
    document = load_json_mapping(path, "review packet")
    if document.get("schema_version") != REVIEW_PACKET_VERSION:
        raise ReviewError(
            f"review packet schema_version must be {REVIEW_PACKET_VERSION!r}"
        )
    packet = ReviewPacket(document=document)
    packet.verify_digest()
    return packet


@dataclass(frozen=True)
class ReviewApproval:
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.document["approval_digest"])

    def verify_digest(self) -> None:
        _require_digest(self.document, "approval_digest", "review approval")


def _validate_reviewed_at(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError("approval.reviewed_at must be an RFC 3339 timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReviewError("approval.reviewed_at must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ReviewError("approval.reviewed_at must include a timezone")
    return value.strip()


def build_review_approval(
    packet: ReviewPacket,
    *,
    approved_by: str,
    reviewed_at: str,
    checklist: Mapping[str, bool],
    acknowledged_risks: list[str],
    note: str | None = None,
) -> ReviewApproval:
    """MCP-502: approve one exact, unblocked packet with every checklist item."""
    packet.verify_digest()
    if packet.document.get("status") != "ready_for_approval":
        blockers = [item.get("id") for item in packet.document.get("blockers") or []]
        raise ReviewError(
            "cannot approve a blocked review packet: " + ", ".join(map(str, blockers))
        )
    reviewer = approved_by.strip() if isinstance(approved_by, str) else ""
    if not reviewer:
        raise ReviewError("approval.approved_by must name the reviewer")
    timestamp = _validate_reviewed_at(reviewed_at)

    unknown_checks = sorted(set(checklist) - REQUIRED_CHECKLIST)
    missing_checks = sorted(REQUIRED_CHECKLIST - set(checklist))
    if unknown_checks or missing_checks:
        raise ReviewError(
            f"approval checklist mismatch; missing={missing_checks}, unknown={unknown_checks}"
        )
    not_accepted = sorted(name for name, accepted in checklist.items() if accepted is not True)
    if not_accepted:
        raise ReviewError(
            "approval checklist items must all be true: " + ", ".join(not_accepted)
        )

    expected_risks = {item["id"] for item in packet.document["metadata_risks"]}
    acknowledged = set(acknowledged_risks)
    if acknowledged != expected_risks:
        raise ReviewError(
            "approval risk acknowledgements mismatch; "
            f"missing={sorted(expected_risks - acknowledged)}, "
            f"unknown={sorted(acknowledged - expected_risks)}"
        )
    if note is not None and not isinstance(note, str):
        raise ReviewError("approval.note must be a string when present")

    document: dict[str, Any] = {
        "schema_version": REVIEW_APPROVAL_VERSION,
        "review_packet_digest": packet.digest,
        "approved_by": reviewer,
        "reviewed_at": timestamp,
        "checklist": {name: True for name in sorted(REQUIRED_CHECKLIST)},
        "acknowledged_risks": sorted(acknowledged),
        "note": note,
    }
    document["approval_digest"] = sha256_json(document)
    return ReviewApproval(document=document)


def write_review_approval(approval: ReviewApproval, path: Path) -> Path:
    approval.verify_digest()
    return write_canonical_json(approval.document, path)


def load_review_approval(path: Path, packet: ReviewPacket) -> ReviewApproval:
    document = load_json_mapping(path, "review approval")
    if document.get("schema_version") != REVIEW_APPROVAL_VERSION:
        raise ReviewError(
            f"review approval schema_version must be {REVIEW_APPROVAL_VERSION!r}"
        )
    approval = ReviewApproval(document=document)
    approval.verify_digest()
    if document.get("review_packet_digest") != packet.digest:
        raise ReviewError("review approval covers a different review packet")
    # Re-run the semantic checks; a self-consistent digest must not make a malformed approval
    # acceptable.
    approved_by = document.get("approved_by")
    reviewed_at = document.get("reviewed_at")
    if not isinstance(approved_by, str):
        raise ReviewError("approval.approved_by must name the reviewer")
    if not isinstance(reviewed_at, str):
        raise ReviewError("approval.reviewed_at must be an RFC 3339 timestamp")
    rebuilt = build_review_approval(
        packet,
        approved_by=approved_by,
        reviewed_at=reviewed_at,
        checklist=document.get("checklist") or {},
        acknowledged_risks=document.get("acknowledged_risks") or [],
        note=document.get("note"),
    )
    if rebuilt.document != document:
        raise ReviewError("review approval is not in canonical form")
    return approval
