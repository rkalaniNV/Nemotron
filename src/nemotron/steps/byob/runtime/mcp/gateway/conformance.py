"""Producing the attestation the gateway serves at `GET /v1/conformance`.

This side states what the gateway can actually demonstrate, and nothing more. The temptation
in a module like this is to write `"L2"` because the deployment is intended to be certifiable;
the discipline is that the level is computed from the checks that were really run, so an
unimplemented probe lowers the claim instead of being assumed to pass.

Nothing here decides whether the pack may publish. The gateway is the subject under test, and
a subject grading itself is not evidence, so it declares `locally_verified` and leaves the
consequence to `benchmark_families/bfcl/conformance.py`, which requires the independently
supplied reports whose digests this document cites. Both sides share one schema so the document
cannot be valid to produce and invalid to read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.conformance import (
    ATTESTATION_KIND,
    MCP_PROFILE_VERSION,
    PROVIDER_KIND_MCP,
    ConformanceAttestation,
    attestation_digest,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.config import McpOracleConfig
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    GatewayArtifacts,
    GatewayIdentity,
)

# The issuer name identifies which suite produced the evidence, so a verifier configured to
# trust a signed release can tell one issuer from another.
GATEWAY_EVIDENCE_ISSUER = "bfcl-mcp-conformance-v1"

# P1-P3 establish discovery, P4 establishes executable L1, and P5-P11 establish L2.
# P7 and P8 are conditional, but they still have to be present as either pass or an explicit
# not_applicable with a reason. Absence never means not applicable.
DISCOVERY_PROBES = ("P1", "P2", "P3")
EXECUTABLE_PROBE = "P4"
L2_PROBES = tuple(f"P{index}" for index in range(5, 12))
CONDITIONAL_PROBES = frozenset({"P7", "P8"})


@dataclass(frozen=True)
class ProbeOutcome:
    """One probe result, as observed rather than as intended."""

    id: str
    requirement: str
    status: str
    reason: str | None = None

    def as_check(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "requirement": self.requirement,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConformanceEvidence:
    """Everything the gateway actually ran, plus the artifact digests it can prove."""

    probes: tuple[ProbeOutcome, ...]
    # A separate document from the probe results: it records the suite, its version, and the
    # exact gateway artifact it exercised, which is what stops a passing report from being
    # reused for a different build.
    suite: Mapping[str, Any] = field(default_factory=dict)

    @property
    def probe_report(self) -> dict[str, Any]:
        return {"probes": [probe.as_check() for probe in self.probes]}

    @property
    def report_digest(self) -> str:
        return attestation_digest(self.probe_report)

    def gateway_report(
        self,
        gateway_artifact_digest: str,
        *,
        effective_content_digest: str,
        tool_catalog_digest: str,
    ) -> dict[str, Any]:
        return {
            "issuer": GATEWAY_EVIDENCE_ISSUER,
            "gateway_artifact_digest": gateway_artifact_digest,
            "effective_content_digest": effective_content_digest,
            "tool_catalog_digest": tool_catalog_digest,
            "suite": dict(self.suite),
        }

    def suite_digest(
        self,
        gateway_artifact_digest: str,
        *,
        effective_content_digest: str,
        tool_catalog_digest: str,
    ) -> str:
        return attestation_digest(
            self.gateway_report(
                gateway_artifact_digest,
                effective_content_digest=effective_content_digest,
                tool_catalog_digest=tool_catalog_digest,
            )
        )


def discovery_evidence(report: DiscoveryReport) -> ConformanceEvidence:
    """Lift the L0 discovery attestation into conformance evidence, unchanged.

    Discovery already recorded `P1` to `P3` against a real server. Re-deriving them here
    would let two code paths disagree about the same observation, so they are copied.
    """
    checks = report.document.get("checks", [])
    if not isinstance(checks, list) or not checks:
        raise ValueError(
            "the discovery report carries no probe results, so there is no evidence to "
            "attest to; refusing to serve an attestation with an empty check list"
        )
    probes = tuple(
        ProbeOutcome(
            id=str(entry["id"]),
            requirement=str(entry.get("requirement", "required")),
            status=str(entry["status"]),
            reason=entry.get("reason"),
        )
        for entry in checks
    )
    return ConformanceEvidence(
        probes=probes,
        suite={
            "kind": "discovery",
            "profile_version": report.document.get("profile_version"),
            "discovery_report_digest": report.document.get("report_digest"),
        },
    )


def _attained_level(probes: Sequence[ProbeOutcome], *, state_observability: str) -> str:
    """Derive the level from evidence, never from intent."""
    by_id = {probe.id: probe for probe in probes}

    def satisfies(identifier: str) -> bool:
        probe = by_id.get(identifier)
        if probe is None:
            return False
        if probe.status == "pass":
            return True
        return (
            identifier in CONDITIONAL_PROBES
            and probe.requirement == "conditional"
            and probe.status == "not_applicable"
            and isinstance(probe.reason, str)
            and bool(probe.reason.strip())
        )

    if not all(satisfies(identifier) for identifier in DISCOVERY_PROBES):
        return "L0"
    if not satisfies(EXECUTABLE_PROBE):
        return "L0"
    if state_observability != "complete":
        # Without complete observable state, replay and isolation cannot be proven, which is
        # the whole basis of a publishable verdict.
        return "L1"
    if not all(satisfies(identifier) for identifier in L2_PROBES):
        return "L1"
    return "L2"


def state_observability_for(
    config: McpOracleConfig,
    evidence: ConformanceEvidence,
) -> str:
    """Derive completeness from the control plane or from the probes that prove a projection."""
    if config.control.state_strategy == "control_tool":
        return "complete"
    passed = {probe.id for probe in evidence.probes if probe.status == "pass"}
    return "complete" if {"P6", "P10", "P11"} <= passed else "diagnostic"


def read_only_boundary_for(
    config: McpOracleConfig,
    evidence: ConformanceEvidence,
) -> str | None:
    """Name a mode-C boundary only after P6 verified that exact mechanism."""
    if config.mode != "C":
        return None
    p6_passed = any(
        probe.id == "P6" and probe.status == "pass" for probe in evidence.probes
    )
    boundary = evidence.suite.get("read_only_boundary")
    if p6_passed and boundary in {
        "upstream_authorization",
        "immutable_snapshot_sandbox",
    }:
        return str(boundary)
    return None


def build_attestation(
    config: McpOracleConfig,
    report: DiscoveryReport,
    artifacts: GatewayArtifacts,
    identity: GatewayIdentity,
    evidence: ConformanceEvidence,
) -> dict[str, Any]:
    """Build the exact document `GET /v1/conformance` returns."""
    validated = artifacts.validated()
    observability = state_observability_for(config, evidence)
    document: dict[str, Any] = {
        "schema_version": ATTESTATION_KIND,
        "provider_kind": PROVIDER_KIND_MCP,
        "profile_version": MCP_PROFILE_VERSION,
        "level": _attained_level(evidence.probes, state_observability=observability),
        # The same digest `GET /v1/metadata` serves, from the same function, so the two
        # routes cannot describe different builds.
        "effective_content_digest": identity.content_digest,
        "gateway_artifact_digest": validated.gateway_artifact_digest,
        "shim_artifact_digest": validated.shim_artifact_digest,
        "tool_catalog_digest": report.tool_catalog_digest,
        "server_content_digest": report.document["identity"].get("server_content_digest"),
        "snapshot_digest": validated.snapshot_digest,
        "probe_report_digest": evidence.report_digest,
        "gateway_conformance_report_digest": evidence.suite_digest(
            validated.gateway_artifact_digest,
            effective_content_digest=identity.content_digest,
            tool_catalog_digest=report.tool_catalog_digest,
        ),
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": GATEWAY_EVIDENCE_ISSUER,
        "state_observability": observability,
        "read_only_boundary": read_only_boundary_for(config, evidence),
        "checks": [probe.as_check() for probe in evidence.probes],
    }
    # Parsing what we just built is cheap and catches a schema drift here before it reaches a
    # pack that pinned the digest of a document BFCL can no longer read.
    ConformanceAttestation.from_mapping(document, source="gateway conformance")
    return document


def attestation_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize canonically, because the pinned digest is over these exact bytes."""
    return str(canonical_json(dict(document))).encode("utf-8")
