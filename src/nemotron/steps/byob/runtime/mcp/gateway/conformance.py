"""Producing the attestation the gateway serves at `GET /v1/conformance`.

This side states what the gateway can actually demonstrate, and nothing more. The temptation
in a module like this is to write `"L2"` because the deployment is intended to be certifiable;
the discipline is that the level is computed from the checks that were really run, so an
unimplemented probe lowers the claim instead of being assumed to pass.

Nothing here decides whether the pack may publish. The gateway is the subject under test, and
a subject grading itself is not evidence, so it declares `locally_verified` and leaves the
consequence to `benchmark_families/bfcl/conformance.py`, which caps a self-reported claim at
`L1`. Both sides share one schema so the document cannot be valid to produce and invalid to
read.
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

# Probes that must pass before an endpoint can claim to be certifiable. Discovery covers
# P1-P3 today; the rest are execution probes, and each one absent is a reason this gateway
# reports `L1` rather than a reason to report `L2` quietly.
L2_REQUIRED_PROBES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")


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
    def report_digest(self) -> str:
        return attestation_digest({"probes": [probe.as_check() for probe in self.probes]})

    def suite_digest(self, gateway_artifact_digest: str) -> str:
        return attestation_digest(
            {
                "issuer": GATEWAY_EVIDENCE_ISSUER,
                "gateway_artifact_digest": gateway_artifact_digest,
                "suite": dict(self.suite),
            }
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
    passed = {probe.id for probe in probes if probe.status == "pass"}
    if not {"P1", "P2", "P3"} <= passed:
        return "L0"
    if state_observability != "complete":
        # Without complete observable state, replay and isolation cannot be proven, which is
        # the whole basis of a publishable verdict.
        return "L1"
    if not set(L2_REQUIRED_PROBES) <= passed:
        return "L1"
    return "L2"


def state_observability_for(config: McpOracleConfig) -> str:
    """Complete state requires the server to hand over episode state, not a projection."""
    return "complete" if config.control.state_strategy == "control_tool" else "diagnostic"


def read_only_boundary_for(config: McpOracleConfig) -> str | None:
    # Only a read-only deployment names a boundary; anywhere else the field is null rather
    # than a reassuring string.
    if config.mode != "C":
        return None
    return "immutable_snapshot_sandbox" if config.fixtures.direction == "snapshot" else None


def build_attestation(
    config: McpOracleConfig,
    report: DiscoveryReport,
    artifacts: GatewayArtifacts,
    identity: GatewayIdentity,
    evidence: ConformanceEvidence,
) -> dict[str, Any]:
    """Build the exact document `GET /v1/conformance` returns."""
    validated = artifacts.validated()
    observability = state_observability_for(config)
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
            validated.gateway_artifact_digest
        ),
        "gateway_evidence_kind": "locally_verified",
        "gateway_evidence_issuer": GATEWAY_EVIDENCE_ISSUER,
        "state_observability": observability,
        "read_only_boundary": read_only_boundary_for(config),
        "checks": [probe.as_check() for probe in evidence.probes],
    }
    # Parsing what we just built is cheap and catches a schema drift here before it reaches a
    # pack that pinned the digest of a document BFCL can no longer read.
    ConformanceAttestation.from_mapping(document, source="gateway conformance")
    return document


def attestation_bytes(document: Mapping[str, Any]) -> bytes:
    """Serialize canonically, because the pinned digest is over these exact bytes."""
    return canonical_json(dict(document)).encode("utf-8")
