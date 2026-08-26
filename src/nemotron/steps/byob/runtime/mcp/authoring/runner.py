"""One intake run: discover, sanitize, derive, and record.

The order is the argument. Identity is pinned before any evidence is written, so a bundle
can never describe one catalog while the pack points at another. The pack files are written
only after the bundle survives hygiene, so a blocked run leaves no artifact that looks
authored. Provenance is written last, because it is the only file that claims the others
exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.authoring.evidence import (
    EvidenceBundle,
    build_evidence_bundle,
    write_evidence_bundle,
)
from nemotron.steps.byob.runtime.mcp.authoring.intake import (
    LoadedMcpIntake,
    load_mcp_intake,
)
from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    EmittedArtifact,
    emit_pack_artifacts,
)
from nemotron.steps.byob.runtime.mcp.authoring.provenance import (
    IntakeProvenance,
    build_intake_provenance,
    write_intake_provenance,
)
from nemotron.steps.byob.runtime.mcp.config import TrustedExecutablePolicies
from nemotron.steps.byob.runtime.mcp.discovery import (
    ConnectionFactory,
    DiscoveryReport,
    discover_mcp_oracle,
    write_discovery_report,
)
from nemotron.steps.byob.runtime.mcp.gateway.identity import (
    GatewayArtifacts,
    GatewayIdentity,
    build_gateway_identity,
)

PACK_DIRECTORY_NAME = "pack"
EVIDENCE_FILE_NAME = "evidence_bundle.json"
DISCOVERY_FILE_NAME = "discovery_report.json"
PROVENANCE_FILE_NAME = "intake_provenance.json"


@dataclass(frozen=True)
class IntakeResult:
    intake: LoadedMcpIntake
    report: DiscoveryReport
    identity: GatewayIdentity
    bundle: EvidenceBundle
    provenance: IntakeProvenance
    artifacts: list[EmittedArtifact]
    output_root: Path

    @property
    def pack_root(self) -> Path:
        return self.output_root / PACK_DIRECTORY_NAME


async def run_intake(
    intake_path: Path,
    output_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    executable_policies: TrustedExecutablePolicies | None = None,
    connection_factory: ConnectionFactory | None = None,
    allow_insecure_localhost: bool = False,
) -> IntakeResult:
    """Turn a reviewed MCP intake declaration into a reviewable pack draft."""
    intake = load_mcp_intake(
        intake_path,
        allow_insecure_localhost=allow_insecure_localhost,
    )
    report = await discover_mcp_oracle(
        intake.oracle,
        environ=environ,
        executable_policies=executable_policies,
        connection_factory=connection_factory,
    )
    identity = build_gateway_identity(
        intake.oracle.value,
        report,
        GatewayArtifacts(
            gateway_artifact_digest=intake.value.gateway.gateway_artifact_digest,
            shim_artifact_digest=intake.value.gateway.shim_artifact_digest,
            snapshot_digest=intake.value.gateway.snapshot_digest,
        ),
    )
    bundle = build_evidence_bundle(intake, report, identity)

    root = output_root.resolve()
    artifacts = emit_pack_artifacts(
        intake,
        report,
        identity,
        root / PACK_DIRECTORY_NAME,
    )
    write_discovery_report(report, root / DISCOVERY_FILE_NAME)
    evidence_path = write_evidence_bundle(bundle, root / EVIDENCE_FILE_NAME)
    provenance = build_intake_provenance(
        intake,
        report,
        bundle,
        artifacts,
        output_root=root,
        evidence_path=evidence_path,
    )
    write_intake_provenance(provenance, root / PROVENANCE_FILE_NAME)
    return IntakeResult(
        intake=intake,
        report=report,
        identity=identity,
        bundle=bundle,
        provenance=provenance,
        artifacts=artifacts,
        output_root=root,
    )
