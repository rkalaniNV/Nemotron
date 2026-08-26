"""What the intake phase did, recorded so a later reviewer can re-derive it.

The drafting phase will add model identity, prompt hashes, and approvals to a record of its
own. This one covers the half that ran before any model existed, and says so explicitly:
`model` is null because nothing was inferred here, not because the field was forgotten. A
provenance record that leaves the distinction to the reader is the one that lets an
unattributed inference through later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.mcp.authoring.evidence import EvidenceBundle
from nemotron.steps.byob.runtime.mcp.authoring.intake import LoadedMcpIntake
from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    PENDING_PACK_ARTIFACTS,
    EmittedArtifact,
)
from nemotron.steps.byob.runtime.mcp.discovery import DiscoveryReport
from nemotron.steps.byob.runtime.mcp.errors import McpProtocolError
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)

INTAKE_PROVENANCE_VERSION = "bfcl-mcp-intake-provenance-v1"
MCP_INTAKE_ADAPTER_VERSION = "1.0.0"


@dataclass(frozen=True)
class IntakeProvenance:
    document: dict[str, Any]

    def verify_digest(self) -> None:
        claimed = self.document.get("record_digest")
        unsigned = {
            key: value for key, value in self.document.items() if key != "record_digest"
        }
        observed = sha256_json(unsigned)
        if claimed != observed:
            raise McpProtocolError(
                "intake provenance was modified after record_digest was computed: "
                f"claimed {claimed!r}, observed {observed!r}"
            )


def build_intake_provenance(
    intake: LoadedMcpIntake,
    report: DiscoveryReport,
    bundle: EvidenceBundle,
    artifacts: list[EmittedArtifact],
    *,
    output_root: Path,
    evidence_path: Path,
) -> IntakeProvenance:
    """Record the inputs, the outputs, and what remains unauthored."""
    bundle.verify_digest()
    report.verify_digest()
    root = output_root.resolve()
    identity = bundle.document["identity"]
    document: dict[str, Any] = {
        "schema_version": INTAKE_PROVENANCE_VERSION,
        "phase": "intake",
        "attained_level": "L0",
        "adapter": {
            "name": "nemotron-bfcl-mcp-intake",
            "version": MCP_INTAKE_ADAPTER_VERSION,
        },
        "pack": dict(bundle.document["pack"]),
        "mode": bundle.document["mode"],
        # Digests rather than absolute paths: the same reviewed inputs must produce the
        # same record on any host, and a path is not evidence of content.
        "inputs": {
            "intake_config_digest": identity["intake_config_digest"],
            "mcp_oracle_config_digest": identity["source_config_digest"],
            "discovery_report_digest": identity["discovery_report_digest"],
        },
        "identity": dict(identity),
        "oracle": dict(bundle.document["oracle"]),
        "evidence_bundle": {
            "path": evidence_path.resolve().relative_to(root).as_posix(),
            "digest": bundle.bundle_digest,
        },
        "artifacts": [artifact.as_dict(root=root) for artifact in artifacts],
        "excluded_tools": list(bundle.document["catalog"]["exclusions"]),
        "review": {
            "status": bundle.document["status"],
            "advisory_findings": list(bundle.document["review"]["advisory"]),
            # Filled by the human act, not by this run.
            "approvals": [],
        },
        # No model read anything in this phase. The drafting phase records its own.
        "model": None,
        "pending_artifacts": list(PENDING_PACK_ARTIFACTS),
    }
    document["record_digest"] = sha256_json(document)
    return IntakeProvenance(document=document)


def write_intake_provenance(provenance: IntakeProvenance, path: Path) -> Path:
    provenance.verify_digest()
    return write_canonical_json(provenance.document, path)
