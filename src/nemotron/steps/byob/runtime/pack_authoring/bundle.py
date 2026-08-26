"""Reading an evidence bundle, and the approval gate in front of it.

The drafting phase sees a file, never a server. That is what makes this side auditable: the
exact bytes a human approved are the exact bytes the model reads, and the digest proves it.

Two gates live here. The digest gate catches a bundle edited after review. The approval gate
refuses to draft from a bundle nobody signed, because the whole point of flagging suspicious
tool text for a human is lost if the next phase runs anyway.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

EVIDENCE_BUNDLE_VERSION = "bfcl-mcp-evidence-v1"
APPROVAL_VERSION = "bfcl-authoring-approval-v1"


class BundleError(Exception):
    """Raised when a bundle cannot be trusted as authoring input."""


@dataclass(frozen=True)
class ToolEvidence:
    """One tool as the drafting model is allowed to see it."""

    published_name: str
    description: str
    parameters: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any] | None
    mutates: bool
    requires_confirmation: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        properties = self.parameters.get("properties")
        if not isinstance(properties, Mapping):
            return frozenset()
        return frozenset(str(name) for name in properties)

    @property
    def required_parameters(self) -> tuple[str, ...]:
        required = self.parameters.get("required")
        if not isinstance(required, list):
            return ()
        return tuple(sorted(str(name) for name in required))


@dataclass(frozen=True)
class EvidenceView:
    """A verified bundle, plus the accessors the generators actually need."""

    document: dict[str, Any]
    path: Path

    @property
    def digest(self) -> str:
        return str(self.document["bundle_digest"])

    @property
    def pack_id(self) -> str:
        return str(self.document["pack"]["pack_id"])

    @property
    def attained_level(self) -> str:
        return str(self.document["attained_level"])

    @property
    def vocabulary(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.document["vocabulary"].items()}

    @property
    def unresolved_unknowns(self) -> frozenset[str]:
        """Field names the bundle says nothing can be inferred about yet."""
        return frozenset(
            str(entry["field"]) for entry in self.document.get("unknowns", [])
        )

    @property
    def tools(self) -> tuple[ToolEvidence, ...]:
        entries: list[ToolEvidence] = []
        for entry in self.document["tools"]:
            schemas = entry["untrusted_schemas"]
            entries.append(
                ToolEvidence(
                    published_name=str(entry["published_name"]),
                    description=str(entry["description"]["untrusted_text"]),
                    parameters=schemas["parameters"],
                    output_schema=schemas["output_schema"],
                    annotations=schemas["annotations"],
                    mutates=bool(entry["declared"]["mutates"]),
                    requires_confirmation=bool(
                        entry["declared"]["requires_confirmation"]
                    ),
                )
            )
        return tuple(entries)

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(tool.published_name for tool in self.tools)

    def tool(self, published_name: str) -> ToolEvidence:
        for tool in self.tools:
            if tool.published_name == published_name:
                return tool
        raise BundleError(f"no tool named {published_name!r} in the evidence bundle")


def _verify_digest(document: Mapping[str, Any], source: Path) -> None:
    claimed = document.get("bundle_digest")
    unsigned = {key: value for key, value in document.items() if key != "bundle_digest"}
    observed = sha256_json(unsigned)
    if claimed != observed:
        raise BundleError(
            f"evidence bundle {source} was modified after its digest was computed: "
            f"claimed {claimed!r}, observed {observed!r}"
        )


def load_evidence_bundle(path: Path) -> EvidenceView:
    """Load and verify a bundle without deciding whether it may be used."""
    source = path.resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read evidence bundle {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleError(f"evidence bundle must be a JSON object: {source}")
    version = raw.get("schema_version")
    if version != EVIDENCE_BUNDLE_VERSION:
        raise BundleError(
            f"evidence bundle {source} declares schema_version {version!r}; "
            f"this drafting phase reads {EVIDENCE_BUNDLE_VERSION!r}"
        )
    for required in ("bundle_digest", "tools", "pack", "vocabulary", "identity"):
        if required not in raw:
            raise BundleError(f"evidence bundle {source} has no {required!r}")
    _verify_digest(raw, source)
    if not raw["tools"]:
        raise BundleError(f"evidence bundle {source} selected no tools to draft against")
    return EvidenceView(document=raw, path=source)


@dataclass(frozen=True)
class Approval:
    """A human's recorded decision about one exact bundle."""

    approved_by: str
    bundle_digest: str
    acknowledged_findings: tuple[str, ...]
    note: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_version": APPROVAL_VERSION,
            "approved_by": self.approved_by,
            "bundle_digest": self.bundle_digest,
            "acknowledged_findings": list(self.acknowledged_findings),
            "note": self.note,
        }


def load_approval(path: Path, bundle: EvidenceView) -> Approval:
    """Load an approval and prove it refers to this bundle and its open flags."""
    source = path.resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read approval {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BundleError(f"approval must be a JSON object: {source}")
    if raw.get("approval_version") != APPROVAL_VERSION:
        raise BundleError(
            f"approval {source} must declare approval_version {APPROVAL_VERSION!r}"
        )
    approved_by = raw.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise BundleError(f"approval {source} must name who approved the bundle")
    digest = raw.get("bundle_digest")
    if digest != bundle.digest:
        # An approval of a different bundle is the failure this gate exists to catch: it
        # is how reviewed text gets swapped for unreviewed text between the two phases.
        raise BundleError(
            f"approval {source} covers bundle {digest!r}, not {bundle.digest!r}"
        )
    acknowledged = raw.get("acknowledged_findings", [])
    if not isinstance(acknowledged, list) or not all(
        isinstance(item, str) for item in acknowledged
    ):
        raise BundleError(f"approval {source} acknowledged_findings must be strings")
    advisory = {
        f"{finding['location']}:{finding['code']}"
        for finding in bundle.document.get("review", {}).get("advisory", [])
    }
    unacknowledged = sorted(advisory - set(acknowledged))
    if unacknowledged:
        # Every flag a human was asked about has to be answered by name. A blanket
        # approval would let a newly appearing flag ride along on an old decision.
        raise BundleError(
            "approval does not acknowledge every flagged finding: "
            + ", ".join(unacknowledged)
        )
    unknown = sorted(set(acknowledged) - advisory)
    if unknown:
        raise BundleError(
            "approval acknowledges findings this bundle does not contain: "
            + ", ".join(unknown)
        )
    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        raise BundleError(f"approval {source} note must be a string when present")
    return Approval(
        approved_by=approved_by.strip(),
        bundle_digest=bundle.digest,
        acknowledged_findings=tuple(sorted(acknowledged)),
        note=note,
    )
