"""What the drafting phase inferred, from which evidence, under whose approval.

The intake record says no model was involved. This one says exactly which model was, which
prompt version it ran under, which request hash each answer is filed against, and which human
approved the evidence it read. Together they cover the whole path from a server's catalog to
a pack draft, which is what makes a later reviewer able to disagree with a specific step
instead of distrusting the artifact as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import Approval, EvidenceView
from nemotron.steps.byob.runtime.pack_authoring.drafts import DraftBundle
from nemotron.steps.byob.runtime.pack_authoring.model_client import AuthoringModel

DRAFT_PROVENANCE_VERSION = "bfcl-authoring-draft-provenance-v1"
DRAFTING_ADAPTER_VERSION = "1.0.0"


class ProvenanceError(Exception):
    """Raised when a provenance record does not describe the artifacts beside it."""


@dataclass(frozen=True)
class DraftProvenance:
    document: dict[str, Any]

    def verify_digest(self) -> None:
        claimed = self.document.get("record_digest")
        unsigned = {
            key: value for key, value in self.document.items() if key != "record_digest"
        }
        observed = sha256_json(unsigned)
        if claimed != observed:
            raise ProvenanceError(
                "draft provenance was modified after record_digest was computed: "
                f"claimed {claimed!r}, observed {observed!r}"
            )


def build_draft_provenance(
    evidence: EvidenceView,
    approval: Approval,
    model: AuthoringModel,
    drafts: DraftBundle,
    *,
    assertions_compiled: bool,
    compilation_refusals: tuple[str, ...] = (),
) -> DraftProvenance:
    """Record the drafting phase and what it still could not produce."""
    documents = drafts.as_documents()
    document: dict[str, Any] = {
        "schema_version": DRAFT_PROVENANCE_VERSION,
        "phase": "drafting",
        "adapter": {
            "name": "nemotron-bfcl-pack-authoring",
            "version": DRAFTING_ADAPTER_VERSION,
        },
        "pack": dict(evidence.document["pack"]),
        "evidence": {
            "bundle_digest": evidence.digest,
            "attained_level": evidence.attained_level,
            "unresolved_unknowns": sorted(evidence.unresolved_unknowns),
        },
        "approval": approval.as_dict(),
        "model": model.as_provenance(),
        "calls": [record.as_dict() for record in drafts.calls],
        # Each artifact is digested so a later edit to a draft is visible rather than
        # inheriting the trust of the run that generated it.
        "artifact_digests": {
            name: sha256_json(value) for name, value in sorted(documents.items())
        },
        "assertions_compiled": assertions_compiled,
        "compilation_refusals": list(compilation_refusals),
        "blocked_on": sorted(
            {
                field
                for case in drafts.validation_cases.cases
                for field in case.blocked_on
            }
            | {
                field
                for template in drafts.task_templates.templates
                for field in template.blocked_on
            }
            | {
                field
                for spec in drafts.assertions.assertions
                for field in spec.blocked_on
            }
        ),
    }
    document["record_digest"] = sha256_json(document)
    return DraftProvenance(document=document)


def write_draft_provenance(provenance: DraftProvenance, path: Path) -> Path:
    provenance.verify_digest()
    return write_canonical_json(provenance.document, path)
