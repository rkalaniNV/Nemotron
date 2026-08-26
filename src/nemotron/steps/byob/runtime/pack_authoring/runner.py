"""One drafting run: verify, approve, draft, compile what can be compiled, record.

The gates come first and the writes come last, so a run that is going to be refused leaves
nothing behind that looks like output. Drafts are written as YAML next to the pack rather
than into it, because they are still proposals: `task_templates.yaml` inside a pack directory
would be loaded by the pipeline as though a human had written it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_text,
    write_text_atomic,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import (
    Approval,
    EvidenceView,
    load_approval,
    load_evidence_bundle,
)
from nemotron.steps.byob.runtime.pack_authoring.compile_assertions import (
    CompilationError,
    compile_assertions,
)
from nemotron.steps.byob.runtime.pack_authoring.drafts import (
    DraftBundle,
    DraftingContext,
    draft_all,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    StructuredCaller,
)
from nemotron.steps.byob.runtime.pack_authoring.provenance import (
    DraftProvenance,
    build_draft_provenance,
    write_draft_provenance,
)

DRAFT_DIRECTORY_NAME = "drafts"
CACHE_FILE_NAME = "authoring_io_cache.jsonl"
PROVENANCE_FILE_NAME = "draft_provenance.json"
ASSERTIONS_FILE_NAME = "assertions.py"


@dataclass(frozen=True)
class DraftingResult:
    evidence: EvidenceView
    approval: Approval
    drafts: DraftBundle
    provenance: DraftProvenance
    output_root: Path
    assertions_path: Path | None
    compilation_refusals: tuple[str, ...]

    @property
    def draft_root(self) -> Path:
        return self.output_root / DRAFT_DIRECTORY_NAME


def _dump_yaml(document: object) -> str:
    return str(
        yaml.safe_dump(
            document,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
            width=100,
        )
    )


def run_drafting(
    bundle_path: Path,
    approval_path: Path,
    output_root: Path,
    model: AuthoringModel,
    *,
    caller: StructuredCaller | None = None,
) -> DraftingResult:
    """Draft the pack artifacts an approved evidence bundle can support."""
    evidence = load_evidence_bundle(bundle_path)
    approval = load_approval(approval_path, evidence)

    root = output_root.resolve()
    context = DraftingContext(
        evidence=evidence,
        model=model,
        cache=ImmutableModelIOCache(root / CACHE_FILE_NAME),
        run_dir=root / "model_runs",
        caller=caller,
    )
    drafts = draft_all(context)

    source: str | None = None
    refusals: tuple[str, ...] = ()
    try:
        source = compile_assertions(drafts.assertions)
    except CompilationError as exc:
        refusals = exc.reasons

    draft_root = root / DRAFT_DIRECTORY_NAME
    for name, document in drafts.as_documents().items():
        write_text_atomic(_dump_yaml(document), draft_root / f"{name}.yaml")

    assertions_path: Path | None = None
    if source is not None:
        assertions_path = write_text_atomic(source, draft_root / ASSERTIONS_FILE_NAME)

    provenance = build_draft_provenance(
        evidence,
        approval,
        model,
        drafts,
        assertions_compiled=source is not None,
        compilation_refusals=refusals,
    )
    write_draft_provenance(provenance, root / PROVENANCE_FILE_NAME)
    if assertions_path is not None and source is not None:
        # Prove the file on disk is the source that was digested into provenance.
        written = assertions_path.read_text(encoding="utf-8")
        if sha256_text(written) != sha256_text(source):
            raise CompilationError(
                ["compiled assertions.py on disk does not match the compiled source"]
            )
    return DraftingResult(
        evidence=evidence,
        approval=approval,
        drafts=drafts,
        provenance=provenance,
        output_root=root,
        assertions_path=assertions_path,
        compilation_refusals=refusals,
    )
