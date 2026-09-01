"""Bind certified source, compiled drafts, and reviewed semantics into a candidate pack.

Drafting stops at proposals. `runner.py` writes its plans *next to* a pack rather than into
one, and `compile_assertions.py` emits only trace predicates, because a model that has
never probed a server must not be able to state what that server returns. What a loadable
pack still needs on top of that — slots bound to fixture columns, turn policies, per
language user turns — is deliberately absent from the draft schema for the same reason.

So this step is a binder rather than an author. Everything mechanical is derived from
artifacts that are already trusted: pack identity from verified evidence, `backend.py`,
`tools.json`, and `fixtures.json` from the exact source tree certification fingerprinted,
and `assertions.py` from drafts that compiled without a blocker. The semantics a model
cannot supply arrive as one reviewed supplement, and every tool and assertion that
supplement names is checked back against those artifacts, so it cannot introduce a tool the
source never published or an assertion nobody compiled.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.assertion_capabilities import (
    AssertionCapabilityError,
    read_literal_assertion_capabilities,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    SourceEvidenceDocument,
    load_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    LocalPythonError,
    inspect_local_python_package,
)

SUPPLEMENT_VERSION: Literal[
    "bfcl-candidate-pack-supplement-v1"
] = "bfcl-candidate-pack-supplement-v1"
RECORD_VERSION: Literal[
    "bfcl-candidate-pack-record-v1"
] = "bfcl-candidate-pack-record-v1"
PACK_DIRECTORY_NAME = "pack"
RECORD_FILE_NAME = "candidate_pack_provenance.json"
ASSERTIONS_FILE_NAME = "assertions.py"
DRAFT_PROVENANCE_FILE_NAME = "draft_provenance.json"

# Copied verbatim from the certified source tree. `fixtures.json` is optional there, so it
# is the only one that may be absent.
_SOURCE_FILES: tuple[tuple[str, bool], ...] = (
    ("backend.py", True),
    ("tools.json", True),
    ("fixtures.json", False),
)


class PackAssemblyError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


class CandidatePackSupplement(BaseModel):
    """The pack semantics no draft can express, supplied once and reviewed as a unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bfcl-candidate-pack-supplement-v1"]
    languages: tuple[StrictStr, ...]
    clock: StrictStr
    task_templates: tuple[dict[str, Any], ...]
    validation_cases: tuple[dict[str, Any], ...]
    absent_ids: dict[StrictStr, tuple[StrictStr, ...]] = {}
    primary_keys: dict[StrictStr, StrictStr] = {}
    assistant_turn_templates: dict[StrictStr, dict[StrictStr, StrictStr]] = {}
    confirmation: dict[StrictStr, StrictStr] | None = None
    system_prompt: StrictStr | None = None

    @field_validator("languages")
    @classmethod
    def _languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("supplement must declare at least one language")
        if len(set(value)) != len(value):
            raise ValueError("supplement languages must be unique")
        if any(not language.strip() for language in value):
            raise ValueError("supplement languages must be non-empty")
        return value

    @field_validator("clock")
    @classmethod
    def _clock(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("supplement clock must be non-empty")
        return value

    @field_validator("task_templates", "validation_cases")
    @classmethod
    def _non_empty(cls, value: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        if not value:
            raise ValueError("supplement must declare at least one entry")
        return value


@dataclass(frozen=True)
class AssembledCandidatePack:
    output_root: Path
    pack_root: Path
    record_path: Path
    record: Mapping[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.pack_root / "manifest.yaml"


def load_candidate_pack_supplement(path: Path) -> CandidatePackSupplement:
    """Read the reviewed supplement, refusing anything the schema does not name."""
    source = path.resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PackAssemblyError(
            "supplement_unreadable",
            f"cannot read pack supplement {source}: {type(exc).__name__}: {exc}",
            recovery="supply a readable YAML supplement",
        ) from exc
    if not isinstance(document, dict):
        raise PackAssemblyError(
            "supplement_invalid",
            "pack supplement must be a mapping",
            recovery=f"write a {SUPPLEMENT_VERSION} mapping",
        )
    try:
        return CandidatePackSupplement.model_validate(document)
    except ValueError as exc:
        raise PackAssemblyError(
            "supplement_invalid",
            f"pack supplement does not satisfy {SUPPLEMENT_VERSION}: {exc}",
            recovery="correct the supplement against the documented schema",
        ) from exc


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackAssemblyError(
            "draft_provenance_unreadable",
            f"cannot read {label} {path}: {type(exc).__name__}: {exc}",
            recovery="assemble from a drafting run that completed",
        ) from exc
    if not isinstance(document, dict):
        raise PackAssemblyError(
            "draft_provenance_unreadable",
            f"{label} must be a mapping",
            recovery="assemble from a drafting run that completed",
        )
    return document


def _verified_evidence(evidence_path: Path) -> SourceEvidenceDocument:
    try:
        return load_source_evidence(evidence_path)
    except ValueError as exc:
        raise PackAssemblyError(
            "evidence_invalid",
            f"cannot load verified source evidence: {exc}",
            recovery="assemble from the evidence bundle intake published",
        ) from exc


def _bind_source(
    source_root: Path,
    evidence: SourceEvidenceDocument,
) -> None:
    """Refuse unless the source tree is the one certification fingerprinted."""
    try:
        inspection = inspect_local_python_package(
            source_root,
            allowed_roots=(source_root,),
        )
    except LocalPythonError as exc:
        raise PackAssemblyError(
            "source_package_invalid",
            f"cannot inspect the local Python source package: {exc}",
            recovery="assemble from the source package intake certified",
        ) from exc
    if inspection.source_identity_digest != sha256_json(
        evidence.identity.model_dump(mode="json")
    ):
        raise PackAssemblyError(
            "source_identity_mismatch",
            "source package identity differs from the verified evidence",
            recovery="assemble from the exact source revision intake certified",
        )


def _compiled_assertion_names(assertions_path: Path) -> frozenset[str]:
    try:
        declared = read_literal_assertion_capabilities(assertions_path)
    except AssertionCapabilityError as exc:
        raise PackAssemblyError(
            "compiled_assertions_invalid",
            f"compiled assertions do not declare readable capabilities: {exc}",
            recovery="recompile the assertion specifications",
        ) from exc
    if not declared:
        raise PackAssemblyError(
            "compiled_assertions_invalid",
            "compiled assertions declare no capabilities",
            recovery="recompile the assertion specifications",
        )
    return frozenset(declared)


def _bind_drafts(draft_root: Path, evidence: SourceEvidenceDocument) -> Path:
    """Return the compiled assertions of a drafting run of this exact evidence."""
    provenance_path = draft_root.parent / DRAFT_PROVENANCE_FILE_NAME
    if not provenance_path.is_file():
        raise PackAssemblyError(
            "draft_provenance_missing",
            f"no drafting record beside the draft root: {provenance_path}",
            recovery="assemble from the output root a drafting run wrote",
        )
    provenance = _load_json_mapping(provenance_path, "draft provenance")
    recorded = provenance.get("evidence")
    bundle_digest = (
        recorded.get("bundle_digest") if isinstance(recorded, dict) else None
    )
    if bundle_digest != evidence.bundle_digest:
        raise PackAssemblyError(
            "draft_evidence_mismatch",
            "drafts were produced from a different evidence revision",
            recovery="redraft the current evidence revision before assembling",
        )
    if provenance.get("blocked_on"):
        raise PackAssemblyError(
            "draft_blocked",
            "drafts are still blocked on "
            + ", ".join(str(item) for item in provenance["blocked_on"]),
            recovery="certify the probes that resolve those unknowns and redraft",
        )
    if not provenance.get("assertions_compiled"):
        raise PackAssemblyError(
            "compiled_assertions_missing",
            "the drafting run compiled no assertions",
            recovery="resolve the compilation refusals and redraft",
        )
    assertions_path = draft_root / ASSERTIONS_FILE_NAME
    if not assertions_path.is_file():
        raise PackAssemblyError(
            "compiled_assertions_missing",
            f"no compiled assertions in the draft root: {assertions_path}",
            recovery="redraft so compilation writes assertions.py",
        )
    return assertions_path


def _referenced_tools(supplement: CandidatePackSupplement) -> set[str]:
    names: set[str] = set()
    for template in supplement.task_templates:
        for key in ("required_tools", "tools_present"):
            names.update(_string_items(template.get(key)))
        for milestone in _mapping_items(template.get("assistant_milestones")):
            tool = milestone.get("tool")
            if isinstance(tool, str) and tool.strip():
                names.add(tool)
    for case in supplement.validation_cases:
        tool = case.get("tool")
        if isinstance(tool, str) and tool.strip():
            names.add(tool)
    return names


def _referenced_assertions(supplement: CandidatePackSupplement) -> set[str]:
    names: set[str] = set()
    for template in supplement.task_templates:
        names.update(_string_items(template.get("success_assertions")))
    return names


def _string_items(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return ()
    return [item for item in value if isinstance(item, str) and item.strip()]


def _mapping_items(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, list):
        return ()
    return [item for item in value if isinstance(item, Mapping)]


def _check_supplement_references(
    supplement: CandidatePackSupplement,
    *,
    evidence: SourceEvidenceDocument,
    assertion_names: frozenset[str],
) -> None:
    published = {tool.published_name for tool in evidence.tools}
    if unknown := sorted(_referenced_tools(supplement) - published):
        raise PackAssemblyError(
            "supplement_tool_unknown",
            "supplement names tools the certified source never published: "
            + ", ".join(unknown),
            recovery="reference only the tools the evidence bundle publishes",
        )
    if unknown := sorted(_referenced_assertions(supplement) - assertion_names):
        raise PackAssemblyError(
            "supplement_assertion_unknown",
            "supplement names assertions the drafts never compiled: "
            + ", ".join(unknown),
            recovery="draft and compile those assertion specifications first",
        )


def _manifest_document(
    supplement: CandidatePackSupplement,
    *,
    evidence: SourceEvidenceDocument,
    has_fixtures: bool,
) -> dict[str, Any]:
    paths: dict[str, str] = {
        "tools": "tools.json",
        "backend": "backend.py",
        "templates": "task_templates.yaml",
        "assertions": ASSERTIONS_FILE_NAME,
        "validation_cases": "validation_cases.yaml",
    }
    if has_fixtures:
        paths["fixtures"] = "fixtures.json"
    manifest: dict[str, Any] = {
        "pack_id": evidence.pack.pack_id,
        "version": evidence.pack.version,
        "languages": list(supplement.languages),
        "clock": supplement.clock,
        "paths": paths,
    }
    if supplement.absent_ids:
        manifest["absent_ids"] = {
            collection: list(identifiers)
            for collection, identifiers in sorted(supplement.absent_ids.items())
        }
    if supplement.primary_keys:
        manifest["primary_keys"] = dict(sorted(supplement.primary_keys.items()))
    if supplement.assistant_turn_templates:
        manifest["assistant_turn_templates"] = {
            name: dict(sorted(turns.items()))
            for name, turns in sorted(supplement.assistant_turn_templates.items())
        }
    if supplement.confirmation is not None:
        manifest["confirmation"] = dict(sorted(supplement.confirmation.items()))
    if supplement.system_prompt is not None:
        manifest["system_prompt_path"] = supplement.system_prompt
    return manifest


def _write_yaml(document: Any, path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def assemble_candidate_pack(
    *,
    evidence_path: Path,
    source_root: Path,
    draft_root: Path,
    supplement_path: Path,
    output_root: Path,
) -> AssembledCandidatePack:
    """Write one candidate oracle pack, or refuse and name the binding that failed."""
    final_root = output_root.resolve()
    if final_root.exists():
        raise PackAssemblyError(
            "pack_output_exists",
            f"candidate pack output already exists: {final_root}",
            recovery="assemble into a fresh directory so no earlier pack is overwritten",
        )
    evidence = _verified_evidence(evidence_path)
    if evidence.source_adapter.kind != "local_python":
        raise PackAssemblyError(
            "adapter_not_supported",
            f"candidate pack assembly covers local_python, not {evidence.source_adapter.kind}",
            recovery="assemble local Python sources, or compile the pack by hand",
        )
    source = source_root.resolve()
    _bind_source(source, evidence)
    assertions_path = _bind_drafts(draft_root.resolve(), evidence)
    assertion_names = _compiled_assertion_names(assertions_path)
    supplement = load_candidate_pack_supplement(supplement_path)
    _check_supplement_references(
        supplement,
        evidence=evidence,
        assertion_names=assertion_names,
    )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(
        dir=final_root.parent,
        prefix=f".{final_root.name}.staging-",
    )
    try:
        root = Path(staging.name)
        pack_root = root / PACK_DIRECTORY_NAME
        pack_root.mkdir()
        copied: list[str] = []
        for name, required in _SOURCE_FILES:
            origin = source / name
            if not origin.is_file():
                if required:
                    raise PackAssemblyError(
                        "source_package_invalid",
                        f"certified source package has no {name}",
                        recovery="assemble from a complete local Python source package",
                    )
                continue
            # copyfile rather than copy2: a pack containing a symlink is refused later,
            # and metadata from the source tree is not part of the pack contract.
            shutil.copyfile(origin, pack_root / name)
            copied.append(name)
        shutil.copyfile(assertions_path, pack_root / ASSERTIONS_FILE_NAME)
        _write_yaml(
            [dict(template) for template in supplement.task_templates],
            pack_root / "task_templates.yaml",
        )
        _write_yaml(
            [dict(case) for case in supplement.validation_cases],
            pack_root / "validation_cases.yaml",
        )
        _write_yaml(
            _manifest_document(
                supplement,
                evidence=evidence,
                has_fixtures="fixtures.json" in copied,
            ),
            pack_root / "manifest.yaml",
        )
        record = _record_document(
            pack_root,
            evidence=evidence,
            source=source,
            supplement_path=supplement_path.resolve(),
            assertion_names=assertion_names,
        )
        write_canonical_json(record, root / RECORD_FILE_NAME)
        root.replace(final_root)
    except Exception:
        staging.cleanup()
        raise
    staging.cleanup()
    return AssembledCandidatePack(
        output_root=final_root,
        pack_root=final_root / PACK_DIRECTORY_NAME,
        record_path=final_root / RECORD_FILE_NAME,
        record=record,
    )


def _record_document(
    pack_root: Path,
    *,
    evidence: SourceEvidenceDocument,
    source: Path,
    supplement_path: Path,
    assertion_names: Sequence[str] | frozenset[str],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": RECORD_VERSION,
        "adapter_kind": evidence.source_adapter.kind,
        "pack": evidence.pack.model_dump(mode="json"),
        "evidence_digest": evidence.bundle_digest,
        "source_identity_digest": sha256_json(
            evidence.identity.model_dump(mode="json")
        ),
        "source_root": str(source),
        "supplement_digest": _digest_file(supplement_path),
        "compiled_assertions": sorted(assertion_names),
        "pack_files": {
            path.relative_to(pack_root).as_posix(): _digest_file(path)
            for path in sorted(pack_root.rglob("*"))
            if path.is_file()
        },
    }
    document["record_digest"] = sha256_json(document)
    return document
