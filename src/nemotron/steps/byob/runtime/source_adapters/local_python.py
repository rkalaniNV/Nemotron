"""Static A0 identity for reviewed local-Python authoring sources."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import platform
import re
import sys
import sysconfig
import tokenize
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    PackTrustError,
    assert_pack_allowed,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    AdapterProbeObservation,
    CertificationProbe,
    CertificationRefusalCode,
    ProbeExecutionRecord,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    ADAPTER_CONTRACT_VERSION,
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    IdentityArtifact,
    SourceIdentity,
    ToolEvidence,
)
from nemotron.steps.byob.runtime.source_adapters.reviewed_catalog import (
    ReviewedCatalogError,
    load_reviewed_tool_catalog,
)

LOCAL_PYTHON_LOCK_VERSION: Literal[
    "bfcl-python-dependency-lock-v1"
] = "bfcl-python-dependency-lock-v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMPORT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class LocalPythonError(ValueError):
    """Stable fail-closed error raised before local A0 evidence can be issued."""

    def __init__(self, code: str, detail: str) -> None:
        try:
            self.code = CertificationRefusalCode(code).value
        except ValueError as exc:
            raise ValueError(f"unknown local Python refusal code {code!r}") from exc
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LockedDependency(_StrictModel):
    import_name: StrictStr
    distribution: StrictStr
    version: StrictStr
    artifact_digest: StrictStr

    @field_validator("import_name")
    @classmethod
    def _import_name(cls, value: str) -> str:
        if not _IMPORT_NAME.fullmatch(value):
            raise ValueError("dependency import_name must be a dotted Python name")
        return value

    @field_validator("distribution", "version")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dependency distribution and version must be non-empty")
        return value

    @field_validator("artifact_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("dependency artifact_digest must be lowercase SHA-256")
        return value


class PythonDependencyLock(_StrictModel):
    schema_version: Literal["bfcl-python-dependency-lock-v1"]
    dependencies: tuple[LockedDependency, ...]

    @field_validator("dependencies")
    @classmethod
    def _canonical_dependencies(
        cls,
        value: tuple[LockedDependency, ...],
    ) -> tuple[LockedDependency, ...]:
        names = [item.import_name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("dependency import names must be unique")
        if names != sorted(names):
            raise ValueError("dependencies must be sorted by import_name")
        return value

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


@dataclass(frozen=True)
class LocalPythonInspection:
    package_root: Path
    backend_path: Path
    dependency_lock: PythonDependencyLock
    descriptor: AdapterDescriptor
    identity: SourceIdentity
    tools: tuple[ToolEvidence, ...]
    import_closure: tuple[str, ...]
    execution_records: tuple[ProbeExecutionRecord, ...]

    @property
    def source_identity_digest(self) -> str:
        return sha256_json(self.identity.model_dump(mode="json"))


def _regular_file(
    root: Path,
    name: str,
    *,
    missing_code: str,
) -> Path:
    candidate = root / name
    if candidate.is_symlink():
        raise LocalPythonError(
            "source_path_escape",
            f"{name} must not be a symlink",
        )
    try:
        resolved = assert_pack_allowed(candidate, (root,))
    except PackTrustError as exc:
        raise LocalPythonError("source_path_escape", str(exc)) from exc
    if not resolved.is_file():
        raise LocalPythonError(missing_code, f"missing reviewed {name}")
    return resolved


def _load_json(path: Path, *, label: str, code: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LocalPythonError(code, f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except LocalPythonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalPythonError(
            code,
            f"cannot load {label}: {type(exc).__name__}",
        ) from exc


def _load_dependency_lock(path: Path) -> PythonDependencyLock:
    try:
        return PythonDependencyLock.model_validate(
            _load_json(
                path,
                label="dependency-lock.json",
                code="dependency_lock_invalid",
            )
        )
    except LocalPythonError:
        raise
    except ValueError as exc:
        raise LocalPythonError("dependency_lock_invalid", str(exc)) from exc


def _module_identity(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root)
    parts = list(relative.parts)
    filename = parts.pop()
    if filename == "__init__.py":
        module = ".".join(parts)
        package = module
    else:
        module = ".".join([*parts, Path(filename).stem])
        package = ".".join(parts)
    return module, package


def _checked_python_path(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise LocalPythonError(
            "source_path_escape",
            f"import closure contains symlink {path.relative_to(root)}",
        )
    try:
        resolved = assert_pack_allowed(path, (root,))
    except PackTrustError as exc:
        raise LocalPythonError("source_path_escape", str(exc)) from exc
    if not resolved.is_file():
        raise LocalPythonError(
            "import_path_ambiguous",
            f"resolved import is not a file: {path.relative_to(root)}",
        )
    return resolved


def _resolve_local_module(root: Path, module: str) -> tuple[Path, ...]:
    if not module or not _IMPORT_NAME.fullmatch(module):
        return ()
    parts = module.split(".")
    module_file = root.joinpath(*parts).with_suffix(".py")
    package_file = root.joinpath(*parts, "__init__.py")
    file_exists = module_file.is_file() or module_file.is_symlink()
    package_exists = package_file.is_file() or package_file.is_symlink()
    if file_exists and package_exists:
        raise LocalPythonError(
            "import_path_ambiguous",
            f"both module and package exist for import {module!r}",
        )
    target = package_file if package_exists else module_file if file_exists else None
    if target is None:
        return ()
    parents: list[Path] = []
    for index in range(1, len(parts)):
        directory = root.joinpath(*parts[:index])
        init_path = directory / "__init__.py"
        if directory.is_dir() and not (init_path.is_file() or init_path.is_symlink()):
            raise LocalPythonError(
                "namespace_package_ambiguous",
                f"namespace package imports are not allowed: {'.'.join(parts[:index])}",
            )
        if init_path.is_file() or init_path.is_symlink():
            parents.append(_checked_python_path(root, init_path))
    return tuple([*parents, _checked_python_path(root, target)])


def _source_tree(path: Path, *, root: Path) -> tuple[ast.AST, str]:
    try:
        raw = path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
    except (OSError, SyntaxError, UnicodeError, LookupError) as exc:
        raise LocalPythonError(
            "source_encoding_invalid",
            f"cannot decode {path.relative_to(root)}: {type(exc).__name__}",
        ) from exc
    try:
        tree = ast.parse(text, filename=path.relative_to(root).as_posix())
    except SyntaxError as exc:
        raise LocalPythonError(
            "source_syntax_invalid",
            f"cannot parse {path.relative_to(root)} at line {exc.lineno}",
        ) from exc
    return tree, f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _absolute_from_import(
    node: ast.ImportFrom,
    *,
    package: str,
    source: str,
) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = package.split(".") if package else []
    remove = node.level - 1
    if not package_parts or remove >= len(package_parts):
        raise LocalPythonError(
            "import_path_ambiguous",
            f"invalid relative import in {source}",
        )
    base = package_parts[: len(package_parts) - remove]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _import_closure(
    root: Path,
    backend: Path,
    lock: PythonDependencyLock,
) -> tuple[tuple[str, str], ...]:
    allowed_external = tuple(item.import_name for item in lock.dependencies)
    queue: deque[Path] = deque([backend])
    visited: dict[str, str] = {}

    def enqueue_or_validate(module: str, *, source: str) -> tuple[Path, ...]:
        local = _resolve_local_module(root, module)
        if local:
            queue.extend(local)
            return local
        top = module.partition(".")[0]
        if top in sys.stdlib_module_names or top in sys.builtin_module_names or any(
            module == allowed or module.startswith(f"{allowed}.")
            for allowed in allowed_external
        ):
            return ()
        if _resolve_local_module(root, top):
            raise LocalPythonError(
                "import_path_ambiguous",
                f"local import {module!r} from {source} cannot be resolved exactly",
            )
        raise LocalPythonError(
            "undeclared_import",
            f"import {module!r} from {source} is neither local, stdlib, nor locked",
        )

    while queue:
        path = _checked_python_path(root, queue.popleft())
        relative = path.relative_to(root).as_posix()
        if relative in visited:
            continue
        tree, source_digest = _source_tree(path, root=root)
        visited[relative] = source_digest
        _, package = _module_identity(root, path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                dynamic = (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"__import__", "compile", "eval", "exec"}
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                )
                if dynamic:
                    raise LocalPythonError(
                        "dynamic_import",
                        f"dynamic code/import operation in {relative}:{node.lineno}",
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "importlib" or alias.name.startswith("importlib."):
                        raise LocalPythonError(
                            "dynamic_import",
                            f"importlib is not allowed in {relative}:{node.lineno}",
                        )
                    enqueue_or_validate(alias.name, source=relative)
            elif isinstance(node, ast.ImportFrom):
                base = _absolute_from_import(
                    node,
                    package=package,
                    source=relative,
                )
                if base == "importlib" or base.startswith("importlib.") or (
                    base == "builtins"
                    and any(alias.name == "__import__" for alias in node.names)
                ):
                    raise LocalPythonError(
                        "dynamic_import",
                        f"dynamic import API is not allowed in {relative}:{node.lineno}",
                    )
                local_base = enqueue_or_validate(base, source=relative)
                if local_base:
                    for alias in node.names:
                        if alias.name != "*":
                            candidate = f"{base}.{alias.name}"
                            local_child = _resolve_local_module(root, candidate)
                            if local_child:
                                queue.extend(local_child)
    return tuple(sorted(visited.items()))


def _runtime_identity() -> dict[str, Any]:
    return {
        "implementation": sys.implementation.name,
        "implementation_version": list(sys.implementation.version),
        "cache_tag": sys.implementation.cache_tag,
        "python_version": platform.python_version(),
        "platform": sysconfig.get_platform(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
    }


def _descriptor(timeout_s: float) -> AdapterDescriptor:
    return AdapterDescriptor(
        contract_version=ADAPTER_CONTRACT_VERSION,
        kind="local_python",
        implementation_name="bfcl.local_python",
        implementation_version="1.0.0",
        capabilities=(
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.PIN_IDENTITY,
        ),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.READ_ONLY,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.IDENTITY_ONLY,
            max_calls=1,
            timeout_s=timeout_s,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.NONE, timeout_s=timeout_s),
    )


def inspect_local_python_package(
    package_path: Path,
    *,
    allowed_roots: tuple[Path, ...],
    timeout_s: float = 10.0,
) -> LocalPythonInspection:
    """Build an A0 identity without importing or executing package code."""
    if package_path.is_symlink():
        raise LocalPythonError(
            "source_path_escape",
            "local Python package path must not be a symlink",
        )
    try:
        root = assert_pack_allowed(package_path, allowed_roots)
    except PackTrustError as exc:
        raise LocalPythonError("source_path_escape", str(exc)) from exc
    if not root.is_dir():
        raise LocalPythonError(
            "source_package_invalid",
            "local Python package path must be a directory",
        )
    backend = _regular_file(root, "backend.py", missing_code="source_package_invalid")
    tools_path = _regular_file(
        root,
        "tools.json",
        missing_code="reviewed_schema_missing",
    )
    lock_path = _regular_file(
        root,
        "dependency-lock.json",
        missing_code="dependency_lock_missing",
    )
    lock = _load_dependency_lock(lock_path)
    try:
        catalog = load_reviewed_tool_catalog(tools_path)
    except ReviewedCatalogError as exc:
        raise LocalPythonError(exc.code, exc.detail) from exc
    closure = _import_closure(root, backend, lock)

    fixtures_digest: str | None = None
    fixtures_path = root / "fixtures.json"
    if fixtures_path.exists() or fixtures_path.is_symlink():
        fixtures_path = _regular_file(
            root,
            "fixtures.json",
            missing_code="source_package_invalid",
        )
        fixtures = _load_json(
            fixtures_path,
            label="fixtures.json",
            code="fixture_metadata_invalid",
        )
        if not isinstance(fixtures, dict):
            raise LocalPythonError(
                "fixture_metadata_invalid",
                "fixtures.json must be an object",
            )
        fixtures_digest = sha256_json(fixtures)

    runtime = _runtime_identity()
    runtime_digest = sha256_json(runtime)
    closure_document = [
        {"path": relative, "digest": digest} for relative, digest in closure
    ]
    effective_document = {
        "schema_version": "bfcl-local-python-identity-v1",
        "source_files": closure_document,
        "reviewed_tool_catalog_digest": catalog.digest,
        "fixtures_digest": fixtures_digest,
        "dependency_lock_digest": lock.digest,
        "runtime": runtime,
    }
    effective_digest = sha256_json(effective_document)
    artifacts = [
        IdentityArtifact(role="backend", digest=dict(closure)["backend.py"]),
        IdentityArtifact(role="dependency_lock", digest=lock.digest),
        IdentityArtifact(role="reviewed_tool_catalog", digest=catalog.digest),
        IdentityArtifact(role="runtime", digest=runtime_digest),
    ]
    if fixtures_digest is not None:
        artifacts.append(IdentityArtifact(role="fixtures", digest=fixtures_digest))
    identity = SourceIdentity(
        subject=f"local-python:{root.name}",
        effective_content_digest=effective_digest,
        source_config_digest=sha256_json(
            {
                "schema_version": "bfcl-local-python-source-v1",
                "backend": "backend.py",
                "tools": "tools.json",
                "fixtures": "fixtures.json" if fixtures_digest is not None else None,
                "dependency_lock": "dependency-lock.json",
                "dependency_lock_digest": lock.digest,
            }
        ),
        artifacts=tuple(sorted(artifacts, key=lambda item: item.role)),
    )
    static_evidence = {
        "effective_content_digest": effective_digest,
        "source_files": closure_document,
        "runtime_digest": runtime_digest,
        "dependency_lock_digest": lock.digest,
    }
    records = (
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=CertificationProbe.IDENTITY_INTEGRITY,
                status="pass",
                evidence=static_evidence,
            ),
            observed_calls=0,
            elapsed_s=0.0,
            cleanup_status="not_required",
        ),
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=CertificationProbe.CATALOG_INTEGRITY,
                status="pass",
                evidence={"reviewed_tool_catalog_digest": catalog.digest},
            ),
            observed_calls=0,
            elapsed_s=0.0,
            cleanup_status="not_required",
        ),
    )
    return LocalPythonInspection(
        package_root=root,
        backend_path=backend,
        dependency_lock=lock,
        descriptor=_descriptor(timeout_s),
        identity=identity,
        tools=catalog.tools,
        import_closure=tuple(relative for relative, _ in closure),
        execution_records=records,
    )
