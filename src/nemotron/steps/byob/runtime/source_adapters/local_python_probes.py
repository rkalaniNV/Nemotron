"""BFCL-owned A1/A2 probes for reviewed local-Python sources."""

from __future__ import annotations

import ast
import copy
import io
import re
import time
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    ProcessWorker,
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
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    HeldOutLeakageError,
    scan_held_out_terms,
)
from nemotron.steps.byob.runtime.source_adapters.local_python import (
    LocalPythonInspection,
    inspect_local_python_package,
)

LOCAL_PROBE_PLAN_VERSION: Literal[
    "bfcl-local-probe-plan-v1"
] = "bfcl-local-probe-plan-v1"
_SAFE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SAFE_STDLIB = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)


class LocalProbeError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        try:
            self.code = CertificationRefusalCode(code).value
        except ValueError as exc:
            raise ValueError(f"unknown local probe refusal code {code!r}") from exc
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalProbeCase(_StrictModel):
    case_id: StrictStr
    tool: StrictStr
    arguments: dict[str, Any]
    expectation: Literal["success", "structured_error", "timeout"]
    expected_state_change: StrictBool | None = None
    expected_error_code: StrictStr | None = None

    @field_validator("case_id", "tool")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("probe case ids and tools must be safe identifiers")
        return value

    @field_validator("arguments")
    @classmethod
    def _canonical_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            sha256_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe arguments must be canonical JSON") from exc
        return value

    @model_validator(mode="after")
    def _expectation_fields(self) -> LocalProbeCase:
        if self.expectation == "success":
            if self.expected_state_change is None or self.expected_error_code is not None:
                raise ValueError(
                    "success cases require expected_state_change and no error code"
                )
        elif self.expectation == "structured_error":
            if not (self.expected_error_code and self.expected_error_code.strip()):
                raise ValueError("structured-error cases require expected_error_code")
            if self.expected_state_change not in {None, False}:
                raise ValueError("structured-error cases cannot expect state mutation")
        elif (
            self.expected_state_change is not None
            or self.expected_error_code is not None
        ):
            raise ValueError("timeout cases cannot claim an outcome or state change")
        return self


class LocalProbePlan(_StrictModel):
    schema_version: Literal["bfcl-local-probe-plan-v1"]
    clock: StrictStr
    seed: StrictInt
    fixtures: dict[str, Any] | None
    cases: tuple[LocalProbeCase, ...]
    confirmation_parameter: StrictStr = "confirm"
    status_field: StrictStr = "status"
    pending_status: StrictStr = "awaiting_confirmation"
    error_path: tuple[StrictStr, ...] = ("error", "code")

    @field_validator("clock")
    @classmethod
    def _clock(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("probe clock must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("probe clock must include an explicit timezone")
        return value

    @field_validator("fixtures")
    @classmethod
    def _fixtures(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        try:
            sha256_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("probe fixtures must be canonical JSON") from exc
        return value

    @field_validator("cases")
    @classmethod
    def _cases(
        cls,
        value: tuple[LocalProbeCase, ...],
    ) -> tuple[LocalProbeCase, ...]:
        if not value:
            raise ValueError("a local probe plan requires cases")
        ids = [case.case_id for case in value]
        if len(ids) != len(set(ids)) or ids != sorted(ids):
            raise ValueError("probe cases must have unique, sorted case_id values")
        if sum(case.expectation == "success" for case in value) > 16:
            raise ValueError("a probe plan supports at most 16 success cases")
        if sum(case.expectation == "structured_error" for case in value) > 8:
            raise ValueError("a probe plan supports at most 8 structured-error cases")
        if sum(case.expectation == "timeout" for case in value) > 1:
            raise ValueError("a probe plan supports at most one timeout case")
        return value

    @field_validator(
        "confirmation_parameter",
        "status_field",
        "pending_status",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("probe vocabulary strings must be non-empty")
        return value

    @field_validator("error_path")
    @classmethod
    def _error_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("error_path must contain non-empty path segments")
        return value

    @property
    def digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


@dataclass(frozen=True)
class LocalProbeRun:
    descriptor: AdapterDescriptor
    plan_digest: str
    records: tuple[ProbeExecutionRecord, ...]


def local_runtime_descriptor(timeout_s: float = 10.0) -> AdapterDescriptor:
    """Return the expanded descriptor whose digest is distinct from UA-801 A0."""
    return AdapterDescriptor(
        contract_version=ADAPTER_CONTRACT_VERSION,
        kind="local_python",
        implementation_name="bfcl.local_python",
        implementation_version="1.0.0+a2",
        capabilities=tuple(sorted(AdapterCapability, key=lambda item: item.value)),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.READ_ONLY,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.RESET_ISOLATED,
            max_calls=128,
            timeout_s=timeout_s,
        ),
        cleanup=CleanupSemantics(
            kind=CleanupKind.PROCESS,
            timeout_s=timeout_s,
        ),
    )


def _validate_plan(
    inspection: LocalPythonInspection,
    plan: LocalProbePlan,
) -> None:
    tools = {tool.published_name: tool for tool in inspection.tools}
    unknown = sorted({case.tool for case in plan.cases} - set(tools))
    if unknown:
        raise LocalProbeError(
            "reviewed_schema_invalid",
            "probe plan names unknown tools: " + ", ".join(unknown),
        )
    success_by_tool = {
        name: [
            case
            for case in plan.cases
            if case.tool == name and case.expectation == "success"
        ]
        for name in tools
    }
    missing = sorted(name for name, cases in success_by_tool.items() if not cases)
    if missing:
        raise LocalProbeError(
            "result_shape_incomplete",
            "success coverage is missing tools: " + ", ".join(missing),
        )
    mutation_missing = sorted(
        name
        for name, tool in tools.items()
        if tool.mutates
        and not any(case.expected_state_change is True for case in success_by_tool[name])
    )
    if mutation_missing:
        raise LocalProbeError(
            "mutation_declaration_mismatch",
            "mutating tools lack a state-changing case: " + ", ".join(mutation_missing),
        )


def _validate_execution_surface(inspection: LocalPythonInspection) -> dict[str, Any]:
    root = inspection.package_root
    local_top_levels = {
        Path(relative).parts[0].removesuffix(".py")
        for relative in inspection.import_closure
    }
    checked: list[str] = []
    for relative in inspection.import_closure:
        path = root / relative
        raw = path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        tree = ast.parse(raw.decode(encoding), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                raise LocalProbeError(
                    "probe_unsafe",
                    f"execution policy rejects __builtins__ access in {relative}",
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise LocalProbeError(
                    "probe_unsafe",
                    f"execution policy rejects dunder access in {relative}",
                )
            if isinstance(node, ast.Import):
                names = [alias.name.partition(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = (
                    []
                    if node.level
                    else [(node.module or "").partition(".")[0]]
                )
            else:
                names = []
            for name in names:
                if name and name not in local_top_levels and name not in _SAFE_STDLIB:
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects import {name!r} in {relative}",
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects {node.func.id}() in {relative}",
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {"fork", "open", "popen", "run", "spawn", "system"}
                ):
                    raise LocalProbeError(
                        "probe_unsafe",
                        f"execution policy rejects .{node.func.attr}() in {relative}",
                    )
        checked.append(relative)
    return {
        "policy": "bfcl-local-least-privilege-v1",
        "checked_files": checked,
        "allowed_stdlib": sorted(_SAFE_STDLIB),
    }


def _extract_path(value: Any, path: Sequence[str]) -> Any:
    current = value
    for segment in path:
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _json_shape(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return {"type": "depth_limit"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        shapes = {
            sha256_json(_json_shape(item, depth=depth + 1)): _json_shape(
                item, depth=depth + 1
            )
            for item in value
        }
        return {"type": "array", "items": [shapes[key] for key in sorted(shapes)]}
    if isinstance(value, Mapping):
        return {
            "type": "object",
            "properties": {
                str(key): _json_shape(child, depth=depth + 1)
                for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            },
        }
    raise LocalProbeError(
        "probe_evidence_invalid",
        f"backend returned non-JSON value {type(value).__name__}",
    )


def _record(
    probe: CertificationProbe,
    *,
    started: float,
    calls: int,
    status: Literal["pass", "fail", "not_applicable"],
    evidence: Any,
    reason: str | None = None,
    cleanup_status: Literal["passed", "failed", "not_required"] = "passed",
) -> ProbeExecutionRecord:
    return ProbeExecutionRecord(
        observation=AdapterProbeObservation(
            probe=probe,
            status=status,
            evidence=evidence,
            reason=reason,
        ),
        observed_calls=calls,
        elapsed_s=float(time.monotonic() - started),
        cleanup_status=cleanup_status,
    )


def run_local_python_probes(
    inspection: LocalPythonInspection,
    plan: LocalProbePlan,
    *,
    allowed_roots: tuple[Path, ...],
    held_out_sensitive_terms: Sequence[str] = (),
    timeout_s: float = 10.0,
    timeout_probe_s: float = 0.25,
) -> LocalProbeRun:
    """Run bounded local probes and return observations, never a tier or report."""
    _validate_plan(inspection, plan)
    plan_findings = scan_held_out_terms(
        plan.model_dump(mode="json"),
        sensitive_terms=held_out_sensitive_terms,
        location="$.local_probe_plan",
    )
    if plan_findings:
        raise HeldOutLeakageError(plan_findings)
    current = inspect_local_python_package(
        inspection.package_root,
        allowed_roots=allowed_roots,
        timeout_s=timeout_s,
    )
    if current.identity != inspection.identity:
        raise LocalProbeError(
            "identity_drift",
            "local source identity changed before probes",
        )
    policy_evidence = _validate_execution_surface(current)
    descriptor = local_runtime_descriptor(timeout_s)
    worker = ProcessWorker(default_timeout_s=timeout_s, worker="process")
    fixture_source = current.package_root / "fixtures.json"
    fixture_source_path = fixture_source if fixture_source.is_file() else None

    def episode(
        task_id: str,
        steps: list[dict[str, Any]],
        *,
        tool_timeout: float = timeout_s,
    ) -> list[Any]:
        return worker.run_episode(
            backend_path=current.backend_path,
            fixtures=copy.deepcopy(plan.fixtures),
            clock_iso=plan.clock,
            seed=plan.seed,
            task_id=task_id,
            steps=steps,
            import_root=current.package_root,
            import_timeout_s=timeout_s,
            reset_timeout_s=timeout_s,
            tool_timeout_s=tool_timeout,
            assertion_timeout_s=timeout_s,
            episode_timeout_s=max(
                timeout_s + 2.0,
                timeout_s + tool_timeout * max(1, len(steps)),
            ),
            fixture_source_path=fixture_source_path,
        )

    records: dict[CertificationProbe, ProbeExecutionRecord] = {}
    records[CertificationProbe.IDENTITY_INTEGRITY] = _record(
        CertificationProbe.IDENTITY_INTEGRITY,
        started=time.monotonic(),
        calls=0,
        status="pass",
        evidence={
            "source_identity_digest": inspection.source_identity_digest,
            "execution_policy": policy_evidence,
        },
        cleanup_status="not_required",
    )

    catalog_started = time.monotonic()
    try:
        inspection_output, listed = episode(
            "ua802-catalog",
            [{"op": "inspect_backend"}, {"op": "list_tools"}],
        )
        required_symbols = {"call_tool", "get_state", "list_tools", "reset"}
        symbols_ok = isinstance(inspection_output, Mapping) and all(
            inspection_output.get(name) is True for name in required_symbols
        )
        listed_names = (
            sorted(listed)
            if isinstance(listed, list)
            and all(isinstance(name, str) for name in listed)
            and len(listed) == len(set(listed))
            else []
        )
        reviewed_names = sorted(tool.published_name for tool in current.tools)
        catalog_ok = symbols_ok and listed_names == reviewed_names
        records[CertificationProbe.CATALOG_INTEGRITY] = _record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=1,
            status="pass" if catalog_ok else "fail",
            evidence={
                "backend_symbols_complete": symbols_ok,
                "listed_names": listed_names,
                "reviewed_names": reviewed_names,
            },
            reason=None if catalog_ok else "catalog_mismatch",
        )
    except TimeoutError:
        records[CertificationProbe.CATALOG_INTEGRITY] = _record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=1,
            status="fail",
            evidence={"backend_symbols_complete": False},
            reason="probe_timeout",
        )
    except Exception as exc:  # noqa: BLE001
        records[CertificationProbe.CATALOG_INTEGRITY] = _record(
            CertificationProbe.CATALOG_INTEGRITY,
            started=catalog_started,
            calls=1,
            status="fail",
            evidence={"error_type": type(exc).__name__},
            reason="probe_failed",
        )

    success_cases = [
        case for case in plan.cases if case.expectation == "success"
    ]
    error_cases = [
        case for case in plan.cases if case.expectation == "structured_error"
    ]
    timeout_cases = [
        case for case in plan.cases if case.expectation == "timeout"
    ]
    success_observations: list[dict[str, Any]] = []
    executable_started = time.monotonic()
    executable_reason: str | None = None
    try:
        for case in success_cases:
            output = episode(
                f"ua802-success-{case.case_id}",
                [
                    {"op": "reset"},
                    {"op": "get_state"},
                    {
                        "op": "call_tool",
                        "name": case.tool,
                        "arguments": case.arguments,
                    },
                    {"op": "get_state"},
                ],
            )
            before, result, after = output[1], output[2], output[3]
            if scan_held_out_terms(
                {"result": result, "state": after},
                sensitive_terms=held_out_sensitive_terms,
            ):
                raise LocalProbeError(
                    "probe_evidence_invalid",
                    "observed success output overlaps held-out material",
                )
            success_observations.append(
                {
                    "case_id": case.case_id,
                    "tool": case.tool,
                    "result_digest": sha256_json(result),
                    "result_shape": _json_shape(result),
                    "before_state_digest": sha256_json(before),
                    "after_state_digest": sha256_json(after),
                    "state_changed": sha256_json(before) != sha256_json(after),
                }
            )
    except TimeoutError:
        executable_reason = "probe_timeout"
    except LocalProbeError as exc:
        executable_reason = exc.code
    except Exception:
        executable_reason = "probe_failed"
    records[CertificationProbe.EXECUTABLE_OBSERVATION] = _record(
        CertificationProbe.EXECUTABLE_OBSERVATION,
        started=executable_started,
        calls=len(success_observations),
        status="pass" if executable_reason is None else "fail",
        evidence={"observations": success_observations},
        reason=executable_reason,
    )

    error_started = time.monotonic()
    if not error_cases:
        records[CertificationProbe.STRUCTURED_ERROR_SHAPE] = _record(
            CertificationProbe.STRUCTURED_ERROR_SHAPE,
            started=error_started,
            calls=0,
            status="not_applicable",
            evidence={"applicable": False, "error_path": list(plan.error_path)},
            reason="no_structured_error_case",
        )
    else:
        error_observations: list[dict[str, Any]] = []
        error_reason: str | None = None
        try:
            for case in error_cases:
                output = episode(
                    f"ua802-error-{case.case_id}",
                    [
                        {"op": "reset"},
                        {
                            "op": "call_tool",
                            "name": case.tool,
                            "arguments": case.arguments,
                        },
                    ],
                )
                result = output[1]
                if scan_held_out_terms(
                    result,
                    sensitive_terms=held_out_sensitive_terms,
                ):
                    raise LocalProbeError(
                        "probe_evidence_invalid",
                        "observed error output overlaps held-out material",
                    )
                observed_code = _extract_path(result, plan.error_path)
                if observed_code != case.expected_error_code:
                    error_reason = "structured_error_mismatch"
                error_observations.append(
                    {
                        "case_id": case.case_id,
                        "result_shape": _json_shape(result),
                        "error_code_digest": sha256_json({"code": observed_code}),
                        "matched": observed_code == case.expected_error_code,
                    }
                )
        except TimeoutError:
            error_reason = "probe_timeout"
        except LocalProbeError as exc:
            error_reason = exc.code
        except Exception:
            error_reason = "probe_failed"
        records[CertificationProbe.STRUCTURED_ERROR_SHAPE] = _record(
            CertificationProbe.STRUCTURED_ERROR_SHAPE,
            started=error_started,
            calls=len(error_observations),
            status="pass" if error_reason is None else "fail",
            evidence={"applicable": True, "observations": error_observations},
            reason=error_reason,
        )

    sequence = [
        {
            "op": "call_tool",
            "name": case.tool,
            "arguments": case.arguments,
            "turn_index": index,
        }
        for index, case in enumerate(success_cases)
    ]
    deterministic_started = time.monotonic()
    deterministic_reason: str | None = None
    deterministic_evidence: dict[str, Any] = {}
    try:
        first = episode(
            "ua802-determinism-a",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        second = episode(
            "ua802-determinism-b",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        first_digest = sha256_json(first)
        second_digest = sha256_json(second)
        deterministic_evidence = {
            "first_digest": first_digest,
            "second_digest": second_digest,
        }
        if first_digest != second_digest:
            deterministic_reason = "reset_nondeterministic"
    except TimeoutError:
        deterministic_reason = "probe_timeout"
    except Exception:
        deterministic_reason = "probe_failed"
    records[CertificationProbe.RESET_DETERMINISM] = _record(
        CertificationProbe.RESET_DETERMINISM,
        started=deterministic_started,
        calls=2 * len(success_cases),
        status="pass" if deterministic_reason is None else "fail",
        evidence=deterministic_evidence or {"completed": False},
        reason=deterministic_reason,
    )

    isolation_started = time.monotonic()
    isolation_reason: str | None = None
    isolation_evidence: dict[str, Any] = {}
    try:
        baseline = episode(
            "ua802-isolation-baseline",
            [{"op": "reset"}, {"op": "get_state"}],
        )[-1]
        episode(
            "ua802-isolation-mutation",
            [{"op": "reset"}, *sequence, {"op": "get_state"}],
        )
        fresh = episode(
            "ua802-isolation-fresh",
            [{"op": "reset"}, {"op": "get_state"}],
        )[-1]
        baseline_digest = sha256_json(baseline)
        fresh_digest = sha256_json(fresh)
        isolation_evidence = {
            "baseline_digest": baseline_digest,
            "fresh_digest": fresh_digest,
        }
        if baseline_digest != fresh_digest:
            isolation_reason = "episode_state_leakage"
    except TimeoutError:
        isolation_reason = "probe_timeout"
    except Exception:
        isolation_reason = "probe_failed"
    records[CertificationProbe.EPISODE_ISOLATION] = _record(
        CertificationProbe.EPISODE_ISOLATION,
        started=isolation_started,
        calls=len(success_cases),
        status="pass" if isolation_reason is None else "fail",
        evidence=isolation_evidence or {"completed": False},
        reason=isolation_reason,
    )

    confirmation_tools = {
        tool.published_name: tool for tool in current.tools if tool.requires_confirmation
    }
    confirmation_started = time.monotonic()
    if not confirmation_tools:
        records[CertificationProbe.CONFIRMATION_SAFETY] = _record(
            CertificationProbe.CONFIRMATION_SAFETY,
            started=confirmation_started,
            calls=0,
            status="not_applicable",
            evidence={"applicable": False},
            reason="no_confirmation_tools",
        )
    else:
        confirmation_observations: list[dict[str, Any]] = []
        confirmation_reason: str | None = None
        try:
            for name in sorted(confirmation_tools):
                case = next(case for case in success_cases if case.tool == name)
                arguments = {**case.arguments, plan.confirmation_parameter: False}
                output = episode(
                    f"ua802-confirmation-{case.case_id}",
                    [
                        {"op": "reset"},
                        {"op": "get_state"},
                        {"op": "call_tool", "name": name, "arguments": arguments},
                        {"op": "get_state"},
                    ],
                )
                before, result, after = output[1], output[2], output[3]
                pending = (
                    isinstance(result, Mapping)
                    and result.get(plan.status_field) == plan.pending_status
                )
                unchanged = sha256_json(before) == sha256_json(after)
                confirmation_observations.append(
                    {
                        "tool": name,
                        "pending": pending,
                        "state_unchanged": unchanged,
                    }
                )
                if not pending or not unchanged:
                    confirmation_reason = "probe_unsafe"
        except TimeoutError:
            confirmation_reason = "probe_timeout"
        except Exception:
            confirmation_reason = "probe_failed"
        records[CertificationProbe.CONFIRMATION_SAFETY] = _record(
            CertificationProbe.CONFIRMATION_SAFETY,
            started=confirmation_started,
            calls=len(confirmation_observations),
            status="pass" if confirmation_reason is None else "fail",
            evidence={
                "applicable": True,
                "observations": confirmation_observations,
            },
            reason=confirmation_reason,
        )

    timeout_started = time.monotonic()
    if not timeout_cases:
        records[CertificationProbe.TIMEOUT_CLEANUP] = _record(
            CertificationProbe.TIMEOUT_CLEANUP,
            started=timeout_started,
            calls=0,
            status="fail",
            evidence={"timeout_case_present": False},
            reason="probe_missing",
        )
    else:
        timeout_case = timeout_cases[0]
        timed_out = False
        fresh_ok = False
        try:
            episode(
                f"ua802-timeout-{timeout_case.case_id}",
                [
                    {"op": "reset"},
                    {
                        "op": "call_tool",
                        "name": timeout_case.tool,
                        "arguments": timeout_case.arguments,
                    },
                ],
                tool_timeout=timeout_probe_s,
            )
        except TimeoutError:
            timed_out = True
        except Exception:
            pass
        try:
            episode(
                "ua802-timeout-recovery",
                [{"op": "reset"}, {"op": "get_state"}],
            )
            fresh_ok = True
        except Exception:
            fresh_ok = False
        timeout_ok = timed_out and fresh_ok
        records[CertificationProbe.TIMEOUT_CLEANUP] = _record(
            CertificationProbe.TIMEOUT_CLEANUP,
            started=timeout_started,
            calls=1,
            status="pass" if timeout_ok else "fail",
            evidence={
                "timeout_observed": timed_out,
                "business_call_attempts": 1,
                "process_cleanup_completed": timed_out,
                "fresh_episode_succeeded": fresh_ok,
                "unknown_commit_state_preserved": timed_out,
            },
            reason=None if timeout_ok else "cleanup_failed",
            cleanup_status="passed" if timeout_ok else "failed",
        )

    mutation_started = time.monotonic()
    mutation_findings = []
    tools_by_name = {tool.published_name: tool for tool in current.tools}
    for observation, case in zip(success_observations, success_cases, strict=False):
        changed = observation["state_changed"]
        declared = tools_by_name[case.tool].mutates
        if changed != case.expected_state_change or (changed and not declared):
            mutation_findings.append(case.case_id)
    mutation_ok = (
        executable_reason is None
        and len(success_observations) == len(success_cases)
        and not mutation_findings
    )
    records[CertificationProbe.MUTATION_DECLARATION] = _record(
        CertificationProbe.MUTATION_DECLARATION,
        started=mutation_started,
        calls=len(success_observations),
        status="pass" if mutation_ok else "fail",
        evidence={
            "checked_cases": [item["case_id"] for item in success_observations],
            "mismatched_cases": mutation_findings,
        },
        reason=None if mutation_ok else "mutation_declaration_mismatch",
    )

    shape_started = time.monotonic()
    covered = sorted({item["tool"] for item in success_observations})
    expected_tools = sorted(tools_by_name)
    shape_ok = executable_reason is None and covered == expected_tools
    records[CertificationProbe.RESULT_SHAPE_COVERAGE] = _record(
        CertificationProbe.RESULT_SHAPE_COVERAGE,
        started=shape_started,
        calls=len(success_observations),
        status="pass" if shape_ok else "fail",
        evidence={
            "covered_tools": covered,
            "expected_tools": expected_tools,
            "result_shapes": {
                item["case_id"]: item["result_shape"] for item in success_observations
            },
        },
        reason=None if shape_ok else "result_shape_incomplete",
    )

    final = inspect_local_python_package(
        current.package_root,
        allowed_roots=allowed_roots,
        timeout_s=timeout_s,
    )
    if final.identity != current.identity:
        records[CertificationProbe.IDENTITY_INTEGRITY] = _record(
            CertificationProbe.IDENTITY_INTEGRITY,
            started=time.monotonic(),
            calls=0,
            status="fail",
            evidence={"identity_changed_during_probes": True},
            reason="identity_drift",
            cleanup_status="not_required",
        )
    return LocalProbeRun(
        descriptor=descriptor,
        plan_digest=plan.digest,
        records=tuple(records[probe] for probe in CertificationProbe),
    )
