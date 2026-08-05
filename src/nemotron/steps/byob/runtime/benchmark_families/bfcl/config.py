"""BFCL family-local configuration (not ByobConfig / MCQ)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# runtime/benchmark_families/bfcl/config.py → …/steps/byob
BYOB_ROOT = Path(__file__).resolve().parents[3]

# Benchmark row schemas this build can write. A config naming anything else would
# publish a manifest that promises a shape the parquet does not have.
DEFAULT_BENCHMARK_SCHEMA_VERSION = "1.1"
BENCHMARK_SCHEMA_VERSIONS = frozenset({DEFAULT_BENCHMARK_SCHEMA_VERSION})
LINEAGE_POLICIES = frozenset({"strict_separation", "smoke_no_publication"})

_TOP_LEVEL_KEYS = frozenset(
    {
        "family",
        "expt_name",
        "output_dir",
        "oracle_pack",
        "oracle_runtime",
        "lineage",
        "stage",
        "random_seed",
        "ndd_batch_size",
        "input_dir",
        "schema_version",
        "config_status",
        "surface_generation",
        "surface_quality_validation",
        "task_generation",
        "semantic_deduplication_config",
        "exports",
        "reference_benchmark",
        "generation_model_config",
        "judge_model_config",
        "eval_config_path",
        "translation_config_path",
    }
)
_ORACLE_PACK_KEYS = frozenset(
    {
        "manifest_path",
        "backend_path",
        "endpoint_config_path",
        "fixtures_path",
        "task_templates_path",
        "assertions_path",
        "validation_cases_path",
    }
)
_ORACLE_RUNTIME_KEYS = frozenset(
    {
        "clock",
        "tool_timeout_s",
        "assertion_timeout_s",
        "import_timeout_s",
        "reset_timeout_s",
        "episode_timeout_s",
        "worker",
        "allowed_roots",
    }
)
_LINEAGE_KEYS = frozenset({"policy", "profile_influenced_surface", "judge_advisory", "roles"})
_LINEAGE_ROLES = frozenset({"profile", "paraphrase", "surface_judge"})
_LINEAGE_ROLE_KEYS = frozenset({"enabled", "model_config"})
SURFACE_GENERATION_KEYS = frozenset(
    {
        "language",
        "model_paraphrase_enabled",
        "paraphrases_per_template",
        "preserve_slot_values",
        "prevent_tool_name_leakage",
    }
)
_SURFACE_QUALITY_KEYS = frozenset({"enabled", "drop_authority"})
_TASK_GENERATION_KEYS = frozenset({"tasks_per_category"})
_DEDUP_KEYS = frozenset({"enabled", "model_identifier"})
_EXPORT_KEYS = frozenset({"bfcl_json", "nemo_evaluator_bundle"})


def _reject_unknown(mapping: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} has unknown keys: {', '.join(unknown)}")


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean, got {type(value).__name__}")
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}, got {value}")
    return value


def _require_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite, got {value!r}")
    return converted


def _placeholder_paths(value: Any, path: str = "") -> list[str]:
    """Return config paths whose values still carry replacement sentinels."""
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _placeholder_paths(child, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in _placeholder_paths(child, f"{path}[{index}]")
        ]
    if isinstance(value, str) and "REPLACE_ME_" in value:
        return [path]
    return []


def _validate_clock(raw: str) -> None:
    """Reject clocks the oracle worker cannot parse or that lack an offset."""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"oracle_runtime.clock must be an ISO-8601 timestamp, got {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(
            f"oracle_runtime.clock must carry a UTC offset so runs stay reproducible, got {raw!r}"
        )


def _resolve_path(raw: str | Path | None, *, base: Path = BYOB_ROOT) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class OraclePackRef:
    """Paths into an executable oracle pack."""

    manifest_path: Path
    backend_path: Path | None = None
    endpoint_config_path: Path | None = None
    fixtures_path: Path | None = None
    task_templates_path: Path | None = None
    assertions_path: Path | None = None
    validation_cases_path: Path | None = None


@dataclass(frozen=True)
class OracleRuntimeConfig:
    """Process-worker timeouts, frozen clock, and pack trust roots."""

    clock: str
    tool_timeout_s: float = 5.0
    assertion_timeout_s: float = 5.0
    import_timeout_s: float = 10.0
    reset_timeout_s: float = 5.0
    episode_timeout_s: float = 60.0
    worker: str = "process"
    allowed_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.worker not in {"process", "thread"}:
            raise ValueError(f"oracle_runtime.worker must be 'process' or 'thread', got {self.worker!r}")
        _validate_clock(self.clock)
        for name in (
            "tool_timeout_s",
            "assertion_timeout_s",
            "import_timeout_s",
            "reset_timeout_s",
            "episode_timeout_s",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"oracle_runtime.{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class LineageRole:
    enabled: bool = False
    model_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class LineageConfig:
    policy: str
    profile_influenced_surface: bool | None = False
    judge_advisory: bool | None = None
    roles: dict[str, LineageRole] = field(default_factory=dict)


@dataclass
class BfclConfig:
    """Validated BFCL generation config.

    Kept family-local (conscious deviation from shared ``runtime/config.py``)
    so MCQ fields such as ``hf_dataset`` / ``target_source_mapping`` cannot leak in.
    """

    family: str
    expt_name: str
    output_dir: Path
    oracle_pack: OraclePackRef
    oracle_runtime: OracleRuntimeConfig
    lineage: LineageConfig
    stage: str = "all"
    random_seed: int | None = None
    ndd_batch_size: int = 32
    input_dir: Path | None = None
    schema_version: str | None = None
    config_status: str | None = None
    surface_generation: dict[str, Any] = field(default_factory=dict)
    surface_quality_validation: dict[str, Any] = field(default_factory=dict)
    task_generation: dict[str, Any] = field(default_factory=dict)
    semantic_deduplication_config: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    reference_benchmark: dict[str, Any] | None = None
    generation_model_config: dict[str, Any] | None = None
    judge_model_config: dict[str, Any] | None = None
    eval_config_path: str | None = None
    translation_config_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BfclConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, dict):
            raise ValueError(f"BFCL config must be a YAML mapping: {config_path}")
        _reject_unknown(data, _TOP_LEVEL_KEYS, "BFCL config")

        family = data.get("family")
        if family != "bfcl":
            raise ValueError(f"BfclConfig requires family='bfcl', got {family!r}")

        for key in ("expt_name", "output_dir", "oracle_pack", "oracle_runtime", "lineage"):
            if key not in data:
                raise ValueError(f"BFCL config missing required field {key!r}: {config_path}")
        if not isinstance(data["expt_name"], str) or not data["expt_name"].strip():
            raise ValueError("expt_name must be a non-empty string")
        # The run directory is named after the experiment, and every artifact hash and
        # cache path is stated relative to it. A separator or a dot segment would move
        # the run somewhere the config does not name, including back out of output_dir.
        if (
            data["expt_name"] != data["expt_name"].strip()
            or data["expt_name"] in {".", ".."}
            or Path(data["expt_name"]).parts != (data["expt_name"],)
        ):
            raise ValueError(
                f"expt_name must be a single directory name, got {data['expt_name']!r}; "
                "point output_dir at the parent instead"
            )
        if not isinstance(data["output_dir"], str) or not data["output_dir"].strip():
            raise ValueError("output_dir must be a non-empty path string")

        schema_version = data.get("schema_version")
        if schema_version is not None and str(schema_version) not in BENCHMARK_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version {schema_version!r} is not a schema this build writes; "
                f"use one of {', '.join(sorted(BENCHMARK_SCHEMA_VERSIONS))}"
            )

        config_status = data.get("config_status")
        if config_status not in {None, "template", "resolved"}:
            raise ValueError(
                f"config_status must be 'template' or 'resolved' when present, got {config_status!r}"
            )
        if config_status == "resolved":
            placeholders = _placeholder_paths(data)
            if placeholders:
                raise ValueError(
                    "resolved BFCL config still contains REPLACE_ME_* values at: "
                    + ", ".join(placeholders)
                )

        pack_raw = data["oracle_pack"]
        if not isinstance(pack_raw, dict) or "manifest_path" not in pack_raw:
            raise ValueError("oracle_pack.manifest_path is required")
        _reject_unknown(pack_raw, _ORACLE_PACK_KEYS, "oracle_pack")

        runtime_raw = data["oracle_runtime"]
        if not isinstance(runtime_raw, dict):
            raise ValueError("oracle_runtime must be a mapping")
        _reject_unknown(runtime_raw, _ORACLE_RUNTIME_KEYS, "oracle_runtime")
        if "clock" not in runtime_raw:
            raise ValueError("oracle_runtime.clock is required")
        if not isinstance(runtime_raw["clock"], str):
            raise ValueError(
                "oracle_runtime.clock must be a quoted ISO-8601 string; "
                f"got {type(runtime_raw['clock']).__name__}"
            )

        allowed_raw = runtime_raw.get("allowed_roots")
        if allowed_raw is None:
            allowed_raw = []
        if not isinstance(allowed_raw, list):
            raise ValueError("oracle_runtime.allowed_roots must be a list when present")
        if any(not isinstance(item, str) or not item.strip() for item in allowed_raw):
            raise ValueError("oracle_runtime.allowed_roots entries must be non-empty path strings")
        allowed_roots = tuple(
            p for p in (_resolve_path(item) for item in allowed_raw) if p is not None
        )
        if not allowed_roots:
            # Default trust root: checked-in BYOB data directory.
            allowed_roots = ((BYOB_ROOT / "data").resolve(),)

        oracle_runtime = OracleRuntimeConfig(
            clock=runtime_raw["clock"],
            tool_timeout_s=_require_number(
                runtime_raw.get("tool_timeout_s", 5.0), "oracle_runtime.tool_timeout_s"
            ),
            assertion_timeout_s=_require_number(
                runtime_raw.get("assertion_timeout_s", 5.0),
                "oracle_runtime.assertion_timeout_s",
            ),
            import_timeout_s=_require_number(
                runtime_raw.get("import_timeout_s", 10.0),
                "oracle_runtime.import_timeout_s",
            ),
            reset_timeout_s=_require_number(
                runtime_raw.get("reset_timeout_s", 5.0), "oracle_runtime.reset_timeout_s"
            ),
            episode_timeout_s=_require_number(
                runtime_raw.get("episode_timeout_s", 60.0),
                "oracle_runtime.episode_timeout_s",
            ),
            worker=str(runtime_raw.get("worker", "process")),
            allowed_roots=allowed_roots,
        )

        lineage_raw = data["lineage"]
        if not isinstance(lineage_raw, dict) or "policy" not in lineage_raw:
            raise ValueError("lineage.policy is required")
        _reject_unknown(lineage_raw, _LINEAGE_KEYS, "lineage")
        policy = str(lineage_raw["policy"])
        if policy not in LINEAGE_POLICIES:
            raise ValueError(
                f"lineage.policy must be one of {', '.join(sorted(LINEAGE_POLICIES))}, got {policy!r}"
            )
        roles_raw = lineage_raw.get("roles")
        if roles_raw is None:
            roles_raw = {}
        roles: dict[str, LineageRole] = {}
        if not isinstance(roles_raw, dict):
            raise ValueError("lineage.roles must be a mapping")
        _reject_unknown(roles_raw, _LINEAGE_ROLES, "lineage.roles")
        for name, role in roles_raw.items():
            if not isinstance(role, dict):
                raise ValueError(f"lineage.roles.{name} must be a mapping")
            _reject_unknown(role, _LINEAGE_ROLE_KEYS, f"lineage.roles.{name}")
            enabled = _require_bool(
                role.get("enabled", False), f"lineage.roles.{name}.enabled"
            )
            model_config = role.get("model_config")
            if model_config is not None and not isinstance(model_config, dict):
                raise ValueError(f"lineage.roles.{name}.model_config must be a mapping or null")
            roles[name] = LineageRole(enabled=enabled, model_config=model_config)

        profile_influenced = lineage_raw.get("profile_influenced_surface", False)
        if profile_influenced is not None:
            profile_influenced = _require_bool(
                profile_influenced, "lineage.profile_influenced_surface"
            )
        judge_advisory = lineage_raw.get("judge_advisory")
        if judge_advisory is not None:
            judge_advisory = _require_bool(judge_advisory, "lineage.judge_advisory")

        lineage = LineageConfig(
            policy=policy,
            profile_influenced_surface=profile_influenced,
            judge_advisory=judge_advisory,
            roles=roles,
        )

        oracle_pack = OraclePackRef(
            manifest_path=_resolve_path(pack_raw["manifest_path"]),  # type: ignore[arg-type]
            backend_path=_resolve_path(pack_raw.get("backend_path")),
            endpoint_config_path=_resolve_path(pack_raw.get("endpoint_config_path")),
            fixtures_path=_resolve_path(pack_raw.get("fixtures_path")),
            task_templates_path=_resolve_path(pack_raw.get("task_templates_path")),
            assertions_path=_resolve_path(pack_raw.get("assertions_path")),
            validation_cases_path=_resolve_path(pack_raw.get("validation_cases_path")),
        )

        output_dir = _resolve_path(data["output_dir"])
        assert output_dir is not None
        run_output = (output_dir / data["expt_name"]).resolve()
        pack_root = oracle_pack.manifest_path.parent
        try:
            run_output.relative_to(pack_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "output_dir/expt_name must be outside the oracle pack root so generated "
                "artifacts cannot change the pack fingerprint"
            )

        strict_sections = (
            ("surface_generation", SURFACE_GENERATION_KEYS),
            ("surface_quality_validation", _SURFACE_QUALITY_KEYS),
            ("task_generation", _TASK_GENERATION_KEYS),
            ("semantic_deduplication_config", _DEDUP_KEYS),
            ("exports", _EXPORT_KEYS),
        )
        sections: dict[str, dict[str, Any]] = {}
        for section, allowed in strict_sections:
            raw_section = data.get(section)
            if raw_section is None:
                raw_section = {}
            if not isinstance(raw_section, dict):
                raise ValueError(f"{section} must be a mapping")
            _reject_unknown(raw_section, allowed, section)
            sections[section] = raw_section

        surface = sections["surface_generation"]
        for key in (
            "model_paraphrase_enabled",
            "preserve_slot_values",
            "prevent_tool_name_leakage",
        ):
            if key in surface:
                _require_bool(surface[key], f"surface_generation.{key}")
        if "paraphrases_per_template" in surface:
            _require_int(
                surface["paraphrases_per_template"],
                "surface_generation.paraphrases_per_template",
                minimum=0,
            )
        if "language" in surface and (
            not isinstance(surface["language"], str) or not surface["language"].strip()
        ):
            raise ValueError("surface_generation.language must be a non-empty string")

        quality = sections["surface_quality_validation"]
        for key in ("enabled", "drop_authority"):
            if key in quality:
                _require_bool(quality[key], f"surface_quality_validation.{key}")
        task_generation = sections["task_generation"]
        if "tasks_per_category" in task_generation:
            _require_int(
                task_generation["tasks_per_category"],
                "task_generation.tasks_per_category",
                minimum=1,
            )
        dedup = sections["semantic_deduplication_config"]
        if "enabled" in dedup:
            _require_bool(dedup["enabled"], "semantic_deduplication_config.enabled")
        if "model_identifier" in dedup and (
            not isinstance(dedup["model_identifier"], str)
            or not dedup["model_identifier"].strip()
        ):
            raise ValueError(
                "semantic_deduplication_config.model_identifier must be a non-empty string"
            )
        exports = sections["exports"]
        for key, value in exports.items():
            _require_bool(value, f"exports.{key}")
        if data.get("random_seed") is not None:
            _require_int(data["random_seed"], "random_seed")
        if "ndd_batch_size" in data:
            _require_int(data["ndd_batch_size"], "ndd_batch_size", minimum=1)

        return cls(
            family="bfcl",
            expt_name=str(data["expt_name"]),
            output_dir=output_dir,
            oracle_pack=oracle_pack,
            oracle_runtime=oracle_runtime,
            lineage=lineage,
            stage=str(data.get("stage", "all")),
            random_seed=data.get("random_seed"),
            ndd_batch_size=int(data.get("ndd_batch_size", 32)),
            input_dir=_resolve_path(data.get("input_dir")),
            schema_version=data.get("schema_version"),
            config_status=config_status,
            surface_generation=dict(sections["surface_generation"]),
            surface_quality_validation=dict(sections["surface_quality_validation"]),
            task_generation=dict(sections["task_generation"]),
            semantic_deduplication_config=dict(
                sections["semantic_deduplication_config"]
            ),
            exports=dict(sections["exports"]),
            reference_benchmark=data.get("reference_benchmark"),
            generation_model_config=data.get("generation_model_config"),
            judge_model_config=data.get("judge_model_config"),
            eval_config_path=data.get("eval_config_path"),
            translation_config_path=data.get("translation_config_path"),
            raw=data,
        )
