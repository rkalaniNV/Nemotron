"""Resolve and load an allowlisted oracle pack."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig, OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    EndpointConfig,
    load_endpoint_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import assert_pack_allowed


@dataclass(frozen=True)
class ResolvedPackPaths:
    pack_root: Path
    manifest_path: Path
    tools_path: Path
    fixtures_path: Path | None
    templates_path: Path
    assertions_path: Path
    validation_cases_path: Path
    system_prompt_path: Path | None
    backend_path: Path | None
    endpoint_config_path: Path | None
    endpoint_ca_bundle_path: Path | None = None
    held_out_path: Path | None = None


@dataclass
class LoadedPack:
    paths: ResolvedPackPaths
    manifest: dict[str, Any]
    tools: list[dict[str, Any]]
    fixtures: dict[str, Any] | None
    templates: list[dict[str, Any]]
    validation_cases: list[dict[str, Any]]
    endpoint_config: EndpointConfig | None = None
    held_out: dict[str, Any] | None = None


def _as_path(pack_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = pack_root / path
    return path.resolve()


def resolve_pack_paths(config: BfclConfig) -> ResolvedPackPaths:
    """Resolve pack file paths from manifest + config overrides."""
    return resolve_declared_pack_paths(config.oracle_pack, config.oracle_runtime.allowed_roots)


def resolve_declared_pack_paths(
    ref: OraclePackRef,
    allowed_roots: Sequence[Path],
) -> ResolvedPackPaths:
    """Resolve one pack's files from its manifest plus the declared overrides.

    Split out from :func:`resolve_pack_paths` so that a later consumer — an eval
    run verifying the pack a published benchmark was generated from — resolves
    the pack through this exact logic instead of reimplementing it. There is one
    definition of which files a pack consists of, and therefore one definition of
    what its fingerprint covers.
    """
    allowed_roots = tuple(allowed_roots)
    manifest_path = assert_pack_allowed(ref.manifest_path, allowed_roots)
    pack_root = manifest_path.parent
    # Ensure the whole pack tree is under allowlist (not just the manifest file).
    assert_pack_allowed(pack_root, allowed_roots)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a mapping: {manifest_path}")

    paths_block = manifest.get("paths") or {}
    if ref.backend_path is not None and ref.endpoint_config_path is not None:
        raise ValueError("oracle_pack cannot declare both backend_path and endpoint_config_path")
    if paths_block.get("backend") and paths_block.get("endpoint"):
        raise ValueError("manifest paths cannot declare both backend and endpoint")

    def pick(override: Path | None, key: str, default_name: str) -> Path:
        if override is not None:
            path = override
        elif paths_block.get(key):
            path = _as_path(pack_root, paths_block[key])  # type: ignore[arg-type]
        else:
            path = pack_root / default_name
        assert path is not None
        return assert_pack_allowed(path, allowed_roots)

    tools_path = pick(None, "tools", "tools.json")
    fixtures_path = None
    fixtures_candidate = pack_root / "fixtures.json"
    if ref.fixtures_path is not None:
        fixtures_path = assert_pack_allowed(ref.fixtures_path, allowed_roots)
    elif paths_block.get("fixtures"):
        fixtures_path = assert_pack_allowed(
            _as_path(pack_root, paths_block["fixtures"]),  # type: ignore[arg-type]
            allowed_roots,
        )
    elif fixtures_candidate.exists():
        fixtures_path = assert_pack_allowed(fixtures_candidate, allowed_roots)

    templates_path = pick(ref.task_templates_path, "templates", "task_templates.yaml")
    assertions_path = pick(ref.assertions_path, "assertions", "assertions.py")
    validation_cases_path = pick(ref.validation_cases_path, "validation_cases", "validation_cases.yaml")
    system_prompt_path = None
    if manifest.get("system_prompt_path"):
        system_prompt_path = assert_pack_allowed(
            _as_path(pack_root, manifest["system_prompt_path"]),  # type: ignore[arg-type]
            allowed_roots,
        )
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"missing system prompt file: {system_prompt_path}")
    held_out_path = None
    if manifest.get("held_out") is not None:
        if not isinstance(manifest["held_out"], str) or not manifest["held_out"].strip():
            raise ValueError("manifest held_out must be a non-empty path string")
        held_out_path = assert_pack_allowed(
            _as_path(pack_root, manifest["held_out"]),  # type: ignore[arg-type]
            allowed_roots,
        )
        if not held_out_path.is_file():
            raise FileNotFoundError(f"missing held-out policy file: {held_out_path}")

    backend_path = None
    endpoint_config_path = None
    if ref.backend_path is not None:
        backend_path = assert_pack_allowed(ref.backend_path, allowed_roots)
    elif ref.endpoint_config_path is not None:
        endpoint_config_path = assert_pack_allowed(ref.endpoint_config_path, allowed_roots)
    elif paths_block.get("backend"):
        backend_path = assert_pack_allowed(
            _as_path(pack_root, paths_block["backend"]),  # type: ignore[arg-type]
            allowed_roots,
        )
    elif paths_block.get("endpoint"):
        endpoint_config_path = assert_pack_allowed(
            _as_path(pack_root, paths_block["endpoint"]),  # type: ignore[arg-type]
            allowed_roots,
        )
    elif (pack_root / "backend.py").exists():
        backend_path = assert_pack_allowed(pack_root / "backend.py", allowed_roots)
    elif (pack_root / "endpoint_config.yaml").exists():
        endpoint_config_path = assert_pack_allowed(
            pack_root / "endpoint_config.yaml", allowed_roots
        )

    if (backend_path is None) == (endpoint_config_path is None):
        raise ValueError("oracle pack must declare exactly one of backend.py or endpoint_config.yaml")
    endpoint_ca_bundle_path = None
    if endpoint_config_path is not None:
        if not endpoint_config_path.is_file():
            raise FileNotFoundError(f"missing endpoint config: {endpoint_config_path}")
        endpoint_ca_bundle_path = load_endpoint_config(
            endpoint_config_path,
            allowed_roots=allowed_roots,
        ).ca_bundle_path
    elif backend_path is not None and not backend_path.is_file():
        raise FileNotFoundError(f"missing backend module: {backend_path}")

    for required in (tools_path, templates_path, assertions_path, validation_cases_path):
        if not required.exists():
            raise FileNotFoundError(f"missing required pack file: {required}")

    return ResolvedPackPaths(
        pack_root=pack_root,
        manifest_path=manifest_path,
        tools_path=tools_path,
        fixtures_path=fixtures_path,
        templates_path=templates_path,
        assertions_path=assertions_path,
        validation_cases_path=validation_cases_path,
        system_prompt_path=system_prompt_path,
        backend_path=backend_path,
        endpoint_config_path=endpoint_config_path,
        endpoint_ca_bundle_path=endpoint_ca_bundle_path,
        held_out_path=held_out_path,
    )


IGNORED_PACK_DIRS = frozenset({"__pycache__", ".git", ".ipynb_checkpoints"})

# What a two-step confirmation looks like on the wire. The names are the pack's to
# choose — a Vietnamese backend may prefer `xac_nhan` — while the shape is the
# pipeline's: one boolean argument, and one status a tool returns while it waits.
DEFAULT_CONFIRMATION_PROTOCOL = {
    "parameter": "confirm",
    "status_field": "status",
    "pending_status": "awaiting_confirmation",
}


def confirmation_protocol(manifest: dict[str, Any]) -> dict[str, str]:
    """Resolve the pack's confirmation vocabulary over the framework defaults."""
    declared = manifest.get("confirmation")
    if declared is None:
        return dict(DEFAULT_CONFIRMATION_PROTOCOL)
    if not isinstance(declared, dict):
        raise ValueError("manifest confirmation must be a mapping")
    unknown = set(declared) - set(DEFAULT_CONFIRMATION_PROTOCOL)
    if unknown:
        raise ValueError("manifest confirmation has unknown keys: " + ", ".join(sorted(unknown)))
    resolved = dict(DEFAULT_CONFIRMATION_PROTOCOL)
    for key, value in declared.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest confirmation.{key} must be a non-empty string")
        resolved[key] = value
    return resolved


def pack_files(paths: ResolvedPackPaths) -> list[Path]:
    """Return every file the pack could read, sorted, plus declared outside files.

    The whole tree counts, not just the declared entry points: a backend that imports
    a helper module or reads a policy file changes what the oracle does, and a
    fingerprint blind to those files would let a stale gold report be reused.
    """
    collected: dict[Path, None] = {}
    for path in sorted(paths.pack_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_PACK_DIRS for part in path.relative_to(paths.pack_root).parts):
            continue
        collected[path] = None
    for declared in (
        paths.manifest_path,
        paths.tools_path,
        paths.fixtures_path,
        paths.templates_path,
        paths.assertions_path,
        paths.validation_cases_path,
        paths.system_prompt_path,
        paths.backend_path,
        paths.endpoint_config_path,
        paths.endpoint_ca_bundle_path,
        paths.held_out_path,
    ):
        if declared is not None and declared.is_file():
            collected[declared] = None
    return sorted(collected)


def pack_fingerprint(paths: ResolvedPackPaths) -> str:
    """Hash every pack input so cached stage reports can be detected as stale."""
    entries: dict[str, Path] = {}
    for path in pack_files(paths):
        try:
            logical_name = f"tree/{path.relative_to(paths.pack_root).as_posix()}"
        except ValueError:
            # Declared files outside the pack tree receive a semantic name below;
            # an absolute path would make identical packs hash differently by host.
            continue
        entries[logical_name] = path
    declared = {
        "manifest": paths.manifest_path,
        "tools": paths.tools_path,
        "fixtures": paths.fixtures_path,
        "templates": paths.templates_path,
        "assertions": paths.assertions_path,
        "validation_cases": paths.validation_cases_path,
        "system_prompt": paths.system_prompt_path,
        "backend": paths.backend_path,
        "endpoint": paths.endpoint_config_path,
        "endpoint_ca_bundle": paths.endpoint_ca_bundle_path,
        "held_out": paths.held_out_path,
    }
    for role, path in declared.items():
        if path is None or not path.is_file():
            continue
        try:
            path.relative_to(paths.pack_root)
        except ValueError:
            entries[f"declared/{role}"] = path

    digest = hashlib.sha256()
    for logical_name, path in sorted(entries.items()):
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def project_model_facing_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip pack-local x-* keys; keep OpenAI-style provider surface."""
    projected: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        entry: dict[str, Any] = {
            "type": tool.get("type", "function"),
            "function": {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            },
        }
        if "strict" in function:
            entry["function"]["strict"] = function["strict"]
        projected.append(entry)
    return projected


TURN_POLICIES = frozenset(
    {
        "single_turn",
        "missing_slot",
        "confirmation",
        "correction",
        "multi_tool",
        "dependent_call",
        "negative_path",
        "clarify_only",
        "irrelevant",
    }
)


def normalize_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize aliases and required defaults on templates."""
    normalized: list[dict[str, Any]] = []
    for template in templates:
        item = dict(template)
        template_id = item.get("template_id")
        if "user_turn_templates" not in item and "first_turn_templates" in item:
            item["user_turn_templates"] = item.pop("first_turn_templates")
        # Paraphrase is optional: the block only carries surface guards today, and a
        # template that declares none is guarded by the run-wide defaults.
        item.setdefault("paraphrase", {})
        if not isinstance(item["paraphrase"], dict):
            raise ValueError(f"template {template_id!r} paraphrase must be a mapping")
        paraphrase = item["paraphrase"]
        if "allowed" in paraphrase and not isinstance(paraphrase["allowed"], bool):
            raise ValueError(
                f"template {template_id!r} paraphrase.allowed must be a boolean"
            )
        if "max_variants" in paraphrase and (
            not isinstance(paraphrase["max_variants"], int)
            or isinstance(paraphrase["max_variants"], bool)
            or paraphrase["max_variants"] < 0
        ):
            raise ValueError(
                f"template {template_id!r} paraphrase.max_variants must be a non-negative integer"
            )
        item.setdefault("call_order", "strict")
        item.setdefault("success_assertions", [])
        if not isinstance(item["success_assertions"], list):
            raise ValueError(f"template {template_id!r} success_assertions must be a list")
        # A policy name drives which gates run, so a typo would quietly remove one.
        policy = item.get("turn_policy")
        if not isinstance(policy, str) or policy not in TURN_POLICIES:
            raise ValueError(
                f"template {template_id!r} declares unknown turn_policy {policy!r}; "
                f"expected one of {', '.join(sorted(TURN_POLICIES))}"
            )
        call_order = item.get("call_order", "strict")
        if call_order not in {"strict", "any", "prefix"}:
            raise ValueError(
                f"template {template_id!r} declares unknown call_order {call_order!r}; "
                "expected one of any, prefix, strict"
            )
        # A dependent call's producer must return before the consumer reads it, so the
        # comparison contract is always order-strict.
        if policy == "dependent_call" and call_order != "strict":
            raise ValueError(
                f"template {template_id!r} is dependent_call but declares call_order "
                f"{call_order!r}; dependent_call requires call_order: strict"
            )
        prefix = item.get("call_order_prefix")
        if call_order == "prefix":
            # The prefix counts required tools, so a value past their number would
            # describe an order the template cannot have and read as strict instead.
            required_count = len(item.get("required_tools") or [])
            if not isinstance(prefix, int) or not 1 <= prefix <= required_count:
                raise ValueError(
                    f"template {template_id!r} with call_order: prefix must set "
                    f"call_order_prefix to an integer between 1 and its "
                    f"{required_count} required_tools"
                )
        elif prefix is not None:
            raise ValueError(f"template {template_id!r} sets call_order_prefix without call_order: prefix")
        milestone_ids: set[str] = set()
        for index, milestone in enumerate(item.get("assistant_milestones") or []):
            if not isinstance(milestone, dict):
                raise ValueError(f"template {template_id!r} assistant_milestones[{index}] must be a mapping")
            identifier = milestone.get("id")
            if identifier is None:
                continue
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(
                    f"template {template_id!r} assistant_milestones[{index}].id must be a non-empty string"
                )
            if identifier in milestone_ids:
                raise ValueError(f"template {template_id!r} contains duplicate milestone id {identifier!r}")
            milestone_ids.add(identifier)
        # Guarding a slot needs to know whether the user stated it, and a missing flag
        # would put the slot in neither the preserve nor the omit set.
        for name, slot in (item.get("slots") or {}).items():
            if not isinstance(slot, dict):
                raise ValueError(f"template {template_id!r} slot {name!r} must be a mapping")
            if not isinstance(slot.get("visible_in_first_turn"), bool):
                raise ValueError(
                    f"template {template_id!r} slot {name!r} must declare visible_in_first_turn as a boolean"
                )
        edge_signatures = item.get("edge_signatures") or []
        if (
            not isinstance(edge_signatures, list)
            or any(
                not isinstance(signature, str) or not signature.strip()
                for signature in edge_signatures
            )
        ):
            raise ValueError(
                f"template {template_id!r} edge_signatures must be a list of non-empty strings"
            )
        normalized_edges = [signature.strip() for signature in edge_signatures]
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError(
                f"template {template_id!r} edge_signatures must be unique"
            )
        item["edge_signatures"] = sorted(normalized_edges)
        normalized.append(item)
    return normalized


def _require_unique_string_ids(items: list[Any], field: str, source: str) -> None:
    """Reject missing or duplicate identifiers before dict-indexed stages lose rows."""
    seen: set[str] = set()
    for index, item in enumerate(items):
        value = item.get(field) if isinstance(item, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}[{index}] must declare a non-empty {field!r}")
        if value in seen:
            raise ValueError(f"{source} contains duplicate {field} {value!r}")
        seen.add(value)


def _fixture_primary_key(manifest: dict[str, Any], collection: str, rows: list[dict[str, Any]]) -> str:
    declared = (manifest.get("primary_keys") or {}).get(collection)
    if isinstance(declared, str) and declared:
        return declared
    fields = set().union(*(row.keys() for row in rows)) if rows else set()
    singular = collection[:-1] if collection.endswith("s") else collection
    for candidate in (f"{singular}_id", "id"):
        if candidate in fields:
            return candidate
    candidates = sorted(field for field in fields if field.endswith("_id"))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"held_out.fixtures.{collection} cannot resolve a primary key; declare manifest primary_keys.{collection}"
    )


def load_held_out_policy(
    path: Path | None,
    *,
    source: str | None,
    manifest: dict[str, Any],
    fixtures: dict[str, Any] | None,
    templates: list[dict[str, Any]],
    text: str | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8") if text is None else text) or {}
    if not isinstance(raw, dict):
        raise ValueError("held_out.yaml must be a mapping")
    unknown = sorted(set(raw) - {"version", "fixtures", "templates", "policy"})
    if unknown:
        raise ValueError("held_out.yaml has unknown keys: " + ", ".join(unknown))
    if (
        isinstance(raw.get("version"), bool)
        or not isinstance(raw.get("version"), (str, int, float))
        or not str(raw["version"]).strip()
    ):
        raise ValueError("held_out.yaml version must be a non-empty string or non-boolean number")

    held_fixtures = raw.get("fixtures") or {}
    if not isinstance(held_fixtures, dict):
        raise ValueError("held_out.fixtures must be a mapping")
    normalized_fixtures: dict[str, list[str]] = {}
    absent_ids = manifest.get("absent_ids") or {}
    for collection, identifiers in held_fixtures.items():
        rows = (fixtures or {}).get(collection)
        if not isinstance(rows, list):
            raise ValueError(f"held_out.fixtures names unknown fixture collection {collection!r}")
        if not isinstance(identifiers, list) or any(
            not isinstance(identifier, (str, int, float)) or isinstance(identifier, bool)
            for identifier in identifiers
        ):
            raise ValueError(f"held_out.fixtures.{collection} must be a list of scalar primary ids")
        if not identifiers:
            normalized_fixtures[str(collection)] = []
            continue
        primary_key = _fixture_primary_key(manifest, str(collection), rows)
        available_values = [str(row.get(primary_key)) for row in rows if primary_key in row]
        if len(available_values) != len(set(available_values)):
            raise ValueError(
                f"held_out.fixtures.{collection} cannot identify rows unambiguously: "
                f"{primary_key!r} values must be unique after scalar normalization"
            )
        available = set(available_values)
        normalized = [str(identifier) for identifier in identifiers]
        missing = sorted(set(normalized) - available)
        if missing:
            raise ValueError(f"held_out.fixtures.{collection} contains unknown primary ids: " + ", ".join(missing))
        absent = absent_ids.get(collection)
        absent_values = {str(absent)} if isinstance(absent, str) else {str(item) for item in (absent or [])}
        overlap = sorted(set(normalized) & absent_values)
        if overlap:
            raise ValueError(f"held_out.fixtures.{collection} overlaps manifest absent_ids: " + ", ".join(overlap))
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"held_out.fixtures.{collection} contains duplicate primary ids")
        normalized_fixtures[str(collection)] = sorted(normalized)

    template_ids = {str(template["template_id"]) for template in templates}
    held_templates = raw.get("templates") or []
    if not isinstance(held_templates, list) or any(
        not isinstance(identifier, str) or not identifier.strip() for identifier in held_templates
    ):
        raise ValueError("held_out.templates must be a list of non-empty strings")
    unknown_templates = sorted(set(held_templates) - template_ids)
    if unknown_templates:
        raise ValueError("held_out.templates contains unknown template ids: " + ", ".join(unknown_templates))
    if len(held_templates) != len(set(held_templates)):
        raise ValueError("held_out.templates contains duplicate template ids")

    policy = raw.get("policy") or {}
    if not isinstance(policy, dict):
        raise ValueError("held_out.policy must be a mapping")
    unknown_policy = sorted(set(policy) - {"fixtures_in_backend_state", "seed"})
    if unknown_policy:
        raise ValueError("held_out.policy has unknown keys: " + ", ".join(unknown_policy))
    fixtures_in_state = policy.get("fixtures_in_backend_state", True)
    if not isinstance(fixtures_in_state, bool):
        raise ValueError("held_out.policy.fixtures_in_backend_state must be a boolean")
    seed = policy.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("held_out.policy.seed must be an integer")
    return {
        "version": str(raw["version"]),
        "fixtures": normalized_fixtures,
        "templates": sorted(held_templates),
        "policy": {
            "fixtures_in_backend_state": fixtures_in_state,
            "seed": seed,
        },
        "source": source,
    }


def oracle_runtime_fixtures(
    *,
    manifest: dict[str, Any],
    fixtures: dict[str, Any] | None,
    held_out: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Project pack fixtures into the state an oracle is allowed to observe.

    Generation still needs the complete inventory to bind, validate, and reject
    reserved rows. When a policy keeps those rows out of backend state, this
    projection is supplied to reset and mirrored over the local pack's fixture file
    before Python code crosses the worker boundary.
    """
    if fixtures is None or held_out is None:
        return fixtures
    settings = held_out.get("policy") or {}
    if settings.get("fixtures_in_backend_state", True):
        return fixtures

    projected = dict(fixtures)
    for collection, identifiers in (held_out.get("fixtures") or {}).items():
        rows = fixtures.get(collection)
        if not isinstance(rows, list):
            raise ValueError(
                f"held_out.fixtures names unknown fixture collection {collection!r}"
            )
        primary_key = _fixture_primary_key(manifest, str(collection), rows)
        reserved = {str(identifier) for identifier in identifiers}
        # A row the policy cannot identify stays: withholding it would remove state
        # the pack never reserved.
        projected[collection] = [
            row
            for row in rows
            if not isinstance(row, dict)
            or primary_key not in row
            or str(row[primary_key]) not in reserved
        ]
    return projected


def oracle_reset_fixtures(pack: LoadedPack) -> dict[str, Any] | None:
    """Return the seed state one oracle episode may reset from.

    Each episode gets its own copy, so a mutating backend cannot write back into
    the loaded pack, and rows a policy keeps out of backend state never cross the
    worker or endpoint boundary.
    """
    return copy.deepcopy(
        oracle_runtime_fixtures(
            manifest=pack.manifest,
            fixtures=pack.fixtures,
            held_out=pack.held_out,
        )
    )


def oracle_fixture_source_path(pack: LoadedPack) -> Path | None:
    """Return the fixture file that must be mirrored outside an oracle process."""
    paths = pack.paths
    backend_path = getattr(paths, "backend_path", None)
    fixtures_path = getattr(paths, "fixtures_path", None)
    held_out = getattr(pack, "held_out", None)
    if backend_path is None or fixtures_path is None or held_out is None:
        return None
    settings = held_out.get("policy") or {}
    return Path(fixtures_path) if settings.get("fixtures_in_backend_state", True) is False else None


def _validate_generation_targets(
    config: BfclConfig,
    templates: list[dict[str, Any]],
) -> None:
    """Reject positive mix targets that the template inventory cannot supply."""
    targets = config.task_generation
    difficulty_inventory = {
        str(template.get("difficulty"))
        for template in templates
        if template.get("difficulty") is not None
    }
    difficulty_mix = targets.get("difficulty_mix") or {}
    unavailable = sorted(
        name
        for name, weight in difficulty_mix.items()
        if float(weight) > 0 and name not in difficulty_inventory
    )
    if unavailable:
        raise ValueError(
            "task_generation.difficulty_mix targets unavailable template difficulties: "
            + ", ".join(unavailable)
        )

    user_turn_counts = [
        1 + len(template.get("user_simulator_turns") or [])
        for template in templates
    ]
    turn_inventory = set()
    if any(count == 1 for count in user_turn_counts):
        turn_inventory.add("single_turn")
    if any(count > 1 for count in user_turn_counts):
        turn_inventory.add("multi_turn")
    turn_mix = targets.get("turn_mix") or {}
    unavailable = sorted(
        name
        for name, weight in turn_mix.items()
        if float(weight) > 0 and name not in turn_inventory
    )
    if unavailable:
        raise ValueError(
            "task_generation.turn_mix targets unavailable conversation shapes: "
            + ", ".join(unavailable)
        )

    call_counts = [
        sum(
            1
            for milestone in template.get("assistant_milestones") or []
            if milestone.get("type") == "tool_call"
        )
        for template in templates
    ]
    bucket_inventory = {
        "1" if count == 1 else "2" if count == 2 else "3+"
        for count in call_counts
        if count > 0
    }
    call_mix = targets.get("tool_call_count_mix") or {}
    unavailable = sorted(
        name
        for name, weight in call_mix.items()
        if float(weight) > 0 and name not in bucket_inventory
    )
    if unavailable:
        raise ValueError(
            "task_generation.tool_call_count_mix targets unavailable call-count buckets: "
            + ", ".join(unavailable)
        )


def load_pack(config: BfclConfig) -> LoadedPack:
    paths = resolve_pack_paths(config)
    manifest = yaml.safe_load(paths.manifest_path.read_text(encoding="utf-8")) or {}
    for field in ("pack_id", "version"):
        if not isinstance(manifest.get(field), (str, int, float)) or not str(manifest[field]).strip():
            raise ValueError(f"manifest must declare a non-empty {field!r}")
    tools = json.loads(paths.tools_path.read_text(encoding="utf-8"))
    if not isinstance(tools, list):
        raise ValueError("tools.json must be a JSON array")
    tool_functions = [tool.get("function") if isinstance(tool, dict) else None for tool in tools]
    _require_unique_string_ids(tool_functions, "name", "tools.json functions")

    fixtures = None
    if paths.fixtures_path is not None:
        fixtures = json.loads(paths.fixtures_path.read_text(encoding="utf-8"))
        if not isinstance(fixtures, dict):
            raise ValueError("fixtures.json must be a JSON object")

    templates_raw = yaml.safe_load(paths.templates_path.read_text(encoding="utf-8")) or []
    if not isinstance(templates_raw, list):
        raise ValueError("task_templates.yaml must be a list")
    templates = normalize_templates(templates_raw)
    _require_unique_string_ids(templates, "template_id", "task_templates.yaml")
    _validate_generation_targets(config, templates)

    cases_raw = yaml.safe_load(paths.validation_cases_path.read_text(encoding="utf-8")) or []
    if not isinstance(cases_raw, list):
        raise ValueError("validation_cases.yaml must be a list")
    _require_unique_string_ids(cases_raw, "id", "validation_cases.yaml")
    endpoint_config = (
        load_endpoint_config(
            paths.endpoint_config_path,
            allowed_roots=config.oracle_runtime.allowed_roots,
        )
        if paths.endpoint_config_path is not None
        else None
    )
    held_out = load_held_out_policy(
        paths.held_out_path,
        source=(
            str(manifest.get("held_out")).replace("\\", "/")
            if manifest.get("held_out") is not None
            else None
        ),
        manifest=manifest,
        fixtures=fixtures,
        templates=templates,
    )

    return LoadedPack(
        paths=paths,
        manifest=manifest,
        tools=tools,
        fixtures=fixtures,
        templates=templates,
        validation_cases=cases_raw,
        endpoint_config=endpoint_config,
        held_out=held_out,
    )
