"""Assemble messages, write the benchmark parquets, and stamp the run manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import uuid
from collections import Counter, deque
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import (
    BYOB_ROOT,
    DEFAULT_BENCHMARK_SCHEMA_VERSION,
    BfclConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    pack_fingerprint,
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
    encode_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import STAGE_TABLES
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_parquet_atomic(table: Any, path: Path, parquet: Any) -> None:
    """Replace one parquet only after its temporary file is fully written."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        parquet.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_pack_fingerprint(
    pack: LoadedPack,
    expected: str,
    *,
    phase: str,
    cleanup: tuple[Path, ...] = (),
) -> str:
    """Abort publication if pack inputs drift, removing outputs from this attempt."""
    current = pack_fingerprint(pack.paths)
    if current == expected:
        return current
    for path in cleanup:
        path.unlink(missing_ok=True)
    raise RuntimeError(
        f"oracle pack changed {phase}; refusing to publish content that validation did not certify"
    )


def _require_endpoint_identity(
    config: BfclConfig,
    pack: LoadedPack,
    expected: dict[str, Any] | None,
    *,
    cleanup: tuple[Path, ...] = (),
) -> None:
    """Abort publication when a remote oracle no longer matches validation."""
    if pack.endpoint_config is None:
        return
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

    runtime = config.oracle_runtime
    outputs = ProcessWorker(
        default_timeout_s=runtime.episode_timeout_s,
        worker=runtime.worker,
    ).run_episode(
        endpoint_config=pack.endpoint_config,
        fixtures=None,
        clock_iso=runtime.clock,
        seed=int(config.random_seed or 0),
        task_id="final-output-endpoint-metadata",
        steps=[{"op": "metadata"}],
        import_timeout_s=runtime.import_timeout_s,
        tool_timeout_s=runtime.tool_timeout_s,
        episode_timeout_s=runtime.episode_timeout_s,
    )
    if outputs[0] == expected:
        return
    for path in cleanup:
        path.unlink(missing_ok=True)
    raise RuntimeError(
        "endpoint identity changed during final output; refusing to publish "
        "artifacts from an uncertified oracle revision"
    )


def _write_endpoint_manifest_atomic(
    text: str,
    path: Path,
    *,
    config: BfclConfig,
    pack: LoadedPack,
    expected_endpoint_metadata: dict[str, Any] | None,
    cleanup: tuple[Path, ...],
) -> None:
    """Recheck endpoint identity immediately before replacing the manifest."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        _require_endpoint_identity(
            config,
            pack,
            expected_endpoint_metadata,
            cleanup=cleanup,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(
    value: Any,
    *,
    roots: tuple[tuple[str, Path], ...] | None = None,
) -> Any:
    """Normalize config-only Python values before provenance hashing.

    Paths are recorded relative to the framework root when they live under it, so the
    same config hashes the same on two machines with different checkouts.
    """
    if isinstance(value, Path):
        return _portable_path(value, roots=roots)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(key): _jsonable(child, roots=roots)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(child, roots=roots) for child in value]
    if isinstance(value, str) and value.startswith("/"):
        return _portable_path(Path(value), roots=roots)
    return value


def _portable_path(
    path: Path,
    *,
    roots: tuple[tuple[str, Path], ...] | None = None,
) -> str:
    candidates = roots or (("<byob>", BYOB_ROOT),)
    for label, root in candidates:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return label if relative == Path(".") else f"{label}/{relative.as_posix()}"
    return str(path)


def _config_roots(config: BfclConfig) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = [
        ("<pack>", config.oracle_pack.manifest_path.parent),
        ("<output>", config.output_dir),
        ("<byob>", BYOB_ROOT),
    ]
    roots.extend(
        (f"<allowed_root_{index}>", root)
        for index, root in enumerate(config.oracle_runtime.allowed_roots)
        if root not in {config.oracle_pack.manifest_path.parent, config.output_dir, BYOB_ROOT}
    )
    return tuple(roots)


def _resolved_config(config: BfclConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("raw", None)
    return _jsonable(payload, roots=_config_roots(config))


def _generation_config(config: BfclConfig) -> dict[str, Any]:
    """Portable view of the YAML input, so two hosts with the same logical config match."""
    return _jsonable(config.raw or {}, roots=_config_roots(config))


def generation_mode(config: BfclConfig) -> str:
    """Derive the manifest generation_mode from lineage roles and surface flags."""
    if config.lineage.policy == "smoke_no_publication":
        return "smoke_no_publication"
    paraphrase = (config.lineage.roles or {}).get("paraphrase")
    if paraphrase and paraphrase.enabled and config.surface_generation.get("model_paraphrase_enabled"):
        return "template_plus_paraphrase"
    return "template_only"


def build_messages(
    surface: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    results: list[Any],
) -> list[dict[str, Any]]:
    """Assemble the OpenAI-style message array from surface text and replay results.

    One assistant message per call_group carries every call in that group, and each
    call is followed immediately by its own tool result message.
    """
    if len(results) != len(expected_calls):
        raise ValueError(
            f"replay returned {len(results)} results for {len(expected_calls)} expected calls"
        )
    messages: list[dict[str, Any]] = [{"role": "system", "content": surface["system_prompt"]}]
    # Walk the calls in trace order so one assistant message consumes exactly the
    # calls of its own step, even if two steps were to carry the same group value.
    remaining = deque(enumerate(expected_calls))

    call_counter = 0
    for step in surface["steps"]:
        if step["kind"] == "user":
            messages.append({"role": "user", "content": step["content"]})
        elif step["kind"] == "assistant_text":
            messages.append({"role": "assistant", "content": step["content"]})
        else:
            group_value = int(step["call_group"])
            group: list[tuple[int, dict[str, Any]]] = []
            while remaining and int(remaining[0][1]["call_group"]) == group_value:
                group.append(remaining.popleft())
            if not group:
                raise ValueError(f"surface call_group {group_value} has no expected calls")
            tool_calls = []
            tool_messages = []
            for index, call in sorted(group, key=lambda item: item[1]["position_in_group"]):
                call_id = f"call_{call_counter}"
                call_counter += 1
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["function_name"],
                            "arguments": canonical_json(call["arguments"]),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": canonical_json(results[index]),
                    }
                )
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            messages.extend(tool_messages)
    if remaining:
        groups = [call.get("call_group") for _, call in remaining]
        raise ValueError(f"expected calls were not represented by surface steps: call_groups={groups}")
    return messages


def build_row(
    config: BfclConfig,
    pack: LoadedPack,
    task: dict[str, Any],
    surface: dict[str, Any],
    expected_calls: list[dict[str, Any]],
    verdict: dict[str, Any],
    tier: str,
    model_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one benchmark row with its provenance stamped top-level."""
    return {
        "task_id": task["task_id"],
        "template_id": task["template_id"],
        "variant_index": int(task["variant_index"]),
        "messages": build_messages(surface, expected_calls, verdict["results"]),
        "tools": canonical_json(model_tools),
        "expected_tool_calls": [
            {**call, "arguments": encode_arguments(call["arguments"])} for call in expected_calls
        ],
        "success_assertions": list(task.get("success_assertions") or []),
        "fixture_refs": list(task.get("fixture_refs") or []),
        "intent": task.get("intent"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "required_tools": list(task.get("required_tools") or []),
        "required_tools_fingerprint": canonical_json(sorted(task.get("required_tools") or [])),
        "tools_present": list(task.get("tools_present") or []),
        "turn_policy": task.get("turn_policy"),
        "is_multi_turn": bool(task.get("is_multi_turn")),
        "num_tool_calls": int(task.get("num_tool_calls") or 0),
        "call_order": task.get("call_order", "strict"),
        "call_order_prefix": task.get("call_order_prefix"),
        "system_prompt_id": surface["system_prompt_id"],
        "tier": tier,
        "gold_eligible": tier == "gold" and config.lineage.policy != "smoke_no_publication",
        "validated_by": ["schema", "replay", "assertions"]
        if task.get("success_assertions")
        else ["schema", "replay"],
        "pack_id": task["pack_id"],
        "pack_version": task["pack_version"],
        "seed": int(task.get("seed") or 0),
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        # Null, not False: no held-out source is configured, so the row was never
        # checked against one and must not claim it was.
        "held_out_hit": None,
        "src": f"{task['pack_id']}:{task['template_id']}",
        "metadata": canonical_json({"language": surface["language"], "expt_name": config.expt_name}),
    }


def _model_role(name: str, config: BfclConfig) -> dict[str, Any]:
    role = (config.lineage.roles or {}).get(name)
    enabled = bool(role and role.enabled)
    if not enabled:
        return {"alias": None, "provider": None, "model_identity": None, "canonical_id": None, "enabled": False}
    model_config = dict(role.model_config or {}) if role else {}
    return {
        "alias": model_config.get("alias"),
        "provider": model_config.get("provider"),
        "model_identity": {
            "source": model_config.get("source"),
            "model": model_config.get("model"),
            "revision": model_config.get("revision"),
            "weights_digest": model_config.get("weights_digest"),
        },
        "canonical_id": model_config.get("canonical_id"),
        "enabled": True,
    }


def run_final_output(
    config: BfclConfig,
    pack: LoadedPack,
    tasks: list[dict[str, Any]],
    surfaces: dict[str, dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]],
    verdicts: dict[str, dict[str, Any]],
    validation_report: dict[str, Any],
    prompt_bundle: dict[str, Any],
    stage_counts: dict[str, int],
    expected_pack_fingerprint: str,
    trace_drop_reasons: dict[str, str] | None = None,
) -> Path:
    """Write benchmark_raw.parquet, benchmark.parquet, and run_manifest.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tier = str(validation_report.get("tier", "prototype"))
    model_tools = project_model_facing_tools(pack.tools)
    model_tools_by_name = {
        str((tool.get("function") or {}).get("name")): tool for tool in model_tools
    }
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        verdict = verdicts.get(task_id) or {}
        surface = surfaces[task_id]
        if not verdict.get("passed"):
            continue
        task_tools = [
            model_tools_by_name[name]
            for name in task.get("tools_present") or []
            if name in model_tools_by_name
        ]
        rows.append(build_row(config, pack, task, surface, traces[task_id], verdict, tier, task_tools))

    if not rows:
        raise RuntimeError(
            "BFCL final_output has no replay-validated rows; inspect "
            "stage_cache/replay_validated_tasks.parquet"
        )

    cache = stage_cache_dir(config)
    # Hash the intermediates first: a missing stage artifact must stop the run before
    # a published parquet exists without the manifest that explains it.
    stage_artifacts = {
        name.removesuffix(".parquet"): {"content_hash": _file_hash(cache / name)}
        for name in STAGE_TABLES
    }
    tool_artifacts = {
        "tools_internal": {"content_hash": _file_hash(cache / "tools_normalized_internal.json")},
        "tools_model_facing": {"content_hash": _file_hash(cache / "tools_normalized.json")},
    }
    current_pack_fingerprint = _require_pack_fingerprint(
        pack,
        expected_pack_fingerprint,
        phase="before final output was assembled",
    )

    output_dir = Path(config.output_dir) / config.expt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_endpoint_identity(
        config,
        pack,
        validation_report.get("endpoint_metadata"),
    )
    schema = benchmark_schema()
    raw_path = output_dir / "benchmark_raw.parquet"
    _write_parquet_atomic(pa.Table.from_pylist(rows, schema=schema), raw_path, pq)

    # Raw means executable/schema-valid. Publication additionally applies deterministic
    # surface guards; future held-out/judge/dedup stages can narrow it further.
    published = [
        row
        for row in rows
        if not surfaces[str(row["task_id"])]["guard_violations"]
    ]
    benchmark_path = output_dir / "benchmark.parquet"
    _write_parquet_atomic(pa.Table.from_pylist(published, schema=schema), benchmark_path, pq)

    surface_rejections: dict[str, int] = {}
    for surface in surfaces.values():
        if surface["guard_violations"]:
            template_id = str(surface["template_id"])
            surface_rejections[template_id] = surface_rejections.get(template_id, 0) + 1
    trace_drop_reasons = trace_drop_reasons or {}
    trace_drop_summary = dict(sorted(Counter(trace_drop_reasons.values()).items()))
    created_at = datetime.now(timezone.utc)
    schema_version = str(config.schema_version or DEFAULT_BENCHMARK_SCHEMA_VERSION)
    _require_pack_fingerprint(
        pack,
        current_pack_fingerprint,
        phase="while final output was being assembled",
        cleanup=(raw_path, benchmark_path),
    )
    # Hash prepare/validation intermediates the published rows depend on, not only
    # the six stage tables — otherwise a tampered validation report would not show.
    lineage_artifacts = {
        "oracle_validation_report": {
            "content_hash": _file_hash(cache / "oracle_validation_report.json")
        },
        "pack_manifest": {"content_hash": _file_hash(cache / "pack_manifest.json")},
        "fixtures_normalized": {"content_hash": _file_hash(cache / "fixtures_normalized.json")},
        "task_templates_normalized": {
            "content_hash": _file_hash(cache / "task_templates_normalized.yaml")
        },
    }
    manifest = {
        "schema_version": schema_version,
        # The timestamp is part of the id: two runs of the same config are different
        # runs, and a manifest that cannot be told from its predecessor is not lineage.
        "run_id": (
            f"{config.expt_name}-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{prompt_bundle['system_prompt_id'].split(':')[-1][:12]}-{uuid.uuid4().hex}"
        ),
        "created_at": created_at.isoformat(),
        "oracle_clock": config.oracle_runtime.clock,
        "generation_mode": generation_mode(config),
        "pipeline_version": "bfcl-runtime-0.1.0",
        "generation_config_hash": _sha256(canonical_json(_generation_config(config))),
        "resolved_config_hash": _sha256(canonical_json(_resolved_config(config))),
        "runtime": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "pipeline_git_sha": None,
            "dependency_lock_hash": None,
            "worker_image_digest": None,
        },
        "pack": {
            "pack_id": pack.manifest.get("pack_id"),
            "version": pack.manifest.get("version"),
            "content_hash": f"sha256:{current_pack_fingerprint}",
        },
        "oracle": {
            "kind": "endpoint" if pack.endpoint_config is not None else "python",
            "endpoint_metadata": validation_report.get("endpoint_metadata"),
        },
        "reference_benchmark": config.reference_benchmark,
        "prompt_bundle_hash": prompt_bundle["prompt_bundle_hash"],
        "tier": tier,
        "gold_eligible": tier == "gold" and config.lineage.policy != "smoke_no_publication",
        "lineage_policy": config.lineage.policy,
        "profile_influenced_surface": bool(config.lineage.profile_influenced_surface),
        "models": {name: _model_role(name, config) for name in ("profile", "paraphrase", "surface_judge")},
        "judge_advisory": config.lineage.judge_advisory,
        "seeds": {
            "global": int(config.random_seed or 0),
            "derivation": (
                "uint64_be(sha256(canonical_json(global_seed,pack_id,pack_version,"
                "template_id,sorted_fixture_refs,slot_bindings,variant_index))[0:8])"
            ),
        },
        "held_out": {"source": None, "evaluated": False, "rows_dropped": 0},
        "stage_counts": {**stage_counts, "published": len(published)},
        # Guard rejections are counted per template, because a template is what an
        # author fixes; paraphrase does not run, so these are surface guards only.
        "surface_guard_rejections": {"by_template": surface_rejections},
        "trace_drop_rejections": {
            "count": len(trace_drop_reasons),
            "by_reason": trace_drop_summary,
        },
        "artifacts": {
            **tool_artifacts,
            **lineage_artifacts,
            # One entry per stage artifact, so a published row can be traced back to
            # the exact intermediate that produced it.
            **stage_artifacts,
            "benchmark_raw_parquet": {"content_hash": _file_hash(raw_path)},
            "benchmark_parquet": {"content_hash": _file_hash(benchmark_path)},
        },
    }
    _require_pack_fingerprint(
        pack,
        current_pack_fingerprint,
        phase="before the final manifest was stamped",
        cleanup=(raw_path, benchmark_path),
    )
    manifest_path = output_dir / "run_manifest.json"
    _write_endpoint_manifest_atomic(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        manifest_path,
        config=config,
        pack=pack,
        expected_endpoint_metadata=validation_report.get("endpoint_metadata"),
        cleanup=(raw_path, benchmark_path),
    )
    logger.info(
        "BFCL final_output wrote %d rows (%d published) to %s",
        len(rows),
        len(published),
        benchmark_path,
    )
    return benchmark_path
