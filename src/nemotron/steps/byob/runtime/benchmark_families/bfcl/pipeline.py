"""BFCL benchmark-family orchestration.

The generic CLI dispatcher lives in `nemotron.steps.byob.scripts.runtime`.
This module owns the BFCL-specific stage order and cache paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import SURFACE_GENERATION_KEYS

if TYPE_CHECKING:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

logger = logging.getLogger(__name__)

# Reports written to the stage cache are editable, so a stored gold claim is never
# trusted across runs. Validation results are instead remembered for the lifetime of
# this process, keyed by the pack and config fingerprints they were computed from, so
# `stage=all` does not pay for the same episodes twice.
_VALIDATED_THIS_PROCESS: dict[tuple[str, str, str], dict] = {}
_FINAL_ARTIFACTS = (
    "run_manifest.json",
    "run_manifest.json.tmp",
    "benchmark.parquet",
    "benchmark.parquet.tmp",
    "benchmark_raw.parquet",
    "benchmark_raw.parquet.tmp",
)

def _unsupported_requests(config: BfclConfig) -> list[str]:
    """List config features that are requested but that generation cannot honor.

    Generation refuses these instead of ignoring them, so a run never reports
    lineage or quality guarantees that no stage actually applied.
    """
    roles = config.lineage.roles or {}
    requested: list[str] = []
    if (role := roles.get("profile")) and role.enabled:
        requested.append("lineage.roles.profile.enabled")
    if (role := roles.get("paraphrase")) and role.enabled:
        requested.append("lineage.roles.paraphrase.enabled")
    if config.surface_generation.get("model_paraphrase_enabled"):
        requested.append("surface_generation.model_paraphrase_enabled")
    if (role := roles.get("surface_judge")) and role.enabled:
        requested.append("lineage.roles.surface_judge.enabled")
    if config.surface_quality_validation.get("enabled"):
        requested.append("surface_quality_validation.enabled")
    if config.semantic_deduplication_config.get("enabled"):
        requested.append("semantic_deduplication_config.enabled")
    # A claim the manifest would publish but no stage substantiates: no profile shapes
    # the surface and no judge runs, so neither may be asserted.
    if config.lineage.profile_influenced_surface:
        requested.append("lineage.profile_influenced_surface")
    if config.lineage.judge_advisory:
        requested.append("lineage.judge_advisory")
    requested.extend(f"exports.{name}" for name, on in sorted(config.exports.items()) if on)
    requested.extend(
        f"task_generation.{name}"
        for name in sorted(config.task_generation)
        if name != "tasks_per_category"
    )
    # Asking for work a disabled stage would have done. Reading these is what keeps
    # "refuse, do not ignore" true for the whole config and not just the gate flags.
    if config.surface_generation.get("paraphrases_per_template"):
        requested.append("surface_generation.paraphrases_per_template")
    if config.surface_quality_validation.get("drop_authority"):
        requested.append("surface_quality_validation.drop_authority")
    if config.eval_config_path is not None:
        requested.append("eval_config_path")
    if config.translation_config_path is not None:
        requested.append("translation_config_path")
    # Shared BYOB fields that BFCL loads for schema compatibility but never applies.
    # Naming them keeps "refuse, do not ignore" true for leftover MCQ keys too.
    if config.input_dir is not None:
        requested.append("input_dir")
    if config.generation_model_config:
        requested.append("generation_model_config")
    if config.judge_model_config:
        requested.append("judge_model_config")
    if "ndd_batch_size" in (config.raw or {}):
        requested.append("ndd_batch_size")
    # A key no stage reads is almost always a typo for one that matters, so name it
    # rather than letting the run proceed with a setting that had no effect.
    requested.extend(
        f"surface_generation.{name}"
        for name in sorted(config.surface_generation)
        if name not in SURFACE_GENERATION_KEYS
    )
    return requested


def _validate_pack(config: BfclConfig) -> tuple[dict, Path]:
    """Validate the pack, reusing only results this process computed itself."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import pack_fingerprint
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        run_oracle_validation,
        validation_config_fingerprint,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.prepare import prepare_oracle_pack

    pack = prepare_oracle_pack(config)
    report_path = stage_cache_dir(config) / "oracle_validation_report.json"
    fingerprint_before = pack_fingerprint(pack.paths)
    key = (str(report_path), fingerprint_before, validation_config_fingerprint(config))
    remembered = _VALIDATED_THIS_PROCESS.get(key)
    if remembered is not None:
        return remembered, report_path
    report = run_oracle_validation(config, pack)
    fingerprint_after = pack_fingerprint(pack.paths)
    if fingerprint_after != fingerprint_before:
        raise RuntimeError(
            "oracle pack changed while it was being validated; pack code and concurrent "
            "writers must not modify fingerprinted inputs"
        )
    _VALIDATED_THIS_PROCESS[key] = report
    return report, report_path


def _invalidate_final_outputs(config: BfclConfig) -> None:
    """Remove a previous completed run before reusing its experiment directory.

    The manifest is removed first: if this generation later fails, no old manifest can
    make stale parquets look like the result of the failed invocation.
    """
    output_dir = Path(config.output_dir) / config.expt_name
    for name in _FINAL_ARTIFACTS:
        (output_dir / name).unlink(missing_ok=True)


def prepare_bfcl(config_path: str | os.PathLike[str]) -> Path:
    """Normalize and validate the configured oracle pack."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )

    config = BfclConfig.from_yaml(str(config_path))
    report, report_path = _validate_pack(config)
    gold_eligible, tier = derive_pack_tier(report)
    if not gold_eligible:
        logger.warning(
            "BFCL prepare completed but pack is not gold_eligible (tier=%s)",
            tier,
        )
    return report_path


def generate_bfcl(config_path: str | os.PathLike[str], *, skip_until: str | None = None) -> Path:
    """Run the supported BFCL generation phases."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
        load_pack,
        pack_fingerprint,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.executable_replay import (
        run_executable_replay,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import run_expand
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expected_trace import (
        run_expected_trace,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import run_final_output
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.reference_profile import (
        run_reference_profile,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.render import run_render
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.schema_validation import (
        run_schema_validation,
    )
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
        run_state_machine,
    )

    if skip_until is not None:
        raise NotImplementedError(
            f"BFCL does not expose stage resume (skip_until={skip_until!r}); "
            "run stage=prepare followed by stage=generate"
        )

    config = BfclConfig.from_yaml(str(config_path))
    unsupported = _unsupported_requests(config)
    if unsupported:
        raise NotImplementedError(
            "BFCL generate cannot honor these config settings: "
            + ", ".join(unsupported)
            + ". Disable them to run the template-only path."
        )
    _invalidate_final_outputs(config)
    report, _ = _validate_pack(config)

    # Derive the gate from the individual checks so a summary flag cannot stand alone.
    gold_eligible, tier = derive_pack_tier(report)
    report["gold_eligible"] = gold_eligible
    report["tier"] = tier
    if not gold_eligible:
        raise RuntimeError(
            f"BFCL generate refuses non-gold pack (tier={tier!r}). "
            "Re-run stage=prepare and fix oracle_validation failures."
        )
    run_reference_profile(config)

    pack = load_pack(config)
    templates_by_id = {str(template["template_id"]): template for template in pack.templates}

    tasks = run_expand(config, pack)
    expanded_tasks = list(tasks)
    expanded = len(expanded_tasks)
    plans = run_state_machine(config, templates_by_id, tasks)
    surfaces, prompt_bundle = run_render(config, pack, templates_by_id, tasks, plans)
    # Drops an instance whose own fixture data cannot bind a trace, so `tasks` shrinks
    # to the kept set. Schema/replay still receive every expanded task so stage tables
    # keep a joinable row for each drop.
    traces, drop_reasons = run_expected_trace(config, pack, tasks, plans)
    schema_failures = run_schema_validation(
        config, pack, expanded_tasks, traces, skipped=drop_reasons
    )
    verdicts = run_executable_replay(
        config, pack, expanded_tasks, traces, schema_failures, skipped=drop_reasons
    )
    current_fingerprint = pack_fingerprint(pack.paths)
    if current_fingerprint != report.get("pack_fingerprint"):
        raise RuntimeError(
            "oracle pack changed after validation; refusing to publish artifacts derived "
            "from content that the gold report did not certify"
        )

    # Each count reports its own stage, so a reader can tell a backend disagreement from
    # a paraphrase the guards rejected. ``published`` is where the stages meet.
    stage_counts = {
        "expanded": expanded,
        "surface_passed": sum(
            1 for surface in surfaces.values() if not surface["guard_violations"]
        ),
        "trace_derived": len(traces),
        "trace_dropped": len(drop_reasons),
        "schema_passed": sum(
            1
            for task_id, failures in schema_failures.items()
            if not failures and task_id not in drop_reasons
        ),
        "replay_passed": sum(
            1
            for task in expanded_tasks
            if (verdicts.get(str(task["task_id"])) or {}).get("passed")
        ),
    }
    return run_final_output(
        config,
        pack,
        tasks,
        surfaces,
        traces,
        verdicts,
        report,
        prompt_bundle,
        stage_counts,
        trace_drop_reasons=drop_reasons,
        expected_pack_fingerprint=current_fingerprint,
    )
