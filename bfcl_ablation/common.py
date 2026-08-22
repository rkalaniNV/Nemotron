"""Shared plumbing for the BFCL Oracle-Pack ablation ladder.

Every ablation arm runs the *unmodified* production pipeline. An arm is defined by
the pack it feeds in, never by a patch to the generator: if an arm had to change
`benchmark_families/bfcl`, its result would measure the patch rather than the pack.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ABLATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ABLATION_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
BYOB_ROOT = SRC_ROOT / "nemotron" / "steps" / "byob"

PACK_A0 = BYOB_ROOT / "data" / "banking_vn_oracle_pack"
BASE_CONFIG = BYOB_ROOT / "bfcl" / "config" / "banking_vn.yaml"

GENERATED = ABLATION_ROOT / "_generated"
RESULTS = ABLATION_ROOT / "results"

# The files a pack author writes by hand. `run config` is counted separately because
# it lives outside the pack directory.
AUTHORED_PACK_FILES = (
    "backend.py",
    "task_templates.yaml",
    "validation_cases.yaml",
    "assertions.py",
    "tools.json",
    "fixtures.json",
    "manifest.yaml",
)

# stage_cache tables the measurement stage reads, in pipeline order. The funnel is
# derived from these, so the order is load-bearing.
STAGE_TABLES = (
    "task_instances",
    "conversation_plans",
    "rendered_conversations",
    "expected_traces",
    "schema_validated_traces",
    "replay_validated_tasks",
)


def bootstrap() -> None:
    """Put the repo's `src` on the import path without requiring an install."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class ArmResult:
    """Everything one ablation arm produced, located rather than loaded."""

    arm: str
    pack_dir: Path
    config_path: Path
    run_dir: Path

    @property
    def stage_cache(self) -> Path:
        return self.run_dir / "stage_cache"

    @property
    def benchmark(self) -> Path:
        return self.run_dir / "benchmark.parquet"

    @property
    def benchmark_raw(self) -> Path:
        return self.run_dir / "benchmark_raw.parquet"

    @property
    def run_manifest(self) -> Path:
        return self.run_dir / "run_manifest.json"


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_stage_tables(result: ArmResult) -> dict[str, list[dict[str, Any]]]:
    tables = {name: read_parquet(result.stage_cache / f"{name}.parquet") for name in STAGE_TABLES}
    tables["benchmark_raw"] = read_parquet(result.benchmark_raw)
    tables["benchmark"] = read_parquet(result.benchmark)
    return tables


def write_config(
    *,
    arm: str,
    manifest_path: Path,
    output_dir: Path,
    extra_allowed_roots: tuple[Path, ...] = (),
    overrides: dict[str, Any] | None = None,
    minimal: bool = False,
) -> Path:
    """Derive a run config from the reference banking_vn config.

    Deriving rather than hand-writing keeps every arm on the same generation settings,
    so a distribution shift between A0 and A1 can only come from the pack.
    """
    config: dict[str, Any] = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))

    config["expt_name"] = f"bfcl_ablation_{arm}"
    config["output_dir"] = str(output_dir)
    config["oracle_pack"] = {"manifest_path": str(manifest_path)}
    # `smoke_no_publication` forces gold_eligible to False no matter what validation
    # concluded, which would make the A0 publish-rate readout vacuous.
    config["lineage"]["policy"] = "strict_separation"

    roots = list(config["oracle_runtime"].get("allowed_roots") or [])
    roots.extend(str(root) for root in extra_allowed_roots)
    config["oracle_runtime"]["allowed_roots"] = roots

    for key, value in (overrides or {}).items():
        config[key] = value

    minimization: dict[str, Any] = {}
    if minimal:
        from bfcl_ablation.simplify import runconfig

        bootstrap()
        config, dropped, kept = runconfig.minimize(config)
        minimization = {"dropped": dropped, "kept": kept}
    write_config.last_minimization = minimization  # type: ignore[attr-defined]

    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"config_{arm}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def run_arm(
    *,
    arm: str,
    pack_dir: Path,
    extra_allowed_roots: tuple[Path, ...] = (),
    minimal_config: bool = False,
    overrides: dict[str, Any] | None = None,
) -> ArmResult:
    """Generate a benchmark from `pack_dir` and return where its artifacts landed.

    `overrides` exists for one setting: `task_generation.tasks_per_category` is a
    per-category budget the pipeline refuses to run below the number of templates in the
    widest category, so an arm that authors more templates than A0 has to raise it or it
    cannot generate at all. Any other override would make an arm's numbers a property of
    its config rather than of its pack.
    """
    bootstrap()
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl

    output_dir = GENERATED / "runs" / arm
    if output_dir.exists():
        # A stale run would let a removed row survive into this arm's measurement.
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = write_config(
        arm=arm,
        manifest_path=pack_dir / "manifest.yaml",
        output_dir=output_dir,
        extra_allowed_roots=extra_allowed_roots,
        minimal=minimal_config,
        overrides=overrides,
    )
    benchmark = Path(generate_bfcl(config_path))
    return ArmResult(arm=arm, pack_dir=pack_dir, config_path=config_path, run_dir=benchmark.parent)


def count_authored_lines(pack_dir: Path, config_path: Path | None = None) -> dict[str, int]:
    """Count the lines a human had to write, per file.

    Blank lines and comment-only lines still count: they are part of what an author
    reads and maintains, and excluding them would flatter whichever arm happens to be
    less commented.
    """
    counts: dict[str, int] = {}
    for name in AUTHORED_PACK_FILES:
        path = pack_dir / name
        counts[name] = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
    if config_path is not None and config_path.exists():
        counts["run_config"] = len(config_path.read_text(encoding="utf-8").splitlines())
    counts["TOTAL"] = sum(counts.values())
    return counts


_ARM_PREFIX = re.compile(r"^(a\d[a-z]?)_(.+)$")


def result_path(name: str) -> Path:
    """Route an artifact to its arm's directory.

    Artifacts are filed per arm (`results/A0/report.md`) rather than flat
    (`results/a0_report.md`) so one arm's outputs can be published, diffed or archived
    as a unit. The arm is taken from the filename prefix, so every existing call site
    keeps working without knowing about the layout.
    """
    match = _ARM_PREFIX.match(name)
    if match is None:
        RESULTS.mkdir(parents=True, exist_ok=True)
        return RESULTS / name
    arm, remainder = match.groups()
    directory = RESULTS / arm.upper()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / remainder


def dump_result(name: str, payload: Any) -> Path:
    path = result_path(name)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def env_note() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "cwd": os.getcwd(),
    }
