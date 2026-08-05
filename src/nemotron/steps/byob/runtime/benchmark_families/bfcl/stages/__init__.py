"""BFCL stage package."""

from __future__ import annotations

from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig


def stage_cache_dir(config: BfclConfig) -> Path:
    return Path(config.output_dir) / config.expt_name / "stage_cache"
