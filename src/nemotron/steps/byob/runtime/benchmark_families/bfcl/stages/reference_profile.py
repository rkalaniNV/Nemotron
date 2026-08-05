"""Create the reference-profile artifact used by surface generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)


def run_reference_profile(config: BfclConfig) -> dict[str, Any]:
    """Record a disabled profile or reject an unsupported enabled profile."""
    role = (config.lineage.roles or {}).get("profile")
    enabled = bool(role and role.enabled)
    profile = {
        "enabled": enabled,
        "status": "disabled" if not enabled else "unsupported",
        "style_hints": [],
        "profile_influenced_surface": bool(config.lineage.profile_influenced_surface),
    }
    if enabled:
        raise NotImplementedError("Reference-profile extraction is not supported")

    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "reference_profile.json"
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("BFCL reference profile wrote %s", path)
    return profile
