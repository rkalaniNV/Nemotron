"""Resolve and normalize oracle-pack artifacts."""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    LoadedPack,
    load_pack,
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir

logger = logging.getLogger(__name__)


def prepare_oracle_pack(config: BfclConfig) -> LoadedPack:
    """Resolve pack, normalize artifacts, write stage_cache/."""
    pack = load_pack(config)
    cache = stage_cache_dir(config)
    cache.mkdir(parents=True, exist_ok=True)

    tools_internal = pack.tools
    tools_model = project_model_facing_tools(tools_internal)

    (cache / "tools_normalized_internal.json").write_text(
        json.dumps(tools_internal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (cache / "tools_normalized.json").write_text(
        json.dumps(tools_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fixtures_payload: dict[str, Any] = pack.fixtures or {}
    (cache / "fixtures_normalized.json").write_text(
        json.dumps(fixtures_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (cache / "task_templates_normalized.yaml").write_text(
        yaml.safe_dump(pack.templates, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (cache / "validation_cases_normalized.yaml").write_text(
        yaml.safe_dump(pack.validation_cases, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (cache / "pack_manifest.json").write_text(
        json.dumps(pack.manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (cache / "pack_paths.json").write_text(
        json.dumps(
            {
                "pack_root": str(pack.paths.pack_root),
                "manifest_path": str(pack.paths.manifest_path),
                "tools_path": str(pack.paths.tools_path),
                "fixtures_path": str(pack.paths.fixtures_path) if pack.paths.fixtures_path else None,
                "templates_path": str(pack.paths.templates_path),
                "assertions_path": str(pack.paths.assertions_path),
                "validation_cases_path": str(pack.paths.validation_cases_path),
                "system_prompt_path": (
                    str(pack.paths.system_prompt_path) if pack.paths.system_prompt_path else None
                ),
                "backend_path": str(pack.paths.backend_path) if pack.paths.backend_path else None,
                "endpoint_config_path": (
                    str(pack.paths.endpoint_config_path)
                    if pack.paths.endpoint_config_path
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("BFCL prepare wrote normalized pack artifacts to %s", cache)
    return pack
