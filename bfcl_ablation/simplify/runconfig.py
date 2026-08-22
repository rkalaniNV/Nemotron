"""Strip run-config settings that only restate a default.

The reference config spends about forty lines saying `false`, `null`, or the number
the code would have used anyway. Removing them is the cheapest friction in the whole
ladder — but "the config still parses" is not the claim being made. The claim is that
generation produces the same benchmark, so the minimized config is run and compared
like any other arm.

Config minimization is kept as its own step rather than folded into the A1 pack
change, so that a divergence points at either the pack or the config, never both.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import yaml

# Settings whose declared value equals the code's default for this pack. Each is a
# (block, key) pair; key None means the whole block.
DEFAULT_VALUED = (
    ("surface_generation", None),
    ("surface_quality_validation", None),
    ("semantic_deduplication_config", None),
    ("config_status", None),
    ("lineage", "profile_influenced_surface"),
    ("lineage", "judge_advisory"),
    ("lineage", "roles"),
    ("oracle_runtime", "tool_timeout_s"),
    ("oracle_runtime", "assertion_timeout_s"),
    ("oracle_runtime", "import_timeout_s"),
    ("oracle_runtime", "reset_timeout_s"),
    ("oracle_runtime", "episode_timeout_s"),
)


def loadable(config: dict[str, Any]) -> tuple[bool, str]:
    """Whether the strict config parser accepts this document."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        path = handle.name
    try:
        BfclConfig.from_yaml(path)
        return True, ""
    except Exception as error:  # noqa: BLE001 - any parse failure disqualifies the drop
        return False, f"{type(error).__name__}: {error}"
    finally:
        Path(path).unlink(missing_ok=True)


def minimize(config: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[dict[str, str]]]:
    """Drop every default-valued setting the parser still accepts without it.

    Each candidate is tested on its own before the survivors are dropped together, so
    one load-bearing setting cannot mask the rest.
    """
    dropped: list[str] = []
    kept: list[dict[str, str]] = []

    for block, key in DEFAULT_VALUED:
        candidate = copy.deepcopy(config)
        if key is None:
            if block not in candidate:
                continue
            candidate.pop(block)
        else:
            if not isinstance(candidate.get(block), dict) or key not in candidate[block]:
                continue
            candidate[block].pop(key)
        ok, error = loadable(candidate)
        label = block if key is None else f"{block}.{key}"
        if ok:
            dropped.append(label)
        else:
            kept.append({"setting": label, "reason": error[:200]})

    minimal = copy.deepcopy(config)
    for block, key in DEFAULT_VALUED:
        label = block if key is None else f"{block}.{key}"
        if label not in dropped:
            continue
        if key is None:
            minimal.pop(block, None)
        elif isinstance(minimal.get(block), dict):
            minimal[block].pop(key, None)

    ok, error = loadable(minimal)
    if not ok:
        # Individually removable settings that together break the parse: report the
        # interaction rather than shipping a config that does not load.
        return config, [], kept + [{"setting": "<combined>", "reason": error[:200]}]
    return minimal, dropped, kept
