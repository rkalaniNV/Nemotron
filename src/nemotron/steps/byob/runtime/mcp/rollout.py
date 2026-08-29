"""Explicit rollout gate for operations that contact an MCP server."""

from __future__ import annotations

from collections.abc import Mapping

from nemotron.steps.byob.runtime.authoring_workflow.rollout import (
    LEGACY_MCP_ROLLOUT_ENV,
    RolloutPolicyError,
    adapter_rollout_enabled,
    require_adapter_rollout,
)
from nemotron.steps.byob.runtime.mcp.errors import McpConfigError

MCP_FEATURE_ENV = LEGACY_MCP_ROLLOUT_ENV


def mcp_feature_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Compatibility alias for the generic MCP Mode A rollout decision."""
    try:
        return adapter_rollout_enabled("mcp_mode_a", environ=environ)
    except RolloutPolicyError as exc:
        raise McpConfigError(str(exc)) from exc


def require_mcp_feature(environ: Mapping[str, str] | None = None) -> None:
    """Compatibility gate retained for one deprecation window."""
    try:
        require_adapter_rollout("mcp_mode_a", environ=environ)
    except RolloutPolicyError as exc:
        raise McpConfigError(
            f"{exc}; compatibility alias: set {MCP_FEATURE_ENV}=1"
        ) from exc
