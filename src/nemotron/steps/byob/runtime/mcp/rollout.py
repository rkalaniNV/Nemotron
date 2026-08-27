"""Explicit rollout gate for operations that contact an MCP server."""

from __future__ import annotations

import os
from collections.abc import Mapping

from nemotron.steps.byob.runtime.mcp.errors import McpConfigError

MCP_FEATURE_ENV = "BFCL_ENABLE_EXPERIMENTAL_MCP"
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


def mcp_feature_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the explicit rollout decision, rejecting ambiguous spellings."""
    source = os.environ if environ is None else environ
    raw = source.get(MCP_FEATURE_ENV)
    if raw is None:
        return False
    normalized = raw.strip().casefold()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise McpConfigError(
        f"{MCP_FEATURE_ENV} must be one of "
        f"{sorted(_TRUE | _FALSE)}, got {raw!r}"
    )


def require_mcp_feature(environ: Mapping[str, str] | None = None) -> None:
    """Fail closed until an operator explicitly opts into the experimental path."""
    if not mcp_feature_enabled(environ):
        raise McpConfigError(
            "MCP onboarding is experimental and not publication-ready; set "
            f"{MCP_FEATURE_ENV}=1 to enable live discovery or gateway operation"
        )
