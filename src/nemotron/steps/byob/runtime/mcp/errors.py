"""Typed failures for MCP discovery and normalization."""

from __future__ import annotations


class McpIntegrationError(RuntimeError):
    """Base class for BFCL MCP integration failures."""


class McpConfigError(McpIntegrationError, ValueError):
    """The strict MCP oracle configuration is invalid."""


class McpCredentialError(McpIntegrationError, ValueError):
    """A named credential is absent or unsafe to place on the wire."""


class McpExecutablePolicyError(McpIntegrationError, ValueError):
    """A stdio executable does not satisfy its trusted policy."""


class McpTransportError(McpIntegrationError):
    """The MCP transport could not be opened or completed safely."""


class McpProtocolError(McpIntegrationError):
    """The server did not satisfy the negotiated MCP discovery contract."""


class McpCatalogError(McpIntegrationError):
    """The complete selected tool catalog could not be established."""


class McpNormalizationError(McpIntegrationError):
    """A selected MCP tool cannot be represented by BFCL."""


class McpIdentityMismatchError(McpIntegrationError):
    """Observed MCP identity differs from the operator-pinned identity."""
