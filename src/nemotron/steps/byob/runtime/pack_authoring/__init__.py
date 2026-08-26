"""Deriving BFCL pack files from an approved evidence bundle.

This package deliberately imports nothing from `runtime/mcp/`. The drafting phase runs in
the `byob` environment, where Data Designer pins MCP SDK v1, while discovery and the
gateway run in `bfcl-mcp` with SDK v2; the two extras are declared mutually exclusive. The
only thing that crosses between them is a file. Keeping that separation structural rather
than incidental is what stops a later module-level import on the MCP side from breaking
authoring in an environment where the SDK it wants cannot be installed.
"""
