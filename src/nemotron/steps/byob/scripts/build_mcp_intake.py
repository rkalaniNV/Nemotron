#!/usr/bin/env python3
"""Derive a reviewable BFCL pack draft and evidence bundle from an MCP server.

Exit codes: 0 when the draft needs no human attention, 2 when hygiene flagged text a
reviewer must approve before the draft is used, and 1 on failure. The draft is never
publication ready on its own: fixtures, task templates, validation cases, and assertions
still need evidence this phase cannot observe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemotron.steps.byob.runtime.mcp.authoring.pack_artifacts import (
    PENDING_PACK_ARTIFACTS,
)
from nemotron.steps.byob.runtime.mcp.authoring.runner import IntakeResult, run_intake
from nemotron.steps.byob.runtime.mcp.config import load_trusted_executable_policies
from nemotron.steps.byob.runtime.mcp.errors import McpIntegrationError
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import (
    UNTRUSTED_TAG,
    ProseHygieneError,
    quote_untrusted,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intake",
        type=Path,
        required=True,
        help="Path to reviewed mcp_intake.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the pack draft, evidence bundle, and provenance",
    )
    parser.add_argument(
        "--trusted-executables",
        type=Path,
        help="Host-owned policy file required for stdio transport",
    )
    parser.add_argument(
        "--allow-insecure-localhost",
        action="store_true",
        help="Allow debug-only http:// loopback MCP transport; never publication eligible",
    )
    return parser


def _flagged_text(result: IntakeResult) -> list[dict[str, str]]:
    """Show a reviewer the exact text that was flagged, fenced as data."""
    descriptions = {
        f"tools.{tool['published_name']}.description": tool["description"][UNTRUSTED_TAG]
        for tool in result.bundle.document["tools"]
    }
    quoted: list[dict[str, str]] = []
    for finding in result.bundle.document["review"]["advisory"]:
        text = descriptions.get(finding["location"])
        if text is None:
            continue
        quoted.append(
            {
                "location": finding["location"],
                "code": finding["code"],
                "text": quote_untrusted(text),
            }
        )
    return quoted


async def _run(args: argparse.Namespace) -> IntakeResult:
    policies = (
        load_trusted_executable_policies(args.trusted_executables)
        if args.trusted_executables is not None
        else None
    )
    return await run_intake(
        args.intake,
        args.output,
        executable_policies=policies,
        allow_insecure_localhost=args.allow_insecure_localhost,
    )


def _print(document: dict) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    try:
        result = asyncio.run(_run(args))
    except (McpIntegrationError, ProseHygieneError, OSError, ValueError) as exc:
        _print(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
        )
        raise SystemExit(1) from exc

    advisory = result.bundle.document["review"]["advisory"]
    _print(
        {
            "status": result.bundle.document["status"],
            "attained_level": result.bundle.document["attained_level"],
            "pack": result.bundle.document["pack"],
            "oracle": result.bundle.document["oracle"],
            "output_root": str(result.output_root),
            "artifacts": [
                artifact.as_dict(root=result.output_root)
                for artifact in result.artifacts
            ],
            "evidence_bundle_digest": result.bundle.bundle_digest,
            "tool_count": len(result.bundle.document["tools"]),
            "advisory_findings": advisory,
            "flagged_text": _flagged_text(result),
            "unknowns": [item["field"] for item in result.bundle.document["unknowns"]],
            "pending_artifacts": list(PENDING_PACK_ARTIFACTS),
        }
    )
    if advisory:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
