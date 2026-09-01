#!/usr/bin/env python3
"""Assemble a candidate oracle pack from certified source, compiled drafts, and semantics.

This sits between `draft` and `review`. It authors nothing: the pack identity comes from
verified evidence, the oracle files come from the source tree certification fingerprinted,
`assertions.py` comes from drafts that compiled, and the reviewed supplement supplies only
the semantics no draft schema can express.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.pack_authoring.pack_assembly import (
    PackAssemblyError,
    assemble_candidate_pack,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True, help="Draft root directory")
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _print(document: dict[str, object]) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    try:
        assembled = assemble_candidate_pack(
            evidence_path=args.evidence,
            source_root=args.source,
            draft_root=args.drafts,
            supplement_path=args.supplement,
            output_root=args.output,
        )
    except (PackAssemblyError, OSError, ValueError) as exc:
        _print(
            {
                "status": "fail",
                "code": getattr(exc, "code", "candidate_pack_assembly_failed"),
                "reason": str(exc),
                "recovery": getattr(
                    exc,
                    "recovery",
                    "correct the bound inputs and assemble into a fresh directory",
                ),
            }
        )
        raise SystemExit(1) from exc
    _print(
        {
            "status": "assembled",
            "pack": str(assembled.pack_root),
            "manifest": str(assembled.manifest_path),
            "record": str(assembled.record_path),
            "record_digest": assembled.record["record_digest"],
            "compiled_assertions": list(assembled.record["compiled_assertions"]),
            "next_command": "review",
        }
    )


if __name__ == "__main__":
    main()
