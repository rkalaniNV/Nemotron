#!/usr/bin/env python3
"""Draft BFCL pack artifacts from an approved MCP evidence bundle.

Exit codes: 0 when every drafted artifact compiled, 2 when drafts were written but remain
blocked on unknowns that only executable probes can resolve, and 1 on failure. Exit 2 is the
normal outcome at L0 and is not an error: it means the drafts are honest about what nobody
has observed yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.pack_authoring.bundle import BundleError
from nemotron.steps.byob.runtime.pack_authoring.grounding import GroundingError
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    AuthoringModelError,
)
from nemotron.steps.byob.runtime.pack_authoring.runner import run_drafting


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="evidence_bundle.json")
    parser.add_argument(
        "--approval",
        type=Path,
        required=True,
        help="Reviewer approval naming the bundle digest and every flagged finding",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for drafts")
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-canonical-id",
        required=True,
        help="Immutable model identity recorded in provenance",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Authoring defaults to greedy decoding so a rerun reproduces the draft",
    )
    return parser


def _print(document: dict) -> None:
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    model = AuthoringModel(
        alias=args.model_alias,
        provider=args.model_provider,
        model=args.model,
        canonical_id=args.model_canonical_id,
        seed=args.seed,
        inference_parameters={"temperature": args.temperature},
    )
    try:
        result = run_drafting(args.bundle, args.approval, args.output, model)
    except (BundleError, GroundingError, AuthoringModelError, OSError, ValueError) as exc:
        _print(
            {
                "status": "fail",
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "violations": list(getattr(exc, "violations", ())),
            }
        )
        raise SystemExit(1) from exc

    document = result.provenance.document
    _print(
        {
            "status": "drafted",
            "pack": document["pack"],
            "output_root": str(result.output_root),
            "draft_root": str(result.draft_root),
            "model": document["model"]["canonical_id"],
            "calls": [
                {
                    "stage": call["stage"],
                    "served_from_cache": call["served_from_cache"],
                }
                for call in document["calls"]
            ],
            "assertions_compiled": document["assertions_compiled"],
            "compilation_refusals": document["compilation_refusals"],
            "blocked_on": document["blocked_on"],
        }
    )
    if document["blocked_on"] or not document["assertions_compiled"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
