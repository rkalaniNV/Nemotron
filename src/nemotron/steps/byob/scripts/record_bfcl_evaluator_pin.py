#!/usr/bin/env python3
"""Record which weights scored an ablation domain, or that none did.

``pinned`` binds a serving route to an immutable revision or weights digest and
is refused if the identity can move. ``unpinned`` records the absence of a pin as
evidence, which is what keeps ``target_model_pin_missing`` visible in a rollout
instead of being papered over with a route name.

Only the *name* of the credential environment variable is recorded. A value that
looks like a credential is refused, because a pin record is published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.mcp.ablation import AblationError
from nemotron.steps.byob.runtime.mcp.ablation_evaluator_pin import (
    EvaluatorPinError,
    build_pinned_evaluator,
    build_unpinned_evaluator,
    write_evaluator_pin,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pinned = commands.add_parser("pinned", help="Record an immutable evaluator identity.")
    pinned.add_argument("--output", type=Path, required=True)
    pinned.add_argument("--provider", required=True)
    pinned.add_argument("--served-model", required=True)
    pinned.add_argument("--api-base", required=True)
    pinned.add_argument(
        "--credential-env-var",
        required=True,
        help="The environment variable name that holds the credential, never its value.",
    )
    pinned.add_argument("--weight-source", required=True)
    pinned.add_argument("--weight-model", required=True)
    pinned.add_argument("--revision", help="An immutable commit identifier, 40-64 hex characters.")
    pinned.add_argument("--weights-digest", help="sha256:<64 lowercase hex>.")
    pinned.add_argument(
        "--pin-evidence-digest",
        required=True,
        help="Digest of the provider response or attestation the pin was read from.",
    )

    unpinned = commands.add_parser("unpinned", help="Record that no immutable pin exists.")
    unpinned.add_argument("--output", type=Path, required=True)
    unpinned.add_argument(
        "--reason-code",
        choices=("target_evaluation_not_run", "immutable_pin_unavailable"),
        required=True,
    )
    unpinned.add_argument("--detail", required=True)
    unpinned.add_argument(
        "--declared-route",
        help="Required for immutable_pin_unavailable; refused for target_evaluation_not_run.",
    )
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise EvaluatorPinError(f"refusing to overwrite an existing pin record: {args.output}")
    if args.command == "pinned":
        pin = build_pinned_evaluator(
            provider=args.provider,
            served_model=args.served_model,
            api_base=args.api_base,
            credential_env_var=args.credential_env_var,
            weight_source=args.weight_source,
            weight_model=args.weight_model,
            revision=args.revision,
            weights_digest=args.weights_digest,
            pin_evidence_digest=args.pin_evidence_digest,
        )
    else:
        pin = build_unpinned_evaluator(
            reason_code=args.reason_code,
            detail=args.detail,
            declared_route=args.declared_route,
        )
    write_evaluator_pin(pin, args.output)
    return {
        "status": "written",
        "output": str(args.output.resolve()),
        "pin_status": pin.status,
        "evaluator_model": pin.evaluator_model,
        "pin_digest": pin.pin_digest,
    }


def main() -> None:
    try:
        result = _run(_parser().parse_args())
    except (AblationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "fail", "error_type": type(exc).__name__, "reason": str(exc)},
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
