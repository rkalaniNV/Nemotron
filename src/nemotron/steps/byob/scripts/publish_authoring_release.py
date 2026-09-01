#!/usr/bin/env python3
"""Freshly validate and publish one adapter-neutral v2 authoring release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemotron.steps.byob.runtime.authoring_release.handoff import (
    AuthoringHandoffError,
    handoff_frozen_release,
)
from nemotron.steps.byob.runtime.authoring_release.publication import (
    publication_adapter_for_release,
)
from nemotron.steps.byob.runtime.authoring_release.revocation import (
    RevocationRegistryVerifier,
    load_trusted_revocation_key,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--revocation-registry", type=Path)
    parser.add_argument("--revocation-issuer")
    parser.add_argument("--revocation-public-key", type=Path)
    parser.add_argument("--revocation-key-id")
    parser.add_argument("--revocation-minimum-generation", type=int, default=1)
    args = parser.parse_args()
    try:
        revocation_values = (
            args.revocation_registry,
            args.revocation_issuer,
            args.revocation_public_key,
            args.revocation_key_id,
        )
        if any(value is not None for value in revocation_values) and not all(
            value is not None for value in revocation_values
        ):
            raise ValueError(
                "revocation registry, issuer, public key, and key ID "
                "must be supplied together"
            )
        revocation_check = None
        if args.revocation_registry is not None:
            revocation_check = RevocationRegistryVerifier(
                path=args.revocation_registry,
                expected_issuer=args.revocation_issuer,
                trusted_public_keys=load_trusted_revocation_key(
                    args.revocation_public_key,
                    key_id=args.revocation_key_id,
                ),
                minimum_generation=args.revocation_minimum_generation,
            )
        adapter = publication_adapter_for_release(args.release)
        result = handoff_frozen_release(
            args.release,
            args.config,
            adapter=adapter,
            revocation_check=revocation_check,
        )
    except (AuthoringHandoffError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "code": getattr(exc, "code", "publication_failed"),
                    "reason": str(exc),
                    "recovery": getattr(
                        exc,
                        "recovery",
                        "repair the frozen release or publication config and retry",
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            {
                "status": "published",
                "adapter": result.release.adapter_kind,
                "validation_report": str(result.validation_report_path),
                "benchmark": str(result.benchmark_path),
                "benchmark_raw": str(result.raw_benchmark_path),
                "run_manifest": str(result.run_manifest_path),
                "frozen_pack_fingerprint": result.release.pack_fingerprint,
                "run_id": result.run_manifest.get("run_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
