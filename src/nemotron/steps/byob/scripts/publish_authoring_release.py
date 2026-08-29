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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        adapter = publication_adapter_for_release(args.release)
        result = handoff_frozen_release(
            args.release,
            args.config,
            adapter=adapter,
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
