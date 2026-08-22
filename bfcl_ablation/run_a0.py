#!/usr/bin/env python3
"""A0 — human baseline.

Generate the banking_vn pack exactly as it is authored today and measure the result.
Nothing is cut and nothing is derived; A0 is the reference every later arm is
compared against, and the first arm to report any property of the benchmark itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.measurement import metrics, report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack",
        type=Path,
        default=common.PACK_A0,
        help="oracle pack directory to measure (default: the authored banking_vn pack)",
    )
    parser.add_argument("--arm", default="a0", help="arm name used for output paths")
    args = parser.parse_args()

    pack_dir = args.pack.resolve()
    print(f"[{args.arm}] generating from {common.rel(pack_dir)} ...", flush=True)
    result = common.run_arm(arm=args.arm, pack_dir=pack_dir, extra_allowed_roots=(pack_dir.parent,))
    print(f"[{args.arm}] artifacts in {common.rel(result.run_dir)}", flush=True)

    tables = common.load_stage_tables(result)
    loc = common.count_authored_lines(pack_dir, result.config_path)
    payload = metrics.measure(
        arm=args.arm,
        tables=tables,
        pack_dir=pack_dir,
        loc=loc,
        run_manifest=common.read_json(result.run_manifest),
        normalized_templates=result.stage_cache / "task_templates_normalized.yaml",
    )

    json_path = common.dump_result(f"{args.arm}_metrics.json", payload)
    md_path = common.result_path(f"{args.arm}_report.md")
    md_path.write_text(report.render(payload), encoding="utf-8")

    print(report.render(payload))
    print(f"\nwrote {common.rel(json_path)}")
    print(f"wrote {common.rel(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
