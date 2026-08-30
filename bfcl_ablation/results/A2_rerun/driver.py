"""Re-run A2 against the EXISTING A0 baseline, writing every result to scratch.

Why a driver instead of calling run_a2.py directly: run_a2 hard-codes its output names
("a2_metrics.json", "a2_report.md") and common.result_path routes those to
results/A2/ regardless of --arm. Pointing common.RESULTS at a scratch directory is the
only way to run the arm without overwriting the committed results.

The __main__ guard is load-bearing: run_a2 uses ProcessPoolExecutor and the BFCL oracle
uses mp.get_context("spawn"), so an unguarded module body re-executes in every worker.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRATCH = Path("/tmp/claude-2524/-localhome-local-hndo/c87d64e8-2ffe-4ba0-9376-0eb5be228f75/scratchpad/a2rerun")
REPO = Path("/localhome/local-hndo/Nemotron")

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def main() -> int:
    from bfcl_ablation import common

    committed = common.RESULTS
    redirected = SCRATCH / "results"
    redirected.mkdir(parents=True, exist_ok=True)
    common.RESULTS = redirected

    # Prove the redirect took before anything runs.
    probe = common.result_path("a2_metrics.json")
    assert redirected in probe.parents, f"redirect failed: {probe}"
    assert committed not in probe.parents, f"still pointing at committed results: {probe}"
    print(f"[driver] committed results (untouched): {committed}")
    print(f"[driver] this run writes to:            {probe.parent}")

    from bfcl_ablation import run_a2

    # Same A0 baseline the user already ran: _generated/config_a0.yaml +
    # _generated/runs/a0/bfcl_ablation_a0. --arm keeps the per-variant pipeline runs
    # in their own _generated/runs/ directories so nothing collides.
    sys.argv = ["run_a2.py", "--arm", "a2rerun", "--baseline-arm", "a0"]
    return run_a2.main()


if __name__ == "__main__":
    raise SystemExit(main())
