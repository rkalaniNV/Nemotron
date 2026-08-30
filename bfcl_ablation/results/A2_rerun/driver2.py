"""Complete the A2 re-run: build the missing budget-24 baseline, then run A2 in full.

Run 1 stopped at `error: baseline for budget 24 is missing at _generated/runs/sweep24`.
That artifact is produced by sweep_budget.py, which SUMMARY.md's reproduce block lists
before run_a2.py but which nothing in run_a2 creates or checks for up front.

sweep_budget.py writes results/budget_sweep.json, a COMMITTED file, so common.RESULTS is
redirected here too — the same guard run 1 used.

The __main__ guard is load-bearing: both scripts drive the BFCL oracle, which uses
mp.get_context("spawn").
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

    probe = common.result_path("budget_sweep.json")
    assert redirected in probe.parents, f"redirect failed: {probe}"
    assert committed not in probe.parents, f"still pointing at committed results: {probe}"
    print(f"[driver2] committed results (untouched): {committed}")
    print(f"[driver2] this run writes to:            {redirected}")

    sweep_run = common.GENERATED / "runs" / "sweep24" / "bfcl_ablation_sweep24"
    if sweep_run.exists():
        print(f"[driver2] budget-24 baseline already present at {sweep_run}")
    else:
        print("[driver2] building the budget-24 baseline via sweep_budget.py 24 ...", flush=True)
        from bfcl_ablation import sweep_budget

        sys.argv = ["sweep_budget.py", "24"]
        rc = sweep_budget.main()
        if rc not in (0, None):
            print(f"[driver2] sweep_budget failed with {rc}", file=sys.stderr)
            return rc
        if not sweep_run.exists():
            print(f"[driver2] sweep_budget did not create {sweep_run}", file=sys.stderr)
            return 3

    print("[driver2] running A2 in full ...", flush=True)
    from bfcl_ablation import run_a2

    # --skip-generate / --skip-runs reuse run 1's pools and its 40 variant pipeline runs,
    # which are already on disk and deterministic. Same A0 baseline as before.
    sys.argv = [
        "run_a2.py",
        "--arm", "a2rerun",
        "--baseline-arm", "a0",
        "--skip-generate",
        "--skip-runs",
    ]
    return run_a2.main()


if __name__ == "__main__":
    raise SystemExit(main())
