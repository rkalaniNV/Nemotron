#!/usr/bin/env python3
"""Measure what `tasks_per_category` actually buys.

A0 reports the benchmark at one budget. This sweeps the budget so the shape of the
trade is visible: the same knob controls how many entities get exercised and how
many times the same sentence is repeated, and it does not control the policy mix at
all. Knowing where each curve flattens is what sizes A2 and A3.

    PYTHONPATH=src python3 bfcl_ablation/sweep_budget.py 6 12 24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.measurement import metrics  # noqa: E402


def run_budget(pack: Path, budget: int) -> dict:
    original = common.write_config

    def patched(**kwargs):
        path = original(**kwargs)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["task_generation"] = {"tasks_per_category": budget}
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    common.write_config = patched
    try:
        result = common.run_arm(arm=f"sweep{budget}", pack_dir=pack, extra_allowed_roots=(pack.parent,))
    finally:
        common.write_config = original

    tables = common.load_stage_tables(result)
    payload = metrics.measure(
        arm=f"sweep{budget}",
        tables=tables,
        pack_dir=pack,
        loc=common.count_authored_lines(pack, result.config_path),
        run_manifest=common.read_json(result.run_manifest),
    )
    coverage = payload["coverage"]
    distribution = payload["distribution"]
    return {
        "budget": budget,
        "tasks": len(tables["task_instances"]),
        "published": len(tables["benchmark"]),
        "entities_bound": coverage["fixture_entities_bound"],
        "entities_total": coverage["fixture_entities_total"],
        "distinct_surfaces": payload["surface"]["overall"]["distinct_masked"],
        "single_turn_share": distribution["policy_task_share"].get("single_turn", 0.0),
        "policy_counts": distribution["policy_task_counts"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("budgets", nargs="*", type=int, default=[6, 12, 24])
    parser.add_argument("--pack", type=Path, default=common.PACK_A0)
    args = parser.parse_args()

    rows = [run_budget(args.pack.resolve(), budget) for budget in args.budgets]
    common.dump_result("budget_sweep.json", rows)

    header = f"{'budget':>7}{'tasks':>7}{'published':>11}{'entities':>11}{'surfaces':>10}{'single_turn':>13}"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['budget']:>7}{row['tasks']:>7}{row['published']:>11}"
            f"{row['entities_bound']:>7}/{row['entities_total']:<3}"
            f"{row['distinct_surfaces']:>10}{row['single_turn_share']:>12.1%}"
        )
    print()
    print("rare policies by budget:")
    for policy in ("correction", "dependent_call", "multi_tool", "missing_slot", "clarify_only"):
        counts = " ".join(f"{row['budget']}:{row['policy_counts'].get(policy, 0)}" for row in rows)
        print(f"  {policy:16} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
