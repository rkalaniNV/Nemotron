#!/usr/bin/env python3
"""A1 — deterministic simplification.

Shrink the authored pack to the fields no code can infer, rehydrate it back to a
full pack, generate from that, and prove the benchmark did not move. No model is
involved anywhere in this arm.

    A0 pack --shrink--> A1 authored --rehydrate--> A1 full --pipeline--> benchmark
                            |                                               |
                    what a user writes                    compared against A0's benchmark

The primary readout is how much friction disappears at zero generative risk. The
secondary readout, and the more interesting one, is the list of fields that did
*not* survive the round trip: those are the places the plan's cut list is wrong.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common, equivalence  # noqa: E402
from bfcl_ablation.measurement import metrics, report  # noqa: E402
from bfcl_ablation.simplify import milestones, rehydrate, shrink  # noqa: E402


def _verdict_table(findings: list[dict]) -> str:
    counts = Counter(f["verdict"] for f in findings)
    lines = ["| verdict | fields |", "| --- | ---: |"]
    for verdict, count in sorted(counts.items()):
        lines.append(f"| {verdict} | {count} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=common.PACK_A0)
    parser.add_argument("--arm", default="a1")
    parser.add_argument("--baseline-arm", default="a0")
    parser.add_argument(
        "--skip-config-minimization",
        action="store_true",
        help="skip the second sub-step that strips default-valued run-config settings",
    )
    args = parser.parse_args()

    source = args.pack.resolve()
    authored = common.GENERATED / "packs" / f"{args.arm}_authored"
    full = common.GENERATED / "packs" / f"{args.arm}_full"

    print(f"[{args.arm}] shrinking {common.rel(source)} ...", flush=True)
    shrink_report = shrink.shrink_pack(source, authored)

    print(f"[{args.arm}] rehydrating -> {common.rel(full)} ...", flush=True)
    rehydrate_report = rehydrate.rehydrate_pack(authored, full)

    print(f"[{args.arm}] generating ...", flush=True)
    result = common.run_arm(arm=args.arm, pack_dir=full, extra_allowed_roots=(common.GENERATED,))

    baseline = common.ArmResult(
        arm=args.baseline_arm,
        pack_dir=source,
        config_path=common.GENERATED / f"config_{args.baseline_arm}.yaml",
        run_dir=common.GENERATED / "runs" / args.baseline_arm / f"bfcl_ablation_{args.baseline_arm}",
    )
    if not baseline.benchmark.exists():
        print(f"error: baseline arm '{args.baseline_arm}' has not been run; run run_a0.py first", file=sys.stderr)
        return 2

    tables = common.load_stage_tables(result)
    # Friction is measured on what a person writes: the authored pack, not the
    # rehydrated one. Counting the rehydrated pack would report no saving at all.
    loc = common.count_authored_lines(authored, result.config_path)
    baseline_loc = common.count_authored_lines(source, baseline.config_path)

    payload = metrics.measure(
        arm=args.arm,
        tables=tables,
        pack_dir=full,
        loc=loc,
        run_manifest=common.read_json(result.run_manifest),
        normalized_templates=result.stage_cache / "task_templates_normalized.yaml",
    )
    payload["simplification"] = {
        "authored_pack": str(authored),
        "rehydrated_pack": str(full),
        "shrink": shrink_report,
        "rehydrate": rehydrate_report,
        "loc_baseline": baseline_loc,
        "loc_a1": loc,
        "loc_saved": baseline_loc["TOTAL"] - loc["TOTAL"],
        "loc_saved_pct": round(
            100 * (baseline_loc["TOTAL"] - loc["TOTAL"]) / (baseline_loc["TOTAL"] or 1), 1
        ),
    }

    comparison = equivalence.compare(baseline, result, candidate_tables=tables)
    payload["equivalence"] = comparison

    # Sub-step: the same pack, run from a config stripped of default-valued settings.
    # Kept separate from the pack change so a divergence names one cause, not two.
    config_step = None
    if not args.skip_config_minimization:
        print(f"[{args.arm}c] re-running with a minimized run config ...", flush=True)
        minimal = common.run_arm(
            arm=f"{args.arm}c",
            pack_dir=full,
            extra_allowed_roots=(common.GENERATED,),
            minimal_config=True,
        )
        minimal_comparison = equivalence.compare(baseline, minimal)
        minimal_loc = common.count_authored_lines(authored, minimal.config_path)
        config_step = {
            "minimization": getattr(common.write_config, "last_minimization", {}),
            "config_lines_before": loc.get("run_config", 0),
            "config_lines_after": minimal_loc.get("run_config", 0),
            "loc_total": minimal_loc["TOTAL"],
            "loc_saved_vs_a0": baseline_loc["TOTAL"] - minimal_loc["TOTAL"],
            "equivalence": minimal_comparison,
        }
        payload["config_minimization"] = config_step

    common.dump_result(f"{args.arm}_metrics.json", payload)
    md = [
        report.render(payload),
        "## 6. Simplification",
        "",
        "| file | A0 | A1 | saved |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, before in baseline_loc.items():
        after = loc.get(name, 0)
        md.append(f"| {'**TOTAL**' if name == 'TOTAL' else name} | {before} | {after} | {before - after} |")
    md += [
        "",
        f"**{payload['simplification']['loc_saved']} lines removed "
        f"({payload['simplification']['loc_saved_pct']}%) with no model involved.**",
        "",
        "### Field-level derivability",
        "",
        _verdict_table(shrink_report["template_findings"] + shrink_report["manifest_findings"]),
        "",
    ]
    not_derivable = [
        f for f in shrink_report["template_findings"] + shrink_report["manifest_findings"]
        if f["verdict"] in {"not_derivable", "compiler_error"}
    ]
    if not_derivable:
        md += ["Fields that stayed authored because derivation did not reproduce them:", ""]
        for finding in not_derivable:
            where = finding.get("template_id", "manifest")
            md.append(f"- `{where}` / `{finding['field']}` — {finding.get('detail', '')[:300]}")
        md.append("")
    untested = milestones.untested_rules()
    if untested:
        md += [
            "### Milestone-compiler rules the round trip does not vouch for",
            "",
            "The compiler chooses among the milestone lists `_check_policy_shape` would "
            "accept. Every choice below compiles, but `banking_vn` contains no template "
            "that exercises it, so nothing has checked the choice against a human's intent.",
            "",
        ]
        for policy, reason in sorted(untested.items()):
            md.append(f"- `{policy}` — {reason}")
        md.append("")
    md += [
        f"Validation cases: {shrink_report['validation_cases_authored_before']} authored -> "
        f"{shrink_report['validation_cases_authored_after']} authored + "
        f"{shrink_report['validation_cases_generated']} generated.",
        "",
        equivalence.render(comparison),
    ]

    if config_step is not None:
        verdict = config_step["equivalence"]["verdict"]
        md += [
            "## 7. Run-config minimization (separate degree of freedom)",
            "",
            f"Run config {config_step['config_lines_before']} -> "
            f"{config_step['config_lines_after']} lines; total authoring "
            f"{baseline_loc['TOTAL']} -> {config_step['loc_total']} "
            f"({config_step['loc_saved_vs_a0']} saved, "
            f"{round(100 * config_step['loc_saved_vs_a0'] / (baseline_loc['TOTAL'] or 1), 1)}%).",
            "",
            f"Settings dropped: `{'`, `'.join(config_step['minimization'].get('dropped') or []) or 'none'}`",
            "",
            f"Verdict against `{args.baseline_arm}`: **{verdict}**",
            "",
        ]
        for entry in config_step["minimization"].get("kept") or []:
            md.append(f"- kept `{entry['setting']}` — {entry['reason']}")


    text = "\n".join(md)
    common.result_path(f"{args.arm}_report.md").write_text(text, encoding="utf-8")
    common.result_path(f"{args.arm}_vs_{args.baseline_arm}_equivalence.json").write_text(
        __import__("json").dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(text)
    if comparison["verdict"] != "EQUIVALENT":
        return 1
    if config_step is not None and config_step["equivalence"]["verdict"] != "EQUIVALENT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
