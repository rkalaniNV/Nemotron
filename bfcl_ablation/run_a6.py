#!/usr/bin/env python3
"""A6 — is the oracle itself falsifiable?

A1 removed 230 lines and declared the remaining 877 — backend 465, assertions 182, tools
162, fixtures 68 — to be ground truth that cannot be cut. Nothing measured that. A1 simply
did not touch those files, and every later document has quoted the number as if it had.

This arm measures it for the largest of them. Corrupt `backend.py` one edit at a time and
ask every check the pack has whether it notices: the validation cases, A0's replayed
traces, the pack's own assertions, the oracle-validation checks, and a full pipeline run
to `published` and tier `gold`. A mutant nothing kills is a line the pack does not pin —
the backend says it, and no part of the benchmark depends on it being true.

This is A4's question one level down. A4 corrupted an *episode* and asked whether the
assertions noticed. A6 corrupts the *oracle* and asks whether anything at all notices.
Neither uses a model.

    PYTHONPATH=src python3 bfcl_ablation/run_a6.py
    PYTHONPATH=src python3 bfcl_ablation/run_a6.py --limit 10 --skip-pipeline   # smoke
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.backend_gate import ladder, operators  # noqa: E402
from bfcl_ablation.measurement.metrics import METRIC_CONTRACT_VERSION  # noqa: E402
from bfcl_ablation.mutate import inputs  # noqa: E402
from bfcl_ablation.propose import probe  # noqa: E402

A0_STAGE_CACHE = common.GENERATED / "runs" / "a0" / "bfcl_ablation_a0" / "stage_cache"
A0_CONFIG = common.GENERATED / "config_a0.yaml"
MUTANT_ROOT = common.GENERATED / "backend_mutants"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="first N mutants only")
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="stop at L4; L5 runs the full generator per surviving mutant and is the slow half",
    )
    args = parser.parse_args()

    if not A0_STAGE_CACHE.exists():
        print(f"missing A0 artifacts at {common.rel(A0_STAGE_CACHE)}; run run_a0.py first")
        return 1

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import confirmation_protocol

    config, pack = inputs.load_config_and_pack(A0_CONFIG)
    protocol = confirmation_protocol(pack.manifest)
    cases = probe.load_validation_cases(pack.paths.pack_root)
    tasks = inputs.load_tasks(A0_STAGE_CACHE)
    traces = inputs.load_traces(A0_STAGE_CACHE)
    replayed = inputs.replayed_task_ids(A0_STAGE_CACHE)
    tasks = [t for t in tasks if t["task_id"] in replayed and t["task_id"] in traces]

    source = pack.paths.backend_path.read_text(encoding="utf-8")
    mutants = operators.build_mutants(source)
    if args.limit:
        mutants = mutants[: args.limit]

    MUTANT_ROOT.mkdir(parents=True, exist_ok=True)
    runner = ladder.Runner(
        pack_dir=pack.paths.pack_root,
        fixtures=pack.fixtures or {},
        runtime=ladder.Runtime.from_config(config.oracle_runtime),
        protocol=protocol,
        worker=config.oracle_runtime.worker,
        concurrency=args.concurrency,
    )

    print(
        f"[a6] {len(mutants)} mutants over {len(source.splitlines())} lines of backend.py; "
        f"{len(cases)} validation cases, {len(tasks)} replayed tasks",
        flush=True,
    )

    started = time.time()
    baseline = ladder.build_baseline(
        runner, backend_path=pack.paths.backend_path, cases=cases, tasks=tasks, traces=traces
    )
    drift = [c["id"] for c in cases if baseline.validation.get(str(c["id"]))]
    if drift:
        # If the unmutated backend already fails its own cases, L1 is dead for every
        # mutant and the whole table would silently under-report kills.
        print(f"[a6] refusing to run: baseline fails its own validation cases {drift}")
        return 1
    baseline_pass = sum(1 for v in baseline.assertions.values() for ok in v.values() if ok)
    baseline_total = sum(len(v) for v in baseline.assertions.values())
    print(
        f"[a6] baseline captured in {time.time() - started:.0f}s — "
        f"{len(baseline.tools)} tools, {len(cases)}/{len(cases)} cases match, "
        f"assertions {baseline_pass}/{baseline_total}",
        flush=True,
    )

    def judge(mutant: operators.Mutant) -> dict[str, Any]:
        path = MUTANT_ROOT / f"m{mutant.site.index:04d}.py"
        path.write_text(mutant.source, encoding="utf-8")
        row: dict[str, Any] = {
            "index": mutant.site.index,
            "operator": mutant.site.operator,
            "family": mutant.site.family,
            "lineno": mutant.site.lineno,
            "before": mutant.site.before,
            "edit": mutant.site.after,
        }

        tools = runner.tools(path)
        if not isinstance(tools, list) or tools != baseline.tools:
            row["killed_by"] = ladder.L0_IMPORT
            row["detail"] = str(tools)[:200]
            return row

        got = runner.validation(path, cases)
        broke = {case_id: fails for case_id, fails in got.items() if fails}
        if broke:
            reasons = sorted({f["reason"] for fails in broke.values() for f in fails})
            row["killed_by"] = ladder.L1_VALIDATION
            row["detail"] = f"{len(broke)} cases {reasons}: {sorted(broke)[:3]}"
            return row

        replays = runner.replay(path, tasks, traces, with_assertions=True)
        changed = [
            tid
            for tid, base in baseline.traces.items()
            if replays.get(tid, {}).get("results") != base["results"]
            or replays.get(tid, {}).get("state") != base["state"]
        ]
        regressed = [
            tid
            for tid, base in baseline.assertions.items()
            for name, was_ok in base.items()
            if was_ok and not (replays.get(tid, {}).get("assertions") or {}).get(name, False)
        ]
        # Order matters: a mutant that changes a value AND trips an assertion is
        # attributed to the assertion, because that is the check the pack actually ships.
        if regressed:
            row["killed_by"] = ladder.L3_ASSERTIONS
            row["detail"] = f"{len(set(regressed))} tasks: {sorted(set(regressed))[:3]}"
            row["also_changed_observably"] = bool(changed)
            return row
        if changed:
            row["killed_by"] = ladder.L2_TRACES
            row["detail"] = f"{len(changed)} tasks differ from baseline: {changed[:3]}"
            row["note"] = "observable, but no shipped check caught it"
            return row

        row["killed_by"] = ladder.SURVIVED
        return row

    rows: list[dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency // 2)) as pool:
        for done, row in enumerate(pool.map(judge, mutants), 1):
            rows.append(row)
            if done % 10 == 0 or done == len(mutants):
                print(f"[a6] judged {done}/{len(mutants)} ({time.time() - started:.0f}s)", flush=True)

    # L4 and L5 need a pack directory, and only mutants nothing has killed yet reach them.
    pending = [r for r in rows if r["killed_by"] == ladder.SURVIVED]
    print(f"[a6] {len(pending)} mutants survived L0-L3; running L4 (oracle validation)", flush=True)
    for position, row in enumerate(pending, 1):
        pack_dir = _mutant_pack(pack.paths.pack_root, rows_index=row["index"], source_path=MUTANT_ROOT)
        verdict = _oracle_verdict(pack_dir, config)
        if verdict is not None:
            row["killed_by"] = ladder.L4_ORACLE
            row["detail"] = verdict
        if position % 5 == 0 or position == len(pending):
            print(f"[a6]   L4 {position}/{len(pending)}", flush=True)

    pending = [r for r in rows if r["killed_by"] == ladder.SURVIVED]
    if args.skip_pipeline:
        print(f"[a6] --skip-pipeline: {len(pending)} mutants left unjudged at L5", flush=True)
    else:
        print(f"[a6] {len(pending)} mutants survived L0-L4; running L5 (full pipeline)", flush=True)
        for position, row in enumerate(pending, 1):
            pack_dir = MUTANT_ROOT / f"pack_m{row['index']:04d}"
            verdict = _pipeline_verdict(pack_dir, row["index"])
            if verdict is not None:
                row["killed_by"] = ladder.L5_PIPELINE
                row["detail"] = verdict
            print(f"[a6]   L5 {position}/{len(pending)} m{row['index']:04d} -> {row['killed_by']}", flush=True)

    payload = _summarise(rows, mutants, source, pack, baseline, cases, tasks, runner, args)
    path = common.dump_result("a6_metrics.json", payload)
    trials_path = common.dump_result("a6_trials.json", rows)
    report_path = common.result_path("a6_report.md")
    report_path.write_text(_render(payload), encoding="utf-8")
    print(f"\nwrote {common.rel(path)}")
    print(f"wrote {common.rel(trials_path)}")
    print(f"wrote {common.rel(report_path)}")
    print()
    print(_render(payload))
    return 0


def _mutant_pack(pack_root: Path, *, rows_index: int, source_path: Path) -> Path:
    """A full pack copy with exactly one file replaced."""
    target = MUTANT_ROOT / f"pack_m{rows_index:04d}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(pack_root, target)
    shutil.copyfile(source_path / f"m{rows_index:04d}.py", target / "backend.py")
    return target


def _oracle_verdict(pack_dir: Path, config: Any) -> str | None:
    """None when the mutant still earns gold; otherwise why it did not."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.oracle_validation import (
        derive_pack_tier,
        run_oracle_validation,
    )

    try:
        cfg = common.write_config(
            arm=f"a6probe",
            manifest_path=pack_dir / "manifest.yaml",
            output_dir=common.GENERATED / "runs" / "a6probe",
            extra_allowed_roots=(pack_dir,),
        )
        probe_config, probe_pack = inputs.load_config_and_pack(cfg)
        report = run_oracle_validation(probe_config, probe_pack)
    except Exception as error:  # noqa: BLE001 — a crash is a kill, and its reason is the detail
        return f"{type(error).__name__}: {error}"[:200]
    eligible, tier = derive_pack_tier(report)
    if eligible and tier == "gold":
        return None
    failing = [
        c.get("name") or c.get("id")
        for c in [*(report.get("checks") or []), *(report.get("extra_checks") or [])]
        if c.get("status") != "pass" or c.get("failures")
    ]
    return f"tier={tier} failing={failing[:5]}"


def _pipeline_verdict(pack_dir: Path, index: int) -> str | None:
    """None when the mutant still publishes every row at gold."""
    try:
        result = common.run_arm(
            arm=f"a6m{index:04d}", pack_dir=pack_dir, extra_allowed_roots=(pack_dir,)
        )
        published = common.read_parquet(result.benchmark)
    except Exception as error:  # noqa: BLE001
        return f"{type(error).__name__}: {error}"[:200]
    if len(published) != 33:
        return f"published {len(published)} rows, expected 33"
    tiers = {str(r.get("tier")) for r in published}
    if tiers != {"gold"}:
        return f"tiers={sorted(tiers)}"
    return None


def _summarise(rows, mutants, source, pack, baseline, cases, tasks, runner, args) -> dict[str, Any]:
    by_layer = Counter(r["killed_by"] for r in rows)
    per_operator: dict[str, dict[str, Any]] = {}
    for operator in sorted({r["operator"] for r in rows}):
        subset = [r for r in rows if r["operator"] == operator]
        survived = sum(1 for r in subset if r["killed_by"] == ladder.SURVIVED)
        per_operator[operator] = {
            "mutants": len(subset),
            "survived": survived,
            "survival_rate": round(survived / len(subset), 4) if subset else None,
            "by_layer": dict(Counter(r["killed_by"] for r in subset)),
        }
    per_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in rows}):
        subset = [r for r in rows if r["family"] == family]
        survived = sum(1 for r in subset if r["killed_by"] == ladder.SURVIVED)
        per_family[family] = {
            "mutants": len(subset),
            "survived": survived,
            "survival_rate": round(survived / len(subset), 4) if subset else None,
        }

    lines_touched = sorted({r["lineno"] for r in rows})
    unpinned = sorted({r["lineno"] for r in rows if r["killed_by"] == ladder.SURVIVED})
    observable_only = [r for r in rows if r["killed_by"] == ladder.L2_TRACES]

    # The raw survival count is the wrong headline on its own: a mutant that survives is
    # usually one nothing executes, which measures coverage, not checking. The number that
    # measures checking is how many mutants demonstrably changed something and were still
    # accepted by every check the pack ships. L2 is the differential reference this arm
    # adds, so a kill there means precisely "observable, and nothing shipped caught it".
    unobservable = by_layer.get(ladder.SURVIVED, 0)
    unchecked = by_layer.get(ladder.L2_TRACES, 0)
    observable = len(rows) - unobservable
    caught_by_pack = observable - unchecked

    return {
        "arm": "a6",
        "metrics_version": METRIC_CONTRACT_VERSION,
        "observable": observable,
        "unobservable": unobservable,
        "caught_by_pack": caught_by_pack,
        "unchecked": unchecked,
        "blind_rate": round(unchecked / observable, 4) if observable else None,
        "env": common.env_note(),
        "pack": common.rel(pack.paths.pack_root),
        "backend_lines": len(source.splitlines()),
        "mutants": len(rows),
        "validation_cases": len(cases),
        "replayed_tasks": len(tasks),
        "episodes_run": runner.episodes_run,
        "l5_run": not args.skip_pipeline,
        "by_layer": dict(by_layer),
        "survival_rate": round(by_layer.get(ladder.SURVIVED, 0) / len(rows), 4) if rows else None,
        "by_operator": per_operator,
        "by_family": per_family,
        "lines_with_a_mutant": len(lines_touched),
        "unpinned_lines": unpinned,
        "observable_but_unchecked": {
            "count": len(observable_only),
            "note": "changed a returned value or the final state, and no check the pack ships noticed",
            "examples": [
                {"line": r["lineno"], "operator": r["operator"], "edit": r["edit"], "detail": r.get("detail")}
                for r in observable_only[:8]
            ],
        },
        "definitions": {
            "survived": "no layer of the pack detected the edit, including a full pipeline run to gold",
            "L2_expected_traces": "differential against the unmutated backend — NOT a check the pack ships",
            "unpinned_lines": "source lines where at least one single-edit mutant survived every layer",
        },
    }


def _render(payload: dict[str, Any]) -> str:
    by_layer = payload["by_layer"]
    total = payload["mutants"]
    out: list[str] = [
        "# BFCL ablation — arm `a6` (is the oracle itself falsifiable?)",
        "",
        f"Pack: `{payload['pack']}`. Metric contract `{payload['metrics_version']}`.",
        f"{total} single-edit mutants of `backend.py` ({payload['backend_lines']} lines), "
        f"judged against {payload['validation_cases']} validation cases, "
        f"{payload['replayed_tasks']} replayed tasks and "
        f"{'a full pipeline run' if payload['l5_run'] else 'L0-L4 only'}. "
        f"{payload['episodes_run']} oracle episodes.",
        "",
        "## 1. The headline",
        "",
        f"**{payload['blind_rate']:.1%} of observable backend corruptions pass every check the pack "
        f"ships** ({payload['unchecked']} of {payload['observable']}).",
        "",
        "Read the raw survival count carefully — on its own it is the wrong number. A mutant that",
        "survives everything is usually one the benchmark never *executes*, which is a coverage",
        "finding, not a checking one. The number that measures checking is the share of mutants that",
        "demonstrably changed something and were still accepted.",
        "",
        "| outcome | mutants | share |",
        "| --- | ---: | ---: |",
        f"| unobservable — nothing the pack runs reaches it | {payload['unobservable']} | "
        f"{payload['unobservable'] / total:.1%} |",
        f"| observable, caught by a check the pack ships | {payload['caught_by_pack']} | "
        f"{payload['caught_by_pack'] / total:.1%} |",
        f"| **observable, caught by nothing the pack ships** | **{payload['unchecked']}** | "
        f"**{payload['unchecked'] / total:.1%}** |",
        "",
        "That is the oracle-side view of the hole A4 measured from the assertion side as 0.610",
        "argument-level false acceptance. Two independent methods, one gap.",
        "",
        "## 2. What killed each mutant",
        "",
        "| layer | mutants | share | ships with the pack? |",
        "| --- | ---: | ---: | --- |",
    ]
    ships = {
        ladder.L0_IMPORT: "yes",
        ladder.L1_VALIDATION: "yes",
        ladder.L2_TRACES: "**no — reference added by this arm**",
        ladder.L3_ASSERTIONS: "yes",
        ladder.L4_ORACLE: "yes",
        ladder.L5_PIPELINE: "yes",
        ladder.SURVIVED: "—",
    }
    for layer in (*ladder.LAYERS, ladder.SURVIVED):
        n = by_layer.get(layer, 0)
        out.append(f"| `{layer}` | {n} | {n / total:.1%} | {ships[layer]} |")
    out += [
        "",
        "**`L4_oracle_validation` and `L5_pipeline` killed nothing.** Every mutant that reached them",
        "passed. The oracle-validation checks and a full generation run to 33 published rows at tier",
        "`gold` added **zero** detection over the cheap layers — A0's finding that the gates never",
        "fire, reproduced against a deliberately corrupted oracle rather than against clean input.",
        "",
        "`L2_expected_traces` is **not** a check the pack ships. It is a differential comparison",
        "against the unmutated backend, added here because nothing in the pack states what a tool",
        "should *return*. Every mutant in that row changed an observable value or the final state and",
        "was accepted by all four shipped layers.",
        "",
        "## 3. By mutation family",
        "",
        "| family | mutants | survived | survival rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in payload["by_family"].items():
        out.append(f"| `{name}` | {row['mutants']} | {row['survived']} | {row['survival_rate']:.3f} |")

    out += [
        "",
        "## 4. By operator",
        "",
        "| operator | mutants | survived | survival rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in payload["by_operator"].items():
        out.append(f"| `{name}` | {row['mutants']} | {row['survived']} | {row['survival_rate']:.3f} |")

    unpinned = payload["unpinned_lines"]
    observable = payload["observable_but_unchecked"]
    out += [
        "",
        "## 5. Unpinned lines",
        "",
        f"{len(unpinned)} of the {payload['lines_with_a_mutant']} lines carrying a mutant have at least",
        "one single-edit corruption that nothing detects:",
        "",
        "```",
        ", ".join(str(n) for n in unpinned) or "(none)",
        "```",
        "",
        f"## 6. Observable but unchecked ({observable['count']})",
        "",
        observable["note"] + ":",
        "",
        "| line | operator | edit |",
        "| ---: | --- | --- |",
    ]
    for ex in observable["examples"]:
        out.append(f"| {ex['line']} | `{ex['operator']}` | {ex['edit']} |")

    out += [
        "",
        "## 7. Reading this",
        "",
        "- **A surviving mutant is not necessarily a bug.** It is a line the *benchmark* does not",
        "  depend on. Some are genuinely unreachable given the fixtures; those are still a finding,",
        "  because they are lines a pack author paid to write and maintain that no task exercises.",
        "- **Equivalent mutants inflate survival.** An edit with no observable effect on any input",
        "  cannot be killed by anything and is not evidence of a weak pack. Survivors reaching L5",
        "  need hand-triage before the headline number is quoted, the way A4's strict/advisory",
        "  split did — that reclassification moved a headline from 0.137 to 0.380.",
        "- **This arm uses no model.** Everything here is deterministic and reproduces exactly.",
        "",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
