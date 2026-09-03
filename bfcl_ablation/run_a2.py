#!/usr/bin/env python3
"""A2 — LLM surface generation, wording only.

A0's sharpest finding was that 33 tasks collapse to 17 distinct sentences, exactly one
per template, and that `tasks_per_category` does not move that number at any budget.
A2 spends a model on the one input that can move it — the opening user turn — and
holds everything else fixed: same templates, same policies, same slot bindings, same
tools, same expected_tool_calls, same assertions.

Mechanics. The production pipeline renders a template's `user_turn_templates[lang]`,
a single string, so there is no per-task wording hook to reach for. Instead the arm
builds one variant pack per paraphrase index, runs the unmodified pipeline on each,
and assembles the arm's benchmark by taking each task's rows from the run its own
`seed` selects. Every published row is a production row that passed every production
guard; only the choice of which run it came from is A2's. Variant 0 is the authored
sentence verbatim, so N=1 is A0 re-derived through A2's machinery.

Two budgets. The ceiling on distinct surfaces is the sum over templates of
min(N, tasks for that template), so at `tasks_per_category=6` N=10 and N=20 are
indistinguishable (both cap at 33). Budget 24 is run alongside so the upper rungs are
measurable at all.

    PYTHONPATH=src:. python3 bfcl_ablation/run_a2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common, equivalence, llm  # noqa: E402
from bfcl_ablation.measurement import metrics  # noqa: E402
from bfcl_ablation.surface import generate, intent_check  # noqa: E402

LANGUAGE = "vi"
N_RUNGS = (1, 2, 3, 5, 10, 20)
N_MAX = max(N_RUNGS)
BUDGETS = (6, 24)

# Stage tables that carry one row per task, so an assembled arm can take a task's row
# from whichever variant run its seed picked. `task_instances` is excluded: it is
# wording-independent by construction and is the table the assembly is keyed on.
_TASK_KEYED = (
    "conversation_plans",
    "rendered_conversations",
    "expected_traces",
    "schema_validated_traces",
    "replay_validated_tasks",
    "benchmark_raw",
    "benchmark",
)


def variant_arm(budget: int, index: int) -> str:
    return f"a2_b{budget}_v{index}"


def variant_pack_dir(index: int) -> Path:
    return common.GENERATED / "packs" / f"a2_v{index}"


def _run_one(job: tuple[int, int]) -> str:
    """Generate one (budget, variant) run in its own process.

    The budget is applied by patching the config `common.write_config` just wrote,
    the way `sweep_budget` does. Patching a module global is only safe because each
    job owns a fresh interpreter.
    """
    budget, index = job
    original = common.write_config

    def patched(**kwargs):
        path = original(**kwargs)
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["task_generation"] = {"tasks_per_category": budget}
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    common.write_config = patched
    try:
        result = common.run_arm(
            arm=variant_arm(budget, index),
            pack_dir=variant_pack_dir(index),
            extra_allowed_roots=(common.GENERATED,),
        )
    finally:
        common.write_config = original
    return str(result.run_dir)


def arm_result(budget: int, index: int) -> common.ArmResult:
    arm = variant_arm(budget, index)
    return common.ArmResult(
        arm=arm,
        pack_dir=variant_pack_dir(index),
        config_path=common.GENERATED / f"config_{arm}.yaml",
        run_dir=common.GENERATED / "runs" / arm / f"bfcl_ablation_{arm}",
    )


def assemble(
    per_variant: list[dict[str, list[dict]]],
    *,
    pool_size_by_template: dict[str, int],
    n: int,
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Pick each task's rows from the variant run its seed selects.

    A template whose pool came back short caps at what it has, so a rung reports the
    variety it actually got rather than the variety it asked for.
    """
    tasks = per_variant[0]["task_instances"]
    indexed = [
        {name: {str(row["task_id"]): row for row in tables.get(name) or []} for name in _TASK_KEYED}
        for tables in per_variant
    ]

    choice: dict[str, int] = {}
    for task in tasks:
        pool = pool_size_by_template.get(str(task["template_id"]), 1)
        effective = max(1, min(n, pool))
        choice[str(task["task_id"])] = int(task["seed"]) % effective

    merged: dict[str, list[dict]] = {"task_instances": tasks}
    for name in _TASK_KEYED:
        rows = []
        for task in tasks:
            task_id = str(task["task_id"])
            row = indexed[choice[task_id]][name].get(task_id)
            if row is not None:
                rows.append(row)
        merged[name] = rows
    return merged, choice


def surface_ceiling(tasks: list[dict], pool_size_by_template: dict[str, int], n: int) -> int:
    per_template: dict[str, int] = {}
    for task in tasks:
        per_template[str(task["template_id"])] = per_template.get(str(task["template_id"]), 0) + 1
    return sum(
        min(count, max(1, min(n, pool_size_by_template.get(template_id, 1))))
        for template_id, count in per_template.items()
    )


def score_checker(client, pools: dict, templates: dict, tools: list[dict]) -> dict:
    """Score the intent checker on the authored sentences, the paraphrases and the decoys.

    The canonical sentences are in the batch on purpose. They are correct by
    construction, so their flag rate is the checker's own error floor and the only
    honest yardstick for reading the paraphrase rate.
    """
    canonical_rows = [
        {
            "template_id": tid,
            "text": pool["canonical"],
            "variant_index": 0,
            "required_tools": templates[tid].get("required_tools") or [],
            "population": "canonical",
        }
        for tid, pool in sorted(pools["pools"].items())
    ]
    paraphrase_rows = [
        {
            "template_id": tid,
            "text": variant,
            "variant_index": index,
            "required_tools": templates[tid].get("required_tools") or [],
            "population": "paraphrase",
        }
        for tid, pool in sorted(pools["pools"].items())
        for index, variant in enumerate(pool["variants"])
        if index > 0
    ]
    # Decoys the mechanical guards already rejected are excluded: they never reach the
    # checker in production either, so scoring them would inflate its recall.
    shift_rows = [
        {**shift, "population": "shift"}
        for shift in pools["shifts"]
        if shift.get("guard_rejection") is None
    ]
    checker = intent_check.evaluate(
        client,
        canonical=canonical_rows,
        paraphrases=paraphrase_rows,
        shifts=shift_rows,
        tools=tools,
        policy_by_template={tid: str(t.get("turn_policy")) for tid, t in templates.items()},
    )
    checker["shifts_rejected_by_mechanical_guards"] = {
        "count": sum(1 for shift in pools["shifts"] if shift.get("guard_rejection") is not None),
        "of": len(pools["shifts"]),
    }
    return checker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=common.PACK_A0)
    parser.add_argument("--arm", default="a2")
    parser.add_argument("--baseline-arm", default="a0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-generate", action="store_true", help="reuse the pools already on disk")
    parser.add_argument("--skip-runs", action="store_true", help="reuse the variant runs already on disk")
    args = parser.parse_args()

    pack = args.pack.resolve()
    common.bootstrap()
    client = llm.LLMClient(concurrency=16)
    print(f"[{args.arm}] probing {client.base_url} ...", flush=True)
    probe = llm.probe(client)
    print(f"[{args.arm}] model {probe['model']} answered {probe['smoke']!r}", flush=True)

    pools_path = common.GENERATED / "a2_pools.json"
    if args.skip_generate and pools_path.exists():
        pools = json.loads(pools_path.read_text(encoding="utf-8"))
    else:
        print(f"[{args.arm}] generating {N_MAX - 1} paraphrases + 2 intent shifts per template ...", flush=True)
        pools = generate.build(client, pack, language=LANGUAGE, need=N_MAX, shifts_per_template=2)
        pools_path.write_text(json.dumps(pools, indent=2, ensure_ascii=False), encoding="utf-8")
    pool_size = {tid: len(pool["variants"]) for tid, pool in pools["pools"].items()}
    print(f"[{args.arm}] pool sizes: {sorted(set(pool_size.values()))}, "
          f"rejections: {pools['rejection_reasons']}", flush=True)

    for index in range(N_MAX):
        generate.write_variant_pack(pack, variant_pack_dir(index), pools["pools"], index, LANGUAGE)

    jobs = [(budget, index) for budget in BUDGETS for index in range(N_MAX)]
    if args.skip_runs:
        print(f"[{args.arm}] reusing {len(jobs)} variant runs already on disk", flush=True)
    else:
        print(f"[{args.arm}] running {len(jobs)} pipeline runs on {args.workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
            for done, _ in enumerate(pool_exec.map(_run_one, jobs), start=1):
                if done % 5 == 0 or done == len(jobs):
                    print(f"[{args.arm}]   {done}/{len(jobs)} runs complete", flush=True)

    tools = generate.load_tools(pack)
    templates = {str(t["template_id"]): t for t in generate.load_templates(pack)}

    print(f"[{args.arm}] scoring the intent checker ...", flush=True)
    checker = score_checker(client, pools, templates, tools)
    # A2 measures the checker rather than gating on it, so a flagged variant is still
    # published. Carrying the flags into the rung loop is what turns the checker's rate
    # into the delivered benchmark's contamination rate.
    flagged_variants = {
        kind: {
            (row["template_id"], row["variant_index"])
            for row in checker["rows"]["paraphrases"]
            if (intent_check.disagreement_kind(row) == kind if kind != "any" else row["agrees"] is False)
        }
        for kind in ("any", "substituted")
    }

    per_budget: dict[str, dict] = {}
    for budget in BUDGETS:
        results = [arm_result(budget, index) for index in range(N_MAX)]
        tables = [common.load_stage_tables(result) for result in results]

        baseline = (
            common.ArmResult(
                arm=args.baseline_arm,
                pack_dir=pack,
                config_path=common.GENERATED / f"config_{args.baseline_arm}.yaml",
                run_dir=common.GENERATED / "runs" / args.baseline_arm / f"bfcl_ablation_{args.baseline_arm}",
            )
            if budget == 6
            else common.ArmResult(
                arm="sweep24",
                pack_dir=pack,
                config_path=common.GENERATED / "config_sweep24.yaml",
                run_dir=common.GENERATED / "runs" / "sweep24" / "bfcl_ablation_sweep24",
            )
        )
        if not baseline.benchmark.exists():
            print(f"error: baseline for budget {budget} is missing at {baseline.run_dir}", file=sys.stderr)
            return 2

        rungs = []
        for n in N_RUNGS:
            merged, choice = assemble(tables, pool_size_by_template=pool_size, n=n)
            payload = metrics.measure(
                arm=f"{args.arm}_b{budget}_n{n}",
                tables=merged,
                pack_dir=variant_pack_dir(0),
                loc=common.count_authored_lines(variant_pack_dir(0), results[0].config_path),
                run_manifest=common.read_json(results[0].run_manifest),
            )
            comparison = equivalence.compare(
                baseline,
                results[0],
                candidate_tables=merged,
                # A2's entire intervention is the opening sentence; the other four
                # checks still have to hold.
                opening_turn_may_change=True,
            )
            rendered_ok = sum(1 for row in merged["rendered_conversations"] if row.get("accepted") is True)
            template_of = {str(t["task_id"]): str(t["template_id"]) for t in merged["task_instances"]}
            contamination = {
                kind: sum(
                    1
                    for task_id, index in choice.items()
                    if (template_of[task_id], index) in variants
                )
                for kind, variants in flagged_variants.items()
            }
            rungs.append(
                {
                    "n": n,
                    "tasks": len(merged["task_instances"]),
                    "published": len(merged["benchmark"]),
                    "distinct_masked_surfaces": payload["surface"]["overall"]["distinct_masked"],
                    "distinct_raw_surfaces": payload["surface"]["overall"]["distinct_raw"],
                    "surface_ceiling": surface_ceiling(merged["task_instances"], pool_size, n),
                    "surfaces_per_template": payload["surface"]["overall"]["surfaces_per_template"],
                    "task_ids_equal": comparison["task_ids"]["equal"],
                    "expected_tool_calls_equal": comparison["expected_tool_calls"]["equal"],
                    "verdict": (
                        "FROZEN"
                        if comparison["task_ids"]["equal"] and comparison["expected_tool_calls"]["equal"]
                        else "BROKEN"
                    ),
                    "first_turns_changed": comparison["surface"]["tasks_with_changed_first_turn"],
                    "render_accepted": rendered_ok,
                    "render_rejected": len(merged["rendered_conversations"]) - rendered_ok,
                    "gold_eligible": comparison["publication"]["candidate_gold_eligible"],
                    "variants_used": len(set(choice.values())),
                    "published_tasks_intent_flagged": contamination["any"],
                    "published_tasks_intent_substituted": contamination["substituted"],
                    "funnel": payload["funnel"]["steps"],
                    "policy_task_counts": payload["distribution"]["policy_task_counts"],
                    "fixture_entities_bound": payload["coverage"]["fixture_entities_bound"],
                    "equivalence": comparison,
                }
            )
            print(
                f"[{args.arm}] budget {budget:>2} N={n:>2}: "
                f"{rungs[-1]['distinct_masked_surfaces']:>3} surfaces "
                f"(ceiling {rungs[-1]['surface_ceiling']}), {rungs[-1]['verdict']}",
                flush=True,
            )

        per_variant_guard_drops = {
            variant_arm(budget, index): [
                row.get("guard_violations")
                for row in tbl["rendered_conversations"]
                if row.get("accepted") is not True
            ]
            for index, tbl in enumerate(tables)
        }
        per_budget[str(budget)] = {
            "budget": budget,
            "baseline_arm": baseline.arm,
            "rungs": rungs,
            "pipeline_guard_rejections": {k: v for k, v in per_variant_guard_drops.items() if v},
        }

    payload = {
        "arm": args.arm,
        # A2 emits a bespoke schema rather than the shared measurement one, but its
        # diversity and coverage figures are compared against A0's, so it carries the
        # same contract stamp. It was the last arm without one; METRICS.md has required
        # it since version 1.0 and nothing enforced it, which is exactly how a
        # definition change would have been read as a benchmark change.
        "metrics_version": metrics.METRIC_CONTRACT_VERSION,
        "env": common.env_note(),
        "llm": {**probe, "stats": client.stats()},
        "design": {
            "language": LANGUAGE,
            "n_rungs": list(N_RUNGS),
            "budgets": list(BUDGETS),
            "variant_0_is_authored_sentence": True,
            "assembly": (
                "One pipeline run per paraphrase index; each task's rows are taken from run "
                "seed % min(N, pool_size(template)). Every row is production output."
            ),
        },
        "generation": {
            "requested_per_template": pools["requested_per_template"],
            "pool_sizes": pools["pool_sizes"],
            "templates_without_pool": pools["templates_without_pool"],
            "rejection_reasons": pools["rejection_reasons"],
            "rejection_rate": round(
                len(pools["rejections"]) / max(1, len(pools["rejections"]) + sum(pools["pool_sizes"].values()) - len(pools["pool_sizes"])),
                4,
            ),
            "rejections": pools["rejections"],
        },
        "intent_check": checker,
        "budgets": per_budget,
    }
    path = common.dump_result("a2_metrics.json", payload)

    report = render(payload)
    common.result_path("a2_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {common.rel(path)}")

    broken = [
        (budget, rung["n"])
        for budget, block in per_budget.items()
        for rung in block["rungs"]
        if rung["verdict"] != "FROZEN"
    ]
    if broken:
        print(f"A2 BROKEN at {broken}", file=sys.stderr)
        return 1
    return 0


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1%}"


def render(payload: dict) -> str:
    generation = payload["generation"]
    checker = payload["intent_check"]
    lines = [
        "# A2 — LLM surface generation (wording only)",
        "",
        f"Model: `{payload['llm']['model']}`. "
        f"{payload['llm']['stats']['calls_made']} calls made, "
        f"{payload['llm']['stats']['cache_hits']} served from cache.",
        "",
        "## 1. Paraphrase generation",
        "",
        f"Requested {generation['requested_per_template']} paraphrases per template "
        f"on top of the authored sentence. Pool sizes (incl. index 0): "
        f"{sorted(set(generation['pool_sizes'].values()))}.",
        "",
        "| rejection reason | variants |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(generation["rejection_reasons"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} |")
    if not generation["rejection_reasons"]:
        lines.append("| none | 0 |")
    lines += ["", f"Rejection rate: {generation['rejection_rate']:.1%} of everything the model returned.", ""]

    lines += [
        "## 2. Surface diversity vs. frozen ground truth",
        "",
        "| budget | N | tasks | distinct masked surfaces | ceiling | % of ceiling | surfaces/template | task_id equal | expected_tool_calls equal | verdict | published tasks on a substituted-intent surface |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for block in payload["budgets"].values():
        for rung in block["rungs"]:
            lines.append(
                f"| {block['budget']} | {rung['n']} | {rung['tasks']} | "
                f"{rung['distinct_masked_surfaces']} | {rung['surface_ceiling']} | "
                f"{rung['distinct_masked_surfaces'] / rung['surface_ceiling']:.0%} | "
                f"{rung['surfaces_per_template']} | "
                f"{'YES' if rung['task_ids_equal'] else 'NO'} | "
                f"{'YES' if rung['expected_tool_calls_equal'] else 'NO'} | {rung['verdict']} | "
                f"{rung['published_tasks_intent_substituted']}/{rung['published']} |"
            )
    lines += [
        "",
        "Ceiling is `sum over templates of min(N, pool size, tasks for that template)`; a rung that "
        "hits it has extracted all the variety the budget allows. The shortfall is collision: "
        "the variant a task gets is `seed % effective_N`, an unbalanced draw, so two tasks on the "
        "same template can land on the same variant while another goes unused. That is why a rung "
        "can lose ground to the rung below it — the assignment is not monotone in N.",
        "",
    ]

    lines += [
        "## 3. Intent check — the number that decides whether any of this is safe",
        "",
        "An independent call reads the sentence plus the tool catalogue and nothing else, and "
        "names the tools the request needs. Disagreement with the template's `required_tools` "
        "flags the sentence.",
        "",
        "| population | n | flagged | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("canonical (authored, correct by construction)", "canonical_false_alarm"),
        ("paraphrases (assumed good)", "paraphrase_false_alarm"),
        ("injected intent shifts (should all be caught)", "shift_recall"),
    ):
        block = checker[key]
        lines.append(f"| {label} | {block['n']} | {block['count']} | {_pct(block['rate'])} |")
    recovered = checker["shift_recovered_target"]
    substitution = checker["paraphrase_substitution"]
    lines += [
        "",
        f"Recall on injected shifts: **{_pct(checker['shift_recall']['rate'])}**. "
        f"False-alarm floor on the authored sentences: {_pct(checker['canonical_false_alarm']['rate'])}. "
        f"Flag rate on generated paraphrases: {_pct(checker['paraphrase_false_alarm']['rate'])}.",
        "",
        f"On {recovered['count']}/{recovered['n']} shifts the checker named exactly the tool the decoy "
        "was steered towards, which is the strong form of catching one.",
        "",
        f"{checker['shifts_rejected_by_mechanical_guards']['count']} of "
        f"{checker['shifts_rejected_by_mechanical_guards']['of']} decoys were stopped by the "
        "placeholder/literal/tool-name guards before the checker saw them.",
        "",
        "### What the flags on paraphrases actually are",
        "",
        f"Flag shapes: `{checker['paraphrase_flag_kinds']}`. "
        f"Only a *substituted* prediction says the sentence now asks for a different tool; that is "
        f"{substitution['count']}/{substitution['n']} paraphrases ({_pct(substitution['rate'])}). "
        "The rest are the checker disagreeing about how many calls the opening turn implies.",
        "",
        "| turn policy | paraphrases | flagged | rate | shapes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for policy, block in sorted(
        checker["paraphrase_by_policy"].items(), key=lambda kv: -(kv[1]["rate"] or 0)
    ):
        lines.append(
            f"| {policy} | {block['n']} | {block['flagged']} | {_pct(block['rate'])} | "
            f"{block['kinds'] or '-'} |"
        )
    lines += [
        "",
        "### What this costs the delivered benchmark",
        "",
        "The checker is a measurement in A2, not a gate: a flagged variant is still published. "
        "The last column of section 2 is therefore the contamination the arm actually shipped.",
        "",
        "| budget | N | published | flagged surface | substituted surface |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for block in payload["budgets"].values():
        for rung in block["rungs"]:
            lines.append(
                f"| {block['budget']} | {rung['n']} | {rung['published']} | "
                f"{rung['published_tasks_intent_flagged']} | "
                f"{rung['published_tasks_intent_substituted']} |"
            )
    lines += ["", f"> {checker['caveat']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
