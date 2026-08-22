#!/usr/bin/env python3
"""A4 — LLM assertions, and the mutation gate that makes them measurable.

A0 reported a 100% publish rate with nothing dropped at any stage, which means no
existing number in this ablation distinguishes a pack that checks its tasks from a
pack that merely runs them. Part 1 builds that number: corrupt each gold episode and
count which corruptions the pack's own assertions let through. Part 2 spends it,
comparing the human assertions against assertions gpt-oss-120b wrote from the same
inputs the author had, and against a second LLM pass told which corruptions survived.

Everything executes through `ProcessWorker.run_episode` — the production isolation
path — including the model-written code. Nothing generated here is exec'd in-process.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.mutate import gate, inputs  # noqa: E402
from bfcl_ablation.mutate.operators import PackContext  # noqa: E402

A0_STAGE_CACHE = common.GENERATED / "runs" / "a0" / "bfcl_ablation_a0" / "stage_cache"
A0_CONFIG = common.GENERATED / "config_a0.yaml"
LLM_PACKS = common.GENERATED / "packs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--skip-llm", action="store_true", help="run Part 1 only")
    parser.add_argument(
        "--llm-attempts", type=int, default=3, help="regenerations allowed per assertion"
    )
    args = parser.parse_args()

    if not A0_STAGE_CACHE.exists():
        print(f"missing A0 artifacts at {common.rel(A0_STAGE_CACHE)}; run run_a0.py first")
        return 1

    config, pack = inputs.load_config_and_pack(A0_CONFIG)
    tasks = inputs.load_tasks(A0_STAGE_CACHE)
    traces = inputs.load_traces(A0_STAGE_CACHE)
    replayed = inputs.replayed_task_ids(A0_STAGE_CACHE)
    tasks = [task for task in tasks if task["task_id"] in replayed and task["task_id"] in traces]
    print(f"[a4] {len(tasks)} replayed tasks from A0", flush=True)

    context = PackContext.from_pack(pack.tools, pack.fixtures or {})
    engine = gate.Gate(
        backend_path=pack.paths.backend_path,
        fixtures=pack.fixtures or {},
        runtime=gate.Runtime.from_config(config.oracle_runtime),
        worker=config.oracle_runtime.worker,
        concurrency=args.concurrency,
    )

    human = gate.Target(
        name="human",
        assertions_path=pack.paths.assertions_path,
        import_root=pack.paths.pack_root,
    )
    started = time.time()
    plans, plan_errors = engine.plan(tasks, traces, target=human, context=context)
    mutations = sum(len(plan.mutations) for plan in plans.values())
    print(
        f"[a4] planned {len(plans)} episodes, {mutations} mutations "
        f"({time.time() - started:.1f}s)",
        flush=True,
    )
    for error in plan_errors:
        print(f"[a4] plan error {error}", flush=True)

    arms: dict[str, dict] = {}
    raw: dict[str, dict] = {}

    # The gate's own control. An assertion that does nothing must score gold 1.0 and
    # false acceptance 1.0 on every operator; anything else means the harness is
    # detecting corruptions by accident and every other row is unreadable.
    null_target = _null_target(sorted({n for t in tasks for n in t["success_assertions"]}))
    raw["null_control"] = engine.score(plans, target=null_target)
    arms["null_control"] = gate.summarize(raw["null_control"], plans)
    _print_arm(arms["null_control"])

    started = time.time()
    raw["human"] = engine.score(plans, target=human)
    arms["human"] = gate.summarize(raw["human"], plans)
    print(f"[a4] human arm scored in {time.time() - started:.1f}s", flush=True)
    _print_arm(arms["human"])

    authoring_report: dict = {}
    if not args.skip_llm:
        from bfcl_ablation.mutate import authoring

        names = sorted({name for plan in plans.values() for name in plan.task["success_assertions"]})
        author = authoring.Author(
            pack_dir=pack.paths.pack_root,
            tools=pack.tools,
            fixtures=pack.fixtures or {},
            templates=pack.templates,
            result_examples=_result_examples(plans),
            task_examples=_task_examples(plans, names),
            attempts=args.llm_attempts,
        )

        started = time.time()
        blind = author.write(names, packs_root=LLM_PACKS, arm="a4_llm_blind")
        print(f"[a4] llm_blind authored in {time.time() - started:.1f}s", flush=True)
        blind_target = gate.Target(
            name="llm_blind", assertions_path=blind.assertions_path, import_root=blind.pack_dir
        )
        raw["llm_blind"] = engine.score(plans, target=blind_target)
        arms["llm_blind"] = gate.summarize(raw["llm_blind"], plans)
        arms["llm_blind"]["authoring"] = blind.report
        _print_arm(arms["llm_blind"])

        started = time.time()
        feedback = gate.surviving_mutations(raw["llm_blind"])
        gold_failures = {
            row["assertion"]
            for row in raw["llm_blind"]["gold"]
            if not row["passed"]
        }
        repaired = author.write(
            names,
            packs_root=LLM_PACKS,
            arm="a4_llm_feedback",
            survivors=feedback,
            gold_failures=sorted(gold_failures),
            previous={name: blind.sources.get(name, "") for name in names},
        )
        print(f"[a4] llm_feedback authored in {time.time() - started:.1f}s", flush=True)
        feedback_target = gate.Target(
            name="llm_feedback",
            assertions_path=repaired.assertions_path,
            import_root=repaired.pack_dir,
        )
        raw["llm_feedback"] = engine.score(plans, target=feedback_target)
        arms["llm_feedback"] = gate.summarize(raw["llm_feedback"], plans)
        arms["llm_feedback"]["authoring"] = repaired.report
        _print_arm(arms["llm_feedback"])

        authoring_report = {
            "model": author.client.model,
            "llm_calls": author.client.stats(),
            "blind": blind.report,
            "feedback": repaired.report,
        }

    payload = {
        "arm": "a4",
        "env": common.env_note(),
        "pack": common.rel(pack.paths.pack_root),
        "tasks_gated": len(plans),
        "mutations_generated": mutations,
        "episodes_run": engine.episodes_run,
        "plan_errors": plan_errors,
        "operator_inventory": _inventory(plans),
        "assertions_loc": {
            "human": len(
                (pack.paths.assertions_path).read_text(encoding="utf-8").splitlines()
            ),
        },
        "arms": arms,
        "authoring": authoring_report,
    }
    path = common.dump_result("a4_metrics.json", payload)
    trials_path = common.dump_result(
        "a4_trials.json", {name: raw[name]["trials"] for name in raw}
    )
    print(f"\nwrote {common.rel(path)}")
    print(f"wrote {common.rel(trials_path)}")
    return 0


def _null_target(names: list[str]) -> gate.Target:
    """A pack whose assertions accept everything, written out and run like any other."""
    pack_dir = LLM_PACKS / "a4_null_control"
    pack_dir.mkdir(parents=True, exist_ok=True)
    body = ['"""Assertions that check nothing. Calibration floor for the mutation gate."""', ""]
    for name in names:
        body.append(f"def {name}(*, state, trace, task, ctx):")
        body.append("    return None")
        body.append("")
    body.append("ASSERTIONS = {" + ", ".join(f'"{name}": {name}' for name in names) + "}")
    (pack_dir / "assertions.py").write_text("\n".join(body) + "\n", encoding="utf-8")
    return gate.Target(
        name="null_control",
        assertions_path=pack_dir / "assertions.py",
        import_root=pack_dir,
    )


def _result_examples(plans: dict[str, gate.TaskPlan]) -> dict[str, dict]:
    """One real result per tool, harvested from the gold replays.

    A pack author owns backend.py and can run it, so the shape of a successful result
    is not privileged information; withholding it would cost the LLM arm points for
    guessing field names rather than for writing weak checks.
    """
    examples: dict[str, dict] = {}
    for task_id in sorted(plans):
        for entry in plans[task_id].trace:
            tool = str(entry["tool"])
            if tool not in examples and isinstance(entry.get("result"), dict):
                examples[tool] = entry["result"]
    return examples


def _task_examples(plans: dict[str, gate.TaskPlan], names: list[str]) -> dict[str, list[dict]]:
    """One task dict per (assertion, template), so slot key names are not guesswork."""
    examples: dict[str, list[dict]] = {name: [] for name in names}
    seen: set[tuple[str, str]] = set()
    for task_id in sorted(plans):
        task = plans[task_id].task
        for name in task.get("success_assertions") or []:
            key = (name, str(task.get("template_id")))
            if name in examples and key not in seen:
                seen.add(key)
                examples[name].append(
                    {
                        field: task.get(field)
                        for field in (
                            "template_id",
                            "turn_policy",
                            "slots",
                            "slots_initial",
                            "slot_updates",
                            "required_tools",
                        )
                    }
                )
    return examples


def _inventory(plans: dict[str, gate.TaskPlan]) -> dict[str, dict]:
    """What the gate could even ask, before anything is scored."""
    rows: dict[str, dict] = {}
    for plan in plans.values():
        for mutation in plan.mutations:
            row = rows.setdefault(
                mutation.operator,
                {"op_class": mutation.op_class, "mode": mutation.mode, "mutations": 0, "tasks": set()},
            )
            row["mutations"] += 1
            row["tasks"].add(str(plan.task["task_id"]))
    return {
        name: {**row, "tasks": len(row["tasks"])}
        for name, row in sorted(rows.items())
    }


def _print_arm(summary: dict) -> None:
    gold = summary["gold"]
    advisory = summary["advisory"]
    print(f"\n=== {summary['target']} ===")
    print(f"gold pass {gold['passed']}/{gold['instances']} ({gold['pass_rate']})")
    print(
        f"advisory-operator detection {advisory['detected']}/{advisory['trials']} "
        f"({advisory['detection_rate']}) -- higher means stricter than the spec"
    )
    print(f"{'operator':<28} {'trials':>6} {'det':>5} {'FA':>4} {'crash':>6} {'FA rate':>8}")
    for name, row in summary["by_operator"].items():
        if not row["trials"]:
            continue
        print(
            f"{name:<28} {row['trials']:>6} {row['detected']:>5} {row['false_accept']:>4} "
            f"{row['crash']:>6} {str(row['false_acceptance_rate']):>8}"
        )
    for label, key in (("all", "by_class"), ("strict", "by_class_strict")):
        for name, row in summary[key].items():
            print(
                f"[{label:<6}] {name:<19} {row['trials']:>6} {row['detected']:>5} "
                f"{row['false_accept']:>4} {row['crash']:>6} {str(row['false_acceptance_rate']):>8}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
