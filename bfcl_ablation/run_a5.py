#!/usr/bin/env python3
"""A5 — does rewording the request change the benchmark's verdict on a model?

Every arm below this one measures benchmark *content*. None runs a model against the
benchmark, so none can say whether a benchmark *conclusion* survives a change to the
benchmark. A2 established that an LLM can raise surface diversity without moving ground
truth; the open question it left is whether a target model scores the same on the new
wording. That is the question here, and it is the last item on the plan's list.

The design is paired. `task_id` is hashed over (pack, template, fixture refs, slot
bindings, variant index) and *not* over the surface, so A0's 33 tasks and A2's 33
paraphrased tasks carry identical ids: every task is its own control, and the statistic
is McNemar's exact test on the discordant pairs rather than a two-sample comparison
that would discard the pairing.

Two verdicts are scored per rollout and reported side by side. `ast_match` compares the
model's calls with `expected_tool_calls`; `assertion` asks the pack's own
`success_assertions`. A4 measured those assertions at 0.610 false acceptance on
argument-level corruptions, so the gap between the two columns is a readout in its own
right, not a redundancy.

    PYTHONPATH=src python3 bfcl_ablation/run_a5.py
    PYTHONPATH=src python3 bfcl_ablation/run_a5.py --a2-run a2_b6_v1   # a second wording
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common  # noqa: E402
from bfcl_ablation.measurement.metrics import METRIC_CONTRACT_VERSION  # noqa: E402
from bfcl_ablation.mutate import gate, inputs  # noqa: E402
from bfcl_ablation.target import score  # noqa: E402
from bfcl_ablation.target.client import TargetClient  # noqa: E402
from bfcl_ablation.target.rollout import Runner  # noqa: E402

A0_RUN = common.GENERATED / "runs" / "a0" / "bfcl_ablation_a0"
A0_CONFIG = common.GENERATED / "config_a0.yaml"
# v6 is the one variant index whose 33 paraphrases the A2 intent checker left entirely
# unflagged, so a wording effect measured here cannot be an intent shift wearing a
# wording's clothes. `--a2-run` swaps it for a second opinion.
DEFAULT_A2_RUN = "a2_b6_v6"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a2-run", default=DEFAULT_A2_RUN)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only, for a smoke run")
    args = parser.parse_args()

    a2_run = common.GENERATED / "runs" / args.a2_run / f"bfcl_ablation_{args.a2_run}"
    for path in (A0_RUN / "benchmark.parquet", a2_run / "benchmark.parquet"):
        if not path.exists():
            print(f"missing {common.rel(path)}; run run_a0.py and run_a2.py first")
            return 1

    config, pack = inputs.load_config_and_pack(A0_CONFIG)
    a0_rows = {r["task_id"]: r for r in common.read_parquet(A0_RUN / "benchmark.parquet")}
    a2_rows = {r["task_id"]: r for r in common.read_parquet(a2_run / "benchmark.parquet")}
    shared = sorted(set(a0_rows) & set(a2_rows))
    if args.limit:
        shared = shared[: args.limit]

    # The published row omits `slots`, which pack assertions read, so the instance table
    # is joined back in. Without it the assertion column would fail for reasons that
    # have nothing to do with the model.
    instances = {
        str(t["task_id"]): t for t in inputs.load_tasks(A0_RUN / "stage_cache")
    }

    reworded = sum(1 for t in shared if _opening(a0_rows[t]) != _opening(a2_rows[t]))
    print(f"[a5] {len(shared)} paired tasks, {reworded} with a different opening turn", flush=True)
    if reworded == 0:
        print("[a5] refusing to run: the two arms carry identical wording, so there is nothing to compare")
        return 1

    client = TargetClient()
    runner = Runner(
        backend_path=pack.paths.backend_path,
        fixtures=pack.fixtures or {},
        assertions_path=pack.paths.assertions_path,
        import_root=pack.paths.pack_root,
        runtime=gate.Runtime.from_config(config.oracle_runtime),
        worker=config.oracle_runtime.worker,
    )
    scorer = gate.Gate(
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

    results: dict[str, dict[str, dict[str, Any]]] = {"a0": {}, "a2": {}}
    trials: list[dict[str, Any]] = []
    started = time.time()

    for arm, rows in (("a0", a0_rows), ("a2", a2_rows)):
        for index, task_id in enumerate(shared, 1):
            row = rows[task_id]
            task = instances.get(str(task_id)) or dict(row)
            rollout = runner.run(
                client=client,
                task=task,
                arm=arm,
                instructions=_system(row),
                user_turns=_user_turns(row),
                tools=json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"],
            )
            expected = _expected(row)
            matched = score.ast_match(
                predicted=rollout.calls,
                expected=expected,
                call_order=str(row.get("call_order") or "strict"),
            )
            passed = _assertions_pass(scorer, human, task, rollout.calls)
            record = {
                "task_id": task_id,
                "arm": arm,
                "template_id": row.get("template_id"),
                "turn_policy": row.get("turn_policy"),
                "category": row.get("category"),
                "call_order": row.get("call_order"),
                "opening_turn": _opening(row),
                "expected_calls": score.canonical_calls(expected),
                "predicted_calls": score.canonical_calls(rollout.calls),
                "ast_match": matched,
                "assertion": passed,
                "model_turns": rollout.turns,
                "stop_reason": rollout.stop_reason,
                "tool_errors": rollout.tool_errors,
                "final_text": rollout.final_text[:400],
            }
            results[arm][task_id] = record
            trials.append(record)
            if index % 10 == 0 or index == len(shared):
                print(f"[a5] {arm}: {index}/{len(shared)} ({time.time() - started:.0f}s)", flush=True)

    payload = {
        "arm": "a5",
        "metrics_version": METRIC_CONTRACT_VERSION,
        "env": common.env_note(),
        "target_model": client.model,
        "endpoint": client.base_url,
        "pack": common.rel(pack.paths.pack_root),
        "wordings": {
            "a0": common.rel(A0_RUN / "benchmark.parquet"),
            "a2": common.rel(a2_run / "benchmark.parquet"),
        },
        "tasks_paired": len(shared),
        "tasks_reworded": reworded,
        "episodes_run": runner.episodes_run + scorer.episodes_run,
        "llm": client.stats(),
        "paired": {
            verdict: score.pair(results["a0"], results["a2"], verdict=verdict)
            for verdict in ("ast_match", "assertion")
        },
        "by_turn_policy": {
            verdict: score.by_group(results["a0"], results["a2"], key="turn_policy", verdict=verdict)
            for verdict in ("ast_match", "assertion")
        },
        "by_category": {
            verdict: score.by_group(results["a0"], results["a2"], key="category", verdict=verdict)
            for verdict in ("ast_match", "assertion")
        },
        "verdict_disagreement": _disagreement(trials),
        "definitions": {
            "ast_match": "model calls equal expected_tool_calls (name + arguments); order honoured unless call_order is 'any'",
            "assertion": "the pack's success_assertions accept the episode the model produced",
            "paired_agreement": "share of tasks where both wordings gave the same verdict",
            "mcnemar_p": "two-sided exact McNemar on discordant pairs; H0 = wording has no effect",
        },
    }

    path = common.dump_result("a5_metrics.json", payload)
    trials_path = common.dump_result("a5_trials.json", trials)
    report_path = common.result_path("a5_report.md")
    report_path.write_text(_render(payload), encoding="utf-8")
    print(f"\nwrote {common.rel(path)}")
    print(f"wrote {common.rel(trials_path)}")
    print(f"wrote {common.rel(report_path)}")
    print()
    print(_render(payload))
    return 0


def _system(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message["role"] == "system":
            return str(message["content"])
    return ""


def _user_turns(row: dict[str, Any]) -> list[str]:
    """Every user turn in order.

    The published `messages` holds the whole gold conversation, assistant calls and tool
    results included. Only the user side is replayed; handing the model the gold
    assistant turns would show it the answer.
    """
    return [str(m["content"]) for m in row["messages"] if m["role"] == "user"]


def _opening(row: dict[str, Any]) -> str:
    turns = _user_turns(row)
    return turns[0] if turns else ""


def _expected(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = row.get("expected_tool_calls") or []
    return sorted(
        (dict(call) for call in calls),
        key=lambda c: (int(c.get("call_group") or 0), int(c.get("position_in_group") or 0)),
    )


def _assertions_pass(
    scorer: gate.Gate, target: gate.Target, task: dict[str, Any], calls: list[dict[str, Any]]
) -> bool:
    """Run the pack's own assertions over the model's episode.

    A task with no assertions cannot be judged this way; it is recorded as False so it
    never silently counts as a pass, and the `verdict_disagreement` block is where that
    shows up.
    """
    names = list(task.get("success_assertions") or [])
    if not names:
        return False
    outcome = scorer.replay(task=task, calls=calls, names=names, target=target)
    return bool(outcome) and all(outcome.values())


def _disagreement(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the pack's assertions and the declared ground truth part company."""
    lenient = [t for t in trials if t["assertion"] and not t["ast_match"]]
    strict = [t for t in trials if t["ast_match"] and not t["assertion"]]
    return {
        "assertion_passed_ast_failed": len(lenient),
        "ast_passed_assertion_failed": len(strict),
        "examples_lenient": [
            {"task_id": t["task_id"], "arm": t["arm"], "expected": t["expected_calls"], "got": t["predicted_calls"]}
            for t in lenient[:5]
        ],
        "examples_strict": [
            {"task_id": t["task_id"], "arm": t["arm"], "expected": t["expected_calls"], "got": t["predicted_calls"]}
            for t in strict[:5]
        ],
    }


def _render(payload: dict[str, Any]) -> str:
    out: list[str] = [
        "# BFCL ablation — arm `a5` (cross-wording target-model evaluation)",
        "",
        f"Target model: `{payload['target_model']}`. Metric contract `{payload['metrics_version']}`.",
        f"{payload['tasks_paired']} paired tasks, {payload['tasks_reworded']} with a different "
        f"opening turn. {payload['episodes_run']} oracle episodes.",
        "",
        "A0 and A2 carry identical `task_id`s — the hash covers pack, template, fixture refs and",
        "slot bindings, not the surface — so each task is its own control and the test is paired.",
        "",
        "## 1. Headline",
        "",
        "| verdict | A0 accuracy | A2 accuracy | delta | paired agreement | discordant | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for verdict in ("ast_match", "assertion"):
        p = payload["paired"][verdict]
        out.append(
            f"| `{verdict}` | {p['accuracy_a0']:.3f} | {p['accuracy_a2']:.3f} | "
            f"{p['delta']:+.3f} | {p['paired_agreement']:.3f} | {p['discordant']} | {p['mcnemar_p']:.4f} |"
        )
    ast = payload["paired"]["ast_match"]
    out += [
        "",
        f"95% CI on A0 accuracy {ast['accuracy_a0_ci95']}, on A2 {ast['accuracy_a2_ci95']} (Wilson).",
        "",
        "`ast_match` is the headline: the model's calls equal `expected_tool_calls`. `assertion`",
        "is the same episode judged by the pack's own `success_assertions`, which A4 scored at",
        "0.610 false acceptance on argument-level corruptions — the gap between the two rows is",
        "that leniency priced on real model output.",
        "",
        "## 2. Contingency (ast_match)",
        "",
        "| | A2 correct | A2 wrong |",
        "| --- | ---: | ---: |",
        f"| **A0 correct** | {ast['contingency']['both_correct']} | {ast['contingency']['a0_only']} |",
        f"| **A0 wrong** | {ast['contingency']['a2_only']} | {ast['contingency']['neither']} |",
        "",
        "Off-diagonal cells are the wording effect. McNemar conditions on exactly those.",
        "",
        "## 3. Per turn policy (ast_match)",
        "",
        "| policy | n | A0 | A2 | delta | flipped down | flipped up |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["by_turn_policy"]["ast_match"].items():
        out.append(
            f"| `{name}` | {row['n']} | {row['accuracy_a0']:.2f} | {row['accuracy_a2']:.2f} | "
            f"{row['delta']:+.2f} | {row['flipped_down']} | {row['flipped_up']} |"
        )
    out += [
        "",
        "Most cells hold one task. A per-cell rate at n=1 is an anecdote with a decimal point;",
        "the counts are printed so no rate is read without its denominator.",
        "",
        "## 4. Where the two verdicts disagree",
        "",
        f"- assertions passed while the calls were wrong: **{payload['verdict_disagreement']['assertion_passed_ast_failed']}**",
        f"- calls were right while assertions failed: **{payload['verdict_disagreement']['ast_passed_assertion_failed']}**",
        "",
        "## 5. What this does not show",
        "",
        "- **One target model, and it is the generator's own family.** A2's paraphrases and this",
        "  model both come from `gpt-oss-120b`, so a model scoring well on its own family's",
        "  wording is a self-preference result as much as a robustness one. A second family is",
        "  the first thing to add.",
        "- **One paraphrase per task.** `--a2-run` selects one variant index; the result is the",
        "  effect of *that* wording, not of paraphrasing in general.",
        "- **Later user turns are replayed, not simulated.** A real user would react to what the",
        "  model actually said. Replaying the canned turns keeps a second model out of the",
        "  measurement path, at the cost of realism on the 8 multi-turn tasks.",
        "- **n = 33.** McNemar on a handful of discordant pairs has low power; a p above 0.05",
        "  here is 'not detected at this n', not 'no effect'.",
        "",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
