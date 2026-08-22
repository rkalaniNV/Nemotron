#!/usr/bin/env python3
"""A3 — LLM task generation (semantics), under a system-controlled policy distribution.

    coverage spec  ->  sampled (category, policy) cell   [system, seeded, never the model]
                            |
                            v
                    LLM proposes task semantics          [tools, records, slots, wording]
                            |
      schema gate -> compile gate -> plan gate           [drop before the pack, not after]
                            |
                    A1 rehydration -> oracle validation  [production per-template contract]
                            |
                    unmodified pipeline -> replay -> assertions -> publish

Everything the model may vary is semantic. The shape of the benchmark — which cells
exist, which are structurally impossible, how many proposals each cell gets — is decided
before the first prompt. That is the only arrangement under which selection bias is
measurable: with a free policy choice, a model that avoids `correction` returns a
benchmark that looks healthy and tests less, and the arm reports its own blind spot as a
finding.

The pipeline is never patched. `generate_bfcl` refuses to run at all on a pack whose
oracle validation failed, so accept/drop has to be resolved *before* generation — which
is why the gates below call the pipeline's own `compile_milestones`, `build_plan` and
`prepare_bfcl` rather than reimplementing their judgement.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bfcl_ablation import common, llm  # noqa: E402
from bfcl_ablation.measurement import metrics, report  # noqa: E402
from bfcl_ablation.propose import bias, generate, probe, sampler, validate  # noqa: E402
from bfcl_ablation.simplify import derive, rehydrate, shrink  # noqa: E402

# Which drop bucket an oracle-validation failure counts against. The validator names a
# reason per template; these map its vocabulary onto the arm's.
VALIDATION_BUCKETS = {
    "missing_tool": "schema_invalid",
    "template_without_success_assertion": "schema_invalid",
    "invalid_conversation_plan": "plan_invalid",
    "representative_trace_schema_mismatch": "expected_trace_invalid",
    "representative_surface_guard_violation": "surface_guard_violation",
    "representative_generation_failed": "generation_failed",
    "representative_replay_failed": "replay_failed",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _semantic_fingerprint(template: dict[str, Any]) -> str:
    """Identify a proposal by what it tests, not by what it is called.

    Two proposals with different `template_id` but the same category, policy, tools,
    slot sources and opening sentence produce the same tasks with different ids. Counting
    them as two would inflate both the accept rate and the coverage.
    """
    return _canonical(
        {
            "category": template.get("category"),
            "turn_policy": template.get("turn_policy"),
            "required_tools": template.get("required_tools"),
            "slots": {
                name: {"source": slot.get("source"), "filter": slot.get("filter")}
                for name, slot in sorted((template.get("slots") or {}).items())
            },
            "user_turn": (template.get("user_turn_templates") or {}).get("vi"),
        }
    )


def _validation_failures(report_json: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Collect per-template failures from an oracle validation report.

    Only the first failure per template is kept: a template is dropped once, and the
    reason that dropped it is the first thing that went wrong, not the cascade after it.
    """
    failures: dict[str, dict[str, str]] = {}
    for check in list(report_json.get("checks") or []) + list(report_json.get("extra_checks") or []):
        if check.get("status") != "fail":
            continue
        for failure in check.get("failures") or []:
            template_id = failure.get("template_id")
            if not template_id or template_id in failures:
                continue
            reason = str(failure.get("reason") or "unspecified")
            bucket = VALIDATION_BUCKETS.get(reason, "slot_source_invalid")
            if reason == "representative_replay_failed" and failure.get("replay_reason") == "assertion_failed":
                bucket = "assertion_failed"
            failures[str(template_id)] = {
                "check": str(check.get("name")),
                "reason": reason,
                "bucket": bucket,
                "detail": str(failure.get("detail") or failure.get("failures") or "")[:400],
            }
    return failures


def _budget_for(templates: list[dict[str, Any]], floor: int) -> int:
    """The per-category task budget the pipeline will accept for this template set."""
    per_category = Counter(str(template.get("category")) for template in templates)
    return max(floor, max(per_category.values(), default=floor))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="a3")
    parser.add_argument("--pack", type=Path, default=common.PACK_A0)
    parser.add_argument("--budget", type=int, default=56, help="proposals requested from the model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-validation-rounds", type=int, default=6)
    args = parser.parse_args()

    common.bootstrap()
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import prepare_bfcl

    source = args.pack.resolve()
    packs = common.GENERATED / "packs"
    base = packs / f"{args.arm}_base"
    seed_full = packs / f"{args.arm}_seed_full"
    authored = packs / f"{args.arm}_authored"
    full = packs / f"{args.arm}_full"

    client = llm.LLMClient(concurrency=args.concurrency)
    endpoint = llm.probe(client)
    print(f"[{args.arm}] llm ok: {endpoint['model']}", flush=True)

    # 1. The pack minus its tasks. A3 replaces task_templates.yaml and nothing else, so
    #    every non-task input is A1's, unchanged.
    print(f"[{args.arm}] shrinking {common.rel(source)} to its A1 authored form ...", flush=True)
    shrink_report = shrink.shrink_pack(source, base)
    rehydrate.rehydrate_pack(base, seed_full)

    tools_raw = json.loads((base / "tools.json").read_text(encoding="utf-8"))
    fixtures = json.loads((base / "fixtures.json").read_text(encoding="utf-8"))
    tools = derive.tool_index(tools_raw)
    base_manifest = yaml.safe_load((base / "manifest.yaml").read_text(encoding="utf-8")) or {}
    assertions_source = (base / "assertions.py").read_text(encoding="utf-8")
    assertion_names = set(re.findall(r"^def (assert_\w+)", assertions_source, flags=re.M))
    # Every collection, not only the ones A0's templates happened to reach: a proposal is
    # free to bind a record A0 never used, and the prompt has to be able to name it.
    primary_keys = derive.best_effort_primary_keys(fixtures)
    absent_ids = derive.derive_absent_ids(fixtures, primary_keys, set(primary_keys))
    canonical_replies = base_manifest.get("user_simulator_templates") or rehydrate.DEFAULT_CANONICAL_REPLIES
    languages = [str(x) for x in (base_manifest.get("languages") or ["vi"])]

    # 2. What the backend actually returns, so `dependent_call` feasibility is an edge
    #    and not a guess about tool counts.
    print(f"[{args.arm}] probing tool result shapes ...", flush=True)
    probed = probe.probe_tool_results(
        pack_dir=seed_full,
        cases=probe.load_validation_cases(seed_full),
        clock_iso=str(base_manifest.get("clock") or "2026-03-02T09:00:00+07:00"),
        seed=args.seed,
    )
    edges = probe.dependency_edges(probed, tools)

    # 3. The coverage target, spent by the sampler and not by the model.
    spec = sampler.build_spec(tools=tools, edges=edges, budget=args.budget, seed=args.seed)
    assignments = spec.assignments
    print(
        f"[{args.arm}] coverage spec: {len(spec.feasible)}/{len(spec.cells)} cells feasible, "
        f"{sum(c.target for c in assignments)} proposals over {len(assignments)} cells",
        flush=True,
    )

    # 4. Proposals.
    context = {
        "tools_raw": tools_raw,
        "fixtures": fixtures,
        "primary_keys": primary_keys,
        "absent_ids": absent_ids,
        "assertions_source": assertions_source,
        "edges": edges,
    }
    print(f"[{args.arm}] proposing ...", flush=True)
    proposals = generate.propose(client, assignments, context)
    print(f"[{args.arm}] llm stats: {client.stats()}", flush=True)

    # 5. Gates. Order matters: a proposal is charged to the first gate that refuses it.
    outcomes: list[dict[str, Any]] = []
    accepted: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    fingerprints: dict[str, str] = {}

    for item in proposals:
        record = {
            "category": item["category"],
            "policy": item["policy"],
            "slot_index": item["slot_index"],
            "template_id": None,
            "status": "dropped",
            "stage": "propose",
            "bucket": "no_proposal",
            "detail": "model returned no proposal for this slot",
        }
        if item["raw"] is None:
            outcomes.append(record)
            continue
        record["template_id"] = (item["raw"] or {}).get("template_id") if isinstance(item["raw"], dict) else None
        try:
            template = validate.validate_proposal(
                item["raw"],
                category=item["category"],
                policy=item["policy"],
                tools=tools,
                fixtures=fixtures,
                assertions=assertion_names,
                seen_ids=seen_ids,
            )
        except validate.Rejected as error:
            record.update(stage="schema", bucket=error.bucket, detail=error.detail)
            outcomes.append(record)
            continue

        template_id = str(template["template_id"])
        record["template_id"] = template_id
        try:
            validate.compile_and_plan(
                template, tools=tools, canonical=canonical_replies, languages=languages
            )
        except validate.Rejected as error:
            record.update(stage="compile", bucket=error.bucket, detail=error.detail)
            outcomes.append(record)
            continue

        fingerprint = _semantic_fingerprint(template)
        if fingerprint in fingerprints:
            record.update(
                stage="dedupe",
                bucket="duplicate_task_content",
                detail=f"same tools, slots and opening sentence as {fingerprints[fingerprint]}",
            )
            outcomes.append(record)
            continue

        seen_ids.add(template_id)
        fingerprints[fingerprint] = template_id
        accepted[template_id] = template
        record.update(status="accepted", stage="accepted", bucket=None, detail="")
        outcomes.append(record)

    # Only accepted records: a rejected proposal may have reused an id, and a later
    # validation drop must not be charged to it.
    by_id = {str(r["template_id"]): r for r in outcomes if r["status"] == "accepted"}
    print(
        f"[{args.arm}] local gates: {len(accepted)}/{len(proposals)} survive "
        f"({dict(Counter(r['bucket'] for r in outcomes if r['bucket']))})",
        flush=True,
    )

    # 6. Oracle validation, which is where the backend gets a say. A failing template is
    #    dropped and the pack revalidated: the validator skips its expensive checks once
    #    a cheap one fails, so one round cannot see every defect.
    rounds: list[dict[str, Any]] = []
    validation_report: dict[str, Any] = {}
    for round_index in range(args.max_validation_rounds):
        templates = [accepted[key] for key in sorted(accepted)]
        if not templates:
            break
        generate.write_authored_pack(base=base, target=authored, templates=templates)
        rehydrate.rehydrate_pack(authored, full)
        budget = _budget_for(templates, floor=6)
        config_path = common.write_config(
            arm=f"{args.arm}v",
            manifest_path=full / "manifest.yaml",
            output_dir=common.GENERATED / "runs" / f"{args.arm}v",
            extra_allowed_roots=(common.GENERATED,),
            overrides={"task_generation": {"tasks_per_category": budget}},
        )
        report_path = Path(prepare_bfcl(config_path))
        validation_report = json.loads(report_path.read_text(encoding="utf-8"))
        failures = _validation_failures(validation_report)
        gold = bool(validation_report.get("gold_eligible"))
        rounds.append(
            {
                "round": round_index,
                "templates": len(templates),
                "tasks_per_category": budget,
                "gold_eligible": gold,
                "tier": validation_report.get("tier"),
                "template_failures": len(failures),
                "unattributed_failures": [
                    {"check": check.get("name"), "failures": check.get("failures")}
                    for check in list(validation_report.get("checks") or [])
                    + list(validation_report.get("extra_checks") or [])
                    if check.get("status") == "fail"
                    and not any(f.get("template_id") for f in check.get("failures") or [])
                ],
            }
        )
        print(
            f"[{args.arm}] validation round {round_index}: {len(templates)} templates, "
            f"gold={gold}, per-template failures={len(failures)}",
            flush=True,
        )
        if gold:
            break
        if not failures:
            print(f"[{args.arm}] validation failed with no attributable template; stopping", flush=True)
            break
        for template_id, failure in failures.items():
            accepted.pop(template_id, None)
            record = by_id.get(template_id)
            if record is not None:
                record.update(
                    status="dropped",
                    stage="oracle_validation",
                    bucket=failure["bucket"],
                    detail=f"{failure['check']}: {failure['reason']}: {failure['detail']}",
                )

    if not validation_report.get("gold_eligible"):
        print(f"[{args.arm}] pack never reached gold; refusing to report generation numbers", file=sys.stderr)
        common.dump_result(
            f"{args.arm}_metrics.json",
            {
                "arm": args.arm,
                "status": "failed_validation",
                "coverage_spec": spec.as_dict(),
                "outcomes": outcomes,
                "validation_rounds": rounds,
            },
        )
        return 1

    # 7. Generation, on the accepted set only.
    templates = [accepted[key] for key in sorted(accepted)]
    budget = _budget_for(templates, floor=6)
    print(f"[{args.arm}] generating from {len(templates)} accepted templates ...", flush=True)
    result = common.run_arm(
        arm=args.arm,
        pack_dir=full,
        extra_allowed_roots=(common.GENERATED,),
        overrides={"task_generation": {"tasks_per_category": budget}},
    )
    tables = common.load_stage_tables(result)

    payload = metrics.measure(
        arm=args.arm,
        tables=tables,
        pack_dir=full,
        loc=common.count_authored_lines(authored, result.config_path),
        run_manifest=common.read_json(result.run_manifest),
        normalized_templates=result.stage_cache / "task_templates_normalized.yaml",
        declared_universe={k: set(v) for k, v in sampler.CATEGORY_TOOLS.items()},
        dependency_edges=edges,
        declared_policies=list(sampler.POLICIES),
    )

    # 8. Post-generation attribution. A template whose representative instance passed can
    #    still lose sibling instances, and that is a different defect from a template that
    #    never worked.
    per_template: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "replay_failed": 0})
    replay_reasons: Counter = Counter()
    for row in tables["replay_validated_tasks"]:
        entry = per_template[str(row.get("template_id"))]
        entry["tasks"] += 1
        if row.get("valid") is not True:
            entry["replay_failed"] += 1
            replay_reasons[str(row.get("reason"))] += 1
    partial = {
        template_id: entry for template_id, entry in sorted(per_template.items()) if entry["replay_failed"]
    }

    accepted_templates = [accepted[key] for key in sorted(accepted)]
    baseline_templates = yaml.safe_load((source / "task_templates.yaml").read_text(encoding="utf-8")) or []

    achieved = Counter(
        (str(t.get("category")), str(t.get("turn_policy"))) for t in accepted_templates
    )
    coverage_rows = []
    for cell in spec.cells:
        got = achieved.get((cell.category, cell.policy), 0)
        coverage_rows.append(
            {
                "category": cell.category,
                "policy": cell.policy,
                "feasible": cell.feasible,
                "structural_reason": cell.reason,
                "target": cell.target,
                "achieved": got,
                "met": got >= cell.target,
            }
        )
    feasible_rows = [row for row in coverage_rows if row["feasible"]]

    payload["proposal"] = {
        "budget": args.budget,
        "proposals_requested": sum(cell.target for cell in assignments),
        "proposals_returned": sum(1 for item in proposals if item["raw"] is not None),
        "accepted": len(accepted),
        "accept_rate": round(len(accepted) / max(len(proposals), 1), 4),
        "drop_buckets": dict(sorted(Counter(r["bucket"] for r in outcomes if r["bucket"]).items())),
        "drop_stages": dict(sorted(Counter(r["stage"] for r in outcomes if r["status"] == "dropped").items())),
        "llm": {**client.stats(), "model": endpoint["model"], "cells_called": len(assignments)},
        "outcomes": outcomes,
    }
    payload["coverage_spec"] = spec.as_dict()
    payload["coverage_achieved"] = {
        "cells_feasible": len(feasible_rows),
        "cells_covered": sum(1 for row in feasible_rows if row["achieved"] > 0),
        "cells_target_met": sum(1 for row in feasible_rows if row["met"]),
        "cells_structural_empty": len(coverage_rows) - len(feasible_rows),
        "coverage_rate": round(
            sum(1 for row in feasible_rows if row["achieved"] > 0) / max(len(feasible_rows), 1), 4
        ),
        "cells": coverage_rows,
    }
    payload["selection_bias"] = {
        "tool_choice": bias.tool_choice(accepted_templates, sampler.CATEGORY_TOOLS),
        "entity_choice": bias.entity_choice(
            tables["task_instances"], fixtures, primary_keys
        ),
        "literal_choice": bias.literal_choice(accepted_templates),
        "assertion_choice": bias.assertion_choice(accepted_templates, sorted(assertion_names)),
        "vacuous_gold": bias.vacuous_gold(accepted_templates, tools),
        "failure_bias": bias.failure_bias(outcomes),
        "vs_a0": bias.compare_with_baseline(accepted_templates, baseline_templates),
    }
    payload["validation_rounds"] = rounds
    payload["post_generation"] = {
        "templates_with_failed_instances": partial,
        "replay_reasons": dict(sorted(replay_reasons.items())),
    }
    payload["backend_probe"] = {
        "tools_probed": sorted(name for name, entry in probed.items() if entry.get("probed")),
        "tools_unprobed": sorted(name for name, entry in probed.items() if not entry.get("probed")),
        "dependency_edges": edges,
    }
    payload["shrink"] = {
        "a0_templates_discarded": len(
            yaml.safe_load((source / "task_templates.yaml").read_text(encoding="utf-8")) or []
        ),
        "validation_cases_authored_after_shrink": shrink_report["validation_cases_authored_after"],
        "note": "A0's templates are discarded; only the non-task half of the A1 pack is reused.",
    }

    common.dump_result(f"{args.arm}_metrics.json", payload)
    text = _render(payload, arm=args.arm)
    common.result_path(f"{args.arm}_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _names(values: list[str]) -> str:
    """Render a list of identifiers inline, without a code span standing in for nothing."""
    return "`" + "`, `".join(values) + "`" if values else "nothing"


def _render(payload: dict[str, Any], *, arm: str) -> str:
    proposal = payload["proposal"]
    coverage = payload["coverage_achieved"]
    selection = payload["selection_bias"]
    funnel = payload["funnel"]

    md = [report.render(payload), "## 6. Proposal accept/drop", ""]
    md.append(
        f"{proposal['proposals_requested']} proposals requested over "
        f"{proposal['llm']['cells_called']} cells; {proposal['proposals_returned']} returned; "
        f"**{proposal['accepted']} accepted ({100 * proposal['accept_rate']:.1f}%)**."
    )
    md += [
        "",
        "The authored line count in section 1 is the size of a pack the model wrote, not "
        "human friction: no person authored a task in this arm. It is comparable with A0's "
        "and A1's line counts only as a measure of how much pack a given number of tasks "
        "costs.",
    ]
    md += ["", "| drop bucket | proposals |", "| --- | ---: |"]
    for bucket, count in proposal["drop_buckets"].items():
        md.append(f"| {bucket} | {count} |")
    md += ["", "| gate | dropped |", "| --- | ---: |"]
    for stage, count in proposal["drop_stages"].items():
        md.append(f"| {stage} | {count} |")

    md += ["", "## 7. Coverage against the spec", ""]
    md.append(
        f"{coverage['cells_feasible']} feasible cells, {coverage['cells_structural_empty']} declared "
        f"structurally empty. {coverage['cells_covered']} feasible cells covered "
        f"({100 * coverage['coverage_rate']:.1f}%), {coverage['cells_target_met']} met their target."
    )
    missed = [row for row in coverage["cells"] if row["feasible"] and row["achieved"] == 0]
    if missed:
        md += ["", "Feasible cells the model could not fill:", ""]
        for row in missed:
            md.append(f"- `{row['category']}` x `{row['policy']}` (target {row['target']})")
    structural = [row for row in coverage["cells"] if not row["feasible"]]
    if structural:
        md += ["", "Cells declared structurally empty, with the claim that makes them so:", ""]
        for row in structural:
            md.append(f"- `{row['category']}` x `{row['policy']}` — {row['structural_reason']}")

    md += ["", "## 8. Selection bias", "", "### Tool choice, against uniform within each category", ""]
    tool = selection["tool_choice"]
    md += ["| tool | observed | expected | obs/exp |", "| --- | ---: | ---: | ---: |"]
    for name, expected in tool["expected_uniform_within_category"].items():
        observed = tool["observed"].get(name, 0)
        ratio = tool["ratio_observed_over_expected"].get(name)
        md.append(f"| {name} | {observed} | {expected} | {ratio if ratio is not None else '-'} |")
    md += [
        "",
        f"Pearson statistic against the conditional-uniform null: "
        f"**{tool['chi_square_vs_conditional_uniform']}**. Tools never required: "
        f"{_names(tool['tools_never_required'])}.",
        "",
        "### Accept rate by policy — bias that survives a controlled sampler",
        "",
        "| policy | proposed | accepted | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for policy, row in selection["failure_bias"]["by_policy"].items():
        md.append(f"| {policy} | {row['proposed']} | {row['accepted']} | {row['accept_rate']} |")
    md += [
        "",
        f"Spread between the easiest and hardest policy: "
        f"**{selection['failure_bias']['accept_rate_spread']}**.",
        "",
        "### Entity choice",
        "",
        f"{selection['entity_choice']['entities_bound']}/{selection['entity_choice']['entities_total']} "
        f"fixture rows bound ({100 * (selection['entity_choice']['coverage'] or 0):.1f}%).",
        "",
        "| collection | rows | bound | TVD from uniform |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in selection["entity_choice"]["by_collection"].items():
        md.append(f"| {name} | {row['rows']} | {row['rows_bound']} | {row['tvd_from_uniform']} |")

    vacuous = selection["vacuous_gold"]
    md += [
        "",
        "### What the oracle could not check",
        "",
        f"{vacuous['unfalsifiable_count']} of {payload['proposal']['accepted']} accepted templates "
        f"({100 * (vacuous['unfalsifiable_share_of_accepted'] or 0):.1f}%) are clarify_only or "
        "irrelevant. Their expected trace is empty by construction and their only available "
        "assertion, `assert_no_tool_called`, passes exactly when the trace is empty — so replay, "
        "determinism and assertions all succeed regardless of what the request says. Their accept "
        "rate is not evidence that the model got them right.",
        "",
    ]
    if vacuous["answerable_but_declined"]:
        md += [
            "Of those, these declare a tool whose every required parameter the customer already "
            "stated, so the request was answerable and the gold behaviour is wrong:",
            "",
        ]
        for row in vacuous["answerable_but_declined"]:
            md.append(f"- `{row['template_id']}` ({row['policy']}) — answerable by `{row['answerable_by']}`")
        md.append("")

    md += ["", "### Against A0's human-authored mix", ""]
    for field, row in selection["vs_a0"].items():
        md.append(
            f"- `{field}`: total variation distance A3 vs A0 = **{row['tvd_a3_vs_a0']}**; "
            f"only A3 uses {_names(row['in_a3_only'])}; "
            f"only A0 uses {_names(row['in_a0_only'])}"
        )

    md += [
        "",
        "## 9. Does the pack still reach gold",
        "",
        f"Validation rounds: {len(payload['validation_rounds'])}. Final tier "
        f"`{payload['validation_rounds'][-1]['tier']}`, gold_eligible="
        f"{payload['validation_rounds'][-1]['gold_eligible']}.",
        "",
        f"Published {funnel['published']}/{funnel['expanded']} expanded tasks "
        f"(publish rate {100 * funnel['publish_rate']:.1f}%), gold rows {funnel['gold_rows']} "
        f"({100 * funnel['gold_rate']:.1f}%).",
        "",
    ]
    partial = payload["post_generation"]["templates_with_failed_instances"]
    if partial:
        md += ["Templates whose representative instance passed but whose siblings did not:", ""]
        for template_id, entry in partial.items():
            md.append(f"- `{template_id}`: {entry['replay_failed']}/{entry['tasks']} instances failed replay")
        md.append("")
    else:
        md += ["No template lost an instance after validation.", ""]
    return "\n".join(md)


if __name__ == "__main__":
    raise SystemExit(main())
