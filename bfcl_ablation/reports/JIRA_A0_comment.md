**Status: Done** — both AC met. Baseline under `bfcl_ablation/results/A0/`; metric definitions documented and versioned (`METRICS.md`, contract `1.0`).

## What A0 is

A measurement stage that reports properties of the **benchmark**, not of the run. The pipeline already tells us generation succeeded — replay was deterministic, fingerprints held, schemas matched — but nothing until now said what the benchmark actually contains.

**How it ran.** Pack `banking_vn` unmodified (9 tools, 17 templates, 6 categories, 9 turn policies), `tasks_per_category: 6`, `random_seed: 42`, no model involved, production pipeline unpatched. Produced 33 tasks. One config deviation: `lineage.policy: strict_separation` rather than `smoke_no_publication`, because the latter forces `gold_eligible: False` regardless of validation and would make the publish readout meaningless.

Definitions are pinned here because A1–A4 compare against these numbers — a silent change to *how* a metric is computed would later look like a *benchmark* change. Every `metrics.json` records its `metrics_version`.

## Metrics and results

**1. Authoring friction** — every line of every authored pack file plus the run config; blank and comment lines count, since excluding them flatters whichever arm is less commented.
→ **1642 lines**; 27.8 template lines each. Of these, **877 (53%) are irreducible ground truth** (`backend.py` 465, `assertions.py` 182, `tools.json` 162, `fixtures.json` 68). The reducible surface is the other 765.

**2. Joint (category x policy) distribution** — task count per cell. An empty cell is classified *derived, never declared*, so a gap cannot be hidden by calling a cell "structurally empty": a cell is structural only if no tool in the category's universe can satisfy the policy.
→ **15 of 54 cells populated** (35 unwritten, 4 structural). `single_turn` **54.5%**; `clarify_only`, `correction`, `dependent_call`, `missing_slot`, `multi_tool` have **one task each**. The mix is an artifact of category budgeting, not a target — budget is spent per category and round-robined across its templates.

**3. Slot and fixture coverage** — a fixture row counts as bound when its primary key appears in some task's `fixture_refs`; never-bound ids are listed explicitly.
→ **17 of 50 entities bound**. `vietqr_payments` 6/6, `disputes` 3/4, `accounts` 3/8, `transactions` 4/16, `cards` 1/4, `transfers` 0/4. Tool coverage **9/9**. Note 12 of the 50 rows are backend state that no slot binds, so the reachable ceiling is 38, not 50.

**4. Utterance diversity** — count of distinct *slot-masked* opening turns. Masking replaces each bound slot value with its slot name by exact substring match, longest first; no case or diacritic folding (the pack is Vietnamese, folding would merge real differences).
→ 33 tasks, 33 distinct raw turns, but only **17 distinct masked — exactly one per template**. `qr_payment` is 6 tasks and 1 sentence. The lexical-shortcut probe is reported as *not runnable* rather than as a number: it needs several phrasings per intent and there is one.

**5. Publish funnel** — rows surviving each stage (expand → state_machine → render → trace → schema → replay → raw → published), each with a per-stage drop-reason breakdown.
→ **33 → 33 at every stage. Nothing dropped.** Publish 100%, gold 100%, all rows tier `gold`.

**Supporting sweep.** `tasks_per_category` is the only generation knob, so I swept it 6 / 12 / 24: tasks 33 → 55 → 91, entities 17 → 25 → 32, **distinct sentences 17 → 17 → 17**, `single_turn` 54.5% → 49.1% → 50.5%.

## What this means

**Nothing is ever dropped, so `publish rate` and `gold` carry no quality information** — they read like quality scores and are throughput scores. That is a gap in what the pipeline measures, not a defect in this pack, and it is the reason A4 exists.

**Surface diversity is unreachable by configuration.** The sweep shows the benchmark has three axes — entity coverage, policy mix, surface diversity — and one knob, which reaches one of them. That is the quantitative case for A2 and also sizes it.

Consequences: (a) stop reporting publish/gold as quality until something measures assertion strictness; (b) the coverage spec must be a target, and needs a third dimension since (category x policy) budgeting alone still starves entity coverage; (c) set coverage targets against the reachable ceiling of 38, not 50.

## Caveats before quoting these numbers

- Structural emptiness is a heuristic, and circular for a category that has no templates yet.
- Fixture coverage does not yet separate "not covered" from "cannot be covered" (hence 17/38 vs 17/50).
- The drop-reason breakdown is implemented but has never emitted a non-empty bucket, since nothing drops. The all-zero funnel table is not evidence it works.
- n = 1 at the pack level: one pack, one domain, one language.

## Artifacts

`bfcl_ablation/reports/BFCL_Ablation_A0.docx` (full report, 14 tables + reproduction and artifact digests) · `results/METRICS.md` (definitions, versioned) · `results/A0/{report.md,metrics.json}` · `results/budget_sweep.json`

Reproduce, no install needed: `PYTHONPATH=src python3 bfcl_ablation/run_a0.py`

## Follow-ups (new tickets)

1. Derive structural emptiness without the circular tool universe — port back A3's declared-universe + backend-probed dependency edges.
2. Split fixture coverage into reachable vs unreachable.
