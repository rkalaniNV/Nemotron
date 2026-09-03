# A2 — reproduction run

**This is not a new arm.** It is A2 re-executed against the same A0 baseline, to answer one
question: does A2 reproduce? The committed `results/A2/` was not touched — `baseline_hashes.txt`
records its SHA-256 before the run, and `sha256sum -c baseline_hashes.txt` passed after it.

## Result: it reproduces

**4,852 metric fields compared; 2 differ, and neither is a measurement.**

| field | committed | re-run | why |
| --- | --- | --- | --- |
| `arm` | `a2` | `a2rerun` | the `--arm` flag |
| `llm.stats.cache_hits` | 426 | 375 | `--skip-generate`; 426 − 375 = 51 = 17 templates × 3 pool-generation calls |
| `llm.stats.calls_made` | 0 | **0** | no model call was made in either run |

Every measured value is bit-identical: all 12 rungs, both budgets, the intent-check table, the
diversity ladder, the equivalence verdicts, the funnel and the coverage figures.

| | budget 6 | budget 24 |
| --- | --- | --- |
| distinct masked surfaces, N = 1…20 | 17 · 23 · 22 · 28 · 29 · **31** | 17 · 33 · 41 · 55 · 67 · **77** |
| surfaces per template at N = 20 | **1.824** | **4.529** |
| verdict | `FROZEN` ×6 | `FROZEN` ×6 |
| published tasks on a substituted-intent surface | 0 · 0 · 2 · 1 · 3 · 2 | 0 · 0 · 3 · 2 · 4 · 3 |

`sweep24` was rebuilt as part of this run and also reproduces: 8 fields compared against the
committed `results/budget_sweep.json`, 0 differ (`tasks` 91, `published` 91).

This is the first demonstrated reproduction of A2. The method section claims every arm re-runs
from cache; until now no arm had shown it.

## Three things the re-run surfaced that a passing run does not

**1. A2 has an undeclared hard dependency on `sweep_budget.py`.** The first attempt died with
`error: baseline for budget 24 is missing at _generated/runs/sweep24/bfcl_ablation_sweep24` —
after generating every paraphrase pool and completing 40 pipeline runs, because nothing checks
for the baseline up front. `SUMMARY.md`'s reproduce block lists `sweep_budget.py 6 12 24` before
`run_a2.py`, so the ordering is right, but it is never stated as a dependency and `README.md`'s
"Running it" block omits `run_a2.py` entirely. A clean clone followed by the README cannot run A2.

**2. `--arm` does not isolate a run.** `_run_one` writes its variant runs as `a2_b{budget}_v{index}`,
hard-coded rather than derived from `args.arm`, and the report paths are the literals
`"a2_metrics.json"` and `"a2_report.md"`, which `common.result_path` routes to `results/A2/`
whatever `--arm` says. Running A2 twice side by side is therefore impossible without patching
`common.RESULTS` — which is what `driver.py` here does, and why it exists. This run did rewrite
`_generated/runs/a2_b*_v*`; those are gitignored, documented as regenerable, and regenerated
identically.

**3. The 426 / 375 discrepancy between the write-ups is not an error.** `results/A2/report.md`
records 426 and `experiments/a2.md:119` records 375; both are correct, and they describe different
invocations — 375 is the `--skip-generate` path, which is exactly what `a2.md` means by "on
re-run". The gap is provenance, not arithmetic: two documents record two different runs without
saying so.

## What this does not show

Reproducibility is not fitness. Every open issue with A2 reproduced too, exactly:

- **The arm still publishes intent-substituted tasks.** The checker is a measurement, not a gate
  (`run_a2.py:250`), and `flagged_variants` is only ever counted, never filtered. 8 of 12 rungs
  ship at least one.
- **The `FROZEN` verdict still gates on 2 of the 5 checks** — `task_ids` and `expected_tool_calls`
  only (`run_a2.py:325-329`), although the comment above it says the other four "still have to
  hold". `conversation_plans.equal` is computed, stored and `True` at every rung, and excluded
  from the verdict. Note also that `task_id` is hashed over
  `(pack_id, pack_version, template_id, fixture_refs, slot_bindings, variant_index)` and does not
  cover the surface, so neither gated check can fail under an intervention that only rewords the
  opening turn.
- **Variant assignment is still `seed % N`**: budget 6 / N=3 scores 22, below N=2's 23.
- **`metrics_version` is still absent** from the payload. *(Fixed after this run: `run_a2.py`
  now stamps it, so a future reproduction will differ from this artifact by exactly that field.
  The finding is left as written — this file records what the run found, not what is true now.)*

A reproduction says the pipeline is deterministic given the same code and cache. It says nothing
about whether the numbers it reproduces are the right ones to publish.

## Reproducing this

```bash
cd <repo root>
PYTHONPATH=src python3 bfcl_ablation/results/A2_rerun/driver.py    # A2 against the existing a0
PYTHONPATH=src python3 bfcl_ablation/results/A2_rerun/driver2.py   # + builds the budget-24 baseline
```

Both drivers redirect `common.RESULTS` to a scratch directory, so neither can overwrite
`results/A2/` or `results/budget_sweep.json`. Both carry an `if __name__ == "__main__"` guard,
which is load-bearing: `run_a2` uses `ProcessPoolExecutor` and the oracle uses
`mp.get_context("spawn")`, so an unguarded module body re-executes in every worker.

Edit the `SCRATCH` constant at the top of each driver to change where output lands.

| file | what |
| --- | --- |
| `metrics.json`, `report.md` | this run's output |
| `budget_sweep_24.json` | the rebuilt budget-24 baseline |
| `baseline_hashes.txt` | SHA-256 of the committed `results/A2/`, taken before the run |
| `driver.py`, `driver2.py` | the two invocations |
| `run.log` | the second run's console output |
