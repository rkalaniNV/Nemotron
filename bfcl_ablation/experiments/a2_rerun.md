# A2R — the reproduction run

**Question:** the method section claims every arm re-runs from cache and produces the same
benchmark. Had any arm ever shown it?

No. Until this run, reproducibility was a property the study asserted about itself. A2R is A2
re-executed against the same A0 baseline to test that assertion, and — more usefully — to find
out what a *passing* run cannot tell you.

```bash
PYTHONPATH=src python3 bfcl_ablation/results/A2_rerun/driver.py    # A2 against the existing a0
PYTHONPATH=src python3 bfcl_ablation/results/A2_rerun/driver2.py   # + rebuilds the budget-24 baseline
```

**This is not a new rung of the ladder.** It opens no degree of freedom. It is a control, and it
is filed here because the thing it controls for — "the numbers reproduce, therefore the numbers
are right" — is the most natural wrong inference a reader can draw from this study.

Both drivers redirect `common.RESULTS` to a scratch directory, so neither can overwrite
`results/A2/`. `baseline_hashes.txt` records the committed results' SHA-256 before the run;
`sha256sum -c` passed after it.

---

## Insights

### 1. It reproduces, exactly

**4,852 metric fields compared. 2 differ, and neither is a measurement.**

| field | committed | re-run | why |
| --- | --- | --- | --- |
| `arm` | `a2` | `a2rerun` | the `--arm` flag |
| `llm.stats.cache_hits` | 426 | 375 | `--skip-generate`; 426 − 375 = 51 = 17 templates × 3 pool-generation calls |
| `llm.stats.calls_made` | 0 | **0** | no model call was made in either run |

Every measured value is bit-identical: all 12 rungs, both budgets, the intent-check table, the
diversity ladder, the equivalence verdicts, the funnel and the coverage figures. `sweep24` was
rebuilt as part of the run and also reproduces — 8 fields against the committed
`results/budget_sweep.json`, 0 differ.

*So what:* the claim is now demonstrated rather than assumed, for one arm. It is worth noting
which arm: A2 is the one that calls a model 375 times. Its determinism rests entirely on the
disk cache, which is the weakest reproducibility story in the study, and it still held.

### 2. Reproducibility and validity are independent axes

Every open defect in A2 reproduced perfectly along with its numbers:

- **The arm still publishes intent-substituted tasks.** The checker is a measurement, not a gate
  (`run_a2.py:250`); `flagged_variants` is counted, never filtered. 8 of 12 rungs ship at least one.
- **Variant assignment is still `seed % N`**: budget 6 / N=3 scores 22, below N=2's 23.
- **`metrics_version` was still absent** from the payload. (Fixed since; A2 now stamps it.)

*So what:* **a green reproduction is not a green light.** This is the sentence the arm exists to
produce. A re-run says the pipeline is deterministic given the same code and cache; it says
nothing about whether the numbers it reproduces are the right ones to publish.

### 3. The sharpest instance: a verdict that cannot fail

`FROZEN` came back 12 of 12 — which is exactly what a verdict incapable of failing would do.

`run_a2.py:325-329` gates the verdict on `task_ids` and `expected_tool_calls` only. `task_id` is
hashed over `(pack_id, pack_version, template_id, fixture_refs, slot_bindings, variant_index)` and
**does not cover the surface**, and A2 changes *only* the surface. `expected_tool_calls` derives
from frozen tools and slots. **Neither gated check is structurally capable of failing under this
intervention.** Twelve `FROZEN` verdicts carry zero bits about whether ground truth moved.

The check that would carry signal — `conversation_plans.equal` — is computed, stored, and `True`
at every rung. It is excluded from the verdict expression, despite the comment directly above it
saying the other four checks "still have to hold".

*So what:* this is a different failure from a weak check. It is a check that cannot fire,
reported as though it had passed. A0 found gates that never fire on real data; this is a gate that
could not fire on any data.

### 4. Two defects only a re-run could find

- **A2 has an undeclared hard dependency on `sweep_budget.py`.** The first attempt died with
  `baseline for budget 24 is missing` — *after* generating every paraphrase pool and completing 40
  pipeline runs, because nothing checks for the baseline up front. `SUMMARY.md`'s reproduce block
  happens to list `sweep_budget.py` first, so the ordering is right by accident; it is never stated
  as a dependency, and `README.md`'s "Running it" block omitted `run_a2.py` entirely. **A clean
  clone following the README cannot run A2.**
- **`--arm` does not isolate a run.** `_run_one` writes variant runs as `a2_b{budget}_v{index}`,
  hard-coded rather than derived from `args.arm`, and the report paths are the literals
  `"a2_metrics.json"` / `"a2_report.md"`, which `common.result_path` routes to `results/A2/`
  whatever `--arm` says. Running A2 twice side by side is impossible without patching
  `common.RESULTS` — which is what `driver.py` does, and why it exists.

*So what:* neither is visible from a passing run, a code review, or the report. Both are the kind
of defect that only appears when someone tries to run the thing a second time, somewhere else.

### 5. A provenance discrepancy that is not an error

`results/A2/report.md` records 426 cache hits and `experiments/a2.md:119` records 375. Both are
correct: 375 is the `--skip-generate` path, which is what `a2.md` means by "on re-run". The gap is
**provenance, not arithmetic** — two documents recording two different invocations without saying
so.

*So what:* small, but it is the same class of defect as the stale A4 write-up found later in the
study: a number in prose with no link back to the run that produced it.

---

## Full numbers

| | value |
| --- | --- |
| metric fields compared | 4,852 |
| fields differing | 2 (`arm`, `llm.stats.cache_hits`) |
| measurements differing | **0** |
| model calls made | 0 (both runs served entirely from cache) |
| `sweep24` fields compared | 8, 0 differ (`tasks` 91, `published` 91) |
| committed `results/A2/` after the run | SHA-256 unchanged |

## Limitations

- **One arm, one machine, one cache.** This shows A2 is deterministic given the same code and the
  same `_generated/llm_cache`. It does not show the cache could be *rebuilt* to the same content:
  that would need the vLLM server to return identical completions, and the cache key does not
  include `system_fingerprint`, so a rebuilt server would reuse stale answers undetected.
- **It is a control, not a rung.** No degree of freedom is opened, so nothing here says anything
  about the pack, the model or the benchmark.
- **`metrics_version` was absent when this ran.** A2 stamps it now, so a future re-run will differ
  from the artifact this one compared against — by exactly that field.

## Artifacts

`results/A2_rerun/REPRODUCTION.md` (the run's own write-up), `metrics.json`, `report.md`,
`budget_sweep_24.json` (the rebuilt budget-24 baseline), `baseline_hashes.txt` (SHA-256 of the
committed `results/A2/`, taken before the run), `driver.py` / `driver2.py` (the two invocations),
`run.log`
