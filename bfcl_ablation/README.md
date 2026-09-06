# BFCL Oracle-Pack ablation

How much of an Oracle Pack can be removed without weakening the benchmark?

This directory implements seven rungs of the ablation ladder, one reproduction control,
and an independent A7 quality gate. A0–A6 run the **unmodified** production pipeline
(`runtime/benchmark_families/bfcl`) or audit its oracle. An arm is defined by the pack
and config it feeds in, never by a patch to the generator — a patched generator would
measure the patch instead of the pack. A7 is not another optimization rung: it audits
whether the frozen A0–A6 evidence supports the study's claims and a public release.

```
A0  human baseline               measure the current state         no model
A1  deterministic simplification  no LLM                           no model
A2  LLM surface generation        wording only                     gpt-oss-120b
A3  LLM task generation           semantics                        gpt-oss-120b
A4  LLM assertions                last                             gpt-oss-120b
A5  target-model evaluation       does a conclusion survive A2?     gpt-oss-120b
A6  backend mutation gate         is the oracle falsifiable?        no model
A7  independent quality gate      audit frozen A0-A6 evidence       no model
```

A0–A4 measure the benchmark's *content*. **A5 is the only arm that measures a model on it**,
and it closes a loop: A2 showed wording can change without ground truth moving, and A5 asks
whether the *score* moves anyway. **A6 turns the question on the oracle itself** — it corrupts
`backend.py` and asks whether anything in the pack notices.

**A7 separates study validity from release readiness.** It recomputes denominators
from stored trials where possible, imports versioned human labels for semantic claims,
and returns `INCONCLUSIVE` rather than inventing evidence when labels or full-layer
outcomes are missing.

`results/A2_rerun/` is a **control**, not a rung: A2 re-executed to show it reproduces, and to
show that every one of its known defects reproduces with it. See
[experiments/a2_rerun.md](experiments/a2_rerun.md).

**Per-arm findings live in [experiments/](experiments/)** — one document per rung, insights
first. This file is the methodology and the how-to-run.

A2, A3, A4 and A5 call a local vLLM server (`openai/gpt-oss-120b` at
`http://127.0.0.1:8000/v1`, overridable via `BFCL_ABLATION_LLM_URL` and
`BFCL_ABLATION_LLM_MODEL`). Every call is disk-cached under `_generated/llm_cache`, so a
re-run reproduces the same benchmark and the cache doubles as the record of what the model was
asked and what it answered.

## Running it

No install required; the BFCL family imports with `pyarrow`, `pydantic`, `pyyaml`
and `rich` alone.

```bash
cd <repo root>
PYTHONPATH=src python3 bfcl_ablation/run_a0.py     # baseline, ~1 min
PYTHONPATH=src python3 bfcl_ablation/run_a1.py     # simplify + verify, ~3 min
PYTHONPATH=src python3 bfcl_ablation/sweep_budget.py 6 12 24   # REQUIRED before run_a2.py
PYTHONPATH=src python3 bfcl_ablation/run_a2.py     # paraphrase ladder, 40 pipeline runs, ~25 min
PYTHONPATH=src python3 bfcl_ablation/run_a3.py     # sampled cells + LLM task proposals, ~20 min cold
PYTHONPATH=src python3 bfcl_ablation/run_a4.py     # mutation gate + LLM assertions, ~8 min cold
PYTHONPATH=src python3 bfcl_ablation/run_a5.py     # target model on A0 vs A2 wording, ~5 min cold
PYTHONPATH=src python3 bfcl_ablation/run_a6.py     # 151 backend mutants through every check, ~35 min
PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py   # artifact-only meta-audit, no model/pipeline
```

`run_a2.py` has a hard dependency on `sweep_budget.py`: it needs the budget-24 baseline at
`_generated/runs/sweep24/`, and it does not check for it until after generating every paraphrase
pool and completing 40 pipeline runs. The reproduction control found this the expensive way —
see [experiments/a2_rerun.md](experiments/a2_rerun.md).

`run_a5.py` needs A0 *and* an A2 variant run on disk; `--a2-run a2_b6_v1` selects a different
paraphrase. Its tool calls go through `/v1/responses`, because `/chat/completions` on a vLLM
started without `--enable-auto-tool-choice --tool-call-parser openai` returns `tool_calls: null`
and would silently score every task as "called nothing".

`run_a4.py --skip-llm` runs the mutation gate alone and needs no model. It reads
A0's `stage_cache`, so `run_a0.py` has to have run first.

`run_a1.py` exits non-zero if A1 is not equivalent to A0, so it works as a CI
regression test on auto-derivation.

`run_a7.py` reads `results/A0` through `results/A6`; it does not rerun an arm. Generate
the human-review queue with:

```bash
PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py \
  --emit-label-template bfcl_ablation/results/A7/human_labels.template.yaml
```

Without `--labels`, semantic checks remain `INCONCLUSIVE`. The default command is
report-only and exits zero after a successful audit; `--strict` exits non-zero unless
artifact integrity and release readiness both pass.

Reports land in `results/` as Markdown and JSON. Generated packs, configs and run
artifacts land in `_generated/` and are disposable.

```
common.py            arm plumbing: config synthesis, pipeline invocation, LOC counting
measurement/         A0 — every metric, and the Markdown renderer   (STEP 1)
simplify/            A1 — derivers, milestone compiler, shrink/rehydrate, config minimizer
equivalence.py       A0-vs-A1 proof obligation
propose/             A3 — coverage spec + controlled sampler, backend result probe,
                     proposal gates, selection-bias measurement
mutate/              A4 — mutation operators, the assertion gate, LLM assertion authoring
target/              A5 — tool-calling client, model rollout loop, paired scoring
backend_gate/        A6 — backend mutation operators and the kill ladder
quality_gate/        A7 — artifact audit, human-label contract, thresholds and report
results/             per-arm reports
```

## A0 — human baseline

Generates `banking_vn` exactly as authored and reports what the pipeline never does:
a property of the benchmark itself.

| readout | value |
| --- | --- |
| authoring | **1642 lines**, 17 templates, 6 categories, 9 policies |
| tasks | 33 generated, 33 published, all `gold` |
| policy mix | `single_turn` **54.5%**; `correction`, `dependent_call`, `multi_tool`, `missing_slot`, `clarify_only` — one task each |
| joint coverage | 15/54 (category x policy) cells populated; 35 unwritten, 4 structurally empty |
| fixture coverage | **17/50** entities ever bound; `transfers` 0%, `cards` 25%, `transactions` 25% |
| surface | 33 tasks collapse to **17 distinct slot-masked utterances** — exactly one per template |
| funnel | no row is dropped at any stage |

Three of these matter more than the LOC figure:

**Surface diversity is the binding constraint.** `user_turn_templates.{lang}` is a
single string, so a category holding one template emits `tasks_per_category`
identical sentences. `qr_payment` is six tasks and one sentence. This is what A2
exists to fix, and it is why the lexical-shortcut probe is reported as *not runnable*
at A0 rather than reported as a number: a generalization gap needs held-out phrasings
of the same intent, and there are none.

**The policy mix is an artifact of category budgeting, not a target.**
`tasks_per_category` budgets the category and round-robins across its templates, so
a policy's share is decided by how many templates happen to share its category. This
is the measurement behind the plan's §3.2 coverage spec.

**A 100% publish rate carries no signal.** Nothing is dropped between expansion and
publication, so the funnel cannot currently distinguish a strong pack from a
permissive one. That is the A4 concern arriving early: no gate measures assertion
strictness, so a permissive assertion looks exactly like a perfect run.

Structural emptiness is inferred from the union of `tools_present` across a
category's existing templates. That definition is circular for a category nobody has
written templates for, and the report says so inline rather than in a footnote.

### What the only available knob actually moves

`PYTHONPATH=src python3 bfcl_ablation/sweep_budget.py 6 12 24`

| `tasks_per_category` | tasks | fixture entities | distinct surfaces | `single_turn` |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 33 | 17/50 | **17** | 54.5% |
| 12 | 55 | 25/50 | **17** | 49.1% |
| 24 | 91 | 32/50 | **17** | 50.5% |

Tripling the budget triples the tasks and nearly doubles entity coverage, and leaves
the number of distinct sentences **exactly where it started**. Surface diversity is
pinned to the template count, and the policy mix barely moves either. The benchmark
has one knob, and it reaches one of its three axes.

That is the quantitative case for A2, and it also sizes it. With `n` paraphrases per
template the ceiling is `Σ_template min(n, tasks_for_that_template)`:

| budget | n=1 | n=2 | n=3 | n=5 | n=10 | n=20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 17 | 26 | 30 | 32 | **33** | **33** |
| 12 | 17 | 34 | 46 | 53 | **55** | **55** |
| 24 | 17 | 34 | 48 | 72 | 90 | 91 |

So the plan's A2-1 / A2-5 / A2-10 / A2-20 ladder **must raise the task budget in
lockstep**. At the current budget of 6, A2-10 and A2-20 both produce 33 surfaces and
are indistinguishable — the top two rungs would measure nothing.

Twelve of the fifty fixture entities (`transfers`, `fee_schedule`,
`transfer_scenarios`) are bound by no slot in any template. They are backend state
rather than task subjects, so the ceiling is 38, not 50. The coverage metric does not
yet separate "unreachable by design" from "not covered yet".

## A1 — deterministic simplification

```
A0 pack --shrink--> A1 authored --rehydrate--> A1 full --pipeline--> benchmark
                        |                                                |
                what a user writes                        compared against A0's
```

**A field is dropped only when rehydration provably reproduces it.** Anything the
derivation gets wrong stays authored and is reported as `not_derivable`, so the cut
list is measured on the real pack rather than argued in advance.

| | A0 | A1 | A1 + minimized config |
| --- | ---: | ---: | ---: |
| task_templates.yaml | 473 | 393 | 393 |
| validation_cases.yaml | 199 | 71 | 71 |
| manifest.yaml | 50 | 28 | 28 |
| run config | 43 | 43 | 17 |
| backend / assertions / tools / fixtures | 877 | 877 | 877 |
| **TOTAL** | **1642** | **1412** | **1386** |

**256 lines (15.6%) removed with no model involved**, and every arm verified
`EQUIVALENT`:

| check | result |
| --- | --- |
| `set(task_id)` equal | yes, 33 vs 33 |
| `expected_tool_calls` equal | yes, element-wise over 33 tasks |
| gold eligibility preserved | yes |
| opening user turn identical | yes |

Field-level verdicts: **39 derived exactly**, 6 structure-derived with reworded
surface, **1 not derivable**.

### What is derived

| field | rule |
| --- | --- |
| `manifest.paths` | filename convention |
| `manifest.primary_keys` | the unique, never-null identifier field, proved not guessed |
| `manifest.absent_ids` | `<PREFIX>-ABSENT-<n>` from the collection's own id format |
| `assistant_milestones` | compiled from (policy, required_tools, call_order, slots, tool contract) |
| `user_simulator_turns` | compiled shape + pack-wide canonical wording |
| `mutates` | `x-mutates` of the required tools |
| `call_order: strict` | the default |
| `paraphrase` | `pack_loader` already defaults it to `{}` |
| ~20 validation cases | success + not-found per tool, plus unconfirmed and mis-typed `confirm` for confirming tools |
| 26 config lines | settings that only restate a default |

### What the plan got wrong

Four corrections the run surfaced, each of them a place the plan's cut list is not
safe as written:

1. **`paraphrase` is not a dead field.** `render.py:237` and `paraphrase.py:110,425`
   read it. It is droppable because `pack_loader.py:327` defaults it to `{}`, not
   because nothing consumes it.

2. **`primary_keys` cannot be dropped under the *existing* heuristic.**
   `expand.primary_key_for` falls back to "the single `*_id` field", which resolves
   `vietqr_payments` (key `payment_ref`, foreign key `transaction_id`) to the foreign
   key. A1 replaces that with a uniqueness-and-non-null proof, which is what makes
   the field genuinely omittable.

3. **`absent_ids` is not free to generate.** An absent id is a bound slot value, so
   it enters `slot_bindings` and therefore the `task_id` hash. Auto-generation is
   equivalence-preserving only when the generated string equals what the author wrote.
   It does here; that is a fact verified per pack, not a property of the derivation.

4. **`call_order` is only derivable in one direction.** `strict` is the default and
   drops cleanly. `any` — `bn_balance_and_card_parallel` — is a claim that two calls
   may be issued in one batch, which no schema carries. Deriving it wrongly silently
   converts one parallel call group into two sequential ones; the equivalence check
   caught exactly that during development.

Two policies additionally need input no schema implies. Both now cost one line each
instead of a hand-written milestone block:

- `corrects:` — which slot a correction replaces, and with what
- `depends_on:` — which argument is read from an earlier result, and at what path
  (`transactions.0.transaction_id` is knowledge about the backend's response shape)

### What A1 does change

`user_simulator_turns` wording moves to pack-wide canonical templates, so **8 tasks
have reworded later turns**. Opening turns are byte-identical, and the equivalence
check treats a changed opening turn as a failure while reporting a changed simulator
reply as the intended trade. The plan's claim that A1 leaves the user surface
untouched holds for the request, not for the replies.

## Open items

Resolved by this implementation:

- **Milestone compiler input contract** — confirmed as
  (policy, required_tools, call_order, slots, tool contract), with `corrects` and
  `depends_on` as the two explicit human inputs. This fixes the A1/A3 boundary.
- **A1 is genuinely zero-generative-risk** — no model, and equivalence is proved
  rather than assumed.

Still open, and needed before A2:

- **Intent-preservation check.** No oracle exists for "does this paraphrase still
  mean the same request". Without it A2's ranking results carry an uncontrolled
  confound. Its catch rate must be measured against injected intent-shifted
  paraphrases, not assumed.
- **Joint (category x policy) coverage spec.** A0 now reports the matrix and marks
  structurally-empty cells, but the *target* and the declaration format for empty
  cells are not yet defined.
