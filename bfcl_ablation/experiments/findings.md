# Findings across the ladder

Five arms, one pack (`banking_vn`), one model (`gpt-oss-120b`). Every arm ran the unmodified
production pipeline; an arm is defined by the pack and config it feeds in.

---

## The one sentence

**Every quality gate the pipeline has is about mechanism — did it replay deterministically, is
the fingerprint intact, did the schema match — and none is about content.** Three arms show this
independently and by different mechanisms: A0 finds the gates never fire, A4 finds the assertions
do not check returned values, A3 finds a badly skewed benchmark passes anyway.

They are **not a causal chain**, and an earlier draft wrongly presented them as one. A3 did not
exploit A4's gap — it leans on `assert_no_tool_called` for 14 of its 20 templates, which A4
scored at 0.000 false acceptance. They are also not statistically independent: A4 gates exactly
A0's 33 tasks from A0's stage cache.

**A4 did not explain A0 either — until A5.** A0 replayed uncorrupted traces, so nothing reached
the assertions in a failing condition, and the link stayed inferred. A5 closes it with an
observation: a target model, given A2's paraphrase of two `confirmation` tasks, emitted an extra
call; `expected_tool_calls` caught it and the pack's assertions passed all 33 tasks in both
wordings. Assertion agreement 1.000, ground-truth agreement 0.939.

---

## The thread

### A0 — the gates never fire

Publish rate is **100%** at every task budget: 33/33, 55/55, 91/91. Nothing is dropped at any
stage. So "publish rate" and "gold" look like quality readouts and carry no information.

A0 also found the benchmark has three axes — entity coverage, policy mix, surface diversity —
and exactly one knob, which reaches one of them. Raising `tasks_per_category` from 6 to 24
triples the tasks and doubles entity coverage while leaving the number of distinct sentences
**exactly at 17**, one per template.

### A4 — the assertions accept almost anything about values

False-acceptance rate on mutations an assertion must catch:

| assertions | call level | argument level | state level |
| --- | ---: | ---: | ---: |
| human (182 lines) | 0.380 | **0.610** | 0.000 |
| null control (empty) | 1.000 | 1.000 | 1.000 |

The suite verifies *that the right tool was called* and *that the final state is right*. It
does not verify *that the values reported back are correct*: it misses a perturbed number
87.5% of the time, a transfer executed twice 85.7% of the time, and a dependent call pair
executed in reverse 100% of the time. `+1` and a large delta
score identically, so the value is not being compared at all.

This is consistent with A0's 100% publish rate but does not establish it: A0 never fed a
corrupted trace to an assertion. Demonstrating the link would require running a model that
produces wrong values against the published benchmark and showing it publishes clean.

### A3 — a skewed benchmark passes anyway, by a different route

An LLM asked to propose tasks, with the policy sampler under system control, produced a pack
that reached **gold with 100% publish rate** and was:

- **68% no-tool-call** (39.3% `clarify_only` + 28.6% `irrelevant`)
- missing **5 of 9 tools**, including **both mutating tools** — no state change is ever exercised
- binding **11 of 50** fixture entities, `transactions` 1/16

Accept rate by policy ran from **100%** (`irrelevant`) to **0%** (`confirmation`, `correction`,
`dependent_call`). The sampler proposed hard policies evenly; the gate kept only the easy ones.

This is indistinguishable from A0 by every gate the pipeline runs. The mechanism is task
selection rather than assertion weakness: **14 of the 20 accepted templates (70%) are
unfalsifiable by the executable oracle** — they assert that nothing happened, which an oracle
cannot disprove. Two are labelled `clarify_only` although the pack holds a tool that answers them.

### A5 — and it happens on real model output

The same 33 tasks in two wordings, one target model, paired by `task_id` (which does not cover
the surface, so both arms share ids exactly):

| verdict | A0 | A2 | paired agreement |
| --- | ---: | ---: | ---: |
| calls equal `expected_tool_calls` | 0.970 | 0.909 | 0.939 |
| the pack's own `success_assertions` | 0.970 | 0.970 | **1.000** |

Two `confirmation` tasks flipped. A0 says *"chuyển **ngay**"* (transfer *immediately*), A2 says
*"**mong** chuyển"* (*would like to* transfer); without the urgency marker the model checked the
fee first. The transfer is identical and correct — the call set is not. The assertions passed
both.

The failure mode is `inject_extra_call`, which A4 had scored at **0.882** false acceptance on
synthetic corruptions. It is the operator A4 rated most dangerous, and it is the one that showed
up first in the wild.

Two honest qualifications: the accuracy delta is **not significant** (McNemar p = 0.50 on 2
discordant pairs), and the paraphrase preserved tool-level intent — A2's checker passed it
correctly. What changed was register. **Intent preservation is not behavioural equivalence**, and
no arm before this one could tell the difference.

### A6 — and the oracle itself is barely checked

A4 corrupted an *episode* and asked whether the assertions noticed. A6 corrupts the *oracle* —
151 single-edit mutations of `backend.py` — and asks whether anything notices.

| outcome | mutants |
| --- | ---: |
| unobservable — nothing the pack runs reaches it | 44 |
| observable, caught by a check the pack ships | 58 |
| **observable, caught by nothing shipped** | **49** |

**Blind rate 45.8%.** Two independent methods aimed at different objects — one corrupts the
episode, the other the backend — land on the same gap. Neither can be explained as an artefact
of the other.

Three things sharpen it:

- **`run_oracle_validation` and a full pipeline run killed nothing.** Every mutant that reached
  them published 33 rows at tier `gold`, byte-identical to A0's benchmark. The layer is live —
  a pack with a tool deleted *is* caught — it just has nothing to say about a wrong value.
- **Deleting a guard survives 61.5% of the time; inverting the same guard survives 3.8%.** A 16x
  asymmetry on the same 26 sites: the pack is well defended against rejecting good input and
  nearly blind to accepting bad input. That is A4's asymmetry again, in a different artifact.
- **Two of the four confirmed gaps sit on a behaviour the pack wrote a dedicated case for.**
  Dropping `transfer_id` from the `awaiting_confirmation` result changes validation case
  `confirm_false_create_transfer` and is seen by oracle check 6 `confirmation_policy`. Both
  pass, because one pins `result_class` and the other pins `status`. The check exists, is aimed
  correctly, and still cannot see it — because it asserts the shape of the answer, never its
  content.

A6 also found that **41 of 48 survivors are simply unreachable**: whole validation surfaces
(`_check_prefix`, the `_require_str` type guards) are never exercised by any case or task. That
is a coverage finding worth as much as the checking one — lines an author paid to write that the
benchmark never touches.

### A1 and A2 — the parts that worked

**A1**: 256 of 1642 authored lines (15.6%) come off with no model involved, verified
`EQUIVALENT` — same `set(task_id)`, same `expected_tool_calls`, gold preserved. Measured
against only the derivable half of the pack (the other 877 lines are ground truth), that is
33%.

**A2**: paraphrasing raises surface diversity to 1.82 sentences per template at the pack's own
budget, and 4.53 at budget 24 — the larger figure needs the task budget raised too. Ground truth
moved **zero** at all twelve configurations tested.

---

## What each arm changed about the plan

| arm | the plan said | the run found |
| --- | --- | --- |
| A0 | measure the baseline | the budget knob cannot reach surface diversity at all; publish rate is not a quality signal |
| A1 | `paraphrase` is a dead field | it is read by `render.py`; it is droppable only because `pack_loader` defaults it |
| A1 | `primary_keys` → infer + validate | the existing heuristic resolves `vietqr_payments` to its **foreign key**; inference must prove uniqueness |
| A1 | `absent_ids` → generate | an absent id enters the `task_id` hash, so generation is only safe if it reproduces the author's string |
| A1 | milestones ← (policy, intent, tools) | confirmed, plus `call_order: any` is **not** derivable — deriving it silently serialises a parallel call group |
| A2 | intent drift is an open risk | quantified: existing guards catch **0 of 34** injected shifts; the new checker catches **32 of 34** |
| A2 | ladder 1/5/10/20 | only measurable if the task budget rises with N; at budget 6 the top rungs are indistinguishable |
| A3 | policy distribution ← system prevents selection bias | **necessary but not sufficient** — bias re-enters at the accept/drop gate |
| A4 | mutation score alone is insufficient | confirmed: the 0.498 aggregate hides 0.000 at state level and 0.610 at argument level |
| A4 | assertions stay human until evidence | supported — LLM suites are stricter but false-reject 2–3 of 34 valid tasks at 8–9x the code, and the feedback arm also rejects 18 of 26 corruptions that were never defects |

---

## What to do next, in order

1. **Report false-acceptance rate per operator class alongside gold.** It is the readout that
   most directly measures assertion strength; coverage and distribution distinguish packs too.
   Publish rate should stop being presented as quality.
2. **Fix the argument-level blindness in `banking_vn`'s assertions.** They do not compare
   returned values against fixture state. This is a bug in the reference pack, not a research
   finding — and the fix is small.
3. **Make A3 sample the *accepted* distribution.** Retry a cell until its target is met or
   declare it unreachable. Without this, every LLM-authored pack drifts toward no-tool-call
   tasks.
4. **Gate A2 on `substituted` only, never on the raw flag.** Raw gating would drop ~95% of
   `dependent_call` and ~84% of `missing_slot` paraphrases — the checker cannot infer a
   multi-step plan from an opening turn, and those policies are already down to one task each.
5. **Fix A2's variant assignment** (`seed % N` collides; round-robin over a template's
   instances instead). Recovers 3–27% of achievable diversity for free.
6. **Run the lexical-shortcut probe.** It was not runnable at A0 (one phrasing per intent) and
   is runnable now. It is the measurement that says whether the diversity A2 bought is real.

---

## What this does not show

- **One pack, one domain, one language, one model.** `banking_vn`, Vietnamese,
  `gpt-oss-120b`. Every cross-arm claim is a single observation.
- **Target-model evaluation is one model, one paraphrase.** A5 implements the plan's STEP 4 and
  found that the assertions cannot see a wording effect the declared ground truth catches. But
  its accuracy delta rests on 2 discordant pairs (p = 0.50) and its target model is the same
  family that wrote the paraphrases. It shows the gap is real; it does not yet say how large.
- **Self-preference is untouched.** It needs at least two generator families.
- **Small n in places.** A4's state-level result rests on 6 trials; A3's `confirmation` accept
  rate on 3 proposals; A2's false-alarm floor on 17 sentences.
- **A2 and A3 do not compose.** They are independent arms. A combined run is not part of this
  ladder.

---

## Where the numbers live

| | report | machine-readable |
| --- | --- | --- |
| A0 | [a0.md](a0.md) | `results/A0/`, `results/budget_sweep.json` |
| A1 | [a1.md](a1.md) | `results/A1/` |
| A2 | [a2.md](a2.md) | `results/A2/`, `results/A2_rerun/` |
| A3 | [a3.md](a3.md) | `results/A3/` |
| A4 | [a4.md](a4.md) | `results/A4/` |
| A5 | [a5.md](a5.md) | `results/A5/` |
| A6 | [a6.md](a6.md) | `results/A6/` |

Metric definitions are fixed and versioned in [`results/METRICS.md`](../results/METRICS.md);
every `metrics.json` records the `metrics_version` it was computed under. Arms recorded under
different versions are not comparable.

Every model call is cached under `_generated/llm_cache`, so a re-run reproduces these numbers
and the cache is the record of what the model was asked and what it answered.
