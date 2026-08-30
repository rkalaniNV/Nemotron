# BFCL Oracle-Pack Ablation — Method and Results

One page for the whole study: what was asked, how it was run, what came out, and what it does
not show. Per-arm detail is in [`experiments/`](experiments/); circulation copies are in
[`reports/`](reports/); metric definitions are in [`results/METRICS.md`](results/METRICS.md).

Every figure here was recomputed from the stored artifacts by an independent verification pass
before publication. Where that pass corrected an earlier claim, the corrected version is what
appears below.

---

## In one paragraph

Authoring one BFCL Oracle Pack costs ~1,600 hand-written lines, and that cost is the adoption
barrier. Five ablations asked how much of it can be removed — by code, then by an LLM — without
weakening the benchmark. **14.0% comes off with no model involved and is provably equivalent**
(a further 1.6 points comes from stripping default-valued config, verified as a separate step).
**An LLM raises linguistic diversity 1.8× at the pack's own task budget and 4.5× if the budget
is raised with it, without moving ground truth.** But answering the question first required
building the ability to tell, and that measurement produced the study's main result: **three
independent demonstrations that the pipeline's gates check mechanism, not content** — nothing is
ever dropped; the assertions accept 61% of corrupted argument values; and an LLM asked to author
tasks produced a benchmark 68% of which calls no tool at all, 70% of whose templates the
executable oracle cannot falsify, which was nonetheless stamped gold.

---

## The question

The BFCL pipeline's verification half — executable replay, double-replay determinism, the
fingerprint chain, process isolation — is sound and stays **out of scope**. Reducing friction
must not reduce what those gates prove.

But those gates check **mechanism**, not **content**. They prove a benchmark is reproducible,
untampered and safely executed. They prove nothing about whether it is diverse, well-covered or
hard. So the study has two layers:

1. How much authoring can be removed — by code, then by a model?
2. What does "without weakening the benchmark" even mean, and can we measure it?

Layer 2 turned out to be the prerequisite, and the more consequential half.

---

## Method

### The ladder

Five arms on the same pack (`banking_vn`: 9 tools, 17 templates, 6 categories, 9 turn policies).
**Each rung opens exactly one degree of freedom**, so when a result moves, the responsible change
is unambiguous.

| arm | what changes | model |
| --- | --- | --- |
| **A0** | nothing — measure the human baseline | no |
| **A1** | deterministic fields removed, re-derived in code | no |
| **A2** | user wording only; everything else frozen | yes |
| **A3** | task semantics; policy sampler system-controlled | yes |
| **A4** | assertions, plus the mutation gate that makes them measurable | yes |

### Two rules held throughout

- **LLM proposes, oracle disposes.** A model may only produce artifacts an independent source can
  refute. The backend stays human-written and executable, and the generator never runs it. A
  consistency check whose two halves were both written by one model in one pass is not a check.
- **If it can be inferred deterministically, do not use an LLM.** Using a model for mechanical
  content adds a failure mode for no benefit.

### Experimental hygiene

- **The production pipeline is never patched.** An arm is defined by the pack and config it feeds
  in. A1 writes an ordinary full pack to disk and the unmodified generator reads it, so every arm
  goes through identical code. A patched generator would measure the patch.
- **Every model call is disk-cached**, keyed by model + prompt + temperature (0) + seed (0),
  storing the reply, reasoning trace, token usage and server fingerprint. A re-run reproduces the
  same benchmark, and the cache is the record of what was asked. Model: `openai/gpt-oss-120b`,
  local vLLM.
- **Metric definitions are pinned in `METRICS.md` and stamped `metrics_version: 1.0`** into the
  result files of A0, A1 and A3 (shared measurement schema) and of A4 (bespoke schema, stamped
  explicitly). **A2 emits a bespoke schema and still carries no version stamp**, and **no code
  compares versions across arms**. Version comparability is a convention enforced by reading the
  contract, not a mechanism. That is a gap, listed under next steps.

### The proof obligation

Any arm claiming to preserve ground truth must pass five checks, not two:

| check | what it proves |
| --- | --- |
| `set(task_id)` equal | both arms bound the same records to the same slots |
| `expected_tool_calls` equal | both assert the same calls with the same arguments |
| `conversation_plans` equal | both compile the same conversation shape |
| validation-case coverage held | every `(tool, outcome)` pair still probed |
| opening user turn identical | the request the model is scored on is unchanged |

The third and fourth were added on review. `task_id` is hashed over pack, template, fixture refs
and slot bindings — it **does not cover `assistant_milestones`** — so a compiler emitting
`ask_confirm` where the author wrote `ask_for_slot`, tool calls untouched, would have passed the
original gate while changing the conversation under test.

The fifth is contract-dependent: A1 keeps the request byte-identical, A2 rewords it by design, so
the gate takes a flag rather than assuming one contract. The other four are unconditional.

**Caveat on check four in A2:** it reads `validation_cases.yaml` off each arm's pack directory,
and A2's variant packs carry the file unchanged from source. It therefore passes trivially at
every A2 rung and is not an independent check there. It is load-bearing only for A1, which
actually rewrites that file.

---

## Results

### A0 — Human baseline

| readout | value |
| --- | --- |
| authoring cost | **1642 lines** (backend 465, templates 473, validation 199, assertions 182, tools 162, fixtures 68, manifest 50, run config 43) |
| scale | 17 templates, 6 categories, 9 policies, 27.8 template lines each |
| output | 33 tasks, 33 published, all tier `gold` |
| policy mix | `single_turn` **54.5%**; `clarify_only`, `correction`, `dependent_call`, `missing_slot`, `multi_tool` — **one task each** |
| joint coverage | **15 of 54** cells populated; 4 are structurally empty, so 15 of 50 feasible |
| fixture coverage | **17 of 50** entities bound — but 12 rows are backend state no slot can bind, so **17 of 38 reachable (45%)** |
| tool coverage | 9/9 |
| utterance diversity | 33 tasks → **17 distinct slot-masked sentences** (1.0 per template) |
| publish funnel | **33 → 33. Nothing dropped at any of the eight stages.** |

**The only generation knob reaches only one of three axes.** Sweeping `tasks_per_category`:

| budget | tasks | entities | **distinct sentences** | `single_turn` |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 33 | 17/50 | **17** | 54.5% |
| 12 | 55 | 25/50 | **17** | 49.1% |
| 24 | 91 | 32/50 | **17** | 50.5% |

Quadrupling the budget multiplies tasks by 2.8 and entity coverage by 1.9, and leaves the
sentence count exactly where it started — at budget 24 each sentence repeats 5.4 times. Surface
diversity is unreachable by configuration.

**Two caveats on the funnel.** It begins at `expand`, so it cannot see the loss that happens
before it: at budget 6 the config asks for 36 tasks (6 categories × 6) and gets 33, at budget 24
it asks for 144 and gets 91. And the per-stage drop-reason breakdown, though implemented, has
never emitted a non-empty bucket — the all-zero table is not evidence that it works.

### A1 — Deterministic simplification

```
A0 pack --shrink--> A1 authored --rehydrate--> A1 full --pipeline--> benchmark
1642 lines           1412 lines                                 compared against A0
```

**A field is dropped only when rehydration reproduces its semantics.** Anything the derivation
gets semantically wrong stays authored and is reported as `not_derivable`, which turns the cut
list from an argument into a measurement.

| | A0 | A1 | A1 + minimised config |
| --- | ---: | ---: | ---: |
| `task_templates.yaml` | 473 | 393 | 393 |
| `validation_cases.yaml` | 199 | 71 | 71 |
| `manifest.yaml` | 50 | 28 | 28 |
| run config | 43 | 43 | **17** |
| backend / assertions / tools / fixtures | 877 | 877 | 877 |
| **TOTAL** | **1642** | 1412 | **1386** |

**A1 proper removes 230 lines (14.0%)**; stripping default-valued run-config settings removes a
further 26 (to 15.6% total) and is verified as a **separate degree of freedom**, not folded in.
All five gates pass on both.

Field verdicts across 46 fields: **39 reproduced exactly, 6 reproduced in structure but not in
wording, 1 not derivable.** The 6 are dropped from the authored pack even though rehydration does
not reproduce them byte-for-byte — their `content_template` wording is replaced by pack-wide
canonical text. That is a deliberate trade, not an exception to the rule, but it means **"dropped"
does not everywhere mean "identical"**.

The visible consequence: **8 of 33 published tasks (24%) come out with different conversation
text.** Opening turns are byte-identical; later turns are not. For the pack's only `correction`
task, the assistant's two confirmation prompts lose the amount they were quoting. The five gates
cannot see this by construction — it is reported separately.

`validation_cases.yaml` 199 → 71 lines: 23 hand-written probes become 5 authored + 18 generated,
with `(tool, outcome)` coverage held at 22/22.

Read the percentage correctly: **877 lines (53%) are ground truth and cannot be cut.** Against the
765-line reducible surface, A1 removes **33%**. Also note 43 of the 1642 counted lines are a
generated run config, so "hand-written" is ~1,600, not ~1,650.

`run_a1.py` exits non-zero on divergence, so it works unchanged as a CI regression test.

### A2 — LLM surface generation (wording only)

| budget | N=1 | N=2 | N=3 | N=5 | N=10 | N=20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 (33 tasks) | 17 | 23 | 22 | 28 | 29 | **31** |
| 24 (91 tasks) | 17 | 33 | 41 | 55 | 67 | **77** |

**Ground truth held at all twelve configurations** on `task_id`, `expected_tool_calls` and
`conversation_plans` (see the caveat above on the fourth check). Diversity rises from 1.0 to
**1.82 sentences per template at the pack's own budget**, and to **4.53 at budget 24** — the 4.5×
figure requires quadrupling the task budget as well, and should not be quoted alone.

The curve is flat above N=5 at the default budget: the ladder is only measurable if the budget
rises with N. Rungs reach 73–100% of their ceiling (73–97% for the LLM rungs; the two N=1 rungs
are trivially at 100%), and budget 6 / N=3 scores *below* N=2 — variant assignment is `seed % N`,
an unbalanced draw that collides. Paraphrase rejection rate 1.5%, all duplicates; no variant was
rejected for dropping a slot placeholder.

**Intent preservation — the risk the plan named, now measured:**

| population | n | flagged | rate |
| --- | ---: | ---: | ---: |
| injected intent shifts (all should be caught) | 34 | 32 | **94.1%** |
| canonical authored sentences (false-alarm floor) | 17 | 1 | 5.9% |
| generated paraphrases | 323 | 42 | 13.0% |
| — of those, intent actually substituted | 323 | 7 | 2.2% |

The existing placeholder / literal / tool-name guards caught **0 of 34** injected shifts.

Three qualifications the pooled numbers hide:

- **"Caught" means "disagreed", not "diagnosed".** The checker flagged 32 of 34 shifts, but named
  the tool the decoy was actually steered towards in only **15 of 34 (44%)**.
- **All 7 substitutions land on one template.** `bn_create_dispute_single` accounts for every one
  — 7 of its 19 paraphrases (**36.8%**), the one template whose ground-truth call is the mutating
  `create_dispute`. The 2.2% pooled rate averages a concentrated failure over 16 clean templates.
- **The arm ships the contamination.** The checker was a measurement, not a gate, so flagged
  variants were published: 8 of 12 rungs published intent-substituted tasks, up to 4 per rung
  (budget 24, N=10) and up to 15 merely-flagged.

**Do not gate on the raw flag.** Only 7 of 42 flags are `substituted`; the other 35 are
`under_predicted`, concentrated in `dependent_call` (18) and `missing_slot` (16), with 1 in
`confirmation` and none in the remaining six policies. The checker sees one opening turn and is
asked which tools the request needs — for those policies that is unanswerable, so the flag
measures the checker, not the paraphrase.

### A3 — LLM task generation (semantics)

The system sampler chose the policies. **The accept/drop gate then kept only the easy ones:**

| policy | proposed | accepted | rate |
| --- | ---: | ---: | ---: |
| `irrelevant` | 8 | 8 | **100%** |
| `clarify_only` | 7 | 6 | **86%** |
| `negative_path` | 5 | 3 | 60% |
| `single_turn` / `missing_slot` | 6 / 6 | 1 / 1 | 17% |
| `multi_tool` | 7 | 1 | 14% |
| `confirmation` | 3 | 0 | **0%** |
| `correction` | 6 | 0 | **0%** |
| `dependent_call` | 8 | 0 | **0%** |

Accept-rate spread: **1.0 — the maximum possible.** 20 of 56 proposals accepted (35.7%).

The delivered benchmark is **39.3% `clarify_only` + 28.6% `irrelevant` = 68% of tasks calling no
tool at all**. Five of nine tools never called, **including both mutating tools**. Fixture
coverage collapsed to **11/50**. Total variation from the human-authored mix: **0.50** on
`required_tools`, **0.64** on `success_assertions`. Coverage against the joint spec: 17 of 44
feasible cells (38.6%).

**And it reached `gold` with a 100% publish rate (28/28).**

**The mechanism is task selection, not assertion weakness.** 14 of the 20 accepted templates
(**70%**) are *unfalsifiable by the executable oracle* — they assert that nothing happened, which
an oracle cannot disprove. Two of them are worse than vacuous: `txn_status_missing_account_id`
and `txn_status_missing_txn_id` are labelled `clarify_only` while the pack has a tool that
answers them. The pack passes gold not because the gates are blind but because it selected tasks
there is nothing for the gates to check.

Two corrections to the obvious reading:

- **The sampler was not perfectly even.** Proposals ranged 3 to 8 per policy, a 2.7× spread.
- **Bias did not enter only at the gate.** The proposal spec already over-weighted the two
  no-tool policies at 26.8% of proposals against 12.1% of A0's tasks; the gate then amplified
  that to 67.9%. The gate is the dominant amplifier, not the sole source.

`policy distribution ← system` is therefore **necessary but not sufficient**. A3 must sample the
*accepted* distribution — retry a cell until its target is met, or declare it unreachable.

### A4 — Assertions and the mutation gate

Corrupt a replayed episode, ask the task's own assertions whether they notice. An assertion that
still passes is a **false acceptance**: it is not a check. 33 tasks, 265 mutations, 899 oracle
episodes, 11 operators across three classes. Three delivery modes: 7 operators re-execute a
different call sequence against the real backend, 3 inject a corrupted trace through the worker's
existing `trace` override, and 1 resets state while replaying the real trace.

False-acceptance rate on mutations an assertion **must** catch (lower is better; the null control
is an empty suite and sets the ceiling at 1.000):

| assertions | call level | argument level | state level | unmutated pass | advisory rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| **human (182 lines)** | 0.380 | **0.610** | 0.000 | 34/34 | 0.000 |
| LLM, blind (1426 lines) | 0.300 | 0.248 | 0.000 | 32/34 | 0.308 |
| LLM + mutation feedback (1668 lines) | 0.000 | 0.024 | 0.000 | 31/34 | **0.692** |
| null control (does nothing) | 1.000 | 1.000 | 1.000 | 34/34 | 0.000 |

The last column is the counterweight: an arm can always lower false acceptance by
rejecting more, and `llm_feedback` rejects **69%** of the corruptions an assertion is
*right* to accept. Its near-zero false acceptance is a trade, not a free win.

Per operator, the human suite catches a missing call (0.061) and a wrong tool (0.061) and always
notices unchanged final state (0.000) — but **misses a perturbed number 87.5% of the time, a
transfer executed twice 85.7%, a lookup nobody asked for 88.2%, and a dependent pair run in
reverse 100%** (n=1).

`perturb_numeric_plus_one` and `perturb_numeric_large` score **identically at 0.875**: the
blindness is not a tolerance problem, the value is not compared at all.

So the suite verifies that the *required* tools appear and that the final state is right. It does
**not** verify that the call set is right — it accepts extra and duplicated calls — and it does
**not** verify that the values reported back are correct.

Four caveats on the table:

- **The rows have different denominators.** Assertion × task pairs whose *unmutated* episode fails
  are dropped from scoring, so human is scored on 255 strict trials, `llm_blind` on 233 and
  `llm_feedback` on 223. The dropped trials are systematically the ones the LLM suites got wrong.
- **`llm_feedback` is scored in-sample.** It is shown the specific mutations `llm_blind` survived
  and asked to rewrite so those fail, then scored on the same mutation set. Its 0.024 is a
  training-set number, not a held-out one.
- **The gains are not all attributable to feedback.** Human → feedback is 25× at argument level,
  but that bundles two interventions. Feedback alone (`llm_blind` → `llm_feedback`) is **10.2×**.
- **24 of the 265 mutations are excluded** from every rate as advisory
  (`duplicate_call_readonly`), on the ground that an assertion accepting a repeated idempotent
  read is behaving correctly. They are not discarded: rejection rate on them is the
  over-strictness readout in the table's last column.

LLM assertions are not simply better: they **false-reject 2 and 3 of 34 assertion instances**
(across 33 unmutated episodes) where the human suite rejects none, at **7.8× and 9.2× the code**
— and the feedback arm additionally rejects **18 of 26** corruptions that were never defects.
The result that generalises is that **the mutation gate is what makes either kind measurable**.

### A5 — Target-model evaluation across wordings

The same 33 tasks in two wordings — A0's authored sentence and A2's paraphrase — scored on one
target model. `task_id` does not cover the surface, so both arms carry identical ids and every
task is its own control; the statistic is McNemar's exact test on the discordant pairs. The A2
side is `a2_b6_v6`, the one variant whose paraphrases A2's intent checker left entirely
unflagged, honouring A2's own rule that a model comparison must gate on `substituted` first.

| verdict | A0 | A2 | delta | paired agreement | discordant | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ast_match` — calls equal `expected_tool_calls` | 0.970 | 0.909 | **−0.061** | 0.939 | 2 | 0.50 |
| `assertion` — the pack's own `success_assertions` | 0.970 | 0.970 | +0.000 | **1.000** | 0 | 1.00 |

**Two tasks changed answer when the sentence changed; the pack's assertions saw none of it.**
Both flips are `confirmation` tasks where A0 says *"chuyển **ngay**"* (transfer *immediately*)
and A2 says *"**mong** chuyển"* (*would like to* transfer). The urgency marker was doing work:
without it the model inserted a `get_transfer_fee` call before the transfer. The transfer itself
is identical and correct — the call *set* is not.

Three things follow, and the third is the one to act on:

- **The effect is entirely in `confirmation`** — 5 tasks, 1.00 → 0.60. The 18 `single_turn`
  tasks did not move at all. The pooled −0.061 hides the whole result, which is why per-cell
  reporting is mandatory for this arm.
- **Intent preservation is not behavioural equivalence.** A2's checker asks which tools a request
  needs; both wordings need the same one, so it passed the paraphrase and was right to. Nothing
  in A2 can see a change of register that leaves the tool set intact and still changes what a
  model does.
- **Not statistically significant.** p = 0.50 on 2 discordant pairs. The load-bearing number is
  not the accuracy delta but the **agreement gap**: 1.000 by assertion versus 0.939 by declared
  ground truth. That is a statement about what the gates can see, and it does not depend on n.

One control worth stating: the single `dependent_call` task fails under *both* wordings — the
model never chains to `get_transaction_status` — and the assertions **correctly failed it** both
times. The suite is not uniformly blind; it missed the extra call and caught the missing one,
which is exactly the shape A4 measured.

---

## The cross-arm finding

> **Every quality gate the pipeline has is about mechanism; none is about content.**

Three arms demonstrate this **independently and by different mechanisms** — but they are *not* a
causal chain, and an earlier draft of this document wrongly presented them as one:

| arm | what it shows | mechanism |
| --- | --- | --- |
| A0 | the gates never fire | nothing is dropped at any stage, at any budget |
| A4 | the assertions do not check returned values | 61% false acceptance at argument level |
| A3 | a skewed benchmark passes anyway | 70% of its templates are unfalsifiable by the oracle |
| A5 | and it happens on real model output | a reworded request changed the model's call set; assertion agreement 1.000, ground-truth agreement 0.939 |

**A4 did not explain A0 — until A5.** A0's replay stage passed 33/33 on *uncorrupted* traces;
nothing ever reached the assertions in a failing condition, so A0's 100% was consistent with
assertion blindness but not evidence of it. Establishing the link required running the published
benchmark against a model that produces wrong output and showing it publishes clean. **A5 is that
run, and it does.** A target model, given A2's paraphrase of two `confirmation` tasks, emitted an
extra `get_transfer_fee` call; `expected_tool_calls` caught it, and the pack's own assertions
passed all 33 tasks in both wordings. The failure mode is `inject_extra_call`, which A4 had scored
at 0.882 false acceptance on synthetic corruptions. The chain is now observed, not inferred.

**A3 did not exploit A4's gap.** A3 leans on `assert_no_tool_called` for 14 of its 20 templates,
and A4 scored that assertion at **0.000 false acceptance** — one of the strongest in the suite.
A3 exploited a different and arguably worse weakness: selecting tasks with nothing to check.

**They are also not statistically independent.** A4 gates exactly A0's 33 replayed tasks using
A0's stage cache — A4's input *is* A0's output. A3 reuses A0's backend, tools, fixtures and
assertion library. The three are independent in *method*, not in *data*.

---

## What the runs changed about the plan

| the plan said | the run found |
| --- | --- |
| `paraphrase` is a dead field | `render.py` reads it; droppable only because `pack_loader` defaults it to `{}` |
| `primary_keys` → infer + validate | the existing heuristic resolves `vietqr_payments` to its **foreign key**; inference must prove uniqueness, not guess |
| `absent_ids` → generate | an absent id enters the `task_id` hash, so generation is safe only when it reproduces the author's string |
| milestones ← (policy, intent, tools) | confirmed — but `call_order: any` is **not** derivable; deriving it silently serialises a parallel call group |
| intent drift is an open risk | quantified: existing guards catch **0/34**; the checker flags **32/34** but diagnoses only 15/34 |
| A2 ladder 1/5/10/20 | only measurable if the task budget rises with N |
| system-controlled policy sampler prevents selection bias | **necessary but not sufficient** — the gate amplifies bias the sampler only partly avoids |
| mutation score alone is insufficient | confirmed: the 0.498 aggregate hides 0.000 at state level and 0.610 at argument level |
| assertions stay human until evidence | supported — LLM suites are stricter but false-reject valid tasks at ~8–9× the code, the best of them is scored in-sample, and it rejects 69% of corruptions that were never defects |

One error was caught by the harness rather than by review: deriving `call_order: any` as `strict`
would have silently serialised a parallel call group, and the round trip flagged it
`not_derivable`. A second — classifying `inject_extra_call` as a harmless operator, which had
reported the human suite's call-level rate as 0.149 instead of 0.380 — was caught by **review**,
not by the harness, because strict-versus-advisory is a hand-maintained constant with nothing
checking it.

---

## What this does not show

- **n = 1 at the pack level.** One pack, one domain (Vietnamese banking), one model. Every
  cross-arm claim is a single observation.
- **Target-model evaluation covers one model and one paraphrase.** A5 runs `gpt-oss-120b` against
  A0 versus one A2 variant. It is enough to show the assertions cannot see a wording effect; it is
  **not** enough to say whether conclusions are stable in general. That needs a second model
  family and several paraphrases per task, and its headline delta is not significant at n=33.
- **A2's intent checker shares a model family with the generator**, so 94.1% is a self-check and
  an upper bound. It needs re-running against a second family.
- **A4's `llm_feedback` arm is scored in-sample** and its numbers are not held out.
- **A2's fourth equivalence check is vacuous** at every A2 rung (same file on both sides).
- **No power analysis.** With most (category × policy) cells holding 1–6 tasks, per-cell paired
  tests would likely be underpowered.
- **Small n in places.** A4 state level rests on 6 trials and `reorder_calls` on 1; A3
  `confirmation` on 3 proposals; A2's false-alarm floor on 17 sentences.
- **Three milestone-compiler rules are untested** because the pack contains no such case.
- **The funnel's drop-reason breakdown has never fired**, so it is implemented but unverified.
- **`metrics_version` is not enforced** by any code. A4 now stamps it; A2 still does not.
- **A2 and A3 do not compose.** They are independent arms.

---

## Next steps, in order

**Fix first, research later**

1. **`banking_vn`'s assertions do not compare returned values against fixture state.** 61%
   argument-level false acceptance is a bug in the reference pack every future author copies.
   Re-running A4 after the fix is a ready-made regression test.

**Make the findings actionable**

2. **Report false-acceptance rate per operator class alongside `gold`**, and stop presenting
   publish rate as quality. It is the readout that most directly measures assertion strength,
   though not the only one that distinguishes packs — coverage and distribution do too.
3. **Make A3 sample the accepted distribution**, and reject templates the oracle cannot falsify.
   The 70% unfalsifiable share is the finding to act on.

**Close the measurement gaps before target-model runs**

4. Re-run A2's intent checker against a **second model family**; report catch rate per
   (generator, checker) pair, and separate "flagged" from "correctly diagnosed".
5. **Score A4's feedback arm on held-out mutations.**
6. **Power analysis** from a target effect size; if N per cell exceeds budget, drop rungs not N.
7. Test whether round-robin variant assignment closes the 3–27% ceiling shortfall — currently a
   hypothesis, not a measured recovery.
8. Enforce `metrics_version` across arms, or drop the claim.
9. Run the **lexical-shortcut probe**, now that several phrasings per intent exist.

**Then the open research question**

10. **Extend A5, now that it exists.** The paired harness is built and the first run found what
    it was designed to find, but on one model, one paraphrase and 2 discordant pairs. Three
    extensions, in order of what they buy:
    (a) **a second model family** as target, which also decouples A5 from the family that wrote
    the paraphrases; (b) **every unflagged A2 variant** (`--a2-run` over the nine other clean
    indices), turning a point estimate into an effect distribution; (c) **budget 24**, where 91
    tasks give McNemar something to work with. Until (c), report A5's agreement gap, not its
    accuracy delta.

---

## Reproducing

No installation required — the BFCL family imports with `pyarrow`, `pydantic`, `pyyaml` and `rich`
alone. Run from the repository root:

```bash
PYTHONPATH=src python3 bfcl_ablation/run_a0.py            # baseline
PYTHONPATH=src python3 bfcl_ablation/sweep_budget.py 6 12 24
PYTHONPATH=src python3 bfcl_ablation/run_a1.py            # exits non-zero if not equivalent
PYTHONPATH=src python3 bfcl_ablation/run_a2.py            # needs the model endpoint
PYTHONPATH=src python3 bfcl_ablation/run_a3.py
PYTHONPATH=src python3 bfcl_ablation/run_a4.py            # --skip-llm runs the gate with no model
PYTHONPATH=src python3 bfcl_ablation/make_docx.py         # circulation copies
```

A2–A4 use `openai/gpt-oss-120b` at `http://127.0.0.1:8000/v1`, overridable via
`BFCL_ABLATION_LLM_URL` and `BFCL_ABLATION_LLM_MODEL`.

| where | what |
| --- | --- |
| `experiments/a0.md` … `a4.md` | per-arm write-ups, insights first |
| `experiments/findings.md` | cross-arm synthesis |
| `results/A0/` … `A4/` | `report.md` + `metrics.json` per arm |
| `results/METRICS.md` | metric contract, versioned |
| `reports/*.docx` | self-contained circulation copies |
| `_generated/` | packs, configs, run artifacts, LLM cache — disposable, all regenerable |
