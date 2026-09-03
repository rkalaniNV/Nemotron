# Metric definitions — contract version 1.0

Every arm records the version it was computed under, as `metrics_version` in its
`metrics.json`. **Arms recorded under different versions are not comparable.** A later
ablation compares its distribution against A0's, so a silent change to how a metric is
computed would present a definition change as a benchmark change.

**This is recorded, not enforced.** Every arm now stamps the field — A0, A1 and A3 through the
shared measurement schema, and A2, A4, A5 and A6 explicitly, since those emit bespoke schemas.
No code reads the stamps back or compares them, so a comparison across two different contracts
would still run and still look fine. Closing that is a next step, not a solved problem.

Bump the version on any behavioural edit to `measurement/metrics.py` and add a row to
the changelog at the bottom. Adding a *new* metric does not require a bump; changing
how an existing one is computed does.

The machine-readable copy of the load-bearing definitions is emitted into every
`metrics.json` under `definitions`, so an artifact carries its own contract.

---

## Authoring friction

**`loc_total`** — every line of every authored pack file plus the run config.

Counted files: `backend.py`, `task_templates.yaml`, `validation_cases.yaml`,
`assertions.py`, `tools.json`, `fixtures.json`, `manifest.yaml`, and the generated run
config.

Blank lines and comment-only lines **count**. They are part of what an author reads and
maintains, and excluding them would flatter whichever arm happens to be less commented.

For an arm that shrinks the pack (A1), LOC is measured on the **authored** pack — what a
person writes — not on the rehydrated pack the pipeline reads. Measuring the rehydrated
pack would report zero saving.

For an arm whose pack a model wrote (A3), LOC is **not** human friction and must not be
compared against A0/A1 as if it were.

## Distribution

**joint `(category × policy)` matrix** — task count per cell.

An empty cell is classified:

- `empty_structural` — no tool in the category's tool universe can satisfy the policy
- `empty_unwritten` — feasible, nobody wrote one

The classification is **derived, never declared**, so a coverage gap cannot be hidden by
calling a cell empty. Rules per policy:

| policy | feasible when |
| --- | --- |
| `clarify_only`, `irrelevant` | always — needs no tool |
| `confirmation` | some tool in the universe carries `x-requires-confirmation` |
| `multi_tool` | the universe holds ≥2 tools |
| `dependent_call` | some tool returns a value another one requires (producer/consumer edges probed from the backend where available; otherwise falls back to ≥2 tools, which is weaker) |
| `missing_slot`, `correction` | some tool takes a parameter |
| `negative_path` | some parameter can carry a failing value |

**Known limitation.** The category's tool universe defaults to the union of
`tools_present` across the templates that category already has, which is circular for a
category nobody wrote templates for. An arm that declares its universes up front passes
them in and escapes the circle; A3 does this, A0 cannot.

## Coverage

**`fixture_entities_bound`** — a fixture row counts as bound when its primary key value
appears in some task's `fixture_refs`.

**`entities_never_bound`** — rows in a referenced collection that no task ever bound.

**Known limitation.** This does not yet separate *not covered yet* from *cannot be
covered*. In `banking_vn`, 12 of 50 rows (`transfers`, `fee_schedule`,
`transfer_scenarios`) are bound by no slot in any template — they are backend state, not
task subjects — so the real ceiling is 38. Reported coverage is against 50.

**`tools_never_called`** — declared tools absent from every expected trace.

## Surface diversity

**`distinct_masked`** — cardinality of the set of slot-masked opening user turns.

**Masking rule (the load-bearing definition):**

> Replace each bound slot value with `{slot_name}` by **exact substring match**,
> **longest value first**. No case folding. No diacritic folding. No punctuation
> stripping. No tokenisation.

Longest-first prevents a value that contains another from being partly rewritten.
Masking by the bound value rather than by a regex over id-shaped tokens is what makes
the count trustworthy: two tasks differing only by account id collapse to one sentence,
which is exactly the property being measured.

Diacritics are deliberately **not** folded. The pack is Vietnamese, and folding would
merge genuinely different surfaces.

**`surfaces_per_template`** — `distinct_masked / template_count`.

**Ceiling.** With `N` variants per template, distinct masked surfaces cannot exceed
`Σ_template min(N, tasks_for_that_template)`. Any A2 rung should be read against its
ceiling, not against the previous rung.

**Lexical-shortcut probe** — reported as *not runnable* when there is one surface per
template, rather than reported as a number. A generalization gap needs held-out
phrasings of the same intent. When run, it must mask slot values before training, or the
probe learns the `ACC-` prefix rather than the phrasing, and split by **variant** rather
than by task so the same phrasing cannot appear on both sides.

## Publish funnel

Stages, in pipeline order: `expand`, `state_machine`, `render_accepted`,
`expected_trace_derived`, `schema_valid`, `replay_valid`, `benchmark_raw`, `published`.

Each stage reports `rows`, `survival_from_expand`, `lost_here`, and **`drop_reasons`** —
a breakdown of *why* rows were lost at that stage, read from the stage's own reason
column (`guard_violations`, `drop_reason`, `reject_reason`, `reason`/`detail`). A single
publish rate cannot be interpreted without it.

**`publish_rate`** — published rows / expanded task instances.

**`gold_rate`** — published rows carrying `gold_eligible` / published rows.

**Interpretation warning.** A0 measured 100% publish at every budget, and A4 measured a
61% argument-level false-acceptance rate in the assertions that gate it. Publish rate and
gold are therefore **not quality readouts** on this pack. Report them as throughput, and
report false-acceptance rate per operator class as the quality readout.

## Equivalence (A1 and any arm claiming to preserve ground truth)

Five checks, all must pass for `EQUIVALENT`:

| check | what it proves |
| --- | --- |
| `set(task_id)` equal | both arms bound the same records to the same slots |
| `expected_tool_calls` equal, element-wise | both assert the same calls with the same arguments |
| `conversation_plans` equal | both compile the same conversation shape — the milestone check the other two cannot see |
| validation-case coverage held | every `(tool, outcome class)` pair A0 probed is still probed |
| opening user turn identical | the request the model is scored on is unchanged |

`task_id` is content-addressed over `(pack_id, pack_version, template_id,
sorted(fixture_refs), slot_bindings, variant_index)` — it does **not** cover
`assistant_milestones`, which is why the third check exists. The plan projection in
`conversation_plans.steps` already excludes `content_template`, so no further
normalization is applied.

Reworded **later** turns are reported, not failed: A1 moves simulator replies to
pack-wide canonical templates by design.

## Assertion strength (A4)

**`false_acceptance_rate`** — mutated episodes the assertion accepted / mutations tried.
Reported **per operator class** and per assertion. Never as a single aggregate: the
aggregate hides a suite that is strong at call level and blind at argument level, which
is the actual finding on this pack.

**strict vs advisory.** An operator is advisory only when an assertion that accepts its
output is behaving *correctly*. Currently advisory: `duplicate_call_readonly` (repeating
an idempotent read is wasteful, not wrong). Everything else is strict, including
`inject_extra_call` — reading a record the user never asked about is a real defect.

A mutation that makes an assertion raise a non-`AssertionError` is counted as a **crash**,
tracked separately from a detection.

**`null_control`** — an empty assertion suite, which must score 1.000 everywhere. It
validates the harness; a null control below 1.000 means mutations are not reaching the
assertion.

## Backend falsifiability (A6)

**`blind_rate`** — of the mutants of `backend.py` whose effect is *observable* on an input the
pack supplies, the share that every check the pack ships accepted. This is the arm's headline.

**Observable** is decided by a differential replay against the unmutated backend: if a mutant
changes any per-call result or the final state on A0's replayed traces, it is observable. That
comparison is a **reference this arm adds, not a check the pack ships** — nothing in the pack
states what a tool should return — so a mutant caught only by it counts toward `blind_rate`.

**Do not report the raw survival count as the headline.** A mutant that survives every layer is
usually one no task and no validation case executes, which measures *coverage*, not *checking*.
Survivors must be triaged into `equivalent` (no input could observe the edit), `unreachable`
(observable in principle, but not on any input the pack supplies) and `real_gap` (observable on a
supplied input, and every shipped check passed) before any rate is quoted. The triage is recorded
in `results/A6/triage.json`.

**`first_killing_layer`** — the cheapest layer that rejected the mutant, over the fixed ladder
import → validation cases → traces → assertions → oracle validation → full pipeline. Reported as
a histogram, and always alongside which layers ship with the pack and which do not.

## Solve accuracy across wordings (A5)

Every metric above describes the benchmark. These describe a **target model measured on
it**, and are the only ones whose subject is not the pack.

**`ast_match`** — the model's tool calls equal `expected_tool_calls`: same function
names, same arguments, compared after normalising both sides to sorted-key JSON (the
published row encodes arguments as `[[key, canonical_json]]` pairs, a model returns a
dict). Order is part of the claim **unless** the template declares `call_order: any`, in
which case the comparison is order-insensitive. A1 proved `call_order` is not derivable,
so it is read from the template and never inferred.

**`assertion`** — the same episode judged by the pack's own `success_assertions`, run
through `ProcessWorker.run_episode` exactly as the gold replay is. Reported **beside**
`ast_match`, never instead of it: A4 measured this pack's assertions at 0.610
false acceptance on argument-level corruptions, so a model can pass this column while
reporting a fabricated value. The gap between the columns is a readout, not noise.

A task with no `success_assertions` is recorded as `assertion: false` rather than
silently passing. An episode that fails to execute returns no verdict at all, so a
broken harness cannot be read as a wrong model.

**`paired_agreement`** — share of tasks on which both wordings produced the same verdict.
The pairing is exact: `task_id` is hashed over pack, template, fixture refs, slot
bindings and variant index, and **not** over the surface, so the same task carries the
same id under both wordings.

**`mcnemar_p`** — two-sided **exact** McNemar on the discordant pairs, H0 = wording has
no effect. Exact rather than chi-square because the discordant count here is far below
the ~25 the approximation needs. A non-significant p at this n means *not detected*, not
*no effect*.

**Per-cell reporting is mandatory.** Accuracy is reported per `turn_policy` and per
`category` with the count printed alongside every rate. Pooled accuracy is dominated by
`single_turn` (18 of 33 tasks) and hides the policies the ladder exists to compare.

---

## Changelog

| version | change |
| --- | --- |
| 1.0 | Initial contract. Covers A0 baseline; A1–A4 compare against it. A5 solve-accuracy and A6 backend-falsifiability metrics added under the same version: they are new metrics, not changes to existing ones, so no arm's numbers move. |
