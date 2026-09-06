# BFCL ablation aggregation contract

SOV-866 turns the ablation ladder into one reviewable decision: which pipeline
components materially improve benchmark quality and model evaluation. The
contract version is `1.2`, implemented by
[`ablation_aggregation.py`](../runtime/benchmark_families/bfcl/ablation_aggregation.py)
and driven by
[`aggregate_bfcl_ablation.py`](../scripts/aggregate_bfcl_ablation.py).

Aggregation is read-only. It reads one declared input document, recomputes every
comparison, and writes `ablation_summary.json` and `ablation_summary.md`. It
never reads a run tree directly, so a summary cannot silently depend on state
that was not declared as evidence.

## What the ladder compares

Every arm is compared against exactly one declared baseline arm. The ladder
opens one degree of freedom per arm, which is what makes a shifted conclusion
attributable:

| Arm | Intervention | Ticket |
|---|---|---|
| A0 | none; human baseline | SOV-859 |
| A1 | derive mechanically inferable pack fields | SOV-860 |
| A2 | LLM-generated user wording only | SOV-861 |
| Step 4 | paired cross-wording evaluation | SOV-862 |
| A3 / A4 | LLM task proposal and generated assertions | SOV-863 |

The manual / `llm_backend` / `llm_mcp` onboarding comparison is a different
experiment with a different unit of observation. It is published through
[`bfcl-authoring-broader-evaluation.md`](bfcl-authoring-broader-evaluation.md)
and must not be folded into these arms.

## Metric registry is fixed, not supplied

An input may only reference a metric this contract already defines, and it may
not declare that metric's direction. An operator who could name a metric's own
direction could turn a regression into an improvement by editing one field.

The registry covers seven families: `effort`, `quality`, `coverage`,
`task_success`, `failure_codes`, `cost`, and `latency`. Every family except
`effort` is required to appear somewhere in the input; a family measured nowhere
is published under `coverage.unmeasured_families` and holds release readiness at
`incomplete` rather than being omitted.

Each measurement declares one of three kinds, because they support different
claims:

- `deterministic` — an exact count such as authored lines or distinct surfaces.
  The delta is exact, so no statistical test is applied and none is implied.
- `proportion` — a numerator over a denominator, such as validation pass rate or
  task success. Compared with a Newcombe score interval and a two-proportion
  score test.
- `repeated` — several observations of the same quantity. Compared with Welch's
  t-test, but only when both arms carry at least five observations; below that
  the comparison is published as `inconclusive` with an `underpowered` note.

## Material change versus noise

A metric is called material only when the relative delta reaches a
practical threshold, and significant only when the interval excludes zero after
Holm correction across every tested metric. Both must hold before a verdict of
`material_improvement` or `material_regression` is published. A delta that is
statistically detectable but practically negligible is `no_material_change`; a
delta that is large but not separable from noise is `inconclusive`.

Thresholds default per metric and may be overridden only with an explicit
`rationale`, so a threshold chosen to produce a favorable verdict is visible in
the published summary.

## A gate only counts when it could have failed

Each arm may declare truth-preservation gates, and each gate must state whether
it is `sensitive_to_intervention`. A gate that passes but cannot fail under the
intervention it guards carries no information: A2 returned `FROZEN` twelve times
from a verdict computed over `task_id` and `expected_tool_calls`, neither of
which A2 modifies. Recording that as evidence is how an ablation launders an
unchecked risk into a green report.

A passing gate that is not sensitive is published under
`vacuous_pass_gate_ids` and forces `adopt_with_conditions`. A failing gate
rejects the arm regardless of any effort or cost saving.

## Comparability is enforced, not assumed

Aggregation refuses to compare:

- task success across different `task_set_hash` values;
- priced cost across a different currency or pricing snapshot. Only a
  currency-denominated metric needs a `cost_context`; a token count is
  provider-independent, and demanding a pricing snapshot for one would push an
  operator into inventing it;
- the same metric measured as different kinds in the two arms;
- an input whose declared evidence hash does not match the file on disk.

A metric present in one arm and absent in the other is published as
`not_measured` with the known side retained. It is never treated as zero,
because an unmeasured metric and a metric measured at zero support opposite
decisions.

## Arm statuses

An arm is `measured`, `partially_measured`, `deferred` or `blocked`.

`partially_measured` exists for the arm that was executed only in a degraded
design. Such an arm carries evidence and real arm-local measurements, but the
design does not license a delta against the baseline, so the contract records
the numbers and withholds the comparison. Without this status the only options
would be to publish a delta the design cannot support or to discard a real
measurement, and both lose information a reviewer needs.

The status is enforced rather than annotated. A `partially_measured` arm must
declare a `deferral_reason`, must publish no baseline comparison, must publish
at least one arm-local measurement, and can only be recommended
`insufficient_evidence`. Its measurements do not close a coverage gap: they
appear under `coverage.families_measured_without_comparison` while the family
stays in `coverage.unmeasured_families`, so a withheld comparison can never
read as readiness.

Such an arm also publishes its own failure-code profile under status
`arm_only`, with `null` baseline counts and `null` deltas. The profile is what
tells a reviewer *how* the arm fails, and suppressing it for want of a delta
would discard the only failure-code evidence the ladder has.

The status covers two different reasons a comparison can be unavailable, and
the published ladder carries one of each. STEP4 is degraded by accident: the
frozen release it reads publishes exactly one wording per skeleton, so the
paired check it owes could not run on that data at all. STEP4B is degraded by
design: it builds the pairing STEP4 lacked and controls the intervention
against the template wording of its own skeletons, which is a stronger control
than the A0 baseline would be and is also not the A0 baseline. Both withhold
the delta, and only the second one does so because comparing to A0 would be the
wrong question rather than an impossible one.

Evaluation-side measurements are derived rather than transcribed. See
`bfcl-eval-measurements-contract.md`: it reconciles the task success rate it
recomputes against the rate the evaluation report already published, and
refuses to emit anything when the two disagree.

## Trade-offs

A trade-off is not a limitation bullet. It is a gain named alongside what the
gain was purchased with, and the price takes three forms that reviewers
routinely merge, so each is reported separately:

- `measured_costs`: a metric that materially regressed;
- `unverified_by_gates`: gates that did not run, or that passed while being
  incapable of failing under this intervention, so the gain's truth is unproven;
- `unpriced_families`: required families no arm can be compared to the baseline
  in, so any price paid there is invisible.

The derived verdict is one of `gain_against_measured_cost`,
`gain_with_unpriced_risk`, `gain_with_no_observed_cost`, `cost_without_gain` or
`no_gain_observed`. Validation refuses a summary that recommends `adopt` or
`adopt_with_conditions` while naming no gain at all.

## Recommendation policy

The recommendation is derived, never written by hand:

| Condition | Recommendation |
|---|---|
| arm is `partially_measured`, `deferred` or `blocked` | `insufficient_evidence` |
| any declared gate failed | `reject` |
| any metric materially regressed | `reject` |
| material improvement, no open conditions | `adopt` |
| material improvement with vacuous, unrun, unmeasured or inconclusive evidence | `adopt_with_conditions` |
| no material improvement | `retain_baseline` |

Release readiness is `blocked` when any arm is rejected, `incomplete` when any
arm lacks evidence or a required family is unmeasured, and `ready` otherwise.

## Claim boundary

The report always publishes `causal_claim: false`. A material change identifies
the arm that produced it under one baseline, not a causal mechanism, and the
validator refuses a summary that claims otherwise.

## Commands

Aggregate a ladder input and write both reports:

```shell
python -m nemotron.steps.byob.scripts.aggregate_bfcl_ablation \
  --input results/ablation-ladder.json \
  --output-dir results/summary
```

The CLI exits `0` when readiness is `ready` or `incomplete`, `1` when an arm is
rejected, and `2` when the input cannot support a trustworthy conclusion. Both
outputs are content-addressed: rewriting identical bytes is allowed, and
replacing them with different bytes is refused.
