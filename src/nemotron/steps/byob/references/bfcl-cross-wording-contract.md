# BFCL cross-wording stability contract

SOV-862 asks whether benchmark conclusions move when only the user wording
moves. With three or four target models a ranking carries almost no power, so
the sharper question is which individual tasks flip verdict — which needs the
same skeleton scored under both a human and a model wording, paired at task
level.

The contract version is `1.0`, implemented by
`nemotron.steps.byob.runtime.benchmark_families.bfcl.cross_wording_analysis`
and driven by `nemotron.steps.byob.scripts.analyze_bfcl_cross_wording`. Both
are read-only over the release and the evaluation artifacts.

## Why a frozen release usually cannot answer the question

Dedup and balancing publish one variant per skeleton. The published table
therefore contains human-worded and model-worded tasks, but never both wordings
of the same skeleton. On the Banking VN Gold release this is exact: 1392
published skeletons, 0 of which carry both wordings.

That leaves a real but weaker measurement available, and the contract's main job
is to keep the two apart rather than let the weaker one stand in for the
stronger one.

## Three separate readouts

**Paired wording design.** Reported as `available` or `not_available`. When it
is unavailable the report names the missing artifact instead of approximating
it. It also counts the counterpart wordings that were rendered into the release
but never scored, because that number decides whether closing the arm needs an
evaluation pass or a regeneration.

**Replicate verdict-flip floor.** Measured between two scored runs over the same
task set, using exact McNemar over the discordant pairs and pairwise agreement.
This says how often a verdict flips for reasons that have nothing to do with
wording. Without it a wording flip rate cannot be read at all.

**Unpaired wording contrast.** The published human-worded and model-worded tasks
compared as independent groups, overall and stratified by `turn_policy`,
`category` and `difficulty`, with a Newcombe score interval and a two-proportion
score test. Its `design` field is fixed to `unpaired_observational` and it
carries the confound in the report body: which skeletons were published under
which wording was decided by balancing, not by assignment.

The contrast also flags groups saturated at a floor or ceiling under both
wordings. A group where every task passes or every task fails cannot move under
any intervention, so a zero delta there is not evidence that wording does not
matter.

## Fail-closed loading

Analysis refuses to proceed when:

- a scored run does not cover exactly the published task set;
- the published table repeats a `task_id`;
- a published task has no wording provenance in the rendered conversations;
- a wording source is not one of `template` or `model`;
- a task maps to two different wording sources;
- a replicate is byte-identical to the primary run, and so cannot measure a
  floor;
- two runs are supplied under the same `run_id`.

## Conclusion policy

The conclusion is one of `stable`, `unstable`, `underpowered` or
`not_measured`, and is derived rather than written. `stable` and `unstable`
require the paired wording design; validation refuses a report that claims
either without it, so re-signing a forged verdict does not get past the
validator. When the paired design is missing the conclusion is `underpowered`,
which the SOV-862 acceptance criterion accepts as an outcome, and the report
lists what is required to conclude.

## Claim boundary

The unpaired contrast is a measurement, not an assignment. `policy.causal_claim`
is fixed to `false` and validation refuses any other value.

Reports are content-addressed over their semantic content: rewriting identical
bytes is allowed, and replacing them with different bytes is refused.

## Relationship to the ablation ladder

The readout enters the SOV-866 ladder as a `partially_measured` arm: it carries
evidence and a real arm-local task success rate, but the design licenses no
delta against the A0 baseline, and the baseline pack is a different task set
besides. See `bfcl-ablation-aggregation-contract.md` for how that status keeps a
withheld comparison from reading as readiness.
