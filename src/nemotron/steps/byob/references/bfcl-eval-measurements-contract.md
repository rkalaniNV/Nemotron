# BFCL evaluation measurements contract

SOV-866 requires the ablation summary to speak to `task_success`,
`failure_codes`, `cost` and `latency`. Those four families only exist once a
candidate model has been scored, and their numbers sit scattered across an
evaluation run: per-task verdicts in the results table, token usage and per-call
latency in the candidate IO cache, published aggregates in the evaluation
report.

The contract version is `1.0`, implemented by
`nemotron.steps.byob.runtime.benchmark_families.bfcl.eval_measurements` and
driven by `nemotron.steps.byob.scripts.extract_bfcl_eval_measurements`. Both are
read-only.

## Why derive rather than transcribe

A content-addressed ablation report is only as trustworthy as the numbers put
into it. Hand-typed figures cannot be re-checked, so this extraction derives
every value and records the sha256 of each file it read.

## Reconciliation is the load-bearing check

The extraction recomputes the task success rate from the per-task results table
and compares it against the rate the evaluation report already published. A
disagreement means the artifacts supplied do not describe the same run, so the
extraction fails instead of emitting numbers from a mismatched pairing. This is
the one check that catches the most likely operator error — pointing at the
wrong evaluation directory — and it is enforced in validation as well, so a
report whose `reconciliation.agrees` is anything but `true` cannot be published.

## Refusing to average over holes

A completion without token usage, or a selected attempt without a latency
sample, aborts the extraction. The alternative would let a missing observation
contribute an implicit zero, which silently understates a cost or a latency
distribution. Percentiles use nearest-rank without interpolation, so a report
hash does not depend on the platform's floating-point behaviour.

## Latency is reported with its caveat attached

Latency was observed against a shared inference endpoint under unknown
concurrent load. It characterises that run's serving conditions, not the
pipeline, and must not be compared across runs measured at different times. The
flag `latency_context.environment_dependent` travels with the number and
validation refuses any value but `true`, so the caveat cannot be dropped by
editing one field.

## Emitted measurements

| Metric | Family | Unit | Source |
|---|---|---|---|
| `task_success_rate` | `task_success` | ratio | per-task results table, reconciled |
| `input_tokens` | `cost` | tokens | summed `usage.prompt_tokens` |
| `output_tokens` | `cost` | tokens | summed `usage.completion_tokens` |
| `latency_p50_ms` | `latency` | milliseconds | selected-attempt `latency_s` |
| `latency_p95_ms` | `latency` | milliseconds | selected-attempt `latency_s` |

The failure-code distribution is emitted alongside them, counted over tasks
rather than over individual assertions.

Cost is reported in tokens, never in currency, unless a pricing snapshot was
actually captured. Inventing a monetary figure to fill the `cost` family would
be worse than reporting tokens.

Reports are content-addressed over their semantic content: rewriting identical
bytes is allowed, and replacing them with different bytes is refused.
