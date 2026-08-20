# BFCL Eval Scoring Contract

This document defines what a BFCL score *means*. It is referenced by
`scoring.contract` in `eval_config.yaml` and content-hashed into
`eval_config_hash`, so editing it changes the identity of every evaluation that
points at it: two runs that agree on every flag but disagree on the rules below
are not the same evaluation and must not be compared.

Schema version: `1.1`. Sections marked *pinned* are the values a publishable run
must use; a run that relaxes any of them is debug-only and reports
`publication_allowed: false` with the field that caused it.

## What is being scored

Each benchmark row states a user request, the tool definitions the model may
call, the gold tool calls (`expected_tool_calls`) with their turn, group, and
order, and the pack's success assertions. A candidate model sees the request and
the tools. It never sees the gold calls, the assertions, or the oracle results:
those are the answer key.

Two evaluation modes exist, and a report always says which one produced a number.

- `trace`: compare the calls the model proposed against the gold calls. No oracle
  runs, so a trace score says the model asked for the right thing, not that the
  request would have worked.
- `executable`: additionally replay the proposed calls against the oracle pack and
  evaluate the pack's assertions. Only gold-eligible source runs may be published
  in this mode, because only those rows were validated against a real oracle at
  generation time. The eval config must also pin a resolvable `source_oracle`:
  the matching pack manifest plus the exact Python backend or endpoint config.
  The source manifest's `oracle.kind` is lineage, not an executable resource.

## What the score is taken over

A number is only meaningful if the benchmark behind it is identified. Before any
candidate is contacted, the source named by the eval config is verified against
the publication Stage 12 committed: `run_manifest.json` still hashes to what the
config resolved, both benchmark tables hash to every declaration the manifest
makes about them, the published table is a selection of the raw table with no
truth rewritten and no held-out row shipped, and the published rows decode into a
unique task set. Every score cites the resulting `verification_identity`, which
covers hashes, row counts, task ids, and the oracle pack fingerprint, and no path
or timestamp. A run that cannot name the source verification it executed under is
not publishable, because "the model scored 0.71" without a benchmark identity is
not a comparable result.

An `executable` score additionally requires the oracle pack to still fingerprint
to what generation certified, across every file in its tree, and requires the
resource that runs to be the one the pack's own manifest selects. Replaying
against an oracle that changed since generation cannot confirm the gold trace it
certified, so such a run is refused rather than reported with a caveat.

The source is re-checked immediately before execution and the run aborts if
anything moved. Two halves of a run taken against two different benchmarks would
otherwise be averaged into one number.

A translated benchmark is scored under this same contract only if it preserves
every field the contract compares — the tools, the gold calls with their turn,
group, and order, the assertions, and the gating columns. Translation may change
the conversation, the intent, the system prompt, and row metadata. A translation
that changes anything a scorer reads is a different benchmark and its scores are
not comparable to the source's.

## Argument matching

`argument_matching: schema_then_canonical` (*pinned*)

1. **Schema step.** Each argument is interpreted against the tool's declared
   parameter schema from the row's `tools`. When
   `insert_declared_defaults: true`, a parameter the schema declares a default for
   and that neither side supplied is filled with that default on both sides, so a
   model that spells out a default is neither rewarded nor punished for it.
2. **Canonical step.** Both sides are then compared as canonical JSON, by type as
   well as by value. `1`, `1.0`, `"1"`, and `true` are four distinct values: a
   scorer that treated them as equal would accept a limit of `"1"` where the gold
   call passed the integer `1`.

Extra arguments the schema does not declare are a mismatch. Missing required
arguments are a mismatch. Argument order inside an object is irrelevant;
canonical JSON sorts object keys.

`canonical_only` skips the schema step. It exists for debugging packs whose
schemas are incomplete and is not publishable.

## Call selection, grouping, and order

- **Selection** is scored per call: the function name must be one the row's
  `tools` declares, and it must be the name the gold call used.
- **Grouping** (`respect_call_group: true`, *pinned*) requires the calls a model
  emits in one assistant turn to match the gold call group. Parallel calls the
  benchmark asked for in one turn may not be spread across turns, and sequential
  calls may not be collapsed into one.
- **Ordering** (`respect_call_order: true`, *pinned*) requires the gold order
  within a group and across groups. A row that carries `call_order_prefix` scores
  the prefix as ordered and the remainder as unordered, which is how a pack
  declares "these two may happen in either order, but both come after login".

## Task success

`task_success: all_applicable_gates` (*pinned*)

A task counts as a success only when every gate that applies to it passes: tool
selection, argument match, grouping and ordering, and — in `executable` mode —
oracle replay plus every success assertion the pack declared for the row. A gate
that does not apply to a row (a single-call row has no ordering gate) is not
counted for or against it.

`assertions_only` scores just the pack assertions. It answers a different
question ("did the end state come out right, however the model got there") and is
not publishable as a function-calling score.

## Repair

`allow_llm_repair: false` (*pinned*)

No model — including a judge — may rewrite, complete, or reinterpret a candidate's
tool call before it is scored. Repair turns a wrong answer into a right one and
makes the score a property of the repairer rather than the candidate.

## Determinism

Every decoding parameter is pinned by the config (`temperature`, `top_p`,
`max_tokens`, `seed`, `tool_choice`), and every runtime limit is pinned by
`limits`. A truncated or timed-out response is a failed task, not a skipped one:
silently dropping it would let a slow model score higher than a fast wrong one.

Candidate identity is weight identity. `provider` and `model` name the route a
request took; a 40–64 hexadecimal commit in `model_identity.revision` or a
`weights_digest` names what answered. Branches and tags are not accepted as
immutable merely because their current names are absent from a denylist. Model
and revision case is preserved for registries where case is meaningful. Scores
are only comparable across runs that pin the same weights.

## Contamination

`contamination.enforce: true`, `on_violation: fail_run`,
`comparison_set: common_intersection` (*pinned*)

Held-out material is not a scoring dimension; it is a validity precondition. When
a candidate is found to have seen a task, the run fails rather than quietly
dropping the row, and all candidates are scored on the same task set so a
difference between two numbers cannot come from a difference in which rows each
one answered.
