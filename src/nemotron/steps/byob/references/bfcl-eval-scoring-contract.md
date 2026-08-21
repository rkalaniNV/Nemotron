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
the committed publication: `run_manifest.json` still hashes to what the
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

## How candidate output is observed

Each assistant turn is requested through the native OpenAI-compatible
function-calling protocol. The request contains only the model-facing
conversation and ordered tool definitions, plus the sampling parameters pinned
by the candidate config. Gold calls, assertions, fixtures, and unreleased oracle
results never enter the request.

The evaluator records the provider's `message.tool_calls` in order and keeps each
function argument exactly as returned. It parses the argument string once under
strict JSON. Invalid JSON, non-object JSON, missing arguments, wrong types,
unknown function names, and duplicate calls are candidate observations, not
transport failures: they are not retried, coerced, or repaired by another LLM.
Only transient endpoint failures may be retried, under the pinned retry budget
and deadline, and every attempt remains in the hash-verified candidate I/O cache.
Replay uses a committed completion without contacting the endpoint; an
interrupted cache sequence is not silently completed with a new sample. A
cancelled call is an interruption of the run, never an observation of the model,
so it is never scored as one.

## How the conversation advances

A multi-turn task is replayed rather than simulated. The prompt opens as the
system prompts and the first user request; nothing else from the published row is
in it. From there exactly three things may be added: an assistant turn the
candidate itself produced, a tool result the benchmark recorded, and the next
scripted user request.

A recorded tool result is released only after the assistant turn that asked for it
matches the trace, and it is addressed to the id the candidate's own call carried.
Every replayed call must declare type `function`, and ids within an assistant turn
must be non-empty and unique; a missing type is preserved as a mismatch rather
than repaired. Duplicate ids are ambiguous and receive no result.
This is what keeps a wrong call from seeing the right answer: a model that
transfers 999 instead of 100 is not handed the result of the correct transfer. The
next user request is released only after the turn that would have prompted it, so
a model that calls a tool where the trace asked a clarifying question never
receives the answer to the question it did not ask.

For an intermediate text-only turn, producing arbitrary text is not enough to
unlock the next user message: the text must equal the assistant text the published
trace recorded. This intentionally strict transport guard is separate from the
tool-call score and fails closed until the benchmark contract carries a generic,
deterministic semantic milestone richer than raw text.

The tool-result release decision uses the same call comparison this document
defines for scoring. That agreement matters in one direction in particular: a
release gate stricter than the scorer would end an episode the scorer would have
credited, so a correct model would fail a task on transport grounds.

Trace replay does constrain one thing the scoring rules do not. Because a recorded
result is only meaningful at the point the trace reached it, a candidate that
defers a call group to a later assistant turn ends the episode there. Within a
single turn, ordering follows the row's `call_order`.

An episode records what happened rather than a verdict: which turns were asked,
what came back, whether each sent turn advanced, and why it stopped — reaching the
end of the trace, a mismatch, malformed output, an unreachable endpoint, an
ambiguous tool-call id, or a spent turn or episode budget. A turn the episode
budget prevented from being sent is a terminal event, not a fabricated candidate
observation. Every number is derived from that record, so re-scoring never re-asks
the model.

## Argument matching

`argument_matching: schema_then_canonical` (*pinned*)

1. **Schema step.** Each argument is interpreted against the tool's declared
   parameter schema from the row's `tools`. When
   `insert_declared_defaults: true`, a parameter the schema declares a default for
   is filled with that default on whichever side omitted it, so a model that
   spells out a default is neither rewarded nor punished for it. Filling only the
   omitting side is what makes this do anything: filling both sides, or neither,
   would leave the two spellings unequal. Insertion recurses through nested
   objects and arrays and follows local `$ref` and `allOf` schemas.
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
  the configured number of `required_tools` first appearances as ordered and the
  remainder as unordered, which is how a pack declares "these two may happen in
  either order, but both come after login". Repeated calls to one tool do not
  consume additional prefix positions.

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

**What counts as having seen a task.** A benchmark records every model that read
a published row while it was being built: the model that profiled the reference
style, the one that paraphrased the surface, the one that judged surface quality,
and the one that translated the rows when a translation is scored. Each is
recorded with the rows it actually read, because scope matters — a paraphraser
that wrote three rows of fifty exposed three rows, not the benchmark. A
publication that cannot state this inventory is not scorable: a missing role
would read as "no contamination found".

**What counts as being the same model.** A candidate is compared against each
exposed model on the axes the two configs carry, strongest evidence first: two
weights digests settle it in either direction, then an equal operator canonical
id, then an equal serving route, then a normalized model name together with a
revision. Digests settle it only when they measure the same thing — identical
weights hash differently under two algorithms, so digests that disagree about
their algorithm are set aside instead of being read as two different models,
while the same digest recorded with and without its `sha256:` prefix is still one
digest. Model names are compared case-insensitively and with the registry
prefix and punctuation removed, so `meta/Llama-3.3-70B` and
`meta-llama/llama_3.3_70b` are not treated as different models. This
over-matching is deliberate: it turns a comparison that would have read
"different" into "unresolved", which asks the operator for a pinned identity
instead of clearing a candidate that may have written the rows.

**What happens when the comparison cannot settle it.** An unresolved comparison
is neither a violation nor a clearance. It never shrinks a task set on suspicion,
and it never aborts a debug run, but it always blocks publication: a published
score asserts that the candidate had not seen these rows, and an unresolved
comparison cannot support that assertion. Pinning `weights_digest` on both sides
resolves it.

**What the score is then taken over.** Under the pinned settings, a publishable
run has no collision at all — the alternative is refusal, not adjustment. The
debug paths are `exclude_row`, which drops exactly the exposed rows and reports
the reduced set, and `per_candidate`, which lets each candidate keep its own
eligible rows. Both are reported with the task-id hash of what was actually
scored, and neither is publishable: a number over a task set chosen per candidate
is not comparable to a number over the benchmark.
