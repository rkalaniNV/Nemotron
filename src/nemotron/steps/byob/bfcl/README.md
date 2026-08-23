# BYOB BFCL

The `bfcl` benchmark family builds function-calling benchmark artifacts from an
executable oracle pack. Unlike the MCQ family, BFCL does not generate questions
with a model: pack templates define the conversation, while the backend and
assertions establish the expected tool behavior.

## Quick Start

Install the BYOB dependencies:

```bash
uv sync --extra byob
```

Run the bundled tiny reference pack:

```bash
nemotron steps run byob/bfcl \
  -c src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  stage=all \
  family=bfcl
```

For a larger example covering every supported conversation policy, replace
`tiny.yaml` with `banking_vn.yaml`.

Use `stage=prepare` to validate a pack without generating benchmark rows:

```bash
python -m nemotron.steps.byob.scripts.validate_oracle_pack --config <CONFIG>
```

## Pipeline

BFCL supports three stage values:

- `prepare`: normalize and validate the oracle pack.
- `generate`: require a gold-eligible pack, generate tasks, replay them, and
  publish artifacts.
- `all`: run `prepare` followed by `generate`.

Generation runs:

```text
expand -> state_machine -> render -> expected_trace
       -> schema_validation -> executable_replay -> final_output
```

BFCL does not currently support `translate` or `skip_until`.

## Oracle Pack

A runnable pack uses these files:

| File | Purpose |
| --- | --- |
| `manifest.yaml` | Identifies the pack and declares languages, paths, prompts, primary keys, and confirmation behavior. |
| `tools.json` | Defines the model-facing function schemas and pack-local mutation or confirmation metadata. |
| `backend.py` | Implements the local executable oracle. Use this or `endpoint_config.yaml`, never both. |
| `endpoint_config.yaml` | Connects to a BFCL Oracle HTTP v1 service over HTTPS and pins its identity/content digest. Use this or `backend.py`, never both. |
| `fixtures.json` | Supplies deterministic records and initial backend state for generated tasks. |
| `task_templates.yaml` | Declares intents, slot sources, conversation policies, milestones, and success assertions. |
| `assertions.py` | Checks the final backend state and executed trace to decide whether a replay actually succeeded. |
| `validation_cases.yaml` | Provides positive and negative probes for backend/schema alignment, determinism, errors, and confirmation behavior. |

### Direct HTTPS endpoint

An endpoint-backed pack declares `paths.endpoint: endpoint_config.yaml` in its
manifest, or sets `oracle_pack.endpoint_config_path` in the run config. The
endpoint must implement BFCL Oracle HTTP v1:

- `GET /v1/metadata`
- `GET /v1/tools`
- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/calls`
- `GET /v1/sessions/{session_id}/state`
- `DELETE /v1/sessions/{session_id}`

Session creation receives the frozen clock, seed, task id, timeout, and fixtures.
It returns a unique `session_id` plus the oracle identity. The endpoint identity
and `content_digest` must match `endpoint_config.yaml` during validation, replay,
and final publication.

Only HTTPS is accepted. Bearer tokens and custom secret headers are referenced by
environment-variable name; their values are never stored in the pack, report, or
manifest. See
[`../references/bfcl-endpoint-config.example.yaml`](../references/bfcl-endpoint-config.example.yaml)
for a complete configuration example.

Pack code must live under an `oracle_runtime.allowed_roots` entry. Gold
eligibility requires `oracle_runtime.worker: process`; thread mode is available
only for debugging.

Start from the bundled packs under `../data/`:

- `tiny_oracle_pack`: smallest end-to-end example.
- `banking_vn_oracle_pack`: domain-sized example with all supported turn
  policies.

## Outputs

Artifacts are written to `output_dir/expt_name/`:

- `benchmark_raw.parquet`: every schema-valid, replay-valid row, before Stage 10
  drops and before Stage 11 deduplication and balancing.
- `benchmark.parquet`: published benchmark rows.
- `run_manifest.json`: lineage, fingerprints, stage counts, and artifact hashes.
- `exports/bfcl_json/`: optional BFCL question/answer JSONL pair.
- `exports/nemo_evaluator_bundle/`: optional six-file W5 adapter input bundle.
- `exports/export_validation_report.json`: read-back equivalence evidence when
  at least one compatibility export is enabled.
- `stage_cache/`: normalized inputs and one table per generation stage, keyed by
  `task_id`.

Both parquets carry the same schema, and the difference between them is a
selection, never a rewrite. `publication_contract` (`1.0`) re-derives the
publication set from the stage decisions, reads both files back from disk, and
refuses the run unless the published rows are byte-identical to their raw
counterparts across every column — `PUBLICATION_RESTATED_FIELDS` is empty, so a
row that ships is exactly the row the audit table records. Publication order is
the Stage 11 selection rank when deduplication ran, and the raw order otherwise.
`held_out_hit` is `false` on every published row once a held-out policy has been
evaluated, and `null` when no policy was declared. The manifest's `publication`
section reports both row counts, both content hashes, which surface gate
decided, and which ordering applies.

## Configuration

Start from `config/default.yaml` for a new pack. The main settings are:

- `oracle_pack.manifest_path`
- `oracle_runtime.clock`, timeouts, `worker`, and `allowed_roots`
- `task_generation`: category budget, turn/tool limits, and normalized
  `difficulty_mix`, `turn_mix`, and `tool_call_count_mix`
- `surface_generation.language`
- `lineage.policy`
- `exports.bfcl_json` and `exports.nemo_evaluator_bundle`

```yaml
exports:
  bfcl_json: true
  nemo_evaluator_bundle: true
```

Both flags default to `false`. Enabling either makes export read-back validation
part of the Stage 12 publication transaction; the flag is never silently ignored.

Evaluation settings do not live here. They belong in their own `eval_config.yaml`
(see [Evaluation Config](#evaluation-config)), so that changing a candidate model
cannot change the identity of the generated benchmark.

Week 4 contracts are parsed strictly even while their owning stages remain
disabled. Enabled model roles require a non-secret `canonical_id`, and
`strict_separation` requires every enabled role to use a distinct identity.
`reference_benchmark` pins an allowlisted JSONL source by SHA-256:

```yaml
reference_benchmark:
  name: bfcl_vi_style
  samples_path: /data/reference_bfcl_samples.jsonl
  content_hash: sha256:<64-hex-digest>
```

Semantic deduplication accepts `model_identifier`, `n_clusters`, `eps`, and
`remove_duplicates`. When enabled, generation runs it after Stage 10 and applies
the resulting total selection order to `benchmark.parquet`.

Surface quality uses the versioned `1.1` six-check contract:

- Python owns `surface_shape`, `semantic_preservation`, and `leakage`.
- The optional surface judge owns `language_locale`, `fluency_naturalness`,
  and `clarity_coherence`.

The judge contract is surface-only: it cannot label tool correctness, change
arguments, inspect oracle results, or rewrite benchmark truth. A complete
quality result contains exactly one verdict for each check. Python checks are
`passed` or `failed`; judged checks may also be `not_applicable` when the task
policy intentionally permits the observed condition, `not_run` when the judge
is skipped, or `error` when the judge call fails. These states are not quality
failures and are not counted as passes. The deterministic stage can be enabled without a judge, while
`drop_authority: true` requires one. Judge responses carry only a controlled
reason code — no free-text evidence. Turn-policy applicability is checked when
the six results are assembled: intentional ambiguity in `clarify_only` is not a
quality failure.

Python owns the first three checks by mapping the existing render and paraphrase
guards (`must_preserve`, `must_omit`, `must_not_mention`, `novel_literal`,
`expected_result_leakage`, `semantic_shape`). Canonical template surfaces never
fail `unchanged_surface`. Each task gets a complete six-check record. When the
optional judge is disabled the judged checks are `not_run`; when it runs, it
sees only language, user-facing turns, style hints, and the surface rubric, and
`clarify_only` ambiguity is recorded as `not_applicable` before assembly.
Expected-result and novel-literal values remain private guard diagnostics and
are not copied into quality-record evidence.

Drop authority is deliberately asymmetric. A Python failure always drops the
row, because those three checks protect semantics and leakage. A judge failure
drops the row only under `drop_authority: true`; otherwise it is recorded as an
advisory observation that changes nothing. A judge error never decides
anything: an advisory run records it and continues, while an authoritative run
refuses to publish, since a gate that could not answer was never enforced.
If the policy drops every replay survivor, final output also refuses to stamp an
empty benchmark as gold.
Stage 10 writes `stage_cache/surface_validated_tasks.parquet` with one row per
task: identity, contract version, keep/drop authority, six queryable statuses,
and canonical JSON check detail. Nested detail remains JSON text so arbitrary
pack-specific evidence cannot mutate the Arrow schema. Generation still rejects
later-stage features it cannot honor, but Stage 10 now runs between replay and
final output, filters publication rows, and records its report and artifact
hashes in `run_manifest.json`.

The immutable judge I/O cache is shared and append-only, so the manifest does
not hash that changing file as though it belonged to one run. Instead,
`surface_judge_cache_usage.json` records only the request, input, and observed
response hashes this run used (including an empty request list when Python
rejected every surface first), and the manifest hashes that per-run usage file.
The Stage-10 end-to-end tests also run an English warehouse-asset oracle pack
outside the bundled banking and tiny domains; checks and parquet schemas contain
no domain-specific branching.

```yaml
surface_quality_validation:
  contract_version: "1.1"
  enabled: false
  drop_authority: false
```

Stage 11 uses versioned contract `1.0`. It requires exactly one decision for
every Stage-10 survivor, preserves input order, and restricts balancing reports
to eight generic dimensions: `intent`, `category`, `required_tools`,
`tools_present`, `difficulty`, `turn_class`, `tool_call_count`, and
`turn_policy`. Controlled drop reasons distinguish semantic duplicates, balance
quotas, and hard turn/call limits. A selected row cannot carry drop detail, and
selected rows carry `selection_rank` `0..k-1` exactly once so publication order
is total.

Deduplication may only collapse tasks that share one coverage bucket, so every
cluster holds exactly one representative and never mixes `language`,
`turn_policy`, or pack-defined edge signatures. A representative may itself be
dropped on quota only when every member of that cluster is dropped. Validation
also preserves at least one selected row for every complete input coverage
bucket, not merely each language, policy, and edge value independently. Bucket
keys and task identifiers are normalized, because bucket identity is string
equality and an untrimmed variant would otherwise become a phantom bucket.

Stage 11 embeds a text projection rather than the published row. The projection
keeps only user-authored turns, in conversation order, each prefixed with a
`[user]` marker; assistant milestones, tool-call payloads, and oracle results
never reach it. Turn whitespace is collapsed, and every literal the pack bound
into the task, corrections included, is masked to `<slot_name>`, so two tasks
that differ only in a bound id project the same text. Masking matches whole
tokens and longest literals first, so a short value cannot corrupt a word that
merely contains it, and a literal two slots share masks under one deterministic
slot name. Correction aliases are masked as well as canonical slot keys.

Embedding runs through the shared Curator semantic-dedup workflow, with
`task_id` as the id and the projected text as the embedded field. `n_clusters`
is capped so non-trivial inputs retain an average of at least two rows per
partition; setting k equal to the row count would turn pairwise deduplication
into a collection of singletons. The effective value is reported. A set of
fewer than two rows never reaches the embedding backend: one surface cannot
duplicate anything and zero surfaces have nothing to embed. `eps` is constrained
to `(0, 1)`, matching Curator's cosine-similarity threshold and keeping singleton
sentinels below it. Both duplicate flags and cluster membership come from one
Curator run. Stage 11 consumes Curator's K-means-partitioned pairwise artifact:
each row crossing `1 - eps` is linked to the predecessor Curator ranked for it,
and those links form the duplicate clusters. It also requires that the official
Curator duplicate artifact names exactly the same ids. This avoids a second
similarity implementation, comparisons across Curator's candidate partitions,
and an extra global quadratic pass. The run records the settings hash, the
projected-input hash, and a signature over the embeddings themselves, so a
deduplication decision can be traced to the exact model, settings, and vectors
that produced it. A backend result is rejected before it can decide anything if
it fails to cover the embedded ids, if Curator's pairwise and duplicate
artifacts disagree, or if a cluster does not carry exactly one non-duplicate
representative. Embeddings decide duplication only; nothing a model produces
reaches a task's text, calls, arguments, or assertions.

Curator clusters are then partitioned by the complete
`(language, turn_policy, edge_signatures)` coverage bucket and by a hash of the
task's generic executable capabilities (`required_tools`, `tools_present`,
assertion contract, mutation flag, and call-order policy). A Curator
representative in one partition therefore cannot erase the final survivor of
another. Each resulting multi-row partition selects exactly one representative
using, in order: eligibility under the run's `max_turns` and `max_tool_calls`
limits, no judge error or advisory failure, applicable-check status,
coverage rarity, configured surface-source preference, a seeded hash, and
`task_id`. The default source preference is `template` before `model`; a run can
reverse it explicitly. Metadata retains the original Curator cluster, duplicate
flag, predecessor, similarity score, final cluster, capability signature,
coverage bucket, and every deterministic rank component.

Balancing then projects every candidate onto the eight locked dimensions:
`intent`, `category`, canonical `required_tools`, canonical `tools_present`,
`difficulty`, `turn_class`, `tool_call_count`, and `turn_policy`. `max_turns`
and `max_tool_calls` are hard filters; removing the final survivor of a coverage
bucket aborts instead of weakening the lock. One representative per complete
coverage bucket is selected before quotas. Category caps and configured
`difficulty_mix`, `turn_mix`, and `tool_call_count_mix` targets are then applied
without cloning rows. Fractional targets use deterministic largest-remainder
allocation. Selection is a deterministic binary optimization: coverage and
representative lineage are hard constraints, while total category-cap and
cross-dimension target deviation is minimized globally before stable rank breaks
equivalent optima. This avoids both greedy local optima and the former cubic
exchange pass, so a feasible mix is not reported unmet merely because of the
order rows were picked in. Minimal environments use an exact bounded fallback;
production BYOB environments use PuLP/CBC. Targets that inventory or a locked
coverage survivor genuinely prevents are returned with explicit
inventory, target, actual, and reason metadata rather than being silently
claimed as met.

Stage 11 writes `stage_cache/balanced_tasks.parquet` with one row for every
Stage-10 survivor, including Curator lineage, final cluster and representative
IDs, duplicate/selection verdict, drop detail, total publication rank, locked
coverage, and all eight balancing dimensions. It also writes
`stage_cache/dedup_balancing_report.json` with pre/post counts, grouped
selection/drop statistics, target and actual mixes, unmet targets, hard-limit
drops, rare-edge preservation, and the semantic/config/artifact hashes needed
to audit the decision. Both files are replaced atomically.

`remove_duplicates` controls whether duplicate rows must be dropped or may be
retained as annotations. With `false`, semantic similarity alone never drops a
row; subsequent hard limits and balancing quotas still apply and retain their
own drop reasons.

Stage 11 is fail-closed. An embedding/Curator error propagates and no final
parquet or manifest is written. Missing Stage-11 artifacts stop final output,
and an empty selected set cannot become an empty gold benchmark. Infeasible
soft targets are always recorded without inventing rows.
`unmet_target_policy: abort` (the default) leaves the diagnostic Stage-11
artifacts but stops before publication. `publish_non_gold` permits publication
only after setting both manifest and row `gold_eligible` values to false and
recording `stage_eleven_unmet_targets` as the reason.

Enabling Stage 11 requires Stage 10 and all Stage-11 model/clustering settings
to be present, so deduplication never admits an unvalidated or under-specified
run. The manifest embeds the Stage-11 report, model identifier, settings hash,
embedding signature, stage counts, and hashes of both Stage-11 artifacts.

```yaml
semantic_deduplication_config:
  contract_version: "1.0"
  enabled: false
  model_identifier: sentence-transformers/all-MiniLM-L6-v2
  n_clusters: 20
  eps: 0.08
  remove_duplicates: true
  representative_source_preference: [template, model]
```

A pack that references `held_out.yaml` from its manifest is enforced under
versioned contract `1.0` at two points. Stage 4 never binds a reserved template
or fixture row: the reservation is applied while slot candidates are collected,
so a reserved row cannot enter a task at all. A slot whose every matching row is
reserved, a category that falls short of `tasks_per_category` once reservations
are honoured, and a policy that reserves every template all stop generation with
an explicit error instead of quietly publishing a smaller or differently mixed
set. Stage 4 records what it examined and withheld in
`stage_cache/held_out_bindings.json`.

Stage 12 re-scans every executable row against the same policy before anything
is written, comparing the canonical JSON `[collection, primary_id]` references expansion
recorded and the template each row came from. The scan writes
`stage_cache/held_out_scan.json` and stamps `held_out_hit` on published rows, so
the column reports a checked result rather than the null a policy-free run
publishes. Enforcement is fail-closed and abort-only: a single hit stops the run
before `benchmark_raw.parquet`, `benchmark.parquet`, or the manifest exists,
because Stage 11 has already fixed the publication set and silently dropping a
row would break the balance the manifest reports. Missing or mismatched Stage-4
evidence stops publication for the same reason. The manifest carries the policy
lineage, scan counters, Stage-4 counters, and hashes of both artifacts, and
marks audit dimension `B7` `na` when a declared policy reserves nothing.

Compatibility exports are specified by versioned export contract `1.0`, with
`bfcl_json` and `nemo_evaluator_bundle` carrying their own `schema_version`.
The BFCL adapter pins the upstream `BFCL_v4_multi_turn` question/function and
separate ground-truth JSONL envelopes; Nemotron metadata retains assertions,
parallel groups, ordering policy, and provenance that upstream BFCL does not
represent. This is data-format compatibility, not a claim that arbitrary oracle
packs provide BFCL's domain-specific executable classes.

Both writers read one deeply immutable canonical projection of the
published parquet row. Tool definitions decode from canonical JSON text and each
argument decodes from its own canonical JSON value in the Arrow map, then
re-encodes byte-for-byte. The contract therefore distinguishes `"1"`, `1`,
`true`, and `1.0`. It requires row-count and truth-field equivalence, source
benchmark and validation-report hashes, and complete NeMo bundle references for
the dataset schema, metadata, evaluator config, and system-prompt catalog.

`export_projection` (`1.0`) is that single decode path: `benchmark.parquet` is
read once, its schema is checked against the published schema, and every row
becomes a canonical export object. A writer receives the projection, never a
path, so no format can decode the parquet its own way. `project_published_benchmark`
optionally binds the projection to the content hash and publication order the
manifest reports, which stops an export built from a parquet that was replaced
after Stage 12 verified it.

The projection also derives, once, the structure a writer would otherwise
reconstruct. Each assistant message that issues tool calls becomes one call
group, checked against the rendered `messages` by name and argument: a group is
parallel exactly when that message issues more than one call, its `turn_index` is
the ordinal of the assistant message, and `user_turn_index` is the request it
answers. `calls_by_user_turn` keeps an empty slot for a clarifying turn that
triggers no call, so BFCL's per-user-turn ground truth cannot shift answers onto
the wrong request. Projection-level provenance (pack, version, tier, prompts,
languages, turn policies) is derived from every row rather than read off the
first, so rows that disagree stop the export instead of being silently labelled
after row zero.

The `bfcl_json` writer turns that projection into two JSONL files under
`exports/bfcl_json/`: the questions and, beside them in `possible_answer/`, the
expected calls, joined by `id`. Four format decisions are fixed there. JSONL
rather than a JSON array, so a harness streams the benchmark and retries a single
task rather than a whole file. Two files rather than one, because a record that
carries its own answer invites a runner to prompt the model with it. Parallel
calls stay grouped in the Nemotron extension, since upstream's per-turn answer
list is flat and cannot tell two simultaneous calls from two sequential ones.
And the expected calls are the only answer exported: the recorded oracle results
stay under `x-nemotron.messages` as provenance, never in `question` or
`ground_truth`, so a scorer re-executes tools instead of diffing a model's output
against a snapshot of one backend revision. Provenance likewise lives only in the
extension, so rendering `question` cannot leak a pack version or a seed into the
prompt. Bytes are deterministic — sorted keys, no incidental whitespace, `\n`
endings, UTF-8 left unescaped so a Vietnamese surface stays readable — and the
format's `content_hash` covers file names together with bytes, so a renamed file
or a swapped question/answer pair changes the digest.

The `nemo_evaluator_bundle` writer turns the same projection into six files under
`exports/nemo_evaluator_bundle/`: `bundle.json` (the W5 adapter descriptor, which
names the other files and pins the dataset's hash and record count),
`dataset.jsonl` in publication order, `dataset.schema.json`, `metadata.json`,
`evaluator.yaml`, and `system_prompts.json`. Three decisions differ from
`bfcl_json`. `seed_messages` is the only model-input field and contains only
leading system messages plus the first user turn. Gold assistant actions remain in
`reference_trace`; `replay_steps` lets the W5 adapter release recorded tool results
only after the candidate produced the corresponding expected call, then release
the next user turn. This prevents a generic chat adapter from forwarding a full
gold trace as the prompt. The dataset schema is generated from the record model
rather than written by hand, so it cannot drift from the records beside it. And
the declared metrics stop at `tool_selection` and `arguments`, plus
`call_ordering` only when some task expects more than one call: `results` and
`task_success` would need the pack's tools re-executed against oracle state, which
no bundle file provides, and an ordering metric over single-call tasks would
report a perfect score for something it never measured. The adapter task id is
derived from `pack_id`; lossy normalization, including an entirely non-ASCII id,
gets a deterministic hash suffix to avoid collisions. The verbatim `pack_id`
remains in `metadata.json` and `bundle.json`.

`evaluator.yaml` is an adapter input contract, not a standalone NeMo Evaluator
Launcher run config. It explicitly declares that W5 must provide a registered
environment, candidate endpoint, and tool resource service. Publishing a dataset
bundle does not pretend those execution dependencies already exist.

The whole bundle is encoded, digested, and validated in memory before any file is
created, so a projection that cannot be expressed — an unresolvable prompt id, a
row no evaluator record represents — leaves nothing behind to be mistaken for a
bundle. The bundle directory is cleared first, because a file this run did not
write would otherwise travel inside a bundle whose digest never covered it, and
the bytes on disk are re-digested afterwards so a truncated write cannot publish a
descriptor nobody re-checks. `content_hash` and `files` are bundle-relative like
the descriptor, so archiving the directory elsewhere does not invalidate it.

Both writers share one projection per run and write under `exports/`, which is
removed before validation so a run that disables a format cannot inherit the tree
a previous run left behind. Which formats can actually be written is declared once,
as the writer registry Stage 12 dispatches through; config validation reads the same
registry, so a format named in the contract but never wired to a writer is refused
at startup instead of silently producing no file. A writer that fails takes the
export tree and both parquet files with it, since a reader cannot tell a partial
bundle from a complete one, and any later abort in Stage 12 discards the tree for
the same reason.

Writing alone does not certify the export. Stage 12 reads every enabled format
back, checks its tree hash, row count, task order, canonical truth fields, and
format-specific envelopes against the single published projection, then writes
`exports/export_validation_report.json`. Any mismatch aborts publication.
`run_manifest.json` records enabled and disabled formats, schema versions, row
counts, content hashes, source benchmark hash, and the validation-report hash.

Parquet files, exports, validation report, and manifest are built in one staging
directory. Stage 12 promotes payloads only after all validation and drift checks
pass, and moves `run_manifest.json` last as the commit marker. A failure removes
the staging tree and leaves no final manifest or partially published benchmark.

For recovery, never patch a generated export in place. A schema or unsupported
call-layout error requires fixing the pack/template or using a matching consumer
and rerunning Stage 12. A hash/equivalence error requires regenerating the whole
publication. If `run_manifest.json` is absent, treat any adjacent parquet/export
as unpublished and rerun; startup clears abandoned `.stage12-*` attempts.

## Evaluation Config

Evaluation is a separate run over a benchmark that was already published, and its
input is `eval_config.yaml` (schema `1.1`). Start from
[`config/eval.default.yaml`](config/eval.default.yaml); the loader lives in
`runtime/benchmark_families/bfcl/eval/`. It parses, resolves, and hashes the
config. No candidate model is contacted, so an invalid config fails before a
single token is paid for.

The config names a `source_run_manifest`, never a bare parquet: `run_manifest.json`
is Stage 12's commit marker, so a directory holding a benchmark without one holds
unpublished bytes. Which table to read, whether the run is gold-eligible, and
which oracle kind produced it come from that manifest rather than being restated
by the operator, so the two cannot disagree. `executable` mode additionally
requires `source_oracle`: the exact pack manifest and concrete `backend.py` or
endpoint config. Both must exist, their pack id/version and kind must match the
source run, and their bytes enter `eval_config_hash`. A manifest's `oracle.kind`
alone is only lineage; it is not an executable resource. Relative paths resolve
from the eval config's own directory.

Nothing defaults. Every scoring gate, runtime limit, and decoding parameter is
stated, because each one changes what the number means: a model cut off at two
turns did not answer the same question as one given ten. Quoted booleans and
numbers are refused rather than coerced, since a `"false"` that becomes `true`
would silently switch off a correctness gate.

A candidate separates two identities. `provider`, `model`, and `api.base_url` name
the route a request takes; `model_identity` names the weights that answered, and
must pin an immutable `revision` or a `weights_digest`. Branch-style refs such as
`main` or `refs/heads/*` are refused, and without a digest the revision must be a
full 40–64 hexadecimal commit id; an arbitrary branch or tag cannot prove
immutability. Model and revision remain case-sensitive for registries that
distinguish case. Two candidates may not share an alias or resolve to the same
canonical weight identity. Credentials never appear: `api.api_key_env` names an
environment variable, a literal key anywhere in the file is refused, validation
diagnostics never echo string values, and a missing variable is an execution
failure rather than a config error.

`scoring.contract` points at
[`../references/bfcl-eval-scoring-contract.md`](../references/bfcl-eval-scoring-contract.md)
and is content-hashed, so editing what "argument match" means changes the config's
identity. Publication requires the locked gates — `schema_then_canonical`
argument matching, call order and grouping respected, no LLM repair,
`all_applicable_gates` task success, contamination enforced with `fail_run` on a
common intersection, and every artifact written. Relaxing any of them is allowed
only with `publication.requested: false`, and the config then reports each
weakened field in `non_publication_reasons`. Executable publication additionally
requires a gold-eligible source run, since only gold rows were validated against a
real oracle. `outputs.output_dir` must sit outside the generation publication tree,
must be a directory when it already exists, and
`write_resolved_eval_config()` only writes below it. An eval run therefore cannot
overwrite `run_manifest.json` or the benchmark it scores.

Resolution ends in one `eval_config_hash` over the config's *meaning*: referenced
files enter as content hashes, candidates are ordered by alias, modes are
canonically ordered, and absolute paths, output locations, and secret values are
absent. Moving the checkout leaves the hash alone; changing a candidate, a
revision, an inference parameter, a limit, the scoring contract, or the source run
changes it. `write_resolved_eval_config()` writes the same content as an auditable
`resolved_eval_config.json`, with resolved paths kept outside the hashed payload;
relative writer paths resolve below `outputs.output_dir`.

An eval config may also be referenced from the generation config through
`eval_config_path`, or inlined as a legacy `eval` block; both go through the same
validator, and carrying both is refused as ambiguous. Either way the eval input is
excluded from `generation_config_hash` and `resolved_config_hash`: evaluating a new
candidate must not change the identity of the benchmark it was scored on. Until the
eval runner lands, `stage=generate` refuses both keys rather than accepting a
setting no stage of that run applies.

## Source Verification

The eval config records what an operator *named*. `verify_eval_source()` reads it
back from disk and holds it to that record, before any candidate token is spent.
It is the only way a runner obtains a source: there is no constructor for a
`VerifiedEvalSource` that skips verification, so "the runner scored an
unpublished parquet" is not a reachable state.

What it proves, in order:

1. `run_manifest.json` is a Stage 12 commit marker — correctly named, carrying
   every publication field, declaring a schema this build reads — and its bytes
   still hash to what the config resolved. Structure is checked before identity,
   because "this is not a manifest" and "this is a different manifest" call for
   different fixes.
2. Both tables hash to what the publication declares, in all three places that
   declare them: the `publication` section, the `artifacts` section, and the
   resolved eval config. A symlink is refused; a link can be re-pointed at
   another benchmark without changing anything the manifest records.
3. The two tables stand in the relationship `publication_contract` (`1.0`)
   defines, replayed here over the files on disk: the published table *selects*
   raw rows without rewriting truth, in the declared order, and ships no held-out
   row.
4. Every published row decodes under this build's benchmark schema into a unique,
   addressable task index. A row the evaluator cannot decode is not skipped,
   because skipping it would change the task set. Task ids must survive being a
   path component and a log token; non-ASCII letters are fine, path separators,
   control characters, and reserved names are not.
5. For `executable` mode, the oracle pack still fingerprints to what generation
   certified across every file in its tree — a helper module the backend imports
   changes what the oracle does — and the resource that will run is the one the
   *pack's own manifest* selects, never an eval-side override. A Python backend is
   imported in a throwaway process worker to confirm it exposes `list_tools`,
   `reset`, `call_tool`, and `get_state`; an endpoint pack's pinned oracle
   identity must equal the `endpoint_metadata` the source run recorded.
6. Every model that read a published row while it was being built is named,
   together with the rows it read: the `profile`, `paraphrase`, and
   `surface_judge` roles from `models.*`, and the translator when a translation
   is evaluated. Scope comes from the rows wherever the schema records it — a
   profile that shaped nothing and a paraphraser that wrote three of fifty rows
   are both narrower than "the whole benchmark". A manifest that omits the block,
   declares a role this build does not read, enables a role without naming a
   model, or ships a paraphrased or profile-shaped row no role accounts for is
   refused, because a gap in this inventory reads as "no contamination found".

None of these are reimplementations. Publication semantics come from
`publication_contract`, row decoding from `export_projection`, the pack file set
and fingerprint from `pack_loader`, and endpoint identity from `endpoint`. A
verifier that re-derived any of them could disagree with the pipeline that wrote
the artifact, and then the disagreement would be the bug.

Two things are deliberately out of scope. No live endpoint is contacted: that
would make offline trace-only evaluation impossible, and an endpoint's *current*
identity is an execution-time question. And no oracle task is replayed —
verification proves the backend can be driven at all; replay is the runner's job.

A translated benchmark is verified against its source rather than trusted. It
must derive from this run, declare its language, table, and task-id hash, match
its declared bytes, carry exactly the source task ids in publication order, and
leave every truth field byte-identical under canonical JSON. Translation may
change the conversation, the stated intent, the system prompt, and row metadata;
anything a scorer compares against must survive unchanged, since a translation a
candidate can pass while failing the source is not a translation of this
benchmark.

A pass writes `source_verification_report.json` into `outputs.output_dir`,
atomically, listing each check that actually passed. A failure writes
`source_verification_failure.json` instead — a different file name, so no reader
can mistake a diagnosis for a pass by seeing which artifact is present. The
report's `verification_identity` hashes hashes, row counts, task ids, and pack
fingerprints, and no path or timestamp: moving an intact publication tree must
not change what was verified, and changing one byte inside it must.

Finally, `assert_source_unchanged()` runs immediately before execution.
Verification and use are separated in time, and that gap is exactly where a
source gets replaced — a re-run of generation into the same directory, a pack
edited to make a failing task pass. Every recorded hash is recomputed, the pack
fingerprint included, so a run can never span two sources and report one score.

## Contamination Gate

Verification says which benchmark is being scored. `evaluate_contamination()`
decides *who may answer which rows of it*, and returns an `EligibleEvalPlan` —
the second and last handle a runner is given. Asking a candidate a task the gate
excluded is not a reachable state, because the only task list the runner has is
the one on the plan.

Each candidate is compared against each exposure on the axes the two artifacts
actually carry. A generation config only had to name what it *called*: a serving
route and an operator `canonical_id`, with weight identity optional. An eval
config must pin an immutable revision or a weights digest. Neither identity is a
superset of the other, so the comparison weighs evidence strongest first — two
comparable weights digests settle it either way, then an equal operator label,
then an equal serving route, then a normalized model name plus revision — and
returns one of three verdicts:

| Verdict | Meaning | Effect |
| --- | --- | --- |
| `different` | The candidate is provably other weights | Nothing; not recorded |
| `match` | The candidate is the model that read those rows | Violation |
| `unknown` | Neither side pinned enough to tell | Recorded as evidence |

The asymmetry is deliberate. Config validation compares candidates *to each
other* and keeps identifiers case-sensitive, because collapsing two case-variants
would hide a real difference between two candidates. Here the dangerous mistake
is the opposite one, so every comparison is case-insensitive and model names are matched on a
normalized form that drops the registry prefix and punctuation. An `unknown`
verdict costs an operator a pinned identity; a wrong `different` verdict costs the
benchmark its validity. Two digests are only allowed to establish a separation
when they measure the same thing: the generation side's digest is whatever the
pack config wrote, so one recorded without its `sha256:` prefix is still the same
digest, and two under different algorithms settle nothing. Every candidate is
compared before any is refused, so one run names all of them.

Then the policy applies, and it only ever narrows:

- `fail_run` refuses the run on a `match`. That is the locked publication
  setting: a publishable comparison either has no collision or does not happen.
- `exclude_row` drops exactly the rows the exposure covers, and only those. The
  remaining score is honest but covers less than the benchmark, so it is recorded
  as `contamination.excluded_rows:<alias>` and is not publishable.
- `unknown` evidence does neither. It never shrinks a task set on suspicion, and
  it never aborts a debug run. What it always does is block publication — and
  when `publication.requested` is true the refusal happens here rather than
  producing a number that cannot be published.

Intersection is last. Under `common_intersection` every candidate answers the rows
all of them may answer, in publication order, so two scores are comparable by
construction rather than by convention; under `per_candidate` each keeps its own
set, which is why the config contract refuses to call such a run publishable. If
contamination empties either set, the run stops: a benchmark whose surface models
are the candidates cannot be salvaged by scoring zero rows.

A pass writes `contamination_report.json` into `outputs.output_dir`, naming every
exposure, every collision with its task ids, and each candidate's eligible,
excluded, and evaluated rows; a refusal writes `contamination_failure.json` and
removes the stale pass, and the other way round. `plan_identity` hashes the whole
decision and no path or timestamp, and candidates are ordered by alias, so
reordering two candidates in the YAML does not fork the hash. `assert_plan_unchanged()`
re-pins the source and re-derives the decision immediately before the first
request, so a plan that was widened between authorization and execution — a
candidate added, an exclusion dropped, a policy relaxed — cannot be the plan a
runner acts on.

## Native Function-Calling Client

`NativeFunctionCallingClient` is the native transport primitive for one assistant
turn. `build_candidate_request()` sends the ordered model-facing `messages` and
OpenAI-compatible `tools` with every pinned inference parameter; provider-only
fields may enter only through the namespace matching `provider` and
`provider_api_version`, and may not replace a standard field. Credentials are
read from `api_key_env` after cache lookup, so replay needs no secret and no
credential value enters a request hash, diagnostic, or artifact.

The response parser preserves provider order, call ids, function names, and each
raw argument value. JSON arguments are parsed exactly once. Invalid JSON, a JSON
array where an object was required, a missing argument string, and a wrong JSON
type remain distinct model observations; none is repaired, coerced, retried, or
sent to another LLM. Envelope failures under HTTP 200 are likewise model output,
not transient infrastructure errors.

Transport retries only timeouts, connection failures, `408`, `429`, and selected
`5xx` responses, within `limits.max_retries` and the logical candidate deadline.
Backoff jitter is derived from the request hash and `Retry-After` is honored only
inside that deadline. Response bodies are streamed under a fixed size bound.
`candidate_io_cache.jsonl` appends a hash-verified request record, every HTTP
attempt, and a completion marker. A completion replays without network access;
an interrupted sequence is preserved as crash evidence and fails closed rather
than silently calling the model again. Cancelling a run records the abandoned
attempt but never a completion, so a resumed run cannot read an interruption as
the model's answer. Each record is verified once, when it first appears, and a
completion cites its attempts by hash, so one response body is stored once
however many records refer to it.

The native client neither chooses the next user turn nor executes or scores a tool call.

## Deterministic Evaluation Conversation Driver

`run_candidate_episode()` is the orchestrator that strings those turns into
one episode. `build_conversation_script()` selects a row and its conversation plan
from a canonical projection only after the projection's content hash, row count,
and complete task sequence match a `VerifiedEvalSource`. No caller-supplied
identity can stamp a stale row as current, and every candidate replays the
identical source-bound conversation.

A candidate sees only what it has earned. The prompt starts as the leading system
messages and the first user request; from there the only material that may enter
is an assistant turn the candidate itself produced, a recorded tool result the
driver decided to release, and a scripted user request. The conversation object
exposes no general append method, and re-audits provenance before every send, so a
gold assistant turn cannot reach a provider.

A recorded tool result is released only to a call that matches the trace, and is
addressed to the id the candidate's own call carried. Matching is an injected
`ContinuationGate`; `CanonicalCallMatchGate` implements the pinned publication
comparison, including declared-default insertion, so a model that spells out a
default — including one in a nested object, array, local `$ref`, or `allOf` —
is neither rewarded nor punished. Within one assistant turn a
`call_order: any` row accepts a permutation and each result still goes to the call
it actually answers; `strict` requires trace positions, while `prefix` orders only
the configured number of required-tool first appearances and matches the
remainder as a set. Ordering *across* turns is not negotiable in trace replay,
because a recorded result is only meaningful at the point the trace reached it.

An intermediate text-only turn advances only when it reproduces the text the
published trace recorded. This deliberately fail-closed rule prevents arbitrary
prose from unlocking a hidden slot or confirmation; a terminal turn must contain
non-empty plain or structured text. Provider finishes that explicitly mean
truncation or filtering (`length`, `content_filter`, or max-token variants) never
advance, even if the partial payload otherwise matches. Tool calls must declare type `function` and
carry unique non-empty ids — missing types are not repaired, and ambiguous ids
never receive recorded results. Before execution, the candidate's canonical
weights identity must still equal the identity the contamination plan authorized;
reusing an alias for different weights is refused.

An episode returns rather than raises for everything the model can cause — a wrong
tool, wrong arguments, unparseable arguments, a call with no id, an unreachable
endpoint, a malformed envelope, an exhausted turn budget or episode budget — each
as a distinct `status` on a `CandidateEpisode`, alongside the ordered events and
every observed turn. An unauthorized task or a row that is not a replayable
conversation raises instead, because those are bugs in the run rather than facts
about the model. Cancellation propagates without producing an episode, matching the
client: an interruption is not an observation.

The conversation driver derives no number and executes no tool. The results it
releases are the ones benchmark generation replay recorded, so an answer cannot
depend on a live fixture. The executable oracle runner is a separate component.

## Canonical Tool-Call Parser and Trace Scorer

`score_trace_episode(..., plan=...)` turns one recorded episode into
one `TraceTaskScore`. It
is a pure function of evidence and policy: it reads the `CandidateEpisode`, the
`ConversationScript` that produced it, the pinned `EvalScoringConfig`, and the
`EligibleEvalPlan` that authorized the run, and it
contacts no provider, executes no tool, reads no clock, and re-parses no provider
bytes. Scoring the same episode twice therefore reproduces the same `score_hash`,
which is what makes a published number auditable after the endpoint it came from
is gone.

Parsing flattens the episode rather than re-reading it. The client already parsed
the provider's bytes once under strict JSON, so a call whose arguments never
parsed stays unparsed and a turn the episode never sent is listed as unsent
rather than invented as an empty one. The parser refuses an episode whose task id
or script hash is not the conversation it answered: a score taken over mismatched
halves would grade the wrong task.

The comparison behind scoring is the same code the driver's release gate uses. A
gate stricter than the scorer would end an episode the scorer would have
credited, so a correct model would fail a task on transport grounds; the two read
one kernel rather than two implementations of the same prose.

A score names every gate the contract defines, and reports a gate that does not
apply as such rather than omitting it. `tool_selection` and `arguments` measure
coverage of the whole gold trace, so a gold call in a turn the episode never
reached counts against them. `schema_valid`, `call_grouping`, `call_ordering`,
and `text_turn` measure consistency of the turns that were actually asked.
`trace_completion` always applies, which is what makes an unfinished episode a
failed task rather than a skipped one, while `non_candidate_stop` still lets a
report separate an unreachable endpoint from a wrong model. `task_success` is
derived from the gates rather than asserted beside them.

The parser also requires the episode and script to name the same verified source;
a recorded episode cannot be restamped as evidence from another benchmark. It
also requires the episode's plan identity, candidate weights, and task to match
the supplied plan, and requires the scoring policy's content hash to match the
one embedded in that plan. The score derives the complete `eval_config_hash`
from this authorization rather than accepting a caller-provided stamp,
so changing limits, candidate inference, or another measurement input changes
the authorization and requires a new episode.

Two attributions are deliberate. Ordering is judged apart from selection: a turn
that made the trace's calls in the wrong order fails ordering only, and a turn
that called something else fails selection only. And declared defaults are filled
only after the candidate arguments satisfy their declared schema, so a parameter
that is both defaulted and required cannot be laundered into a match or earn a
recorded result. Relaxing
`respect_call_group` or `respect_call_order` drops the corresponding gate but does
not make an unreplayable episode succeed, because replay still holds exactly one
recorded result per gold call.

Nothing here is repaired. `allow_llm_repair` and `task_success: assertions_only`
are refused with a typed error rather than approximated: the first would make the
number a property of the repairer, and the second needs an oracle this scorer does
not have. Oracle replay and pack assertions belong to executable evaluation, and a
trace score never stands in for one.

Human-readable `detail` fields are emitted for diagnosis but excluded from
`score_hash`; stable `reason_code` values and structural verdicts carry semantic
identity. Rewording a diagnostic therefore does not fork otherwise identical
scores.

## Executable Evaluation Evidence

`ExecutableEpisode` (`executable contract` `1.0`) freezes what one live-oracle
task must record before the executable driver and scorer derive any metric. It
binds the candidate, task, authorized plan, full eval config, verified source,
verified oracle, and conversation script by content identity. It then retains
the candidate turns, exactly one `ExecutedToolCall` outcome for every proposed
call, the final-state hash, classified assertion outcomes, and ordered driver
events.

A proposed call never disappears because it could not execute. Invalid JSON,
schema-invalid arguments, missing ids, and undeclared tools are represented as
`not_executed`; attempted calls distinguish normal JSON results, structured
business rejections, tool failures, oracle returns that do not conform to the
tool contract, timeouts, infrastructure failures, and an unknown mutation commit
state. A business rejection is evidence only when it has the certified
`{"error": {"code": ...}}` shape. A non-object oracle return is recorded as a
`malformed_result` outcome that preserves the non-object JSON value, verifies
its type and canonical hash, and keeps it separate from conforming `result`.
Malformed output and commit state are independent: if a mutating call's result
is malformed and its commit cannot be established, the outcome retains both
facts and the episode terminates as `unknown_commit_state`. Assertion failures
remain model outcomes, while assertion import or runtime failures are
infrastructure outcomes.

Obtaining a result and admitting it to the candidate prompt are separate facts.
`released_to_model` records the second one per execution, so a result the driver
obtained but never released — the batch aborted after it, or the episode budget
expired, or a terminal tool-only task needed no following model request — is
retained without claiming the candidate read it. Every nonterminal turn that
advanced must have released its results in the next request; a terminal turn can
complete with unreleased results. Per-turn and per-episode release counts are
derived from the executions rather than restated beside them. A scripted
user-message release is likewise attached to the turn by the released message's
content hash; its episode count is derived rather than trusted as a free-standing
claim.

The contract is frozen and closed. It binds every outcome to the provider call it
records by typed JSON equality, so a coerced argument cannot pass as the value
the provider sent. It validates call-to-outcome ordering, result hashes, that a
tool-execution event cites both the exact outcome and the turn that owns it, that
a scripted user turn is released only by a turn that advanced, and that a
candidate call which never completed carries no envelope to read a finish
reason, assistant content, or tool calls from. Terminal status is checked in both
directions: malformed or unknown-commit evidence cannot be restamped as a
completed episode.

`build_executable_task_spec(...)` creates the only task handle accepted by the
live driver. It checks the complete canonical projection against
`VerifiedEvalSource`, requires the task to be assigned to the candidate by
`EligibleEvalPlan`, binds the verified oracle identity and source clock, and
retains runner-only fixture references, milestones, assertion names, and
mutation policy outside the model-facing seed. Pack-local `x-mutates` and
`x-requires-confirmation` flags are recovered from the verified `tools.json`;
they are never added to the provider tool schema. The assertion task preserves
verified template metadata and reconstructs bound `slots`, `slots_initial`, and
`slot_updates` from what the row published: the verbatim opening surface, the
expected trace, cited fixture rows, and the verified pack's fixture, literal,
enum, range, and absent-id declarations. The opening turn renders pre-correction
values, so it selects typed candidates for `slots_initial`; a model-paraphrased
surface is not read back. A final value no channel settles is named in
`unresolved_slots`, while unknown pre-correction and correction values are named
separately in `unresolved_slots_initial` and `unresolved_slot_updates`; none is
guessed from another phase. The isolated assertion runner tracks which missing
final, initial, or correction values each assertion reads and classifies such a
verdict as an infrastructure error, never a candidate failure, while an assertion
that does not read the missing value still runs normally.

`open_oracle_session(...)` selects a `PythonOracleSession` or
`EndpointOracleSession`. Both run through one persistent process-isolated
`ProcessWorker` episode per task, so reset, calls, state, and assertions share
state while pack modules never enter the evaluator process. Endpoint sessions
are deleted on normal close, timeout, cancellation, and worker failure.
Mutating calls are issued once; a timeout or transport failure becomes unknown
commit state instead of being retried. Successful mutating responses are not
assumed to have committed: the driver compares canonical state snapshots before
and after the call and records both hashes as evidence of `committed` or
`not_committed`. Missing either side yields `unknown_commit_state`.

`run_executable_episode(...)` interleaves native candidate turns with those live
operations. It executes only declared, parseable, schema-valid calls, in
candidate order, and sends canonical live results back under the candidate's
own call IDs. The continuation gate controls release of deterministic scripted
user turns, but never supplies a recorded gold result. Result and user-message
release evidence is committed only when the next candidate request is actually
sent. A terminal tool-only turn records no false release. Final state and pack
assertions are recorded before the session is closed. Source/plan/task lineage is
checked as one authorization unit, and the oracle session is closed even when
that preflight check fails.

`episode_hash` is path-free and time-free. Human `detail` wording and the
cache-replay flag are outside that identity; stable reason codes, canonical
arguments and results, commit verdicts, release verdicts, state identity, and
assertion verdicts remain inside it. Each model excludes only its own `detail`,
so oracle-owned JSON keys named `detail` are still evidence. Trace scores derive
`score_hash` the same way.

Persisting an append-only tool trace, executable scoring, bounded batch
execution, and aggregate reporting remain separate runtime components.

For the complete pack contract, validation rules, turn policies, and schema
requirements, see
[`../references/bfcl-oracle-pack.md`](../references/bfcl-oracle-pack.md). For what
a score means, see
[`../references/bfcl-eval-scoring-contract.md`](../references/bfcl-eval-scoring-contract.md).

## Capability Matrix

Unsupported capabilities stay gated and are rejected rather than silently
ignored.

| Capability | Availability | Responsibility |
| --- | --- | --- |
| Reference profiling | **Implemented** | Normalize content-addressed style samples and create a cached profile without exposing oracle truth. |
| Model paraphrasing | **Implemented** | Produce cached surface variants; Python guards preserve values, hidden slots, tool-name boundaries, turn shape, and deterministic lineage. |
| Surface quality judging | **Implemented** | Map Python guards onto six checks, optionally score surface-only language quality, enforce advisory/drop policy, write the Stage-10 parquet, and filter publication rows with manifest lineage. |
| Semantic deduplication | **Integrated** | Run after surface-quality validation, project masked user text, cluster through Curator, choose and balance coverage-safe representatives, publish in selection-rank order, and retain complete artifact and manifest lineage. |
| Evaluation and scoring | **Partial** | Config, source verification, contamination gating, native function-calling transport (`candidate client` `1.0`), deterministic trace driving/scoring, source-bound executable task projection, process-isolated Python/endpoint oracle sessions, live result conversation driving, pack-assertion execution, and executable evidence (`1.0`) are available. Tool-trace persistence, executable scoring, and bounded batch execution are not yet exposed. |
| Held-out enforcement | **Integrated** | Refuse reserved templates and fixture rows at binding time, re-scan every row before publication, stamp `held_out_hit`, and record policy, counters, and artifact hashes in run lineage. |
| Translation and localization | **Partial** | Localize benchmark surfaces through a BFCL-specific adapter while preserving executable calls and oracle assertions. |
| Additional exports | **Integrated** | Emit, read back, validate, hash, and transactionally publish BFCL JSON and NeMo Evaluator input bundles from one canonical projection. |
| Stage resume | **Partial** | Resume from a verified intermediate artifact without accepting stale pack, endpoint, or config state. |

The final evaluation interface, metric names, artifact schemas, and CLI stage
names may change while implementation is in progress. Until they are promoted
to the supported contract above, `benchmark.parquet` and `run_manifest.json`
remain generation outputs rather than evidence that a target model has been
evaluated.
