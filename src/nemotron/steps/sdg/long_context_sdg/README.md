# Long-Context Synthetic Data Generation

`long_context_sdg` is a domain- and language-neutral pipeline for producing
grounded, multi-turn training conversations. It is split into independent
query-synthesis, conversation-generation, evaluation, and export stages so a
team can replace or extend one stage without coupling it to the others.

The package provides:

- taxonomy-stratified, persona-conditioned synthetic query generation;
- exact query quotas across topics, query archetypes, persona modes, and
  languages;
- 6–40-turn conversations with independently sampled length and retrieval
  targets;
- natural user turns without per-turn intent labels or a scripted semantic
  episode plan;
- sparse retrieval-deadline intervention only when the sampled retrieval target
  would otherwise become infeasible;
- per-turn and per-conversation tool budgets;
- retrieval, memory, reviewed custom tools, and explicitly marked simulated
  tools;
- hidden, source-linked context compression without altering exported messages;
- deterministic validation and optional model judging;
- Data Designer model providers, orchestration, artifact storage, and native
  row-group resume;
- canonical, status-partitioned, and trainer-oriented JSONL artifacts.

## Architecture

```text
reviewed taxonomy + retrieval corpus + Data Designer persona assets
                              |
                     synthesize queries
                              |
                       raw seed JSONL
                              |
                    prepare conversation seeds
                              |
                Data Designer conversation dataset
           user <--> assistant <--> tools + compression
                              |
                    generated record JSONL
                              |
                 validate + optional rejudge
                              |
               canonical/status-partitioned JSONL
                              |
                    trainer-oriented export
```

| Stage | Command | Input | Output |
|---|---|---|---|
| Query synthesis | `synthesize` | Taxonomy, managed personas, retriever, generator, judge | Raw query seed JSONL |
| Seed preparation | `prepare` | Raw seed JSONL | Validated and enriched Data Designer seed JSONL |
| Conversation generation | `generate` | Enriched seeds, role models, tools | Data Designer dataset plus finalized generated JSONL |
| Evaluation | `evaluate` | Finalized generated JSONL | Canonical JSONL, status partitions, summary |
| Export | `export` | Canonical JSONL | Accepted trainer-oriented JSONL |

`synthesize` is optional. Existing reviewed queries can be written directly in
the seed format and passed to `prepare`. Query synthesis can also be used alone.
Future evaluation modules should consume canonical records and write derived
artifacts rather than changing generation state.

## Model providers

Generation uses Data Designer's model-provider system. The YAML `providers` and
`models` entries are converted to `data_designer.config.ModelProvider` and
`data_designer.config.ModelConfig` objects. Custom Data Designer column
generators resolve role aliases with `self.get_model(alias)`:

- query synthesis resolves `query_generation.generator_alias` and
  `query_generation.judge_alias`;
- conversation generation resolves `assistant`, `user`, `compressor`, and
  `judge`.

There is no separate direct model runner in the generation path. The retriever
is a tool service and therefore has its own HTTP adapter. `evaluate --rejudge`
is a post-generation operation and uses the configured judge endpoint directly;
it does not create a Data Designer dataset.

`providers[].api_key_env` names an environment variable. Keep credentials out
of YAML. An empty variable value is valid when an OpenAI-compatible endpoint is
intentionally unauthenticated.

## Installation

Python 3.11–3.13 is supported.

```bash
cd src/nemotron/steps/sdg/long_context_sdg
uv sync --extra dev
```

Create workload-specific configuration files instead of editing the examples:

```bash
cp config/default.yaml config/my-conversations.yaml
cp config/query_generation.example.yaml config/my-queries.yaml
```

The loader reads one complete YAML file and does not merge overlays. Relative
paths are resolved from the configuration file. Unknown keys are rejected.

## End-to-end run

If query synthesis uses managed persona locales, install those Data Designer
assets first:

```bash
uv run data-designer download personas \
  --locale en_IN \
  --locale hi_Deva_IN \
  --locale hi_Latn_IN
```

Set the credential named by `providers[].api_key_env`, configure the model and
retrieval endpoints, then run:

```bash
uv run long-context-sdg synthesize --config config/my-queries.yaml
uv run long-context-sdg prepare --config config/my-conversations.yaml
uv run long-context-sdg generate --config config/my-conversations.yaml
uv run long-context-sdg evaluate --config config/my-conversations.yaml
uv run long-context-sdg export --config config/my-conversations.yaml
```

Useful variants:

```bash
# Replace a different existing synthesized seed file after all checks pass.
uv run long-context-sdg synthesize --config config/my-queries.yaml --force

# Re-run model judging for every structurally valid generated record.
uv run long-context-sdg evaluate --config config/my-conversations.yaml --rejudge

# Apply deterministic evaluation only. Unjudged records remain quarantined.
uv run long-context-sdg evaluate --config config/my-conversations.yaml --no-network
```

`run.mode: preview` previews conversation generation and does not materialize
`paths.generated`. Query synthesis supports create mode only.

## Data Designer resume behavior

The pipeline does not implement a second checkpoint layer. Data Designer owns
dataset progress and resume inside `paths.artifacts`.

| `run.resume` | Behavior |
|---|---|
| `never` | Start a new dataset creation; do not resume prior progress. |
| `always` | Resume a compatible run from its last completed row group; restart from the beginning when no row group completed; raise on incompatible stored config. |
| `if_possible` | Resume compatible state when available; otherwise create a new run. |

Data Designer checks dataset-configuration compatibility. Completed row groups
are reused; an interrupted in-flight row group may be recomputed. Configure a
unique `run.dataset_name` and artifact directory for each independently managed
workload.

After Data Designer completes, the pipeline validates IDs and ordering and
atomically materializes:

- conversation records at `paths.generated`;
- synthesized query seeds at query-generation `paths.seeds`.

Evaluation and export consume these finalized artifacts, not internal Data
Designer storage. If generation is interrupted before final materialization,
rerun the same command and configuration with resume enabled.

## Query synthesis

The query module deliberately separates topic selection from conversation
generation. Its sequence is:

1. Load and validate the reviewed taxonomy.
2. Allocate exact largest-remainder quotas for taxonomy leaves, archetypes,
   persona modes, and locales.
3. Retrieve an evidence pool for each scheduled taxonomy leaf.
4. Filter short chunks, normalize duplicates, and cap chunks per source.
5. Sample a bounded evidence bundle within that leaf.
6. Ask Data Designer's managed `PERSON` sampler for the scheduled locale.
7. Project only selected persona details into the model prompt.
8. Ask the generator model for a natural information need conditioned on the
   topic, persona, archetype, language, and evidence.
9. Apply deterministic leakage, length, script, grounding, and retrievability
   checks.
10. Ask the independent judge model to score the query.
11. Retry the row up to `max_attempts` when a draft fails.
12. Reject publication unless every scheduled candidate is accepted and the
   final batch has no near-duplicate pair.

### Critical view of evidence-first query generation

Starting with random corpus chunks is useful because it grounds queries in the
available corpus and exposes long-tail material. Used globally, however, it can
overrepresent repetitive documents, produce context-shaped trivia, leak answer
phrasing, and destroy topic or language quotas.

This implementation samples evidence only after scheduling a reviewed taxonomy
leaf and persona/archetype/locale quota. Evidence is a grounding constraint, not
the sole source of diversity. Independent judging, overlap limits,
retrievability checks, source caps, and final duplicate detection reduce the
remaining failure modes.

### Query-generation knobs

| Knob | Default | Effect |
|---|---:|---|
| `query_generation.num_queries` | `100` | Exact number that must be accepted for publication. |
| `taxonomy_path` | required | Reviewed hierarchical topic YAML. |
| `generator_alias` | `assistant` | Data Designer model alias used to draft queries. |
| `judge_alias` | `judge` | Independent model alias used to judge drafts. |
| `max_attempts` | `3` | Draft/check/judge attempts within one Data Designer row. |
| `min_judge_score` | `4` | Minimum accepted score on the 1–5 scale. |
| `min_query_chars` / `max_query_chars` | `8` / `400` | Query-length bounds. |
| `archetype_weights` | see YAML | Exact marginal for research, scenarios, comparisons, misconceptions, clarification, and insufficient-evidence needs. |
| `persona_mode_weights` | see YAML | Exact marginal for general-interest, situated-need, and domain-adjacent personas. |
| `persona_locales` | required | Exact language/locale marginal and native persona filters. |
| `evidence.pool_size` | `32` | Retrieved candidates retained per taxonomy leaf. |
| `evidence.bundle_min` / `bundle_max` | `2` / `4` | Evidence chunks shown for one draft. |
| `evidence.max_per_source` | `2` | Source-diversity cap in one bundle. |
| `evidence.min_chunk_chars` | `160` | Discard fragments below this size. |
| `evidence.retrievability_top_k` | `8` | Retrieval depth used by the anchor check. |
| `evidence.max_lexical_overlap` | `0.35` | Maximum normalized query/evidence overlap. |
| `evidence.max_verbatim_tokens` | `12` | Longest permitted copied phrase. |
| `evidence.duplicate_similarity` | `0.85` | Final same-topic/language duplicate threshold. |

Each `persona_locales` item supports:

| Field | Effect |
|---|---|
| `locale` | Data Designer managed persona asset locale. |
| `language` | Expected output language/script label. |
| `weight` | Exact locale marginal. |
| `asset_revision` | Auditable reviewed persona-asset version. |
| `narrative_fields` | Weighted persona narratives available for projection. |
| `attribute_fields` | Structured attributes allowed in prompts. |
| `sex`, `city`, `age_range`, `select_field_values` | Optional native `PERSON` sampler filters. |

### Query-generation paths

| Path | Contents |
|---|---|
| `seeds` | Atomically published raw seeds consumed by conversation preparation. |
| `evidence_manifest` | Reusable normalized evidence pools and provenance. |
| `candidates` | Deterministically scheduled candidate rows. |
| `artifacts` | Data Designer dataset artifacts and native resume state. |
| `report` | Status, attempts, coverage, rejection, and duplicate statistics. |

## Seed format

Minimal seed:

```json
{"query_id":"q-001","naive_query":"Explain the main trade-offs in this topic."}
```

Rich seed fields may include:

| Field | Purpose |
|---|---|
| `query_id` | Stable unique identity. Generated deterministically when omitted. |
| `naive_query` | Opening user request. |
| `language` | Language or script label used by prompts and analysis. |
| `domain` | Domain label for metadata and instructions. |
| `persona` | Compact persona description. |
| `persona_provenance` | Locale, asset revision, sampler identity, and projected fields. |
| `topic`, `taxonomy_id` | Topic metadata and reviewed taxonomy leaf. |
| `query_archetype`, `persona_mode` | Query-diversity labels. |
| `instructions` | Sample-specific behavior layered over global instructions. |
| `turn_budget` | Optional requested length when `honor_seed_turn_budget` is true. |
| `memory` | Initial allowlisted memory state. |
| `metadata` | Additional non-control provenance. |

`prepare` validates JSONL, rejects duplicate IDs, deterministically enriches the
records, and atomically writes `paths.enriched_seeds`.

## Conversation control

Conversation generation does not create an intent trace or prescribe a
semantic purpose for every turn. The user model receives the trajectory and is
asked for a natural follow-up. This permits clarification, demographics,
preferences, corrections, scope changes, brief acknowledgements, and new
evidence needs in any plausible order.

Each episode samples only stable constraints:

- exact conversation length;
- required successful retrieval count;
- maximum retrieval attempts;
- maximum tool calls per turn and per conversation.

Before each assistant turn, the controller checks remaining capacity. Usually it
adds no directive. If deferring retrieval would make the sampled retrieval floor
impossible, it emits a sparse `retrieval_deadline` policy event specifying only
the minimum retrievals required on that turn. This guarantees accepted-data
constraints without teaching a labeled turn-by-turn intent state machine.

### Conversation knobs

| Knob | Default | Effect |
|---|---:|---|
| `episode.turn_budget.min` | `6` | Minimum exact user/assistant turn pairs; allowed range 6–40. |
| `episode.turn_budget.max` | `40` | Maximum exact user/assistant turn pairs; allowed range 6–40. |
| `episode.honor_seed_turn_budget` | `false` | Use a valid seed-specific budget instead of sampling when true. |
| `episode.retrieval_depth_weights` | `{1: .65, 2: .25, 3: .10}` | Distribution for retrieval depth when retrieval is required. |
| `episode.retrieval_calls.min` | `1` | Minimum sampled successful retrieval floor. Set to 0 for episodes that may never retrieve. |
| `episode.retrieval_calls.max` | `12` | Maximum sampled floor and maximum retrieval attempts. |
| `episode.max_steps_per_turn` | `6` | Assistant action-loop steps, including the final-answer step. Must exceed enabled retrieval depth. |
| `episode.max_tool_calls_per_turn` | `3` | Hard cap across all tools during one assistant turn. |
| `episode.max_tool_calls_per_conversation` | `32` | Hard cap across all tools during the episode. |

The sampled retrieval floor is clipped to feasible capacity. A successful
retrieval call—not a failed attempt—counts toward the floor. Every attempted tool
call counts against tool budgets. Set `retrieval_calls.max` no higher than the
conversation tool cap. Per-turn retrieval depth can never exceed both the
per-turn tool cap and the remaining assistant-step capacity.

Examples:

- Broad diversity: `turn_budget: {min: 6, max: 40}`.
- Retrieval-optional data: `retrieval_calls: {min: 0, max: 8}`.
- Retrieval-heavy conversations: raise the retrieval floor and the global tool
  cap together.
- Mostly single-search turns: weight depth 1 heavily.
- Multi-hop evidence: increase weights for depth 2 or 3 and ensure
  `max_steps_per_turn` and per-turn tool capacity are sufficient.

Do not add an intent label merely to control retrieval timing. If a use case
needs a new hard invariant, represent the invariant directly and intervene only
when remaining capacity requires it.

## Complete conversation configuration reference

### `paths`

| Field | Contents |
|---|---|
| `seeds` | Raw seed JSONL. |
| `enriched_seeds` | Prepared Data Designer seed JSONL. |
| `artifacts` | Data Designer generation artifacts and native resume state. |
| `generated` | Atomically materialized canonical generation records. |
| `canonical` | Evaluation output containing all terminal records. |
| `output_dir` | Status partitions and evaluation summary. |
| `export` | Accepted trainer-oriented JSONL. |

### `run`

| Field | Default | Effect |
|---|---:|---|
| `mode` | `preview` in the schema; `create` in the example | Preview or create a Data Designer dataset. |
| `seed` | `7` | Deterministic preparation and per-sample constraint sampling. |
| `num_records` | `0` | First N prepared seeds; 0 means all. |
| `dataset_name` | `long_context_sdg` | Data Designer dataset identity used for artifacts/resume. |
| `resume` | `always` | Data Designer `never`, `always`, or `if_possible` policy. |

### `providers` and `models`

| Field | Effect |
|---|---|
| `providers[].name` | Provider name referenced by model configs. |
| `providers[].endpoint` | OpenAI-compatible base endpoint. |
| `providers[].api_key_env` | Environment-variable name holding the API key. |
| `models[].alias` | Runtime role alias. Conversation generation requires `assistant`, `user`, `compressor`, and `judge`. |
| `models[].model` | Served model identifier. |
| `models[].provider` | Data Designer provider name. |
| `models[].skip_health_check` | Skip provider/model health validation when explicitly needed. |
| `models[].inference_parameters` | Data Designer chat-completion parameters such as temperature and max tokens. |

The four conversation aliases may point to the same endpoint/model or different
ones. Temperature is typically higher for the user simulator and lower for the
compressor and judge.

### `context`

| Field | Default | Effect |
|---|---:|---|
| `compression_threshold` | `32000` | Estimated active tokens that trigger hidden compression. |
| `model_token_limit` | `65536` | Fail-safe active-context boundary. |
| `recent_raw_turns` | `4` | Recent turns retained verbatim after compression. |
| `min_turns_between_compression` | `3` | Compression-event spacing. |
| `compression_token_budget` | `500` | Structured summary budget. |
| `max_reasoning_tokens` | `400` | Maximum retained reasoning tokens when supplied by a model. |

Compression changes only the model-facing active view. Exported messages retain
the original trajectory; `compaction_events` records summaries and provenance.

### `retriever`

| Field | Effect |
|---|---|
| `endpoint`, `method` | Retrieval service request target and GET/POST method. |
| `query_field`, `top_k_field` | Request-key mapping. |
| `top_k` | Default requested result count. |
| `results_path` | Dot path to the response list. |
| `fields.*` | Dot-path mapping for ID, text, title, source, score, URL, and date. |
| `selection` | `ranked`, seeded `sampled`, or source-oriented `diverse`. |
| `timeout_seconds` | Request timeout. |
| `retries`, `backoff_seconds` | Bounded transport retries and backoff. |
| `headers`, `extra_body` | Static service-specific request additions. Do not store secrets here. |

### `tools`

Each tool entry contains an OpenAI function `schema`, a trusted Python
`executor` import path, and optional `executor_kwargs`. The registry validates
names, arguments, budgets, and execution results.

To add a real tool:

1. Implement the executor protocol under `executors/` or another reviewed
   package.
2. Keep side effects idempotent because model/transport retries can repeat
   requests.
3. Add its JSON schema and executor path to YAML.
4. Add deterministic validation and tests for its claims and side effects.
5. Increase tool budgets only if the new behavior requires it.

To add a simulated tool, use the simulated executor and retain the explicit
synthetic marker in its output. Never represent model-invented output as a real
external observation.

### `validation`, `judge`, and `export`

| Field | Effect |
|---|---|
| `validation.require_final_answer_each_turn` | Require each user turn to end with an assistant answer. |
| `judge.enabled` | Run model judging during generation/evaluation. |
| `judge.min_score` | Minimum per-dimension score for acceptance. |
| `judge.dimensions` | Named quality dimensions included in the structured rubric. |
| `export.format` | `messages`, `messages_and_tools`, or `rich`. |

Only `accepted` canonical records are exported.

## Multilingual and domain adaptation

To adapt this package:

1. Replace the taxonomy with reviewed domain leaves and retrieval probes.
2. Point both configs at the domain retrieval service and map its response.
3. Choose persona locales and language quotas supported by installed Data
   Designer assets.
4. Update global/sample instructions for domain style and citation rules.
5. Select role models that reliably follow the target language and structured
   schemas.
6. Adjust query archetypes, evidence gates, length, retrieval targets, and tool
   budgets.
7. Add domain tools through reviewed executors.
8. Extend deterministic validation and judge dimensions for domain risks.
9. Run a bounded sample, manually inspect every record, then scale.

Models may internally reason in a different language; this pipeline validates
observable messages, tool calls, evidence use, and configured script/language
constraints. Do not infer hidden reasoning language from output language.

## Output record anatomy

| Field | Contents |
|---|---|
| `run_id` | Generation identifier derived from the configuration fingerprint. |
| `config_fingerprint` | Hash of generation-affecting configuration. |
| `query_id` | Seed identity. |
| `status` | `accepted`, `rejected`, `quarantine`, or `generation_failed`. |
| `messages` | Full OpenAI-style trajectory; hidden compression is not inserted. |
| `tools` | Tool schemas exposed during generation. |
| `episode_spec` | Stable length, retrieval target, and hard tool budgets. |
| `policy_events` | Sparse retrieval-deadline interventions; normally far fewer than turns. |
| `tool_call_attempts` | Every execution attempt with turn, call ID, tool name, success, and error. |
| `metadata` | Seed, counts, budgets, turn mapping, context history, and rejected calls. |
| `retrieval_transcript` | Queries and returned chunk IDs by turn. |
| `memory_events` | Allowed memory reads and writes. |
| `compaction_events` | Hidden summaries and source provenance. |
| `validation` | Deterministic errors and warnings. |
| `judgment` | Scores, rating, explanation, and gate errors. |

`reasoning_content` is retained when a model supplies it. Confirm trainer support
or transform the export before training.

## Evaluation artifacts

`evaluate` reads `paths.generated`, verifies fingerprint and query uniqueness,
replays deterministic validation, optionally judges, and writes:

- `paths.canonical` with every terminal record;
- `output_dir/accepted.jsonl`;
- `output_dir/rejected.jsonl`;
- `output_dir/quarantine.jsonl`;
- `output_dir/generation_failed.jsonl`;
- `output_dir/summary.json` with status, length, tool, retrieval, language,
  domain, compaction, and sparse policy-event statistics.

Evaluation is repeatable and does not mutate Data Designer artifacts.

## Production operation

- Review and version every taxonomy, prompt, config, and executor.
- Use a separate Data Designer artifact directory and dataset name per workload.
- Store credentials in environment variables or a secret manager.
- Keep retrieval and custom-tool APIs idempotent.
- Start with a small `num_records` and inspect rich output manually.
- Monitor Data Designer row-group progress and final status statistics.
- Budget for multiple model calls per turn, structured corrections, retrieval
  retries, compression, and judging.
- Preserve Data Designer artifacts until final materialization and evaluation
  have been verified.
- Train only from reviewed accepted exports.

## Testing

Tests use deterministic doubles confined to `tests/`; production generation does
not monkeypatch models or retrieval.

```bash
uv run ruff check .
uv run pytest -q
uv lock --check
uv build
```

Before releasing a workload, also run a real bounded integration against the
configured model and retrieval services and inspect the resulting rich records.

## Troubleshooting

**Configuration reports an extra field**

The YAML contains an obsolete or misspelled key. Compare it with the complete
example; unknown fields are intentionally rejected.

**Data Designer refuses to resume**

The stored dataset configuration is incompatible or the requested resume mode
requires state that is unavailable. Restore the original generation config, use
the matching dataset name/artifact directory, or deliberately start a new
dataset identity. Do not edit internal artifacts.

**Generation finished but `generated.jsonl` is absent**

Final materialization happens only after `create` completes and result IDs are
validated. Rerun the same command with native resume enabled.

**No records are generated**

Check that prepared seeds exist, `run.num_records` is not greater than the seed
count, and the Data Designer seed file contains `episode_input`.

**A retrieval deadline cannot be satisfied**

The configured floor exceeds remaining step/tool capacity or retrieval attempts
failed. Inspect `tool_call_attempts`, `rejected_tool_calls`, and
`retrieval_transcript`; align retrieval and tool caps.

**Trajectory is quarantined**

Deterministic validation passed but judgment was unavailable or invalid. Fix
judge access and use `evaluate --rejudge`; regeneration is usually unnecessary.

**Trajectory is rejected**

Inspect `validation.errors`, then `judgment.gate_errors`. Rejection is a quality
outcome, not orchestration failure.

**Compaction fails**

Check compressor structured-output reliability, context length, source IDs, and
token budgets. The active context may not exceed `model_token_limit`.

**Retriever response fields are missing**

Adjust `results_path` and `fields.*`. Dot paths traverse nested objects but do
not index arrays. The resolved result container must be a list of objects.

## File guide

### Configuration and entry points

| File | Responsibility |
|---|---|
| `config/default.yaml` | Complete conversation-generation example and supported knobs. |
| `config/query_generation.example.yaml` | Independent query-synthesis example. |
| `config/taxonomy.example.yaml` | Reviewed hierarchical topic and retrieval-probe contract. |
| `data/queries.jsonl` | Minimal generic raw-seed examples. |
| `pyproject.toml` | Dependencies, CLI, Data Designer plugin registration, lint/test settings. |
| `uv.lock` | Reproducible dependency resolution. |

### Conversation package

| File | Responsibility |
|---|---|
| `__main__.py` | `synthesize`, `prepare`, `generate`, `evaluate`, and `export` CLI. |
| `config.py` | Strict configuration, cross-field constraints, path resolution, fingerprinting. |
| `config_base.py` | Shared unknown-field-rejecting Pydantic base. |
| `service_config.py` | Independent Data Designer model/provider and retriever contracts. |
| `conversation_generation.py` | Public façade for preparation and generation. |
| `pipeline.py` | Data Designer providers/models, seed source, native resume, result materialization. |
| `generator_config.py` | Data Designer custom-column and side-effect-column contract. |
| `generator.py` | Custom Data Designer column generator that runs one episode. |
| `records.py` | Validated canonical JSONL loading and atomic final-artifact writing. |
| `episode_control.py` | Stable episode constraints and sparse retrieval-deadline calculation. |
| `runtime.py` | User/assistant/tool loop, context management, validation, judging, projection. |
| `prompts.py` | Natural user, assistant action, retrieval deadline, final answer, and judge prompts. |
| `schemas.py` | Seeds, messages, episode specs, policy events, tools, judgments, canonical records. |
| `llm.py` | Data Designer facade normalization, transport retries, JSON extraction, schema recovery. |
| `retrieval.py` | Persistent retrying retrieval adapter, field mapping, stable IDs, selection. |
| `tool_registry.py` | Trusted executor loading, JSON-schema argument validation, dispatch. |
| `executors/base.py` | Shared executor protocol, services, state, and errors. |
| `executors/retrieval.py` | Retrieval execution and transcript capture. |
| `executors/memory.py` | Allowlisted episode-local memory. |
| `executors/simulated.py` | Explicitly marked synthetic tool output. |
| `compression.py` | Structured context summaries and provenance validation. |
| `tokens.py` | Token estimation and active-context accounting. |
| `reasoning.py` | Reasoning-length and retrieved-chunk citation checks. |
| `validation.py` | Replayable conversation, policy-event, tool, retrieval, and answer checks. |
| `evaluation.py` | Revalidation, optional rejudging, partitions, summary. |
| `models.py` | Direct judge facade used only by offline evaluation. |
| `exporters.py` | Accepted-only `messages`, `messages_and_tools`, and `rich` exports. |
| `seeds.py` | Stable IDs, deterministic enrichment, duplicate detection, atomic preparation. |
| `plugin.py` | Data Designer custom-column registration. |

### Query-generation package

| File | Responsibility |
|---|---|
| `query_generation/config.py` | Strict standalone query, persona, evidence, provider, model, retriever, and path config. |
| `query_generation/taxonomy.py` | Taxonomy loading, weighted leaf traversal, raw-file hashing. |
| `query_generation/allocation.py` | Largest-remainder exact quotas and deterministic scheduling. |
| `query_generation/evidence.py` | Retrieval pools, eligibility, deduplication, source diversity, bundle sampling. |
| `query_generation/candidates.py` | Evidence-cache reuse, marginal scheduling, fingerprints, stable candidates. |
| `query_generation/personas.py` | Locale keys, managed sampler columns, compact persona projection/provenance. |
| `query_generation/prompts.py` | Draft and independent-judge prompts. |
| `query_generation/validation.py` | Length/script, leakage, retrievability, comparison, duplicate checks. |
| `query_generation/schemas.py` | Taxonomy, candidate, persona, draft, judgment, record schemas. |
| `query_generation/generator_config.py` | Data Designer persona-query custom-column contract. |
| `query_generation/generator.py` | Per-row persona projection, draft/check/judge retry loop, seed conversion. |
| `query_generation/pipeline.py` | Managed persona samplers, Data Designer native resume, reporting, publication. |

### Tests

| File | Responsibility |
|---|---|
| `tests/fixtures.py` | Temporary configs and deterministic model/retrieval/executor doubles. |
| `tests/test_config_seeds_episode.py` | Config, enrichment, variable constraints, sparse deadline policy, prompts. |
| `tests/test_llm_retries.py` | Transport retries and structured JSON extraction/recovery. |
| `tests/test_query_generation.py` | Quotas, evidence, personas, validation, Data Designer record finalization. |
| `tests/test_retrieval_registry_memory.py` | Retrieval mapping, registry, memory policy, simulated marking. |
| `tests/test_runtime_e2e.py` | Full episodes, tool use, natural turns, sparse policy events, compression. |
| `tests/test_tokens_reasoning.py` | Token accounting, compaction spacing, reasoning validation. |
| `tests/test_validation_evaluation_export.py` | Validation, atomic records, fingerprint checks, evaluation, export. |

## Known constraints

- Data Designer resume is row-group-level; an interrupted in-flight group can be
  recomputed.
- Exact conversation length is sampled before generation, not decided by an
  intent plan.
- Sparse policy events enforce retrieval feasibility but do not assign turn
  semantics.
- Model retry and structured-correction counts are internal rather than YAML
  knobs.
- Token counts are estimates and can differ from the serving tokenizer.
- `model_token_limit` is a compression-failure boundary, not a universal server
  context limit.
- Simulated tool output remains synthetic even when explicitly marked.
- Custom executor import paths run trusted Python and require code review.
