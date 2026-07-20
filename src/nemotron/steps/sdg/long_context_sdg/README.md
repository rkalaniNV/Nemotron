# Long-Context Synthetic Data Generation

`long_context_sdg` is a domain- and language-neutral package for creating
grounded, multi-turn, tool-using training conversations with NeMo Data
Designer. It has three independent boundaries:

1. query synthesis creates diverse, persona-conditioned seed queries;
2. conversation generation turns any compatible seed JSONL into a long
   conversation;
3. evaluation and export validate records without changing generation state.

The split is intentional. Query synthesis owns the dataset's semantic mix.
Conversation generation owns runtime safety, context management, and structural
quality. A future evaluator can consume canonical records without importing or
modifying either generator.

## Design principles

- The first user message may be clear, underspecified, poorly phrased,
  search-like, overbroad, or adjacent to the underlying intent.
- A hidden canonical query preserves the intended topic and gives query
  synthesis a measurable retrieval target. The assistant sees only the natural
  first utterance and conversation history.
- Task-shape, surface-form, persona, and evidence labels are audit provenance.
  They are not exposed to the assistant and never force tool behavior.
- There is no per-turn intent trace, episode plan, retrieval deadline, planned
  retrieval turn, or minimum retrieval quota.
- Retrieval and tool-call limits are upper safety bounds. A valid conversation
  may make zero retrieval calls.
- Repeated searches are controlled by lexical-query checks and observed
  evidence gain. These checks do not claim to be a general semantic-similarity
  model.
- Data Designer owns dataset artifacts and resume behavior. The package does
  not add a second conversation checkpoint layer.

## Architecture

```text
taxonomy + corpus evidence + managed personas
                     |
              synthesize queries
                     |
        canonical query + natural first utterance
                     |
              prepare seed JSONL
                     |
       Data Designer conversation generation
        user <-> assistant <-> configured tools
                     |
       generated records + hidden compactions
                     |
          deterministic validation + judge
                     |
         canonical/status-partitioned records
                     |
                   export
```

| Stage | Command | Main input | Main output |
|---|---|---|---|
| Query synthesis | `synthesize` | Query config, taxonomy, personas, evidence endpoint | Raw seed JSONL |
| Seed preparation | `prepare` | Raw seed JSONL | Enriched Data Designer seed JSONL |
| Conversation generation | `generate` | Enriched seeds, role models, tools | Generated record JSONL |
| Evaluation | `evaluate` | Generated records | Canonical records, status partitions, summary |
| Export | `export` | Canonical records | Accepted trainer-oriented JSONL |

`synthesize` is optional. A team may provide reviewed seeds directly and start
with `prepare`.

## Installation

Python 3.11 through 3.13 is supported.

```bash
cd src/nemotron/steps/sdg/long_context_sdg
uv sync --extra dev
```

Create workload-specific configuration files. The loaders are strict and do
not merge overlays; unknown keys are rejected, and relative paths resolve from
the YAML file.

```bash
cp config/query_generation.example.yaml config/my-queries.yaml
cp config/default.yaml config/my-conversations.yaml
```

Keep credentials out of YAML. `providers[].api_key_env` contains the name of an
environment variable, not the secret itself. An explicitly empty variable is
accepted for endpoints that intentionally require no authentication.

## End-to-end run

Install the managed persona locales selected in the query configuration, then
configure the model and evidence endpoints.

```bash
uv run data-designer download personas \
  --locale en_IN \
  --locale hi_Deva_IN \
  --locale hi_Latn_IN

uv run long-context-sdg synthesize --config config/my-queries.yaml
uv run long-context-sdg prepare --config config/my-conversations.yaml
uv run long-context-sdg generate --config config/my-conversations.yaml
uv run long-context-sdg evaluate --config config/my-conversations.yaml
uv run long-context-sdg export --config config/my-conversations.yaml
```

Useful evaluation and publication variants:

```bash
# Replace an existing synthesized seed file only after reviewing the report.
uv run long-context-sdg synthesize --config config/my-queries.yaml --force

# Re-run judging for every deterministically valid record.
uv run long-context-sdg evaluate --config config/my-conversations.yaml --rejudge

# Run deterministic evaluation only; pending judge records stay quarantined.
uv run long-context-sdg evaluate --config config/my-conversations.yaml --no-network
```

`run.mode: preview` previews conversation dataset construction. Use
`run.mode: create` to materialize output. Query synthesis requires `create`.

## Module boundary and seed contract

The query module publishes ordinary seed JSONL. The conversation module reads
that schema but does not import query-generation policy.

Minimal seed:

```json
{"query_id":"q-001","query":"A complete canonical information need","naive_query":"what about this for my setup"}
```

Supported fields:

| Field | Meaning |
|---|---|
| `query_id` | Stable unique ID. Deterministically generated when absent. |
| `query` | Canonical topic formulation used for continuity and synthesis validation. Required. |
| `naive_query` | Visible first user message. Defaults to `query`. |
| `persona` | Role, expertise, style, language, compact description, and provenance. |
| `instructions` | Reviewed sample-level instructions layered over global instructions. |
| `turn_budget` | Optional 6–40 turn count when seed budgets are honored. |
| `memory_seed` | Initial values from the fixed memory allowlist. |
| `query_provenance` | Hidden audit fields such as taxonomy, task shape, surface form, and evidence IDs. |

Only deterministic language instructions are emitted by query synthesis. The
query-drafting model cannot write arbitrary seed instructions or smuggle tool
quotas into conversation generation.

## Query synthesis

The query module schedules diversity before asking a model to write text:

1. validate a reviewed hierarchical taxonomy;
2. allocate exact largest-remainder counts for taxonomy leaves, task shapes,
   surface forms, persona modes, and locales;
3. build a deduplicated evidence pool for each taxonomy leaf;
4. sample a source- and content-diverse evidence bundle appropriate to the task
   shape;
5. project a small, reviewed subset of a managed persona;
6. draft a canonical query, visible first utterance, and hidden evidence needs;
7. run canonical, visible-query, and per-facet retrieval probes;
8. apply leakage, language/script, length, overlap, answerability, and duplicate
   checks;
9. run an independent model judge;
10. retry a rejected row up to `max_attempts`;
11. publish only a complete, unique batch and write realized coverage counts.

This is evidence-conditioned query generation, not unrestricted random-chunk
prompting. The taxonomy and scheduled distributions determine coverage;
evidence makes each item answerable and exposes long-tail corpus material.

### Task-shape knobs

`query_generation.archetype_weights` controls exact dataset-level counts.
`archetype_profiles` maps each task shape to its evidence structure.

| Default task shape | Intended information need |
|---|---|
| `single_lookup` | One focused evidence facet. |
| `comparison` | Two or more independently supported alternatives or dimensions. |
| `timeline` | Multiple time-linked facets. |
| `multi_constraint` | A decision requiring several independent constraints. |
| `conflict_resolution` | Evidence from distinct sources or claims that must be reconciled. |
| `clarification` | A plausible opening where conversational context may naturally be resolved first. |
| `insufficient_evidence` | An essential facet is intentionally unsupported by the sampled bundle. |

Each profile exposes:

| Knob | Effect |
|---|---|
| `evidence_scope` | `conversational`, `single_facet`, or `multi_facet`. |
| `bundle_min`, `bundle_max` | Evidence-bundle range for this task shape. |
| `min_sources` | Minimum distinct sources in the bundle. |
| `min_evidence_needs` | Minimum independent hidden evidence facets. |

For a supported multi-facet query, every evidence need contains its own hidden
`retrieval_probe`. Validation executes each probe and requires it to recover the
declared supporting chunk. Probe pairs that are too similar are rejected. A
single broad canonical search therefore cannot by itself prove that an example
is genuinely multi-facet.

### First-utterance surface forms

`surface_form_weights` controls the exact scheduled mix of visible opening
queries. `surface_form_profiles` controls deterministic acceptance gates.
The seven names below are a strict supported set; an unknown name is rejected
during config loading instead of failing later in prompting.

| Surface form | Generated behavior |
|---|---|
| `well_formed` | Clear, complete, natural request. |
| `underspecified` | Omits material scope, entity, time, comparison basis, or success criteria. |
| `retrieval_rewrite` | Expresses the right need with weak shorthand, vague references, or colloquial terms. |
| `noisy_language` | Plausible spelling, grammar, code-switching, or word-order imperfections; never gibberish or stereotypes. |
| `keyword_fragment` | Terse search-like fragment rather than a polished sentence. |
| `overbroad` | Starts from a broader area that naturally narrows during conversation. |
| `adjacent_intent` | Touches a closely neighboring concern while preserving a recoverable link to the canonical intent. |

For English, `noisy_language` may contain realistic imperfect or non-native
English. For other configured languages, it produces plausible imperfections
appropriate to that language and script. The generator must preserve meaning
well enough for a real conversation to recover; demographic caricatures and
unreadable noise are rejected.

Profiles expose:

| Knob | Effect |
|---|---|
| `minimum_anchor_recall_gap` | Minimum improvement in sampled-anchor recall from `naive_query` to canonical `query`. |
| `require_noncanonical_form` | Reject a visible query identical to the canonical form. |
| `require_topic_overlap` | Require an explicit recoverable lexical connection, used by `adjacent_intent`. |

The retrieval-gap check is important: a label alone is not evidence that a
query benefits from rewriting. The pipeline executes both formulations and
measures their anchor recall.

### Query-generation configuration reference

| Knob | Default | Effect |
|---|---:|---|
| `num_queries` | `100` | Exact candidate count required for publication. |
| `taxonomy_path` | required | Reviewed hierarchical topic YAML. |
| `generator_alias` | `assistant` | Data Designer model alias for drafting. |
| `judge_alias` | `judge` | Independent query-judge alias. |
| `max_attempts` | `3` | Draft/check/judge attempts per candidate. |
| `min_judge_score` | `4` | Minimum 1–5 score on every required dimension. |
| `min_query_chars`, `max_query_chars` | `8`, `400` | Bounds for canonical and visible queries. |
| `archetype_weights` | see YAML | Exact semantic task-shape marginal. |
| `archetype_profiles` | see YAML | Evidence requirements by task shape. |
| `surface_form_weights` | see YAML | Exact first-utterance marginal. |
| `surface_form_profiles` | see YAML | Rewrite-gain and relationship gates. |
| `persona_mode_weights` | see YAML | General-interest, situated-need, and domain-adjacent mix. |
| `persona_locales` | required | Locale/language mix and managed persona projection. |
| `evidence.pool_size` | `32` | Candidate chunks retained per taxonomy leaf. |
| `evidence.bundle_min`, `bundle_max` | `1`, `4` | Global evidence-bundle limits. |
| `evidence.max_per_source` | `2` | Maximum bundle chunks from one source. |
| `evidence.max_pair_similarity` | `0.75` | Maximum content overlap inside one bundle. |
| `evidence.min_chunk_chars` | `160` | Minimum useful chunk length. |
| `evidence.retrievability_top_k` | `8` | Depth for canonical, visible, and facet checks. |
| `evidence.max_lexical_overlap` | `0.35` | Maximum query-to-evidence overlap. |
| `evidence.max_verbatim_tokens` | `12` | Longest allowed copied source span. |
| `evidence.duplicate_similarity` | `0.72` | Final same-language hybrid shingle/unigram/containment duplicate threshold. |
| `evidence.max_probe_similarity` | `0.80` | Maximum similarity between distinct facet probes. |

Each `persona_locales` item provides a Data Designer locale, output language,
weight, reviewed asset revision, permitted narrative and attribute fields, and
optional native sampler filters. Avoid projecting names or irrelevant
demographics into the query.

Query paths:

| Path | Contents |
|---|---|
| `seeds` | Atomically published raw seed JSONL. |
| `evidence_manifest` | Normalized evidence pools and provenance. |
| `candidates` | Deterministically scheduled candidate rows. |
| `artifacts` | Data Designer artifacts and native resume state. |
| `report` | Target/realized coverage, attempts, rejections, and duplicates. |

## Conversation generation

The visible `naive_query` becomes user turn 1. Later user turns are generated
from what was actually said, the persona, and the canonical topic. The assistant
receives the visible transcript, effective instructions, configured tool
schemas, exact retrieved chunk IDs, and bounded retrieval history. It does not
receive query-generation labels.

The assistant decides naturally whether a tool materially improves the current
response. It may clarify, answer conversationally, read a saved preference,
retrieve evidence, combine tools, or stop retrieving. There is no hard-coded
rule that turn 1 is research and no blanket instruction to avoid tools.

### How many tool calls are generated?

There is intentionally no configured minimum.

- Retrieval calls per conversation: `0..episode.max_retrieval_calls` successful
  calls.
- Successful retrieval calls per turn:
  `0..episode.max_retrieval_calls_per_turn`.
- All tool executions per turn: `0..episode.max_tool_calls_per_turn`.
- All tool attempts per conversation:
  `0..episode.max_tool_calls_per_conversation`.

A failed retrieval may retry within a turn because only successful retrievals
consume the retrieval-specific cap; failed executions still consume general
tool-attempt capacity. Independent tools are not blocked merely because the
retrieval cap was reached. For example, a batch may contain one retrieval and
one memory read when both are useful.

### Retrieval novelty and low-gain stopping

The runtime uses two complementary safeguards:

1. Before execution, it rejects a query whose lexical token similarity to an
   earlier successful query exceeds
   `query_lexical_similarity_threshold`.
2. After execution, it measures the fraction of new chunk IDs and lexical
   overlap with prior evidence. A search is marked low gain when either measure
   crosses its configured boundary.

After `max_low_gain_chain` consecutive low-gain outcomes, a lexically related
follow-up is rejected. One lexically distinct exploratory query may still run;
if its observed evidence is also low gain, retrieval is paused for the rest of
the episode. If it produces high-gain evidence, the consecutive chain resets.
This evidence-level hard stop prevents synonymous rewordings from consuming the
remaining retrieval budget while still giving a genuinely different facet one
chance. Memory and other tools remain available. Evaluation reports successful,
low-gain, and rejected redundant calls.

Lexical checks catch exact repeats, reordered words, and high-overlap
rephrasings. They do not reliably identify every synonymous or cross-language
paraphrase. Observed evidence-gain checks provide the second line of defense;
deployments needing embedding or model-based semantic comparison can add it as
a reviewed validator without changing the query/conversation boundary.

### Conversation configuration reference

| Knob | Default | Effect |
|---|---:|---|
| `episode.turn_budget.min` | `6` | Minimum user/assistant turn pairs. |
| `episode.turn_budget.max` | `40` | Maximum user/assistant turn pairs. |
| `episode.honor_seed_turn_budget` | `false` | Use a valid seed budget when true; otherwise sample deterministically. |
| `episode.max_retrieval_calls` | `6` | Successful retrieval upper bound per conversation; never a target. |
| `episode.max_retrieval_calls_per_turn` | `1` | Successful retrieval upper bound per turn. |
| `episode.retrieval_novelty.query_lexical_similarity_threshold` | `0.80` | Pre-execution redundant-query boundary. |
| `episode.retrieval_novelty.evidence_lexical_similarity_threshold` | `0.85` | Prior-evidence overlap that marks low gain. |
| `episode.retrieval_novelty.min_new_chunk_fraction` | `0.50` | Minimum returned chunk-ID novelty. |
| `episode.retrieval_novelty.max_low_gain_chain` | `1` | Low-gain outcomes allowed before only one distinct exploratory retrieval may run. |
| `episode.retrieval_novelty.low_gain_followup_similarity_threshold` | `0.35` | Relationship threshold for continuing a low-gain chain. |
| `episode.max_steps_per_turn` | `6` | Structured assistant correction/action steps. |
| `episode.max_tool_calls_per_turn` | `2` | All-tool attempt cap for one turn. |
| `episode.max_tool_calls_per_conversation` | `16` | All-tool attempt cap for the episode. |
| `context.compression_threshold` | `32000` | Estimated active tokens that trigger hidden compaction. |
| `context.model_token_limit` | `65536` | Fail-safe active-context boundary. |
| `context.recent_raw_turns` | `4` | Recent turns preserved verbatim after compaction. |
| `context.min_turns_between_compression` | `3` | Minimum spacing between compactions. |
| `context.compression_token_budget` | `500` | Structured summary budget. |
| `context.max_reasoning_tokens` | `400` | Maximum retained model reasoning tokens. |
| `validation.require_final_answer_each_turn` | `true` | Require every user turn to end with a tool-free assistant response. |
| `judge.enabled` | `true` | Enable model quality gating. |
| `judge.min_score` | `3` | Minimum score on each configured dimension. |
| `export.format` | `messages_and_tools` | `messages`, `messages_and_tools`, or `rich`. |

Legacy `retrieval_calls: {min: 0, max: N}` is migrated to an upper cap. Any
nonzero legacy minimum is rejected. Legacy novelty field names are migrated,
but new configuration should use the explicit `*_lexical_*` names.
`retrieval_depth_weights` is rejected because evidence/task diversity now
belongs to query synthesis and no longer controls runtime calls.

### Context compression

Compression is hidden model-facing state, not a tool call and not an exported
message. A structured summary records supported facts, exact chunk IDs,
constraints, open questions, and source message IDs. Recent turns remain raw.
The original conversation is retained in the canonical record.

If compression fails below the model limit, generation records a warning and
continues. If it fails at or above the configured model limit, the record fails
instead of silently truncating context.

## Providers and role models

`providers` and `models` are converted to Data Designer model-provider and
model configurations. Conversation generation requires these aliases:

| Alias | Responsibility |
|---|---|
| `assistant` | Assistant actions, tool decisions, and responses. |
| `user` | Natural user follow-ups. |
| `compressor` | Hidden structured context summaries. |
| `judge` | Quality gating and optional rejudging. |

Aliases may share one served model or use separate models. Query synthesis uses
the aliases selected by `generator_alias` and `judge_alias`.

## Retrieval endpoint mapping

The `retriever` section adapts a domain evidence API without domain-specific
code:

| Field | Effect |
|---|---|
| `endpoint`, `method` | Request target and GET/POST method. |
| `query_field`, `top_k_field` | Request-key mapping. |
| `top_k` | Default result count. |
| `results_path` | Dot path to the response list. |
| `fields.*` | Dot paths for ID, text, title, source, score, URL, and date. |
| `selection` | `ranked`, deterministically `sampled`, or source-oriented `diverse`. |
| `timeout_seconds` | Request timeout. |
| `retries`, `backoff_seconds` | Bounded transport recovery. |
| `headers`, `extra_body` | Static service additions; do not place secrets here. |

Missing result IDs are replaced with stable content hashes. Stable IDs are
strongly preferred because citation and novelty checks depend on identity.

## Adding or changing tools

Each YAML tool entry contains an OpenAI function `schema`, a trusted Python
`executor` import path, and optional `executor_kwargs`.

To add a tool:

1. Implement the executor protocol from `executors/base.py`.
2. Validate arguments and return a `ToolResult`.
3. Make external side effects idempotent where possible.
4. Add the JSON schema and executor import path to the conversation YAML.
5. Add deterministic checks for claims or state changes introduced by the
   tool.
6. Add registry, runtime, and replay-validation tests.
7. Raise generic tool caps only when the new workflow requires more capacity.

Retrieval-specific caps apply only to the tool named `retrieve`; memory and
custom tools remain governed by generic tool caps. Simulated results retain an
explicit `_sdg_simulated` marker and must not be presented as real external
observations.

## Adapting to another domain or language

1. Replace `taxonomy.example.yaml` with reviewed topic leaves, exclusions,
   required terms, seed searches, and optional metadata filters.
2. Map the domain evidence endpoint in both YAML files.
3. Choose managed persona locales and pin reviewed asset revisions.
4. Adjust task-shape, surface-form, persona-mode, and language weights.
5. Update global instructions for the target response style and citation
   requirements without adding retrieval quotas.
6. Add domain tools as reviewed executors.
7. Extend deterministic validation and judge dimensions for domain-specific
   risks.
8. Generate a small batch, inspect rich records manually, then scale.

Visible user and assistant text follows the configured language. Query
synthesis performs basic script checks for configured English and Devanagari
outputs. The optional seed instruction keeps `reasoning.think` in English; this
is an output-field requirement, not a claim about a model's private reasoning.

## Data Designer resume behavior

`paths.artifacts` contains Data Designer's native dataset state.

| `run.resume` | Behavior |
|---|---|
| `never` | Start without resuming stored progress. |
| `always` | Resume compatible completed row groups and reject incompatible state. |
| `if_possible` | Resume compatible state when available; otherwise create a new run. |

Use a unique `run.dataset_name` and artifact directory for each workload. An
interrupted in-flight row group may be recomputed. After completion, the package
validates row IDs/order and atomically materializes query seeds or generated
records.

## Outputs and evaluation

Canonical record fields:

| Field | Contents |
|---|---|
| `run_id`, `config_fingerprint`, `query_id` | Reproducibility and identity. |
| `status` | `accepted`, `rejected`, `quarantine`, or `generation_failed`. |
| `messages`, `tools` | Full OpenAI-style trajectory and advertised schemas. |
| `episode_spec` | Length and hard runtime caps; no semantic plan. |
| `tool_call_attempts` | Executed attempts, success, errors, and retrieval-quality metrics. |
| `metadata` | Seed provenance, observed counts, context history, and rejected calls. |
| `retrieval_transcript` | Successful queries, returned chunk IDs, and novelty measurements. |
| `memory_events` | Allowed memory reads and writes. |
| `compaction_events` | Hidden summaries and provenance. |
| `validation` | Deterministic errors and warnings. |
| `judgment` | Model scores, rating, explanation, and gate errors. |

Deterministic validation checks tool schemas, call/result pairing, budgets,
turn structure, query repetition, low-gain chains, final responses, and visible
chunk IDs. The prompt standardizes visible citations as `[[full-exact-ID]]`, so
unknown IDs are rejected regardless of the evidence service's identifier shape;
legacy `h-*` and `chunk-*` forms are also recognized.

`evaluate` writes:

- `paths.canonical` with all terminal records;
- `output_dir/accepted.jsonl`;
- `output_dir/rejected.jsonl`;
- `output_dir/quarantine.jsonl`;
- `output_dir/generation_failed.jsonl`;
- `output_dir/summary.json` with acceptance, turn, tool, successful retrieval,
  low-gain retrieval, and rejected redundant-search statistics.

Only accepted canonical records are exported.

## File guide

Configuration and entry points:

| File | Responsibility |
|---|---|
| `config/default.yaml` | Complete conversation example and all runtime knobs. |
| `config/query_generation.example.yaml` | Complete query-synthesis example. |
| `config/taxonomy.example.yaml` | Domain-neutral hierarchical taxonomy example. |
| `src/long_context_sdg/__main__.py` | `synthesize`, `prepare`, `generate`, `evaluate`, and `export` CLI. |
| `config.py`, `config_base.py` | Strict conversation models, migrations, path resolution, fingerprints. |
| `service_config.py` | Shared provider, model, and evidence-endpoint schemas. |
| `schemas.py` | Public seed, message, tool, compaction, validation, and record schemas. |

Query synthesis:

| File | Responsibility |
|---|---|
| `query_generation/config.py` | Independent query config, surface/task profiles, strict loading. |
| `query_generation/taxonomy.py` | Taxonomy loading and hashing. |
| `query_generation/allocation.py` | Exact largest-remainder scheduling. |
| `query_generation/evidence.py` | Evidence-pool filtering and diverse bundle sampling. |
| `query_generation/personas.py` | Managed persona keys and safe compact projection. |
| `query_generation/candidates.py` | Deterministic cross-marginal candidate construction. |
| `query_generation/schemas.py` | Candidate, draft, evidence-need, judgment, and terminal schemas. |
| `query_generation/prompts.py` | Draft and independent-judge prompts. |
| `query_generation/validation.py` | Leakage, language, rewrite-gain, per-facet, and duplicate checks. |
| `query_generation/generator_config.py` | Data Designer custom-column configuration. |
| `query_generation/generator.py` | Per-row structured generation, retries, judging, and seed creation. |
| `query_generation/pipeline.py` | Data Designer orchestration, final coverage report, atomic publication. |

Conversation generation:

| File | Responsibility |
|---|---|
| `conversation_generation.py` | Public seed-to-conversation module boundary. |
| `seeds.py` | Seed validation, deterministic turn sampling, atomic preparation. |
| `episode_control.py` | Hard-cap episode specification without semantic planning. |
| `prompts.py` | Neutral user/assistant prompts and exact citation allowlists. |
| `runtime.py` | Turn loop, natural tool actions, caps, novelty, compaction, and judging. |
| `generator_config.py`, `generator.py` | Data Designer episode-column configuration and row generator. |
| `pipeline.py` | Data Designer conversation orchestration and result materialization. |
| `models.py`, `llm.py` | Model facade resolution, structured JSON calls, bounded retries. |
| `retrieval.py` | Generic evidence HTTP adapter and response mapping. |
| `tool_registry.py` | Trusted executor loading and JSON-schema argument validation. |
| `executors/base.py` | Executor protocol, services, and conversation state. |
| `executors/retrieval.py` | Live retrieval tool and transcript updates. |
| `executors/memory.py` | Allowlisted read/write memory tool. |
| `executors/simulated.py` | Explicitly marked synthetic tool results. |
| `compression.py`, `tokens.py` | Hidden structured compaction and context accounting. |
| `reasoning.py` | Reasoning-size and chunk-citation checks. |

Evaluation and packaging:

| File | Responsibility |
|---|---|
| `validation.py` | Deterministic trajectory replay validation. |
| `evaluation.py` | Revalidation, optional rejudging, partitioning, and statistics. |
| `records.py` | Canonical record I/O. |
| `exporters.py` | Trainer-facing output formats. |
| `plugin.py` | Data Designer plugin entry points. |
| `tests/` | Deterministic unit and end-to-end regression tests. |

## Testing and release checks

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
uv build
```

Tests use deterministic doubles only inside `tests/`. A production workload
should also run a bounded integration against its configured endpoints and
manually inspect rich records before scaling.

## Troubleshooting

**A config field is rejected as extra**

The key is obsolete or misspelled. Compare with the complete example. Strict
rejection prevents silent changes in dataset behavior.

**Query synthesis accepts too few rows**

Read `paths.report`. Common causes are weak evidence pools, surface forms that
do not create measurable rewrite gain, duplicate facet probes, script mismatch,
or judge scores below the threshold. Fix the taxonomy/evidence/config rather
than forcing partial publication.

**A conversation makes no retrieval calls**

That can be valid. There is no minimum. Check whether the user request and
available context genuinely required external evidence and inspect judge
scores if the omission appears incorrect.

**Searches are rejected as repetitive**

Inspect `metadata.rejected_tool_calls` and `retrieval_transcript`. The assistant
must reuse existing evidence or pursue a substantively different unresolved
facet. Tune lexical and evidence-gain thresholds only after reviewing samples.

**Generation is quarantined**

Deterministic validation passed, but judging was unavailable or invalid. Fix
judge access and run `evaluate --rejudge`; conversation regeneration is usually
unnecessary.

**Data Designer cannot resume**

Restore the matching config/dataset identity or deliberately start a new
artifact location. Do not edit Data Designer's internal state.
