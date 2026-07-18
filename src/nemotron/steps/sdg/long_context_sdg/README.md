# Long-Context Synthetic Data Generation

`long_context_sdg` generates realistic, multi-turn, tool-using conversations for
supervised fine-tuning. The pipeline is domain- and language-neutral: behavior
comes from YAML instructions, seed records, model choices, retrieval mappings,
tool schemas, and executor implementations.

The package provides:

- deterministic episode planning with 6–40-turn conversations;
- a separate opening-intent distribution so the first assistant response does
  not always retrieve;
- required retrieval depths of one, two, or three distinct queries on planned
  research turns;
- optional retrieval, memory, custom real tools, and explicitly marked
  simulated tools;
- hidden, source-linked context compaction without altering exported messages;
- deterministic validation followed by an optional model judge;
- append-only, episode-level checkpoints and safe resume;
- canonical, partitioned, and trainer-oriented JSONL outputs;
- Data Designer orchestration with a prepare, generate, evaluate, and export CLI.

## Architecture

```text
raw seed JSONL
    |
    +-- prepare --> enriched seed JSONL
                       |
                       +-- Data Designer generate ---- EpisodeRunner
                                                        |
                      deterministic plan <--------------+
                                                        |
                 user model <--> assistant model <--> ToolRegistry
                                                        |-- retrieval
                                                        |-- memory
                                                        |-- custom executor
                                                        `-- simulated executor
                                                        |
                               compressor model <-------+ (hidden active view)
                                                        |
                 deterministic validator --> judge model
                                                        |
                                               checkpoint JSONL
                                                        |
                               evaluate --> canonical + status partitions
                                                        |
                                 export --> accepted training JSONL
```

An episode is the atomic unit. Each completed attempt is flushed and `fsync`'d
to the checkpoint before the next seed begins. Compaction changes only the
model-facing active view; the canonical trajectory retains the full original
messages and records compaction events separately.

## Installation

Python 3.11–3.13 is supported.

```bash
cd src/nemotron/steps/sdg/long_context_sdg
uv sync --extra dev
```

Copy the complete example instead of editing it in place:

```bash
cp config/default.yaml config/my-workload.yaml
```

Replace the example model endpoint, model names, credential-variable name, and
retrieval API contract. Do not place credentials directly in YAML.

```bash
export MODEL_API_KEY='...'
```

The loader accepts one complete YAML file; it does not merge overlays.
Configuration paths are resolved relative to the YAML file containing them.
Unknown YAML keys are rejected, so misspellings fail before generation.

## Running the pipeline

Prepare validates, enriches, and atomically replaces the enriched seed file:

```bash
uv run long-context-sdg prepare --config config/my-workload.yaml
```

Generate pending episodes:

```bash
uv run long-context-sdg generate --config config/my-workload.yaml
```

Evaluate the latest attempt for every query and write canonical partitions:

```bash
uv run long-context-sdg evaluate --config config/my-workload.yaml
```

Re-run the model judge for every structurally valid canonical candidate:

```bash
uv run long-context-sdg evaluate --config config/my-workload.yaml --rejudge
```

Evaluate without contacting the judge model. Records still needing a judgment
remain quarantined:

```bash
uv run long-context-sdg evaluate --config config/my-workload.yaml --no-network
```

Export accepted records in `export.format`:

```bash
uv run long-context-sdg export --config config/my-workload.yaml
```

Data Designer is the only supported generation path. The package does not ship
a separate standalone model runner.

## Multilingual generation

The target language is controlled by the global and per-seed `instructions`;
there is no English-only runtime restriction. For example:

```yaml
instructions: >-
  Write every visible user and assistant turn in natural Hindi. Keep exact
  retrieved chunk identifiers unchanged. Translate explanations, but preserve
  source-language quotations and standard technical terms when translation
  would reduce precision. Write the synthetic reasoning.think field in English.
```

The retrieval corpus and generated conversation may use different languages.
Cross-language retrieval quality depends on the embedding model and index, not
this pipeline. Before a large run, probe representative target-language queries
and inspect whether the returned chunks are relevant, authoritative, and current.
An English source corpus can still ground a Hindi conversation when retrieval
and the role models are sufficiently multilingual.

`reasoning.think` is a model-generated, bounded field retained as
`reasoning_content`; it is not an observation of the serving model's hidden
chain of thought. Its language is not guaranteed unless requested and validated.
If downstream training requires English reasoning with target-language visible
turns, state both requirements explicitly and add a language check to dataset
quality control.

## Seed format

The raw seed file is JSONL: one JSON object per line. The only required field is
`query`.

```json
{"query":"Explain the feature and its limitations."}
```

A rich seed can customize one episode:

```json
{
  "query_id": "topic-001",
  "query": "Compare the available options.",
  "naive_query": "Which option should I choose?",
  "persona": {
    "role": "application user",
    "expertise": "novice",
    "style": "concise and curious"
  },
  "instructions": "Answer in French and define specialized terms.",
  "turn_budget": 18,
  "retrieval_depth": 2,
  "memory_seed": {"verbosity": "concise"}
}
```

| Field | Default or constraint | Effect |
|---|---|---|
| `query` | Required, nonempty string | Episode topic and user-simulator anchor. |
| `query_id` | Stable SHA-256-derived ID | Resume key. Explicit IDs should be unique and stable across runs. |
| `naive_query` | `query` | Exact first user message. |
| `persona.role` | `user` | User identity or relationship to the task. |
| `persona.expertise` | `intermediate` | Expected knowledge level. |
| `persona.style` | `natural and curious` | User-message tone and interaction style. |
| `instructions` | Empty | Appended to global `instructions` for this seed. |
| `turn_budget` | Schema default `18`; 6–40 | Used only when `planning.honor_seed_turn_budget` is true. |
| `retrieval_depth` | Sampled if omitted; 1–3 | Number of distinct successful retrieval queries required on each planned research/rewrite turn. |
| `memory_seed` | `{}` | Initial allowlisted preference memory. |

Allowed memory keys are `preferred_language`, `verbosity`, `expertise_level`,
`response_format`, `preferred_units`, `focus_area`, and `citation_style`.
Unknown keys are rejected. Values written during an episode must be JSON scalars.

Preparation rejects duplicate `query_id` values and never curates, translates,
or rewrites the source questions. Its output stores each validated `EpisodeSeed`
as a JSON string under `episode_input`, which is the Data Designer seed contract.

## Planning conversations

Planning is deterministic for `(run.seed, query_id)`. Repeating those inputs with
the same generation configuration produces the same turn budget, retrieval
depth, and intent sequence; model sampling can still vary the text.

### Opening diversity

`first_turn_intents` is sampled only for turn 1. `intents` is sampled for turns
2 through the end. This separation prevents every opening from following the
same pattern.

The first user message itself is always `naive_query`. The opening intent guides
the first assistant response. For example:

- `clarify` can elicit missing task details;
- `user_context` can ask about relevant preferences or circumstances;
- `scope` can establish boundaries;
- `orientation` can provide a high-level map;
- `direct_answer` can answer from established information;
- `misconception_check` can test a premise;
- `example_first` can begin with a concrete illustration;
- `research` and `rewrite` require retrieval.

The intent label is included in the model directive. Clear, descriptive custom
labels are allowed, but only the exact labels `research` and `rewrite` activate
mandatory retrieval in the current planner.

Weights are relative, need not sum to 1, must be nonnegative, and must have a
positive total. Set a weight to zero to disable an intent.

### Planning knobs

| Key | Schema default / bounds | Effect |
|---|---|---|
| `planning.turn_budget.min` | `6`; 6–40 | Smallest sampled conversation length. |
| `planning.turn_budget.max` | `40`; 6–40 and ≥ `min` | Largest sampled conversation length. |
| `planning.honor_seed_turn_budget` | `true` | When false, ignore seed budgets and sample the configured range. The example sets this to false. |
| `planning.retrieval_depth_weights` | `{1: .25, 2: .5, 3: .25}` | Relative distribution for seeds without `retrieval_depth`. Only keys 1–3 are valid. |
| `planning.max_steps_per_turn` | `6`; 2–12 | Maximum assistant action attempts in one turn. Must exceed every enabled retrieval depth so a final-answer step remains. |
| `planning.ensure_retrieval_turn` | `true` | If no research/rewrite intent was sampled, replace one non-opening intent with `research`. Never forces turn 1. |
| `planning.first_turn_intents` | See YAML | Opening assistant-behavior distribution. |
| `planning.intents` | See YAML | Later user/assistant intent distribution. |

With the example configuration, turn budgets vary uniformly from 6 through 40,
seed-level budgets are ignored, and retrieval depth is sampled 65%/25%/10% for
depths 1/2/3.

### Tool-call counts and limits

Let:

- `B` be the selected turn budget;
- `R` be the number of turns whose intent is `research` or `rewrite`;
- `D` be the episode retrieval depth;
- `S` be `max_steps_per_turn`.

The planned mandatory retrieval count is `R × D`.

- With `ensure_retrieval_turn: true`, `R >= 1`, so the mandatory minimum is
  `D` retrieval calls per conversation.
- With it false, the mandatory minimum is zero.
- The theoretical planned maximum is `B × D`, which is 120 at the schema limits
  of 40 turns and depth 3.
- On a required turn, the runtime requests one retrieval per action step until
  `D` distinct successful queries are complete, then reserves a step for the
  final answer. The configuration validator therefore requires `S > D`.
- Failed or empty required retrievals consume steps. At most `S - 1` retrieval
  attempts can precede a successful final answer on that turn.
- On a non-required turn, retrieval is optional. A general assistant action can
  contain multiple tool calls and the runtime then forces a tool-free final
  answer on the next step.

There is no configurable `max_tool_calls_per_action`,
`max_tool_calls_per_turn`, or `max_tool_calls_per_conversation` in version
`0.1.0`. Consequently, the mandatory retrieval count is bounded, but optional
multi-call actions do not have a hard numeric quota. `max_steps_per_turn` is an
action-attempt limit, not a complete cost cap. Enforce an external budget or add
explicit quotas before using untrusted model/tool combinations with strict cost
requirements.

## Complete configuration reference

### `paths`

All paths may be absolute or relative to the config file.

| Key | Purpose |
|---|---|
| `seeds` | Raw input JSONL. |
| `enriched_seeds` | Atomically prepared Data Designer seed JSONL. |
| `checkpoint` | Append-only attempt log. |
| `canonical` | Latest evaluated attempt for each query. |
| `output_dir` | Status partitions and `summary.json`. |
| `export` | Accepted trainer-oriented JSONL. |

Do not reuse one checkpoint path across different generation configurations.

### `run`

| Key | Default / values | Effect |
|---|---|---|
| `mode` | `preview`; `preview` or `create` | Selects the Data Designer execution mode. |
| `seed` | `7` | Deterministic seed enrichment and episode planning. Included in the checkpoint fingerprint. |
| `num_records` | `0`; ≥0 | Zero means all seeds. A positive value limits the ordered input prefix. |
| `retry_failed` | `false` | Regenerate queries whose latest checkpoint attempt is `generation_failed`. |
| `retry_quarantine` | `false` | Regenerate queries whose latest attempt is `quarantine`. Consider `evaluate --rejudge` first when only judging failed. |

Retries append a new attempt; they never rewrite checkpoint history. Evaluation
keeps only the latest attempt for each `query_id` in canonical outputs.

### `instructions`

Global generation policy. State the target language, domain boundaries,
grounding expectations, desired tone, citation rules, safety constraints, and
what the assistant should do when evidence is insufficient. Seed instructions
are appended after the global instructions.

### `providers`

| Key | Effect |
|---|---|
| `name` | Unique provider identifier referenced by models. |
| `endpoint` | Model provider endpoint passed to Data Designer and used for offline rejudging. |
| `api_key_env` | Name of the environment variable containing the provider credential. |

Provider names must be unique. Data Designer may additionally support providers
configured by its own installation.

### `models`

Exactly four aliases are required; additional unique aliases are allowed.

| Alias | Responsibility | Suggested characteristics |
|---|---|---|
| `assistant` | Answers, writes retrieval queries, and chooses optional tools. | Strong instruction following and structured JSON. |
| `user` | Produces turns 2 onward in persona. | Diverse, natural conversational behavior. |
| `compressor` | Creates source-linked summaries of completed context. | Faithful summarization with conservative temperature. |
| `judge` | Scores full trajectories. | Reliable structured evaluation and sufficient context length. |

Each model has:

| Key | Effect |
|---|---|
| `alias` | Runtime role name; aliases must be unique. |
| `model` | Provider-specific model identifier. |
| `provider` | Provider name. |
| `skip_health_check` | Passed to Data Designer. |
| `inference_parameters` | Provider-supported chat parameters such as `temperature` and `max_tokens`. |

Data Designer supplies the configured role models during generation. The
offline rejudge client sends the judge's inference parameters to
`/chat/completions`, except transport-only `timeout` and
`max_parallel_requests`. Runtime model calls are retried up to eight times with
exponential backoff, and structured responses receive up to three
schema-correction attempts. These retry counts are implementation constants,
not YAML knobs in this version.

### `context`

| Key | Default / bounds | Effect |
|---|---|---|
| `compression_threshold` | `32000`; ≥256 and below model limit | Approximate active-token count that makes compaction eligible. |
| `model_token_limit` | `65536`; ≥512 | Episode fails if compression fails at or above this boundary. It is not a universal request preflight limit. |
| `recent_raw_turns` | `4`; 1–20 | Full recent turns retained beside the latest summary. |
| `min_turns_between_compression` | `3`; ≥1 | Minimum spacing between successful compactions. |
| `compression_token_budget` | `500`; ≥100 | Requested summary budget included in the compressor prompt. |
| `max_reasoning_tokens` | `400`; ≥32 | Hard validation limit for `reasoning.think`. |

Token counting uses `cl100k_base` when available and a deterministic estimate as
a fallback. Calibrate thresholds against the chosen models and target language;
token density varies across scripts and tokenizers.

The compressor must identify the covered turn range, source message IDs, user
facts, key facts and their chunk IDs, constraints, and open questions. Unknown
message or chunk references reject the compaction. A failed compaction becomes a
warning below `model_token_limit`; at or above the limit it fails the episode.

### `retriever`

The built-in adapter supports GET and POST JSON APIs.

| Key | Default / bounds | Effect |
|---|---|---|
| `endpoint` | Required | Query API URL. |
| `method` | `POST`; `GET` or `POST` | Sends the request as JSON or query parameters. |
| `query_field` | `query` | Request key for the model-authored query. |
| `top_k_field` | `top_k` | Request key for requested result count. |
| `top_k` | `4`; 1–100 | Default chunks requested when the tool call omits it. |
| `results_path` | `chunks` | Dot-separated path from response root to the result list. Empty means the root. |
| `fields.id` | `id` | Dot-separated chunk-ID mapping. Missing IDs receive stable hashes. |
| `fields.text` | `text` | Required content mapping; empty content is discarded. |
| `fields.title/source/score/url/date` | Same-named fields | Optional normalized metadata mappings. |
| `selection` | `ranked` | `ranked`, deterministic `sampled`, or source-oriented `diverse`. |
| `timeout_seconds` | `45`; >0 | Per-request timeout. |
| `retries` | `4`; 1–20 | HTTP attempts. |
| `backoff_seconds` | `1.0`; ≥0 | Exponential retry base, capped at 15 seconds. |
| `headers` | `{}` | Static request headers. Keep secrets out of committed YAML. |
| `extra_body` | `{}` | Constant request/query fields merged before query and `top_k`. |

Normalized chunks contain `chunk_id`, `content`, `title`, `source`, `score`,
`url`, `date`, and unmapped top-level `metadata`. The client is persistent for
connection reuse and is closed by both orchestration paths.

### `tools`

Each tool definition contains:

| Key | Effect |
|---|---|
| `schema` | OpenAI function-tool schema presented to the assistant and used for argument validation. |
| `executor` | Trusted `module:Class` import path. Configuration files are executable trust boundaries. |
| `executor_kwargs` | Optional constructor keyword arguments. |

The schema must contain a unique `function.name`. The built-in `retrieve` tool
is required because the planner and validator use it for evidence-depth
requirements. Every model-authored call is normalized and JSON-Schema validated
before execution. Unknown tools, invalid arguments, and executor failures are
recorded as rejected calls.

Built-in executors:

- `RetrievalExecutor` calls the configured retrieval client and records query,
  turn, call ID, returned chunk IDs, and success.
- `MemoryExecutor` reads/writes only allowlisted preferences and records memory
  events. Memory is episode-local; no cross-episode storage is provided.
- `SimulatedExecutor` asks the configured simulator model for a JSON value and
  wraps the result with `_sdg_simulated: true` so synthetic results cannot be
  mistaken for real observations.

#### Adding a real tool

1. Implement a class whose constructor accepts `services=` and optional keyword
   arguments.
2. Implement `execute(call, state, context) -> ToolResult`.
3. Put the class in a trusted importable module.
4. Add a precise JSON schema and the `module:Class` path to YAML.
5. Add the required service client to `ExecutionServices` if the standard
   `models` and `retriever` fields are insufficient.
6. Add deterministic unit tests for validation, state changes, failures, and
   serialization.

Minimal executor:

```python
from long_context_sdg.schemas import ToolResult


class CatalogExecutor:
    def __init__(self, *, services, namespace="default"):
        self.client = services.catalog
        self.namespace = namespace

    def execute(self, call, state, context):
        payload = self.client.lookup(
            namespace=self.namespace,
            item_id=call.arguments["item_id"],
        )
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            payload=payload,
        )
```

If a service is added to `ExecutionServices`, update both orchestration paths
that construct it.

#### Adding a simulated tool

Use the simulated executor only when synthetic observations are acceptable:

```yaml
- schema:
    type: function
    function:
      name: catalog_lookup
      description: Look up an item in the catalog.
      parameters:
        type: object
        properties:
          item_id: {type: string}
        required: [item_id]
  executor: long_context_sdg.executors.simulated:SimulatedExecutor
```

Do not use simulated output to claim grounding in an external source. Filter or
separate simulated trajectories if the downstream task expects real tool
observations.

### `validation`

`require_final_answer_each_turn` defaults to true. When enabled, every planned
turn must contain a user message and a nonempty, tool-free assistant answer.

Deterministic validation also checks:

- monotonic turn ordering;
- unique call IDs and matching tool results;
- schema-valid arguments and known tool names;
- no leaked internal compaction tool;
- required retrieval count, nonempty chunk IDs, and distinct normalized queries;
- a final tool-free assistant message;
- bounded reasoning and valid cited chunk IDs.

Structural failure produces `rejected`; an exception that prevents completing
the episode produces `generation_failed`.

### `judge`

| Key | Default / bounds | Effect |
|---|---|---|
| `enabled` | `true` | Run the judge after deterministic validation. |
| `min_score` | `3`; 1–5 | Required score on every configured dimension. |
| `dimensions` | `[]` in schema | Names the judge must score. The example supplies eight quality dimensions. |

Acceptance requires every requested dimension, every score at or above the
threshold, and `rating: success`. A judge exception produces `quarantine`, which
preserves a structurally valid trajectory for later rejudging.

### `export`

`format` supports:

- `messages`: `{messages}` only;
- `messages_and_tools`: `{messages, tools}`;
- `rich`: the complete canonical record.

Only `accepted` canonical records are exported.

## Checkpoint and resume behavior

The checkpoint is append-only JSONL and records terminal episode attempts:
`accepted`, `rejected`, `quarantine`, or `generation_failed`.

Before generation, the pipeline:

1. parses every checkpoint line;
2. rejects mixed configuration fingerprints;
3. derives completed query IDs according to retry policy;
4. writes an ordered pending seed file for Data Designer.

The fingerprint covers generation-affecting configuration and `run.seed`. It
excludes paths plus orchestration-only `mode`, record count, and retry flags.

Checkpoint granularity is one full episode, not one turn. If execution stops
during an episode, that episode starts again; every earlier completed episode is
durable. Do not run multiple writers against the same checkpoint: the in-process
lock does not coordinate separate processes.

Evaluation treats the checkpoint as attempt history and writes only the latest
attempt for each query to canonical and partition files. It writes:

- `canonical` with every latest evaluated attempt;
- `output_dir/accepted.jsonl`;
- `output_dir/rejected.jsonl`;
- `output_dir/quarantine.jsonl`;
- `output_dir/generation_failed.jsonl`;
- `output_dir/summary.json` with counts, acceptance rate, retrieval depths, and
  turn budgets.

## Adapting to another domain or language

Create a complete workload config and change these layers deliberately:

1. **Seeds:** supply representative topics, personas, naive phrasings, and
   optional per-seed constraints. Preserve stable IDs across reruns.
2. **Global instructions:** name the target language, terminology policy,
   audience, grounding boundary, desired style, and insufficient-evidence
   behavior.
3. **Models:** choose role-appropriate context length, structured-output
   reliability, language capability, token limits, and temperatures.
4. **Retrieval mapping:** map the exact request keys, result path, and response
   fields. Verify that stable chunk IDs and useful source metadata are returned.
5. **Tool descriptions:** describe domain semantics precisely; the assistant
   uses descriptions to decide whether an optional call is appropriate.
6. **Intent distributions:** emphasize clarification, context gathering,
   comparison, application, or research to match the downstream product.
7. **Context thresholds:** calibrate against observed token counts in the target
   language and the smallest context window among active roles.
8. **Judge dimensions:** add domain-relevant quality criteria and test that the
   judge returns every requested score consistently.
9. **Small validation run:** inspect retrieval queries, tool payloads, citations,
   opening diversity, compaction continuity, and rejection reasons before a
   large run.

Example language policy:

```yaml
instructions: >-
  Write every user and assistant message in the target language. Define
  specialized terms on first use. Preserve product names exactly. Base factual
  claims only on retrieved evidence, and state when evidence is insufficient.
```

Temperature is a useful diversity control: the user role usually benefits from
more variation than the compressor and judge. Do not rely on temperature alone;
seed personas and intent weights create more controllable diversity.

## Output record anatomy

A rich `CanonicalRecord` contains:

| Field | Contents |
|---|---|
| `run_id` | Orchestration identifier derived from the fingerprint. |
| `config_fingerprint` | Resume-compatibility hash. |
| `query_id` | Seed identity. |
| `status` | Terminal quality state. |
| `messages` | Full OpenAI-style trajectory; compaction is not inserted. |
| `tools` | Tool schemas used during generation. |
| `episode_plan` | Turn intents and retrieval requirements. |
| `metadata` | Query, instructions, budgets, counts, message-turn mapping, context history, and rejected calls. |
| `retrieval_transcript` | Retrieval queries and returned chunk IDs by turn. |
| `memory_events` | Reads and writes without hidden internal state. |
| `compaction_events` | Validated hidden summaries and provenance. |
| `validation` | Deterministic errors and warnings. |
| `judgment` | Model scores, rating, explanation, and gate errors. |

Reasoning is retained as `reasoning_content` when the model supplies it. Confirm
that the intended trainer supports this field, or transform the export before
training.

## Production operation

- Version and review every config used to generate a dataset.
- Store credentials in environment variables or a secret manager.
- Use a unique output tree per fingerprint and workload.
- Run `prepare` before Data Designer generation.
- Start with a small record limit and inspect outputs manually.
- Monitor checkpoint line count and status summary, not export count alone.
- Budget for multiple model calls per turn, retrieval retries, structured-output
  corrections, compression, and judging.
- Keep retrieval and custom-tool APIs idempotent because retries can repeat
  requests.
- Keep executor import paths restricted to reviewed modules.
- Preserve checkpoints until canonical evaluation and export are verified.
- Do not train directly from the append-only checkpoint; retries may create
  multiple attempts for one query.

## Testing

The test suite uses deterministic doubles confined to `tests/`; production
scripts do not monkeypatch models or retrieval.

```bash
uv run ruff check .
uv run pytest -q
```

Important coverage includes seed reproducibility, opening diversity, retrieval
fallback placement, structured JSON recovery, transport retries, tool argument
validation, retrieval normalization, memory allowlists, simulated-result
marking, full runtime trajectories, compression, checkpoint durability,
validation, evaluation, and export.

Before releasing a workload, also run a real bounded integration against the
chosen services and inspect the resulting rich records.

## Troubleshooting

**Configuration fails with an extra-field error**

The YAML contains an unknown or obsolete key. Compare it with
`config/default.yaml`; unknown keys are intentionally not ignored.

**Checkpoint fingerprint mismatch**

A generation-affecting knob changed. Restore the original config or choose a
new checkpoint/output tree. Do not combine incompatible attempts.

**No records are generated**

All selected query IDs are already completed under current retry policy, or
`run.num_records` selected an empty input. Check the checkpoint and seed IDs.

**Research turn exhausts its step budget**

The API returned empty results, repeated normalized queries, invalid structured
output, or executor errors. Inspect `rejected_tool_calls` and the retrieval
transcript. Ensure `max_steps_per_turn` is greater than every enabled depth.

**Trajectory is quarantined**

Deterministic validation passed but the judge was unavailable or invalid. Fix
judge access and use `evaluate --rejudge`; regeneration is usually unnecessary.

**Trajectory is rejected**

Inspect `validation.errors` first, then `judgment.gate_errors`. Rejection is a
quality result, not an orchestration exception.

**Compaction repeatedly fails**

Check compressor structured-output reliability, context length, provenance IDs,
and token budgets. Lower the compression threshold only after confirming the
compressor can summarize the earlier prefix accurately.

**Offline judge call is unauthorized**

Confirm `providers[].api_key_env` names an exported variable. An empty value is
valid only for a judge endpoint intentionally configured without authentication.

**Expected response fields are missing**

Adjust `results_path` and `fields.*`. Dot paths traverse nested objects; they do
not index arrays. The resolved result container must be a list of objects.

## File guide

### Configuration, data, and entry points

| File | Responsibility |
|---|---|
| `config/default.yaml` | Complete domain-neutral template with placeholders and all supported knobs. |
| `data/queries.jsonl` | Minimal raw and rich generic seed examples. |
| `pyproject.toml` | Package metadata, dependencies, console command, Data Designer plugin entry point, and test/lint settings. |
| `uv.lock` | Reproducible dependency resolution. |

### Package modules

| File | Responsibility |
|---|---|
| `src/long_context_sdg/__main__.py` | `prepare`, `generate`, `evaluate`, and `export` CLI. |
| `src/long_context_sdg/checkpoint.py` | Checkpoint parsing, fingerprint verification, resume index, append/flush/fsync. |
| `src/long_context_sdg/compression.py` | Compressor prompts, summary rendering, and source-provenance validation. |
| `src/long_context_sdg/config.py` | Strict Pydantic configuration, constraints, path resolution, and fingerprints. |
| `src/long_context_sdg/evaluation.py` | Latest-attempt selection, deterministic revalidation, optional rejudging, partitions, and summary. |
| `src/long_context_sdg/exporters.py` | Accepted-only messages, messages+tools, and rich JSONL export. |
| `src/long_context_sdg/generator.py` | Data Designer column generator and episode checkpoint adapter. |
| `src/long_context_sdg/generator_config.py` | Data Designer column and side-effect-column contract. |
| `src/long_context_sdg/llm.py` | Sync/async facade normalization, retries, JSON extraction, and structured validation. |
| `src/long_context_sdg/models.py` | Offline rejudge client and provider credential resolution. |
| `src/long_context_sdg/pipeline.py` | Data Designer models/providers, pending-seed construction, and orchestration. |
| `src/long_context_sdg/planning.py` | Seeded intent planning and non-opening retrieval fallback. |
| `src/long_context_sdg/plugin.py` | Data Designer plugin registration. |
| `src/long_context_sdg/prompts.py` | Assistant, retrieval-only, final-only, and user prompt builders. |
| `src/long_context_sdg/reasoning.py` | Reasoning token and chunk-citation validation. |
| `src/long_context_sdg/retrieval.py` | Persistent retrying HTTP adapter, response mapping, stable IDs, and selection. |
| `src/long_context_sdg/runtime.py` | Episode state engine, tool loops, compaction, validation, judging, and canonical projection. |
| `src/long_context_sdg/schemas.py` | Seed, plan, message, tool, compression, judgment, and canonical schemas. |
| `src/long_context_sdg/seeds.py` | Stable IDs, deterministic enrichment, JSONL parsing, duplicate detection, and atomic preparation. |
| `src/long_context_sdg/tokens.py` | Token estimation, active-context accounting, and compaction meter. |
| `src/long_context_sdg/tool_registry.py` | Trusted executor imports, tool-call normalization, schema validation, and dispatch. |
| `src/long_context_sdg/validation.py` | Replayable message/tool/retrieval/final-answer validation. |
| `src/long_context_sdg/executors/base.py` | Shared state, services, context, error, and executor protocol. |
| `src/long_context_sdg/executors/retrieval.py` | Retrieval execution and transcript capture. |
| `src/long_context_sdg/executors/memory.py` | Allowlisted episode-local memory. |
| `src/long_context_sdg/executors/simulated.py` | Explicitly labeled model-simulated tool output. |

### Tests

| File | Responsibility |
|---|---|
| `tests/fixtures.py` | Temporary configs, deterministic model doubles, retrieval double, and custom executor. |
| `tests/test_config_seeds_planning.py` | Enrichment, variable budgets, planning, first-turn diversity, and prompt constraints. |
| `tests/test_llm_retries.py` | Transport retries and robust structured JSON extraction. |
| `tests/test_retrieval_registry_memory.py` | Retrieval mapping, registry validation, memory policy, and simulated marking. |
| `tests/test_runtime_e2e.py` | Full episodes, tool use, hidden compaction, and correction behavior. |
| `tests/test_tokens_reasoning.py` | Token accounting, compaction spacing, and reasoning validation. |
| `tests/test_validation_checkpoint_export.py` | Validation, durable checkpointing, fingerprint enforcement, evaluation, and export. |

## Known constraints

- Checkpoints are episode-level and single-writer.
- Optional tool calls have no hard per-action, per-turn, or per-conversation cap.
- Only `research` and `rewrite` intent labels make retrieval mandatory.
- Model retry and structured-correction counts are not configurable in
  YAML.
- Token counts are estimates and may differ from the serving model tokenizer.
- `model_token_limit` is a compression-failure boundary, not a universal hard
  limit on every request.
- Simulated tool output is synthetic even though it is explicitly marked.
- Custom executor paths execute trusted Python and require code review.
