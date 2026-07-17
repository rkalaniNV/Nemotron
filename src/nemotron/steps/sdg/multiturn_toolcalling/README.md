# mtsdg — Multi-Turn Long-Context Tool-Calling SDG

A [NeMo **Data Designer**](https://developer.nvidia.com/nemo) column-generator plugin that produces **synthetic multi-turn (20–25 turn) tool-calling chat trajectories** for supervised fine-tuning (SFT), starting from a small `queries.jsonl` seed file.

> Package name: `mtsdg`. Entry point registered as the Data Designer plugin `episode-simulator`.

---

## What is this?

Given one seed information-need (a `queries.jsonl` row), the pipeline has two LLM "agents" role-play an entire coherent conversation, turn by turn, against a **real retriever service**, and emits it as an OpenAI-style `messages` list ready for tool-calling SFT.

Key properties, all verified in the code:

- **Fully LIVE / single-pass.** There is **no prescripted plan and no pre-built synthetic corpus**. For each query the LLM agents generate the whole conversation turn-by-turn, and every `retrieve` call hits a live NeMo Retriever HTTP service (real Constitution-of-India passages by default). A vague query returns weak chunks; a precise rewrite returns better ones — so the query-rewrite skill is learned from a genuine signal. (`generator.py`, `runtime.py`, `retriever.py`)
- **Two agents per episode.** A **User Agent** writes the next user message (turn 1 is the seed's `naive_query`; every later turn is improvised in-persona). An **Assistant Agent** emits bounded "think tokens" (reasoning) and then either calls a tool or answers, running a `retrieve → assess → rewrite → retrieve → answer` loop. (`prompts.py`, `generator.py`)
- **Three model tools:** `retrieve` (the live retriever), `memory_read`, `memory_write` (an allow-listed preference store). (`tools/contracts.py`, `runtime.py`)
- **Automatic context compaction.** A `ContextMeter` (tiktoken-based) tracks the active context; at a token threshold (32k in production; a smaller value in simulation) the prefix is silently compacted into a rolling summary. Later turns condition on `[summary + recent raw turns]`, but **neither a `context.compress` tool call nor the summary text ever appears in the emitted chat** — compaction is metadata only. (`tokens.py`, `compression.py`, `generator.py`)
- **Crash-safe.** Each finished episode is appended to an incremental checkpoint (`output/checkpoint.jsonl`) the moment it completes, so a killed run still yields usable data. (`generator.write_checkpoint`)

---

## How it works

```
 data/queries.jsonl                      one QuerySeed per line
        │
        ▼
 pipelines/run.py                         wraps each row as episode_input (JSON)
        │  builds Data Designer config + custom OpenAI provider
        ▼
 Data Designer engine  ── resolves plugin `episode-simulator` (entry point)
        │                  one seed row  ->  one generator column "conversation"
        ▼
 EpisodeSimulatorGenerator.generate(row)  (generator.py)
        │  resolves 4 model aliases (user/assistant/judge/compressor)
        │  builds RetrieverClient
        ▼
 EpisodeRunner.run_episode(...)  ── loop over turn = 1 .. turn_budget
        │
        │   ┌──────────────────────────────────────────────────────────┐
        │   │  turn 1: user_text = seed.naive_query                     │
        │   │  turn n: User Agent LLM improvises next user message      │
        │   │                                                           │
        │   │  Assistant Agent (majority-voted over N):                 │
        │   │    reason (bounded think) -> tool_calls OR final answer   │
        │   │    ├─ retrieve  -> LiveToolExecutor -> real retriever     │
        │   │    ├─ memory_read / memory_write -> validated KV store    │
        │   │    └─ answer (cite chunk ids)                             │
        │   │                                                           │
        │   │  ContextMeter crosses threshold?                          │
        │   │    -> compress prefix into rolling summary (metadata)     │
        │   └──────────────────────────────────────────────────────────┘
        │
        ▼
 assemble + project structured_messages   (assembler.py)
 validate_trajectory + optional judge      (validators.py, core/llm.py)
        │
        ├─► append episode -> output/checkpoint.jsonl   (incremental, crash-safe)
        ▼
 6 side-effect columns per row  ─────►  filter_accepted (accept.py)
        │                                        │
        ▼                                        ▼
 output/trajectories_full.json          output/const_sft.jsonl   ({"messages":[...]})
```

`pipelines/from_checkpoint.py` can rebuild the same SFT file directly from `output/checkpoint.jsonl` if a run was interrupted.

---

## File-by-file reference

### `src/mtsdg/` — package core

| File | Purpose | Key public API |
|------|---------|----------------|
| `__init__.py` | Package marker / docstring only. | — |
| `schemas.py` | All Pydantic data models + the allowed-memory policy. Domain-agnostic. | `QuerySeed`, `PersonaSeed` (input); `RetrievalChunk`; `ReasoningContent`, `EvidenceSelection`, `ClaimAndSupport`; `Message` (+`.to_openai()`); `AssistantTurn`; `CompressionEvent`, `KeyFact`, `UserStatedFact`, `AuthorityRef`, `ToolOutcome`; `TrajectoryJudgment`. Constants: `ALLOWED_MEMORY_KEYS`, `RETRIEVER_TOOL`, `MODEL_TOOLS`, `ALL_TOOLS`. |
| `tokens.py` | Token accounting + the compaction trigger. Uses tiktoken (`cl100k_base`) when available, else a conservative char/word heuristic. | `count_tokens`, `message_tokens`, `context_tokens`, `ContextMeter` (`.add`, `.should_compress`, `.reset_after_compression`, `.history`). |
| `retriever.py` | Thin HTTP client for the live NeMo Retriever (`POST {url}/query`), with backoff/retry. Builds `chunk_id`/`title` from source+page. | `RetrieverClient` (`.query`, `.health`), `prefetch_query_catalog`, `DEFAULT_RETRIEVER_URL`. |
| `runtime.py` | Live tool executor. Resolves `retrieve` against the real retriever and `memory_read`/`memory_write` against a validated allow-listed KV store. Accumulates retrieved chunks for grounding. | `LiveToolExecutor` (`.execute`), `ToolCall` (`.from_openai`), `ToolResult`, `ConversationState`, `ToolError`. |
| `reasoning.py` | Bounded, auditable "think token" gate. Enforces a token budget (default 400) and that every cited chunk was actually returned by `retrieve`. Hard errors (fabricated citation / over-budget) vs. soft warnings (empty scaffold, uncited claim). | `validate_reasoning_content`, `reasoning_to_text`, `ReasoningValidation`, `MAX_REASONING_TOKENS`. |
| `assembler.py` | Deterministic assembly of turn blocks into one ordered message list, and projection to the emitted `structured_messages` (strips bookkeeping). No compress artifacts ever emitted. | `assemble_blocks`, `project_structured_messages`, `render_summary` (used only to condition later turns). |
| `compression.py` | Generates + validates one `context.compress` result over the completed prefix. Drops teacher-invented provenance (unknown message/chunk IDs) rather than failing the whole event. | `generate_compression_event`, `DEFAULT_COMPRESSION_TOKEN_BUDGET`. |
| `validators.py` | Deterministic gates that run before any LLM judge. Provenance checks for compaction; structural checks over assembled messages (tool-call/result pairing, turn monotonicity, unknown tools, and rejecting any leaked `context.compress`). | `validate_compression_event`, `validate_trajectory`, `ValidationReport`. |
| `prompts.py` | All prompt templates. | `SYSTEM_POLICY`, `USER_AGENT_LIVE_PROMPT`, `ASSISTANT_AGENT_SYSTEM_PROMPT`, `CONTEXT_COMPRESSION_PROMPT`, `TRAJECTORY_JUDGE_PROMPT`. |
| `generator.py` | **Heart of the pipeline (Job B).** The DD-independent `EpisodeRunner` orchestrates the whole turn-by-turn live episode (user turn, assistant tool loop, automatic compaction, validation, judging), and the DD wrapper adapts it as a column generator. Also the crash-safe checkpoint writer. | `EpisodeRunner` (`.run_episode`), `EpisodeSimulatorGenerator` (`.generate`), `write_checkpoint`. |
| `generator_config.py` | The Pydantic config for the `episode-simulator` column (all knobs: aliases, thresholds, budgets, checkpoint path) and declares its `required_columns` / `side_effect_columns`. | `EpisodeSimulatorConfig` (`column_type="episode-simulator"`). |
| `accept.py` | Deterministic accept/reject split (Job D). Re-runs the structural + reasoning gates over each row's `structured_messages` and partitions rows (DD has no row-drop primitive). | `filter_accepted`. |
| `model_configs.py` | Model roles, providers, and factories. Supports the built-in `nvidia` provider (Nemotron models) and a custom OpenAI-compatible proxy provider. Also `DirectChatFacade` for out-of-pipeline calls. | `nvidia_provider`, `default_model_configs`, `custom_openai_provider`, `custom_openai_model_configs`, `DirectChatFacade`, `direct_openai_facades`, `api_key_present`. Aliases: `TEACHER`, `USER`, `ASSISTANT`, `JUDGE`, `COMPRESSOR`, `BULK`. |
| `plugin.py` | The Data Designer `Plugin` object (`PluginType.COLUMN_GENERATOR`) tying config → implementation. | `episode_simulator`. |
| `core/__init__.py` | Empty package marker. | — |
| `core/llm.py` | Self-contained LLM utilities (DD + pydantic only). Sync/async-tolerant completion shim (routes through one shared event loop), transient/rate-limit retry with backoff, structured (schema-validated) completion with JSON repair, majority voting over assistant turns, and an `<explanation>/<rating>` inline judge. | `call_llm`, `call_structured`, `call_structured_n`, `majority_vote_tool_calls`, `run_inline_judge`. |
| `tools/__init__.py` | Package marker / docstring. | — |
| `tools/contracts.py` | The JSON tool schemas the agent sees. Three model tools + the app-level `context.compress` (used for validation only; never emitted). | `TOOL_SCHEMAS`, `ALLOWED_TOOLS`, `openai_tools`, `RETRIEVE_SCHEMA`, `MEMORY_READ_SCHEMA`, `MEMORY_WRITE_SCHEMA`, `CONTEXT_COMPRESS_SCHEMA`. |

### `pipelines/` and `scripts/`

| File | Purpose | Key entry |
|------|---------|-----------|
| `pipelines/run.py` | **Live generation driver.** Reads `queries.jsonl`, builds the DD config + custom OpenAI provider, runs `DataDesigner.preview()`, prints per-episode stats, then splits accepted trajectories into the SFT file. | `python pipelines/run.py --queries ... --out ...` |
| `pipelines/from_checkpoint.py` | **Salvage/build SFT from the incremental checkpoint.** Reads `output/checkpoint.jsonl` and writes `{"messages": [...]}` lines for each accepted (or, with `--include-rejected`, every) episode. | `python pipelines/from_checkpoint.py --checkpoint ... --out ...` |
| `scripts/dd_preview_mock.py` | **No-key / no-network smoke test.** Mocks the model endpoint (`FakeFacade`) and the retriever (`FakeRetriever`) and runs the real Data Designer engine end-to-end to prove the plugin is discovered, resolved, and drives a valid trajectory with no compress leakage. | `python scripts/dd_preview_mock.py` |

### `tests/` (expected behavior)

| File | What it pins down |
|------|-------------------|
| `fixtures.py` | `FakeRetriever` (authority chunks for specific queries, distractors for vague ones), `FakeFacade` (prompt-routed fake LLM that drives retrieve→rewrite→answer), `make_query`, `make_fake_models`. |
| `test_generator_e2e.py` | Full episode via fakes: valid trajectory, ≥2 retrieves showing a real rewrite (distractor `doc_10*` then authority `doc_00*`), compaction present in metadata but **not** in chat, reasoning bounded (≤400 tokens) and grounded. |
| `test_runtime.py` | Tool executor: vague vs. specific retrieval, empty-query rejection, memory allow-list enforcement (disallowed keys and non-scalar values rejected), read restricted to allowed keys, unknown tool (`context.compress`) rejected. |
| `test_tokens.py` | Token counting monotonicity, threshold trigger, `min_turns_between` spacing, reset-collapses-to-summary. |
| `test_reasoning_assembler.py` | Fabricated citation = hard error; uncited claim = soft warning; grounded within budget accepted; over-budget rejected; assembler assigns IDs and strips bookkeeping. |
| `test_checkpoint.py` | `write_checkpoint` appends records and is a no-op on an empty path. |

---

## How it plugs into Data Designer

1. **Entry point** — `pyproject.toml` registers the plugin so DD can resolve it by `column_type`:
   ```toml
   [project.entry-points."data_designer.plugins"]
   episode-simulator = "mtsdg.plugin:episode_simulator"
   ```
2. **Plugin object** — `plugin.py` binds config → implementation:
   ```python
   episode_simulator = Plugin(
       impl_qualified_name="mtsdg.generator.EpisodeSimulatorGenerator",
       config_qualified_name="mtsdg.generator_config.EpisodeSimulatorConfig",
       plugin_type=PluginType.COLUMN_GENERATOR,
   )
   ```
3. **Column config** — `EpisodeSimulatorConfig` sets `column_type="episode-simulator"`, requires one input column (`episode_input`, a `QuerySeed` serialized as JSON), and declares **six side-effect columns** it writes per row:

   | Column | Contents |
   |--------|----------|
   | `structured_messages` | The emitted OpenAI-style `messages` list (the SFT target). |
   | `episode_metadata` | query_id, domain, topic, turn_count, threshold, compaction events/triggers, message & chunk counts, user-turn ratings, full tool transcript, reasoning provenance. |
   | `compaction_events` | Hidden rolling-summary provenance (`{summary_id: CompressionEvent}`) — never in the chat. |
   | `trajectory_status` | `True`/`False` — did it pass the deterministic gates. |
   | `trajectory_validation` | `{ok, errors, warnings}` from `validate_trajectory` + reasoning/tool notes. |
   | `trajectory_judgment` | The LLM trajectory judge verdict (or `{"skipped": true}`). |

   The generator resolves four model aliases — `user`, `assistant`, `judge`, `compressor` — from the DD model registry (default all bound to the same model).

---

## How to run

### Prerequisites

- **Python 3.12** (the package requires ≥3.10) in a virtualenv.
- `data-designer`, `pydantic>=2.6`, `tiktoken>=0.7`, `pandas>=2.0` (declared in `pyproject.toml`); `httpx` is used by the retriever/proxy clients; `pytest>=8` for tests.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # installs mtsdg + data-designer + tiktoken + pytest
```

### (a) Run the tests — no API key, no network

```bash
pytest tests/
```
All tests use fake models + a fake retriever, so nothing external is required.

### (b) Prove the plugin works inside real Data Designer — no key, no network

```bash
python scripts/dd_preview_mock.py
```
This mocks the model endpoint and retriever but drives DD's actual engine. Success prints `SUCCESS: live episode-simulator is a working Data Designer plugin.` and asserts a populated `structured_messages` with `trajectory_status=True` and no `context.compress` in the chat.

### (c) Live generation — requires a real key + a reachable retriever

```bash
LLM_API_KEY=sk-...                             \
LLM_BASE_URL=https://inference-api.nvidia.com  \
LLM_MODEL=azure/openai/gpt-5.5                 \
RETRIEVER_URL=http://localhost:8000         \
MAX_PARALLEL=2                                 \
SIM_THRESHOLD=32000                            \
TRAJ_JUDGE=1                                    \
MAJORITY_VOTE_N=1                              \
python pipelines/run.py \
    --queries data/queries.jsonl \
    --out output/const_sft.jsonl
```

Environment variables read by `pipelines/run.py` (and `core/llm.py`):

| Var | Default | Meaning |
|-----|---------|---------|
| `LLM_API_KEY` | — (**required**) | Provider/proxy key for the generation LLM. |
| `LLM_BASE_URL` | `https://inference-api.nvidia.com` | OpenAI-compatible endpoint base URL. |
| `LLM_MODEL` | `azure/openai/gpt-5.5` | Model id bound to every role. |
| `RETRIEVER_URL` | `http://localhost:8000` | Live retriever service. |
| `MAX_PARALLEL` | `2` | DD max parallel requests (kept low; see gotchas). |
| `SIM_THRESHOLD` | `32000` | Token threshold that triggers auto-compaction (also `--sim-threshold`). |
| `TRAJ_JUDGE` | `1` | Run the trajectory judge (`1`/`0`). |
| `MAJORITY_VOTE_N` | `1` | Assistant candidates per turn to majority-vote. |
| `NUM_QUERIES` | all | Limit queries (also `--num-queries`). |
| `RETRIEVE_TOP_K` | `3` | Chunks per retrieve. |
| `MAX_REASONING_TOKENS` | `400` | Think-token budget. |
| `INLINE_JUDGE` | `0` | Also judge each simulated user turn. |
| `CHECKPOINT_PATH` | `output/checkpoint.jsonl` | Incremental checkpoint (reset per run). |
| `LLM_MAX_RETRIES` / `LLM_MAX_RATELIMIT_RETRIES` | `10` / `18` | Backoff retry caps for transient / rate-limit errors. |

Outputs: `output/const_sft.jsonl` (accepted trajectories, `{"messages": [...]}` per line), `output/trajectories_full.json` (all rows with metadata), `output/checkpoint.jsonl` (incremental).

### (d) Build / salvage the SFT file from a checkpoint

If a run was killed before its final write, rebuild the SFT file from whatever completed:

```bash
python pipelines/from_checkpoint.py \
    --checkpoint output/checkpoint.jsonl \
    --out output/const_sft.jsonl
# add --include-rejected to also emit trajectory_status=False episodes
```

---

## Schemas

### Input — one `queries.jsonl` row (`QuerySeed`)

| Field | Type | Notes |
|-------|------|-------|
| `query_id` | str | Unique id. |
| `query` | str | The precise seed information-need. |
| `naive_query` | str | A vaguer first phrasing; used verbatim as the user's turn-1 message and drives the rewrite loop (falls back to `query`). |
| `domain` | str | Free-text hint, e.g. `"indian-constitution"`. |
| `corpus_hints` | list[str] | Optional keywords (metadata). |
| `turn_budget` | int (6–40) | Target number of user turns (typically 20–25). |
| `difficulty` | `easy`/`medium`/`hard` | — |
| `persona` | `{role, expertise, style}` | Drives the User Agent. `expertise` ∈ `novice`/`intermediate`/`expert`. |
| `memory_seed` | dict | Initial durable memory (allowed keys only). |

Allowed memory keys (`ALLOWED_MEMORY_KEYS`): `preferred_language`, `verbosity`, `expertise_level`, `response_format`, `preferred_units`, `focus_area`, `citation_style`.

**Example** (from `data/queries.jsonl`):
```json
{
  "query_id": "q-0001",
  "query": "How can Parliament form a new State or alter the boundaries of an existing State under the Constitution of India?",
  "naive_query": "making new states in India",
  "domain": "indian-constitution",
  "corpus_hints": ["Article 2", "Article 3", "Article 4", "First Schedule", "President recommendation"],
  "turn_budget": 22,
  "difficulty": "hard",
  "persona": {"role": "law student", "expertise": "intermediate", "style": "curious, asks precise follow-ups"},
  "memory_seed": {"preferred_language": "en", "verbosity": "concise", "citation_style": "article-numbers"}
}
```

Two seed files ship in `data/`: `queries.jsonl` (5 Constitution-of-India needs) and `queries_scourt.jsonl` (6 Indian Supreme-Court case needs).

### Output — one `structured_messages` record

An OpenAI-style `messages` list. Roles:

- **`system`** — the baked-in `SYSTEM_POLICY` (message 0).
- **`user`** — the simulated user's turn.
- **`assistant`** — either a tool-calling step (`tool_calls`, empty `content`) or a final answer (`content`, no `tool_calls`). Assistant messages carry **`reasoning_content`** = the bounded natural-language think trace (the trainable field, ≤ `max_reasoning_tokens`).
- **`tool`** — a tool result (`retrieve` returns a JSON list of chunks; `memory_read`/`memory_write` return small JSON), keyed to the assistant call by `tool_call_id` and `name`.

No `context.compress` call and no rolling-summary text ever appear here — compaction lives only in `compaction_events` / `episode_metadata`.

**Example** (abbreviated):
```json
[
  {"role": "system", "content": "You are a helpful research assistant with tools: `retrieve` ..."},
  {"role": "user", "content": "making new states in India"},
  {"role": "assistant", "content": "",
   "reasoning_content": "Need evidence; retrieve first.",
   "tool_calls": [{"id": "call-1-0", "type": "function",
                   "function": {"name": "retrieve",
                                "arguments": "{\"query\": \"making new states in India\", \"top_k\": 3}"}}]},
  {"role": "tool", "name": "retrieve", "tool_call_id": "call-1-0",
   "content": "[{\"chunk_id\": \"doc_10_p3_r1\", \"title\": \"...\", \"content\": \"Article 249 ...\"}]"},
  {"role": "assistant", "content": "",
   "reasoning_content": "Weak/off-topic results; rewrite with Article terms.",
   "tool_calls": [{"id": "call-2-0", "type": "function",
                   "function": {"name": "retrieve",
                                "arguments": "{\"query\": \"Article 3 Parliament form a new State boundaries\", \"top_k\": 3}"}}]},
  {"role": "tool", "name": "retrieve", "tool_call_id": "call-2-0",
   "content": "[{\"chunk_id\": \"doc_00_p1_r1\", \"content\": \"Article 3. Parliament may by law form a new State ...\"}]"},
  {"role": "assistant",
   "content": "Under Article 3, Parliament may by law form a new State or alter boundaries, on the President's recommendation (see doc_00_p1_r1).",
   "reasoning_content": "Refined retrieval returned Article 3/4; answer from doc_00_p1_r1."}
]
```

The `from_checkpoint.py` / `run.py` SFT writers wrap each accepted record as `{"messages": [...]}` per line.

---

## Operational notes / gotchas

- **The retriever is a live HTTP service** you run separately (any NeMo-Retriever-compatible endpoint). Set `RETRIEVER_URL` to it (`POST /query` with `{"query", "num_chunks"}`; `GET /health`). Switch domains by pointing `RETRIEVER_URL` at a retriever over a different corpus and supplying matching seed queries (e.g. `data/queries.jsonl` for the Constitution corpus, `data/queries_scourt.jsonl` for a Supreme-Court case-law corpus). The client retries with backoff since a retriever's vLLM backend can briefly stall under burst load.
- **Memory tool names are underscore, not dotted** — `memory_read` / `memory_write`, *not* `memory.read`. The endpoint's function-calling format rejects dots in tool names. (`retrieve` and the internal `context.compress` are the exceptions; `context.compress` is never actually emitted as a tool call.)
- **`MAX_PARALLEL` is kept low (default 2).** Episodes run concurrently under DD; a high fan-out would burst the shared retriever/model endpoint past its quota and stall the vLLM backend. Raise it only if you have headroom (or split load across keys). Rate-limit/timeout errors are retried automatically with backoff (`core/llm.py`).
- **The incremental checkpoint makes partial runs recoverable.** Every finished episode is `fsync`-appended to `output/checkpoint.jsonl` immediately (thread-safe, best-effort — a checkpoint failure never breaks generation), so a crash, kill, or a run that never reaches DD's final return still leaves usable data. `tail -f output/checkpoint.jsonl` to watch records land; rebuild the SFT file with `pipelines/from_checkpoint.py`.
- **Compaction threshold.** Use `SIM_THRESHOLD=32000` for production-faithful behavior. A smaller value (e.g. the mock uses `1200`) forces 2–3 compactions within a short synthetic episode so the "when to compact" signal is exercised.
- **`response_format` / sampling params.** Structured calls omit `response_format` for the GPT-5.5 proxy (it 400s on `json_object`; the schema is in the prompt and repaired if needed), and temperature/top_p are not forwarded to GPT-5-family models (they reject them). Majority voting uses N separate calls rather than the `n` request param.
