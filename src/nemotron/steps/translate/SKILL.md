---
name: nemotron-translate
description: Translate JSONL or Parquet training corpora with NeMo Curator, including structured chat fields, hosted LLM, NMT, Google, AWS, optional FAITH scoring, and skip-already-translated input rows. Use when preparing multilingual data before data prep, SFT, CPT, or review.
---

# Nemotron Translation

Use this skill when a user wants to translate corpus data, chat records, or row-oriented training artifacts. The concrete step is [`translate/translation`](translation/SKILL.md).

## Default Workflow

1. Install runtime dependencies with `uv sync --extra translation`.
2. Read [`translation/step.toml`](translation/step.toml) for the step contract.
3. Ask for `source_language`, `target_language`, input path, output path, backend, and field path. Do not infer source or target language silently.
4. For downstream training data, start with `output_mode=replaced`, `merge_scores=false`, and `faith_eval.enabled=false`.
5. For audit or quality review, use `output_mode=both` and enable `faith_eval`.
6. Run a two-row smoke test before a large corpus.
7. Validate row count, schema, translated field content, and that secrets were not printed.

For one-shot or eval-style requests, do not end in exploration mode. Deliver a
minimal runnable handoff first, then add optional checks:

- `Decision`: chosen step and scope (`translation-only` or `translation+FAITH`).
- `Config`: key fields or config path bound to a concrete input path/format.
- `Run`: exact command.
- `Output`: expected output directory/artifacts.
- `Env`: required environment variable names only.

Practical stop condition for single-turn runs:

- After a short orient pass, if key constraints are mostly known, stop
  discovery and ship runnable handoff immediately.
- Prefer inline config evidence over waiting for file-generation steps.
- If execution cannot run, still provide blocker-aware handoff with exact `Run`
  and `Output`.
- Do not delegate broad repository audits to subagents for one-shot translation
  requests; use only targeted reads needed for runnable output.

Before finalizing, make the following evidence explicit in user-facing output:

- Which input path/format is used for this run.
- Which incompatible inputs were excluded (for example mixed JSONL/Parquet roots).
- Any observed language mismatch vs requested translation direction, with stated assumption.
- Execution-ready handoff (`Run`, `Output`, `Env`, optional toggles).
- Keep these statements explicit in prose even when config/command implies them.

Add a short `Constraint resolution` section before the handoff:

- `Language`: observed sample language vs requested source language, plus chosen assumption.
- `Model`: exact model variant used (or clearly stated default if user gave family only).
- `Credentials`: env-var names used for auth (names only, never values).

For `translation+FAITH` requests, append a `FAITH handoff` section in the same
response:

- confirm `faith_eval.enabled=true` (and any filter setting assumptions),
- list expected FAITH outputs/artifacts,
- include exact `Run` command and `Output` path.

## Backend Choice

| Need | Backend | Notes |
| --- | --- | --- |
| Structured chat, tool calls, JSON, code, or high formatting fidelity | `llm` | Use OpenAI-compatible endpoint settings under `server`. |
| Large plain-text corpus with local service | `nmt` | Service must expose `GET /health` and `POST /translate`. |
| Managed provider translation | `google` or `aws` | Credentials must come from environment or provider config, not YAML secrets. |
| Quality scoring for any backend | FAITH | FAITH still needs an LLM client even when translation backend is not `llm`. |

## Common Commands

Plain text JSONL through a hosted LLM:

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_llm_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Structured chat records:

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/chat_code_en.jsonl" \
  output_dir="$TR_ROOT/out_chat_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field='messages.*.content' \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=true \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

NMT server:

```bash
uv run --no-sync nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_nmt_hi" \
  source_language=en \
  target_language=hi \
  backend=nmt \
  nmt.server_url="$NMT_SERVER_URL" \
  text_field=text \
  output_mode=replaced \
  merge_scores=false \
  reconstruct_messages=false \
  faith_eval.enabled=false
```

## Patterns To Cite

- [`../patterns/translate-training-corpus.md`](../patterns/translate-training-corpus.md) for inserting translation before prep or training.
- [`../patterns/prefer-llm-for-structured-chat.md`](../patterns/prefer-llm-for-structured-chat.md) for chat, tool-call, JSON, and code-heavy data.
- [`../patterns/prefer-nmt-for-large-corpora.md`](../patterns/prefer-nmt-for-large-corpora.md) for large plain-text translation.
- [`../patterns/enable-faith-for-high-value-data.md`](../patterns/enable-faith-for-high-value-data.md) for quality annotation or filtering.
- [`../patterns/multilingual-tokenizer-check.md`](../patterns/multilingual-tokenizer-check.md) before using translated data for SFT or CPT.

## Guardrails

- Do not build custom readers or writers first. Use Curator `JsonlReader` or `ParquetReader`, `TranslationStage`, and `JsonlWriter` or `ParquetWriter`.
- Do not mix JSONL and Parquet in one input directory.
- If the user provides a mixed-format root, require an explicit include/exclude
  decision in the final plan or handoff.
- Do not use `merge_scores=true` with `output_mode=replaced`; use `output_mode=both` if scores must be merged.
- Do not treat `skip_translated=true` as output-directory resume. It only skips input rows that already contain a non-empty translation column.
- Do not enable FAITH filtering without telling the user that rows may be dropped.
- Keep API keys in environment variables such as `NVIDIA_API_KEY`, `NGC_API_KEY`, AWS credentials, or Google application credentials.
- Never run environment-dump commands (`env`, `printenv`, `set`, broad
  `export` listings) or echo secret variables in shell output.
- For diagnostics, mention only env-var names; keep values fully redacted.

## Common Pitfalls / Troubleshooting

- CLI mismatch or unexpected-argument errors: fall back to the documented
  command shape in this file (`uv run --no-sync nemotron steps translation ...`)
  or confirm valid flags with `--help`. Do not invent unsupported subcommands.
- `ModuleNotFoundError` for translation dependencies: run
  `uv sync --extra translation` first. If an eval environment still misses
  basics like `toml` or `pyyaml`, report the blocker and provide runnable
  handoff instead of stalling in discovery.
- Mixed `.jsonl` and `.parquet` roots: bind `input_path` to one format only and
  explicitly list excluded paths/formats in the handoff.
- Missing `translate/translation` directory or metadata in the runtime
  workspace: treat it as an environment/path issue, state the blocker, and
  provide inline config + exact run command the user can execute in a complete
  checkout.
- Path-not-found during validation: do not retry with guessed path variants.
  First confirm actual created paths, then validate only those files.
- Source language does not match request: state the observed mismatch and the
  assumption used for this run.

## Load More

- [`guide.md`](guide.md) for detailed flow, output modes, FAITH, resume semantics, and validation.
- [`translation/SKILL.md`](translation/SKILL.md) for the concrete step.
- [`translation/config/default.yaml`](translation/config/default.yaml) for starter config.
- [`translation/step.py`](translation/step.py) for the reader -> translation stage -> writer implementation.
