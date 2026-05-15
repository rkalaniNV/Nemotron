# Translate Corpus

This example shows the current Nemotron `translate/translation` flow for
translating a small training corpus and inspecting FAITH translation-quality
scores. It mirrors the layout of the BYOB use-case so an agent has one
copy-pasteable hello-world per pipeline.

The notebook is intentionally aligned with the agentic step structure:

- The reusable step lives in `src/nemotron/steps/translate/translation/`.
- The starter config lives in `src/nemotron/steps/translate/translation/config/`.
- Runtime execution goes through `nemotron steps translation`.
- The notebook creates only example-local inputs, outputs, and a working config.

## Files

| File | Purpose |
| --- | --- |
| `translate_corpus.ipynb` | End-to-end notebook for writing 10 JSONL chat rows, deriving a working config from the packaged translation config, running the step, and previewing translated output + FAITH scores. |
| `assets/` | Example corpus location. The notebook creates `assets/sample_chat.jsonl` (10 rows). |
| `config/` | Example working config location. The notebook writes `en_to_hi.yaml` from the packaged `config/default.yaml`. |
| `outputs/` | Example output location for translated JSONL shards and FAITH metadata. |

## Run

From the repository root:

```bash
uv run jupyter lab use-case-examples/translate-corpus/translate_corpus.ipynb
```

The notebook defaults to a dry command preview so it does not accidentally spend
model quota. Set `RUN_TRANSLATE = True` in the run cell after reviewing the
generated config and credentials. NVIDIA-hosted endpoints use `NGC_API_KEY` or
`NVIDIA_API_KEY`.

The equivalent CLI is:

```bash
uv sync --extra translation
uv run --extra translation nemotron steps translation \
  -c default \
  input_path=use-case-examples/translate-corpus/assets/sample_chat.jsonl \
  output_dir=use-case-examples/translate-corpus/outputs/en_to_hi \
  source_language=en \
  target_language=hi \
  text_field=messages.*.content \
  reconstruct_messages=true \
  faith_eval.enabled=true \
  output_mode=both
```

## What you should see

After a successful run, `outputs/en_to_hi/` contains JSONL shards with:

- `messages` — original chat messages (preserved when `output_mode=both`).
- `translated_messages` — chat messages with translated `content` fields.
- `translated_text` — the flattened translated string per row.
- `faith_score_avg` — FAITH average score (only when `faith_eval.enabled=true`).
- Per-segment FAITH metadata for downstream filtering.

If FAITH scoring is enabled and the configured `server.model` cannot be
reached, the step raises before any rows translate. That is the intended
guardrail for hosted endpoints — verify the model name first.
