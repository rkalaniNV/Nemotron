---
name: translation
description: Translate training or curation corpora for Nemotron workflows using Curator readers, TranslationStage, and writers. Use when the user wants corpus translation for CPT, SFT, or stage0 preparation, especially for text, parquet, chat, or wildcard fields such as messages.*.content.
when_to_use: Use for requests like "translate training data", "translate chat data", "translate before CPT", "translate before SFT", or "translate messages.*.content". Do not use for MCQ benchmark-only translation.
compatibility: Requires the Nemotron Python package plus a working NeMo Curator installation and, depending on backend, either an OpenAI-compatible endpoint, cloud translate credentials, or a local NMT server.
metadata:
  owner: nemotron
  workflow-step: stage0
---

# Translation

Use this skill for corpus translation, not benchmark-only translation.

## Default

1. Read [references/STEP.md](references/STEP.md) for the step contract.
2. Use [assets/default.yaml](assets/default.yaml) as the starting config.
3. Read [references/guide.md](references/guide.md) if backend choice, data shape, or chunking is unclear.
4. Run:

```bash
nemotron steps translation
```

Apply only the overrides needed by the request.

Example direct invocation:

```bash
nemotron steps translation \
  translation.input_path=/workspace/data/*.jsonl \
  translation.output_dir=/workspace/out \
  translation.source_lang=en \
  translation.target_lang=de \
  translation.text_field=messages.*.content \
  translation.reconstruct_messages=true
```

## Gotchas

- For chat or tool-calling transcripts, prefer `backend=llm` and set `text_field=messages.*.content`.
- If the output needs a translated message list, also set `reconstruct_messages=true`.
- Valid JSON tool payloads should survive unchanged inside translated message structures.
- `faith_eval.segment_level=true` is the preferred FAITH mode for long documents; Curator scores segment rows before reassembly and filters only after reassembly.
- `skip_translated=true` only works correctly when `translation_column` points at an existing translated field.
- Curator reader partitioning is file-oriented. For one huge file that still causes memory pressure, generate a one-off preprocessing helper rather than editing the checked-in runtime.

## Validate

- Confirm the output path exists.
- Confirm row counts are stable unless explicit filtering was requested.
- Confirm translated schema matches the requested type.
- If FAITH was enabled, confirm `faith_*` columns exist.
- If tool payloads were present, inspect at least one row to ensure the JSON was preserved.

## Load More Only If Needed

- [references/guide.md](references/guide.md) for backend and chunking decisions
- [references/quality-and-failure-modes.md](references/quality-and-failure-modes.md) for FAITH, resume, and failure behavior
- [references/compute-and-backends.md](references/compute-and-backends.md) for service-level tradeoffs
- [references/data-and-artifacts.md](references/data-and-artifacts.md) for IO and output-shard expectations
