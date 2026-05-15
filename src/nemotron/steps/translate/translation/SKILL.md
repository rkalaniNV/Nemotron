---
name: translation
description: Translate JSONL or Parquet training corpora with NeMo Curator's TranslationStage, preserving structured fields and optionally attaching FAITH translation-quality scores. Use when translating training data, scoring translation quality, or wiring `translate/translation` into a multilingual customization pipeline.
when_to_use: Use for requests like "translate this English JSONL to Hindi for SFT", "translate a chat corpus and keep the messages schema intact", "score existing translations with FAITH", or "swap translation backends from llm to nmt/google/aws". Do not use for translating BYOB MCQ benchmarks (see `src/nemotron/steps/byob/SKILL.md`).
compatibility: Install the optional `translation` extra before running this step. Translation runs on NeMo Curator's experimental `TranslationStage`; FAITH scoring uses the same pipeline through Curator's `AsyncOpenAIClient`.
metadata:
  owner: nemotron
  workflow-step: translate/translation
---

# Corpus Translation + FAITH Scoring

Use this skill to translate training corpora with NeMo Curator while keeping
structured fields (messages, code, JSON payloads) intact, and to optionally
attach FAITH translation-quality scores for downstream filtering.

This step stays a thin wrapper around Curator's reader → `TranslationStage` →
writer pipeline. Do not generate custom pandas chunking or row-level processing
unless a single huge input file requires a one-off preprocessing stage.

## Default

1. Install translation runtime dependencies with `uv sync --extra translation`
   or `pip install ".[translation]"` in the target environment.
2. Read [step.toml](step.toml) for the artifact contract, parameters,
   strategies, and failure modes.
3. Start from [config/default.yaml](config/default.yaml) and override only what
   the user asks for.
4. Ask the user explicitly for `source_language` and `target_language`
   (ISO 639-1 codes). Do not silently default.
5. Run locally with `uv run --extra translation nemotron steps translation
   input_path=... output_dir=... source_language=... target_language=...`.
6. Run on Lepton/Slurm with `uv run nemotron steps run translate/translation -c
   default --batch lepton_translate ...`.

## Change Points

- The agent surface is YAML; do not invent new keys. Mirror the keys read by
  [step.py](step.py) and the existing [config/default.yaml](config/default.yaml).
- Switch backends with `backend=llm|nmt|google|aws`; configure only the
  matching `server` / `nmt` / `google` / `aws` block. Keep secrets out of the
  YAML — pass them through environment variables.
- Toggle FAITH with `faith_eval.enabled` and `faith_eval.threshold`. FAITH
  needs an LLM client even when translation itself uses a non-LLM backend.
- For OpenAI-style chat data, set `text_field=messages.*.content` and
  `reconstruct_messages=true` so the writer emits `translated_messages`.
- For huge single files, generate a one-off chunking pre-step that writes
  smaller shards into a directory; do not push partitioning logic into this
  step.

## Gotchas

- **`text_field=messages.*.content` is the rule for chat corpora.** Pointing
  this step at the top-level `text` field for OpenAI-style records will skip
  the assistant turn. Always inspect the input schema first.
- **FAITH evaluation is gated by an LLM client.** Even when `backend=nmt`,
  `google`, or `aws`, enabling `faith_eval.enabled=true` requires
  `server.url`, `server.model`, and `server.api_key` (or the env var named in
  `server.api_key_env`). Without those, the step raises before a single row
  translates.
- **Hosted model names can be retired.** For real runs, verify the configured
  `server.model` is still listed before scaling. Catch-all defaults like
  `meta/llama-...` will silently 404 if the catalog rotates.
- **Do not mix JSONL and Parquet inside one input directory.** Curator's
  reader is file-format-uniform per partition. Split the dataset or set
  `input_format` explicitly.
- **`segmentation_mode=coarse` preserves valid JSON, fenced code, and
  tool-call payloads.** Switch to `fine` only when the upstream segmenter is
  splitting natural-language paragraphs too aggressively.
- **`output_mode=both` keeps audit metadata.** Use it whenever translated
  fields will gate SFT data; agents downstream should be able to read FAITH
  scores and the raw translation alongside the replaced field.
- **Cloud credentials live in the environment, not in YAML.** For
  `backend=google` and `backend=aws`, configure provider auth through the
  runtime environment and only set project/location/region in YAML.

## Validate

- `uv run nemotron steps show translate/translation` prints the manifest
  and confirms the step is registered.
- `uv run --extra translation nemotron steps translation -c default
  input_path=<10-row sample> output_dir=/tmp/translate_smoke
  source_language=en target_language=hi` runs the smallest possible smoke
  (see [use-case-examples/translate-corpus/](../../../../../use-case-examples/translate-corpus/)).
- For chat data, confirm the writer emits `translated_messages` alongside
  `translated_text` when `reconstruct_messages=true`.
- For FAITH, confirm `faith_score_avg` and per-segment FAITH metadata appear
  in the output records.

## Load More Only If Needed

- [step.toml](step.toml) for parameters, strategies, errors, references.
- [config/default.yaml](config/default.yaml) for the production-shape config.
- [step.py](step.py) for the exact keys the runner reads.
- [docs/customize/steps/translate/translation.md](../../../../../docs/customize/steps/translate/translation.md) for the agent checklist and CLI matrix.
- [skills/nemotron-customize/context/curator-translation-faith.txt](../../../../../skills/nemotron-customize/context/curator-translation-faith.txt) for upstream Curator API extracts (load only when generating exceptional code).
- [use-case-examples/translate-corpus/](../../../../../use-case-examples/translate-corpus/) for a 10-row hello-world.

## Related Steps

- [`byob`](../../byob/SKILL.md) — translates BYOB MCQ benchmarks. Different
  schema; do not point that flow at this step's CLI.
- [`curate/nemo_curator`](../../curate/nemo_curator/SKILL.md) — runs upstream
  filtering before translation.
- [`data_prep/sft_packing`](../../data_prep/sft_packing/SKILL.md) — packs the
  translated chat JSONL for Megatron-Bridge SFT.
