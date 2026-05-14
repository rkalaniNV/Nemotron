---
name: nemotron-translate
description: Route Nemotron translation steps for corpus translation, structured chat translation, cloud/NMT/LLM backends, and FAITH scoring before downstream data_prep or SFT.
---

# Translation

Use this category when a corpus must be translated before SFT, pretraining
prep, BYOB review, or downstream quality filtering.

## Route

| Need | Step |
|---|---|
| JSONL/Parquet corpus translation with optional FAITH scoring | [`translation`](translation/SKILL.md) |

## Workflow

1. Read `translation/step.toml` first. It lists required language codes, input
   format controls, backend configs, FAITH settings, strategies, and errors.
2. Ask for `source_language`, `target_language`, `input_path`, and
   `text_field`; do not silently infer production language pairs.
3. Pick one backend (`llm`, `nmt`, `google`, or `aws`) and keep credentials in
   the runtime environment.
4. Keep `output_mode=both` while validating so translated fields, raw metadata,
   and FAITH scores are inspectable.
5. Validate a small shard before feeding translated data into data_prep or SFT.

## Guardrails

- Do not run one translation step across mixed JSONL and Parquet directories.
- Use coarse segmentation first for structured records, code, JSON, tool
  payloads, and chat messages.
- Keep FAITH scoring model/server settings explicit, even when the translation
  backend is not LLM-based.
