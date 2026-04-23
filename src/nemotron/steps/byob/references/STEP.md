---
id: nemotron.steps.byob
version: 0.1
owner: nemotron
summary: Translate MCQ benchmark rows by adapting them to the generic corpus translation step and restoring BYOB schema after translation.
entrypoint:
  kind: compose
  adapter_module: nemotron.steps.byob.adapter
  translation_step: nemotron.steps.translation
consumes:
  - type: mcq_benchmark_jsonl
produces:
  - type: translated_mcq_benchmark_jsonl
  - type: translation_report
parameters:
  required:
    - byob_translation.dataset_path
    - byob_translation.source_lang
    - byob_translation.target_lang
  important:
    - byob_translation.backend
    - byob_translation.faith_eval.enabled
    - byob_translation.backtranslation_quality_metrics
compute:
  shape: single-node cpu
  optional_services:
    - OpenAI-compatible LLM endpoint
    - Google Cloud Translate
    - AWS Translate
    - local NMT server
source:
  - repo: Nemotron
    path: src/nemotron/steps/byob/adapter.py
  - repo: Curator
    path: nemo_curator/stages/text/translation/
---

# BYOB Benchmark Translation

## What it does

This step provides the stable MCQ schema adapter for benchmark translation.
It flattens `question` and `options` into plain text rows for the generic translation step, then restores the original benchmark schema after translation.

## Strategies

- Preserve row counts during translation and reassembly.
- Use the generic corpus translation step for forward and reverse translation.
- Attach FAITH scores if requested, but do not drop staged rows inline.
- Use round-trip metrics when benchmark quality needs a second check.

## Failure Modes

- missing dataset path
- row count drift if any inline filter drops strings before reassembly
- backend credential or connectivity errors
- malformed MCQ schema
