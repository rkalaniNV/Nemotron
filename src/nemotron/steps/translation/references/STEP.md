---
id: nemotron.steps.translation
version: 0.1
owner: nemotron
summary: Translate training or curation corpora with NeMo Curator's TranslationStage.
entrypoint:
  module: nemotron.steps.translation.scripts.run
  script: src/nemotron/steps/translation/scripts/run.py
config:
  default: src/nemotron/steps/translation/assets/default.yaml
consumes:
  - type: text_jsonl
  - type: chat_jsonl
  - type: text_parquet
produces:
  - type: translated_text_jsonl
  - type: translated_chat_jsonl
  - type: translation_report
parameters:
  required:
    - translation.input_path
    - translation.output_dir
    - translation.source_lang
    - translation.target_lang
  important:
    - translation.backend
    - translation.text_field
    - translation.segmentation_mode
    - translation.faith_eval.enabled
    - translation.skip_translated
compute:
  shape: single-node cpu
  optional_services:
    - OpenAI-compatible LLM endpoint
    - Google Cloud Translate
    - AWS Translate
    - local NMT server
source:
  - repo: Nemotron
    path: src/nemotron/steps/translation/scripts/driver.py
  - repo: Curator
    path: nemo_curator/stages/text/translation/
---

# Corpus Translation

## What it does

This step wires Curator's native reader, `TranslationStage`, and writer into one reusable corpus-translation workflow.
It translates the configured text fields and writes Curator-managed output shards with optional FAITH scores and translation metadata.

## Strategies

- Use wildcard text fields such as `messages.*.content` for chat data.
- Set `reconstruct_messages=true` when the output should include a reconstructed translated message list.
- Use `backend=llm` for structured chat data and tool-calling transcripts.
- Use `backend=nmt` for large plain-text corpora when a local server is available.
- Enable FAITH only when quality evidence matters enough to justify extra cost.
- Use `faith_eval.segment_level=true` when FAITH should score short translation segments instead of full reassembled documents.

## Failure Modes

- Missing `NVIDIA_API_KEY` or cloud credentials for the selected backend
- unsupported input format
- unreachable NMT or LLM server
- low FAITH scores when quality filtering is enabled
- ambiguous directories that mix JSONL and Parquet inputs
