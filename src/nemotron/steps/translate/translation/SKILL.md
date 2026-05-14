---
name: nemotron-translate-translation
description: Configure translate/translation for NeMo Curator corpus translation with LLM, NMT, Google, or AWS backends plus optional FAITH quality scoring.
---

# Corpus Translation

Use `translate/translation` to translate JSONL or Parquet corpora while
preserving structured fields and optional FAITH quality metadata.

Before changing configs or code, read `step.toml` for the artifact contract,
parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume filtered JSONL or homogeneous JSONL/Parquet inputs.
- Produce translated JSONL or Parquet shards.
- For chat records, keep translated messages inspectable before SFT prep.

## Configure

- Ask for `source_language`, `target_language`, `input_path`, and
  `text_field`; do not infer production language pairs silently.
- Set `input_format` explicitly when globs or directories are ambiguous.
- Pick one backend: `llm`, `nmt`, `google`, or `aws`.
- For `backend=llm`, set `server.url`, `server.model`, and either
  `server.api_key_env` or `server.api_key`.
- For FAITH scoring, configure `faith_eval.enabled`,
  `faith_eval.threshold`, and an LLM server/model even when translation uses a
  non-LLM backend.
- Keep `output_mode=both` and `reconstruct_messages=true` when auditability or
  downstream SFT review matters.

## Guardrails

- Do not mix JSONL and Parquet in one input directory.
- Keep provider credentials in the runtime environment, not in config files.
- Use `segmentation_mode=coarse` first for JSON, code, tool payloads, and
  structured chat.
- Split single huge files before running Curator if partition memory pressure
  appears.
