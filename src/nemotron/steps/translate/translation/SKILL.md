---
name: nemotron-translate-translation
description: Configure and run the concrete translate/translation step for JSONL or Parquet corpus translation with NeMo Curator's experimental TranslationStage.
---

# Translation Step

Use `translate/translation` for corpus translation. Before changing configs or code, read `step.toml` to understand the step contract, consumed artifacts, produced artifacts, parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume homogeneous JSONL or Parquet records from `input_path`.
- Translate `text_field`, such as `text` or `messages.*.content`.
- Produce translated JSONL or Parquet shards under `output_dir`.
- Use `output_mode=replaced` for training-ready data.
- Use `output_mode=both` for audit data with metadata and scores.

## Configure

- Set `source_language` and `target_language` explicitly.
- Set `backend` to `llm`, `nmt`, `google`, or `aws`.
- For `backend=llm`, set `server.url`, `server.model`, and `server.api_key_env`.
- For `backend=nmt`, set `nmt.server_url` and verify `/health` and `/translate`.
- For structured chat, set `text_field='messages.*.content'` and `reconstruct_messages=true`.
- For plain text, set `text_field=text` and `reconstruct_messages=false`.
- For FAITH annotation, set `faith_eval.enabled=true` and usually `faith_eval.filter_enabled=false` first.
- For FAITH filtering, confirm with the user because rows can be dropped.

For eval-style or one-shot runs, emit early evidence before deep exploration:

- Step decision and scope (`translate/translation`; translation-only or translation+FAITH).
- Input policy for mixed formats (selected JSONL/Parquet path and excluded incompatible inputs).
- Runnable config stub (inline or file path) and execution-ready handoff skeleton.
- If model variant or source-language evidence is incomplete, state explicit
  assumptions and continue with a runnable default rather than stopping.
- Prefer inline config in the response before optional file writes to reduce
  truncation risk in single-turn runs.
- Keep mixed-format include/exclude and language-mismatch statements explicit in
  plain text, not only implicit in config paths.

## Local Files

- Contract: `src/nemotron/steps/translate/translation/step.toml`
- Runner: `src/nemotron/steps/translate/translation/step.py`
- Config: `src/nemotron/steps/translate/translation/config/default.yaml`
- Guide: `src/nemotron/steps/translate/guide.md`
- Patterns: `src/nemotron/steps/patterns/translate-training-corpus.md`

## Avoid

- Do not mix JSONL and Parquet in one input directory.
- Do not store API keys in config files.
- Do not print environment values or run env-dump commands that may expose
  tokens/keys in tool output.
- Do not use `merge_scores=true` with `output_mode=replaced`.
- Do not treat `skip_translated=true` as output-directory resume.
- Do not add custom chunking to `step.py` for normal use. Split huge single files before this step if needed.
- Do not silently enable FAITH filtering for training data.

## Validate

- Import check: `uv run --no-sync python -c "from nemo_curator.stages.text.experimental.translation import TranslationStage; print(TranslationStage)"`.
- Prompt resource check: verify `translate.yaml` and `faith_eval.yaml` exist under `nemo_curator.stages.text.experimental.translation.prompts`.
- LLM smoke: translate two plain-text JSONL rows with `faith_eval.enabled=false`.
- NMT smoke: call `GET /health`, then translate two rows with `backend=nmt`.
- Chat smoke: translate `messages.*.content` and verify `tool_calls[].function.arguments` remains valid JSON.
- If a command fails with CLI argument errors, do not keep guessing arbitrary
  subcommands. Return to the documented step command template and provide a
  corrected runnable handoff.
- If file validation fails with `FileNotFoundError`, re-check actual output
  paths first and validate only existing files instead of trying guessed roots.

## Runtime prerequisites

- Required Python support packages in the runtime should include parsing/config
  basics such as `toml` and `pyyaml`.
- If these are missing in an eval container, report the environment blocker
  explicitly and continue with a complete handoff (`Decision/Config/Run/Output/Env`)
  rather than extending repository exploration.
- Do not add blocker-only responses: always include a runnable command template
  and expected output path even when execution cannot be performed locally.

## Completion checklist (user-visible)

Before ending a response, ensure these are explicitly present:

1. Runnable config evidence (inline key fields or config path).
2. `Run` command and expected `Output` path.
3. Required `Env` variable names (never values).
4. For FAITH runs, confirm `faith_eval.enabled=true` (or equivalent) and expected FAITH outputs.
5. If blocked by environment/runtime, state blocker and provide workaround-ready handoff.

Recommended response shape:

- `Decision`: step + scope.
- `Constraint resolution`: language mismatch handling, model-variant choice/default, credential env-var names.
- `Config`: runnable key fields or config file path.
- `Mixed-format policy`: explicit include/exclude decision for JSONL vs Parquet.
- `FAITH handoff` (for FAITH requests): `faith_eval.enabled`, expected FAITH outputs, and any filter assumptions.
- `Run`: exact command.
- `Output`: expected translated artifact path.
- `Env`: environment variable names only.

Single-turn guardrail:

- Do not keep exploring after the handoff is already runnable.
- Use additional reads only to resolve a concrete blocker, then return to the
  handoff template.
