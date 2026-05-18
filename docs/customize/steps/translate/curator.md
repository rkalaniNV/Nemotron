# Corpus Translation + FAITH Scoring

This step wraps NeMo Curator's reader -> `TranslationStage` -> writer pipeline.
It should stay a thin wrapper around Curator; do not generate custom chunking or
pandas processing unless a single huge input file needs a one-off preprocessing
stage.

```{step-toml} src/nemotron/steps/translate/curator/step.toml
```

## Agent Checklist

- Ask for `source_language` and `target_language` explicitly.
- Ask for `input_path`, `input_format`, and the field path to translate.
- Choose one backend: `llm`, `nmt`, `google`, or `aws`.
- For `llm`, set `server.url`, `server.model`, and `server.api_key_env` or `server.api_key`.
- For `nmt`, set `nmt.server_url`; tune `batch_size`, `timeout`, and concurrency only if needed.
- For `google`, use environment credentials and set `api_version`, `project_id` for v3, and `location`.
- For `aws`, use environment credentials or an IAM role and set `region`.
- If FAITH is enabled, configure the LLM `server` fields even when translation uses a non-LLM backend.
- Keep `output_mode=both` when users need auditability of translated fields and metadata.

## CLI

Install the Curator-backed translation dependencies before running the step:

```bash
uv sync --extra translation
```

Run the step through the generic step dispatcher with `-o key=value` overrides:

```bash
uv run --extra translation nemotron steps run translate/curator \
  -o input_path=/path/to/source.jsonl \
  -o output_dir=/path/to/translated \
  -o source_language=en \
  -o target_language=hi
```

Use `-c` or `--config` to pass a config name from the step's `config/`
directory or a path to a YAML file. Repeat `-o` (or `--override`) for each
config override; bare ``key=value`` positionals are also still accepted.

For batch executors such as Lepton or Slurm, add ``--batch <profile>``:

```bash
uv run nemotron steps run translate/curator \
  -c default \
  --batch lepton_translate \
  -o input_path=/mnt/lustre-shared/data/source.jsonl \
  -o output_dir=/mnt/lustre-shared/output/translated \
  -o source_language=en \
  -o target_language=hi
```

## Reference Implementation

```{literalinclude} ../../../../src/nemotron/steps/translate/curator/step.py
:language: python
:caption: step.py
```

## Starter Config

```{literalinclude} ../../../../src/nemotron/steps/translate/curator/config/default.yaml
:language: yaml
:caption: config/default.yaml
```
