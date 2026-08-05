# BYOB

Use this README to create or translate benchmark artifacts while keeping benchmark-family logic easy for developers to extend.

## MCQ Developer Journey

BYOB turns domain source documents into benchmark artifacts. Treat the source
corpus as evaluation data, not training data: keep it separate from SDG, SFT, and
CPT inputs so the final benchmark remains held out.

1. Organize source documents by target subject or benchmark slice.
2. Run `prepare` to normalize and stage source data.
3. Run `generate` to produce MCQ benchmark Parquet.
4. Optionally run `translate` to create a target-language benchmark while
   preserving MCQ schema and row identity.
5. Validate row count, schema, answer indexes, and quality filters before using
   the benchmark for model claims.

## MCQ Data And Artifact Flow

```text
domain source documents
  -> byob/mcq stage=prepare
  -> staged benchmark source
  -> byob/mcq stage=generate
  -> benchmark_raw.parquet + benchmark.parquet
  -> byob/mcq stage=translate when target-language eval is needed
```

Final benchmark rows must preserve `question_id`, `question`, `options`,
`answer_index`, `answer`, `cot_content`, `src`, and `category`.

## MCQ Quick Start

1. Install BYOB runtime dependencies with `uv sync --extra byob` or `pip install ".[byob]"` in the target environment.
2. Read [references/STEP.md](references/STEP.md) for the artifact manifest.
3. Start from [mcq/config/default.yaml](mcq/config/default.yaml) for MCQ generation or [mcq/config/translate.yaml](mcq/config/translate.yaml) for translation.
4. Ensure generation configs include `target_source_mapping` and explicit
   `filtering_model_configs`.
5. Run `nemotron steps run byob/mcq -c <CONFIG> stage=prepare family=mcq`.
6. Run `nemotron steps run byob/mcq -c <CONFIG> stage=generate family=mcq`.
7. Translate an existing benchmark with `stage=translate` and a translation config.

## Function-Calling Benchmarks (BFCL)

The `bfcl` family generates function-calling benchmarks from an executable oracle
pack instead of source documents, so its flow differs from MCQ:

```text
oracle pack (tools + backend + fixtures + templates + assertions + validation_cases)
  -> byob/bfcl stage=prepare
  -> stage_cache/ normalized artifacts + oracle_validation_report.json
  -> byob/bfcl stage=generate  (requires a gold-eligible report)
  -> expand -> state_machine -> render -> expected_trace
  -> schema_validation -> executable_replay (reset + replay twice + assertions)
  -> benchmark_raw.parquet + benchmark.parquet + run_manifest.json
```

Each generation stage writes one `stage_cache/` parquet keyed by `task_id`
(`task_instances`, `conversation_plans`, `rendered_conversations`,
`expected_traces`, `schema_validated_traces`, `replay_validated_tasks`), so
joining them shows which stage dropped a task.

- Run the whole slice on the checked-in tiny pack with `nemotron steps run byob/bfcl -c src/nemotron/steps/byob/bfcl/config/tiny.yaml stage=all family=bfcl`.
- Swap in `bfcl/config/banking_vn.yaml` for a domain-sized run: it budgets `tasks_per_category` across six categories and covers every conversation shape the pipeline supports.
- Validate a pack without generating with `python -m nemotron.steps.byob.scripts.validate_oracle_pack --config <CONFIG>`.
- No stage of BFCL generation calls a model: user and assistant turns are rendered from the pack's templates.
- Keep pack code under an `oracle_runtime.allowed_roots` entry; the default root is `data/`.
- Keep `oracle_runtime.worker: process`. `thread` runs pack code in-process for debugging and can never reach the gold tier.
- Read [references/bfcl-oracle-pack.md](references/bfcl-oracle-pack.md) for the pack layout, backend contract, validation-case keys, and tier rules.

## CLI And Config Knobs

### MCQ

Start from `mcq/config/tiny.yaml` for a smoke run, `mcq/config/default.yaml` for
generation, and `mcq/config/translate.yaml` for translation. Developers usually
change:

- `stage`: `prepare`, `generate`, `translate`, or `all`.
- `target_source_mapping`: target subjects mapped to source document roots.
- `filtering_model_configs`: explicit model configs for filtering and dedup.
- `skip_until`: resume from an MCQ stage only when the preceding stage cache exists.
- Translation backend and language settings in the translate config.
- BYOB translation controls under `translation_model_config.stage`
  (`translation_prompt_path`) and `translation_model_config.segment_stage`
  (`max_concurrent_requests`, `health_check`, `dry_run`, `dry_run_log_count`).

Example shape:

```bash
uv run nemotron steps run byob/mcq \
  -c src/nemotron/steps/byob/mcq/config/default.yaml \
  stage=all \
  family=mcq
```

### BFCL

Start from `bfcl/config/tiny.yaml` for a smoke run or
`bfcl/config/banking_vn.yaml` for the bundled domain example. BFCL supports:

- `stage`: `prepare`, `generate`, or `all`.
- `oracle_pack.manifest_path`: executable oracle-pack manifest.
- `oracle_runtime`: clock, process-worker timeouts, and `allowed_roots`.
- `task_generation.tasks_per_category`: category-wide task budget.
- `surface_generation.language`: language rendered from pack templates.

BFCL does not support `translate` or `skip_until`; generation must validate the
current pack and config before publishing output.

## Change Points

- Add new benchmark families under `runtime/benchmark_families/<family>/`.
- Before adding a new family, answer the questions in [references/new-family-checklist.md](references/new-family-checklist.md).
- Register the family in `runtime/benchmark_families/registry.py`.
- Keep `scripts/runtime.py` as a dispatcher only; family-specific schema, prompts, postprocessing, and export code belong in family modules.
- Keep MCQ stage orchestration in `runtime/benchmark_families/mcq/pipeline.py`; do not recreate a top-level `runtime/pipeline.py`.
- Use `adapter.py` only for schema bridging when composing BYOB with other pipeline modules.
- Use Curator experimental translation as the translation backend; BYOB should only flatten/reassemble benchmark-family schema around it.
- Use Curator semantic dedup with `RayDataExecutor`, `RayActorPoolExecutor`, and package-level `SemanticDeduplicationWorkflow`.

## Gotchas

- Do not merge the whole runtime into `scripts/runtime.py`; that blocks future GSM8K-style extensions.
- Do not put MCQ-specific orchestration in top-level `runtime/`; family pipelines belong under `runtime/benchmark_families/<family>/`.
- Keep `question_id`, `question`, `options`, `answer_index`, `answer`, `cot_content`, `src`, and `category` stable in final MCQ parquet outputs.
- Do not drop staged rows inline during translation reassembly. Filtering belongs after rows are restored.
- Do not add a translation mode selector; BYOB translation always uses Curator experimental translation.
- Keep semantic dedup as a two-step flow: compute embeddings first, then run KMeans, pairwise similarity, and duplicate identification.
- For MCQ, resume with `--skip-until` only when the expected cached parquet for the previous stage already exists. BFCL does not support stage resume.
- Use deterministic seeds for sampling and distractor shuffling when comparing benchmark runs.

## Validate

- Run `python -m nemotron.steps.byob.scripts.validate`.
- Run `python -m nemotron.steps.byob.scripts.run --list-families`.
- Confirm final generation outputs `benchmark_raw.parquet` and `benchmark.parquet`.
- Confirm translated outputs preserve row count unless `remove_low_quality` is intentionally enabled.

## Further Reading

- [references/guide.md](references/guide.md) for orchestration details
- [references/benchmark-schema.md](references/benchmark-schema.md) for MCQ schema rules
- [references/bfcl-oracle-pack.md](references/bfcl-oracle-pack.md) for the BFCL oracle-pack contract
- [references/new-family-checklist.md](references/new-family-checklist.md) for GSM8K-style or non-MCQ extensions
- [references/quality-and-filtering.md](references/quality-and-filtering.md) for quality gates
- [patterns/index.yaml](patterns/index.yaml) for BYOB-local routing hints
