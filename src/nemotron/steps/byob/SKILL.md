---
name: byob
description: Generate and translate bring-your-own MCQ benchmarks from domain documents with a modular benchmark-family runtime. Use when a user asks to create an MCQ benchmark, translate a BYOB benchmark, or extend the flow to a new benchmark family such as GSM8K.
when_to_use: Use for requests like "create a custom benchmark from these documents", "run BYOB MCQ generation", "translate the generated benchmark", "add a GSM8K-style BYOB family", or "keep the benchmark schema intact". Do not use for ordinary training-corpus translation.
compatibility: Install the optional `byob` extra before running this step. Generation uses Data Designer and Curator semantic deduplication; translation uses Curator experimental translation.
metadata:
  owner: nemotron
  workflow-step: byob
---

# BYOB

Use this skill to create or translate benchmark artifacts while keeping benchmark-family logic easy for coding agents to change.

## Default

1. When this step skill is loaded, first confirm the caller has already routed
   through the parent `nemotron-customize` skill. Do not use this step skill as
   a replacement for the parent skill activation evidence.
2. Read [references/STEP.md](references/STEP.md) for the artifact contract,
   [references/benchmark-schema.md](references/benchmark-schema.md) for MCQ
   schema, and [references/quality-and-filtering.md](references/quality-and-filtering.md)
   for deduplication and translation quality gates.
3. Present a compact Orient checklist with defaults filled in before running or
   writing anything:
   - Source location and domain/category grouping.
   - Target MCQ count per domain/category.
   - Source language and target language if translation is requested.
   - Runtime/backend: `[local only]` unless the user explicitly approves hosted
     model calls.
   - Output artifact: `[benchmark.parquet]`.
4. If the user has already supplied enough information, do not stop at
   clarification. State the assumptions and deliver a plan; keep any remaining
   questions as a short "Confirm or change" checklist.
5. Install BYOB runtime dependencies with `uv sync --extra byob` or
   `pip install ".[byob]"` in the target environment.
6. Start from [config/default.yaml](config/default.yaml) for MCQ generation or
   [config/translate.yaml](config/translate.yaml) for translation.
7. Run `nemotron byob --family mcq --stage prepare --config CONFIG`.
8. Run `nemotron byob --family mcq --stage generate --config CONFIG`.
9. Translate an existing benchmark with `--stage translate` and a translation config.

## Catalog Routing

Use this BYOB-specific routing only after the parent `nemotron-customize` skill
has identified the request as a BYOB benchmark task.

1. Read the upstream step catalog before implementation:
   - Required catalog source: `src/nemotron/steps/STEPS.md`.
   - In eval workspaces, verify and read
     `/workspace/repo/src/nemotron/steps/STEPS.md` or
     `/workspace/src/nemotron/steps/STEPS.md`.
   - Do not use `skills/nemotron-customize/STEPS.md` as the primary routing
     source. If it exists, treat it only as a last-resort diagnostic after every
     upstream path fails.
   - Verification check before fallback:
     `test -f src/nemotron/steps/STEPS.md || test -f /workspace/repo/src/nemotron/steps/STEPS.md || test -f /workspace/src/nemotron/steps/STEPS.md`.
2. Read `skills/nemotron-customize/context/index.toml` or
   `context/index.toml`, then load the BYOB pack
   `byob-benchmark-curator-translation.txt` once for generation and translation
   planning.
3. If every upstream catalog path is absent, state that the upstream catalog
   could not be read, use the embedded BYOB catalog excerpt below, and keep the
   response at planning level until the catalog issue is resolved.

Embedded BYOB catalog excerpt for `STEPS.md` fallback:

| Step | Description | Consumes | Produces |
| --- | --- | --- | --- |
| `byob` | Generate and translate BYOB MCQ benchmark parquet artifacts from domain documents with an extensible benchmark-family runtime. | `benchmark_source_corpus`, optional `benchmark_parquet` | `mcq_benchmark_parquet`, optional `translated_mcq_benchmark_parquet` |

When using this fallback, the plan still must explicitly include `question`,
`options`, `answer_index`, `answer`, `category`, forward translation,
backtranslation, and round-trip `sacrebleu`, `chrf`, `ter`.

## Dependency Preflight

Before writing or running a BYOB implementation, check the environment once:

```bash
python - <<'PY'
import importlib.util

required = [
    "datasets",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
    "sentencepiece",
    "sacrebleu",
    "ray",
    "nemo_curator",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing BYOB dependencies: " + ", ".join(missing))
print("BYOB dependency preflight OK")
PY
```

If dependencies are missing, do not replace Curator semantic deduplication with
TF-IDF, rapidfuzz, difflib, or another ad hoc implementation unless the user
explicitly asks for a degraded local prototype. Instead, add the missing
packages to the project or eval environment and explain that production BYOB
requires Hugging Face libraries, SentenceTransformers, SentencePiece, Curator,
Ray, and SacreBLEU-compatible metric support.

## Plan Contract

For BYOB requests, the plan must be detailed enough to execute without another
round of repository discovery. Include:

- The benchmark family (`mcq`) and stages to run: `prepare`, `generate`, and
  `translate` only when requested.
- The exact final MCQ parquet schema fields: `question_id`, `question`,
  `options`, `answer_index`, `answer`, `cot_content`, `src`, and `category`.
- Domain/category handling: preserve the user's category grouping through
  generation, deduplication, translation, and export.
- Deduplication implementation: Curator semantic dedup with
  `RayDataExecutor`, `RayActorPoolExecutor`, and package-level
  `SemanticDeduplicationWorkflow`.
- Translation quality, when translating: forward translation, backtranslation,
  and round-trip `sacrebleu`, `chrf`, and `ter` checks. Do not substitute an
  ad hoc string-similarity metric for these quality gates.
- Validation: run `python -m nemotron.steps.byob.scripts.validate`, confirm
  `nemotron byob --list-families`, and check row count/schema of final parquet
  outputs.
- A dependency line naming `datasets`, `huggingface_hub`, `transformers`,
  `sentence_transformers`, `sentencepiece`, `sacrebleu`, `ray`, and
  `nemo_curator` for data/model loading, translation quality metrics, and
  Curator semantic deduplication.

If the user only asks for a plan, stop after this plan. Do not generate files,
run benchmarks, or call model endpoints until the user approves.

## Script Safety

Prefer the BYOB CLI and runtime modules over generated one-off Python. When a
small validation helper is needed, write a file and run it; avoid complex
multi-line `python -c` strings because shell escaping easily corrupts f-strings
and backslashes.

Use this copy-pasteable validation helper shape:

```python
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "question_id",
    "question",
    "options",
    "answer_index",
    "answer",
    "cot_content",
    "src",
    "category",
}


def validate_benchmark(path: str) -> None:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing benchmark parquet: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required MCQ columns: {missing}")
    if frame.empty:
        raise ValueError("Benchmark parquet is empty")

    print(f"Validated {len(frame)} rows at {parquet_path}")


if __name__ == "__main__":
    validate_benchmark("benchmark.parquet")
```

## Change Points

- Add new benchmark families under `runtime/benchmark_families/<family>/`.
- Before adding a new family, answer the questions in [references/new-family-checklist.md](references/new-family-checklist.md).
- Register the family in `runtime/benchmark_families/registry.py`.
- Keep `scripts/runtime.py` as a dispatcher only; family-specific schema, prompts, postprocessing, and export code belong in family modules.
- Keep MCQ stage orchestration in `runtime/benchmark_families/mcq/pipeline.py`; do not recreate a top-level `runtime/pipeline.py`.
- Use `adapter.py` only for schema bridging when composing BYOB with other skills.
- Use Curator experimental translation as the translation backend; BYOB should only flatten/reassemble benchmark-family schema around it.
- Use Curator semantic dedup with `RayDataExecutor`, `RayActorPoolExecutor`, and package-level `SemanticDeduplicationWorkflow`.

## Gotchas

- Do not merge the whole runtime into `scripts/runtime.py`; that blocks future GSM8K-style extensions.
- Do not put MCQ-specific orchestration in top-level `runtime/`; family pipelines belong under `runtime/benchmark_families/<family>/`.
- Keep `question_id`, `question`, `options`, `answer_index`, `answer`, `cot_content`, `src`, and `category` stable in final MCQ parquet outputs.
- Do not drop staged rows inline during translation reassembly. Filtering belongs after rows are restored.
- Do not add a translation mode selector; BYOB translation always uses Curator experimental translation.
- Keep semantic dedup as a two-step flow: compute embeddings first, then run KMeans, pairwise similarity, and duplicate identification.
- Resume with `--skip-until` only when the expected cached parquet for the previous stage already exists.
- Use deterministic seeds for sampling and distractor shuffling when comparing benchmark runs.
- Do not run destructive cleanup such as `rm -rf`. Use a fresh output
  directory, intentionally overwrite generated files, or ask before deleting.
- Do not print environment variables, API keys, full source documents, or
  generated regulated-data rows. Refer to file paths, schemas, and aggregate
  counts instead.
- Treat private legal, banking, insurance, medical, government, or other
  regulated corpora as local-only by default. Do not upload source documents to
  hosted services unless the user explicitly approves that backend.
- Keep tool use narrow: read this file and the BYOB references first, then only
  inspect runtime/config files needed for the selected stages. Avoid broad file
  maps, package listings, or unrelated benchmark docs unless a concrete error
  requires them.

## Validate

- Run `python -m nemotron.steps.byob.scripts.validate`.
- Run `nemotron byob --list-families`.
- Confirm final generation outputs `benchmark_raw.parquet` and `benchmark.parquet`.
- Confirm translated outputs preserve row count unless `remove_low_quality` is intentionally enabled.

## Load More Only If Needed

- [references/guide.md](references/guide.md) for orchestration details
- [references/benchmark-schema.md](references/benchmark-schema.md) for MCQ schema rules
- [references/new-family-checklist.md](references/new-family-checklist.md) for GSM8K-style or non-MCQ extensions
- [references/quality-and-filtering.md](references/quality-and-filtering.md) for quality gates
- [patterns/index.yaml](patterns/index.yaml) for skill-local routing hints
