# Skill-Local Step Catalog

This file is a compact, skill-local fallback step catalog for isolated eval
workspaces where `src/nemotron/steps/STEPS.md` may be unavailable.

## BYOB

- **Step id:** `byob`
- **Use when:** The user asks to build or translate an MCQ benchmark from
  domain/private documents.
- **Primary context pack:** `context/byob-benchmark-curator-translation.txt`
- **Output artifact:** `benchmark.parquet` (or translated benchmark parquet when
  translation is requested)

## BYOB Required Plan Elements

When planning with `byob`, explicitly include:

1. **MCQ schema fields**
   - `question`
   - `options`
   - `answer_index`
   - `answer`
   - `category`
   - and final parquet columns: `question_id`, `cot_content`, `src`
2. **Deduplication implementation**
   - Curator semantic dedup with `RayDataExecutor`,
     `RayActorPoolExecutor`, and `SemanticDeduplicationWorkflow`
3. **Translation quality flow** (if translating)
   - forward translation
   - backtranslation
   - round-trip metrics: `sacrebleu`, `chrf`, `ter`
4. **Privacy default**
   - local-only processing unless user explicitly approves hosted execution

## Minimal BYOB Workflow

1. Read `context/index.toml` to resolve BYOB context packs.
2. Read `context/byob-benchmark-curator-translation.txt`.
3. Produce Orient checklist with defaults and missing parameters.
4. Produce stage plan with schema, dedup, translation metrics, and output path.
5. Validate final parquet schema and row integrity before completion.
