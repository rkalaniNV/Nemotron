---
name: byob
description: Adapt BYOB MCQ benchmarks to the generic Nemotron translation skill by flattening question and option fields, restoring the original benchmark schema after translation, and optionally composing reverse translation plus text-quality metrics. Use when the user wants benchmark translation without changing MCQ structure.
when_to_use: Use for requests like "translate the MCQ benchmark", "translate only the BYOB benchmark", "keep the benchmark schema intact", or "attach round-trip metrics after benchmark translation". Do not use for general corpus translation.
compatibility: Requires the Nemotron Python package. Uses the translation skill for actual translation runtime and Curator TextQualityMetricStage for optional round-trip metrics.
metadata:
  owner: nemotron
  workflow-step: stage4
---

# BYOB

Use this skill for MCQ benchmark translation only. It is an adapter skill, not a standalone translation runtime.

## Default

1. Read [references/STEP.md](references/STEP.md) for the step contract.
2. Use `nemotron.steps.byob.adapter.flatten_mcq_records(...)` to stage `question` and `options`.
3. Run the generic `translation` skill over the staged rows.
4. Use `nemotron.steps.byob.adapter.restore_mcq_records(...)` to rebuild the benchmark schema.
5. If round-trip quality is requested, run a second translation pass with reversed languages and score with Curator `TextQualityMetricStage`.

Example composition:

```python
from nemotron.steps.byob import flatten_mcq_records, restore_mcq_records
from nemotron.steps.translation import translate_data

staged_rows, index = flatten_mcq_records(source_records)
# write staged_rows, run translation skill, then restore
restored = restore_mcq_records(source_records, index, translated_rows, target_lang="hi")
```

## Gotchas

- Never drop staged rows inline during translation. Reassembly needs a one-to-one mapping between source strings and translated rows.
- Keep FAITH row filtering disabled while rows are staged. Review attached `faith_*` values after restore instead.
- Preserve `answer` and stable row identifiers exactly.
- `options` may be either a dict or a list; restore the original shape.
- Use the generic translation skill for runtime behavior. Do not build a second translation pipeline under BYOB.

## Validate

- Confirm restored row count matches source row count.
- Confirm `question`, `options`, and `answer` exist on every restored row.
- Confirm translated rows still preserve the original `options` container shape.
- If metrics were requested, confirm `score_*` columns were attached after the reverse pass.

## Load More Only If Needed

- [references/guide.md](references/guide.md) for orchestration guidance
- [references/benchmark-schema.md](references/benchmark-schema.md) for accepted input shapes
- [references/quality-and-filtering.md](references/quality-and-filtering.md) for FAITH and round-trip quality rules
