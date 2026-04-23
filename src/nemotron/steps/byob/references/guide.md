# BYOB Translation Guide

## When to choose this step

Choose this step when the user wants translated benchmark questions, not translated training corpora.

## Why this is separate from corpus translation

The BYOB path only keeps schema-specific logic:
- only `question` and `options` are translated
- `answer` and benchmark identifiers are preserved
- the benchmark schema must be restored exactly after translation

The actual translation runtime should still be the generic corpus translation step.

## Recommended flow

1. Load benchmark rows.
2. Use `nemotron.steps.byob.adapter.flatten_mcq_records(...)` to stage `question` and `options` as plain text rows.
3. Run the generic translation step with `text_field=text`.
4. Use `nemotron.steps.byob.adapter.restore_mcq_records(...)` to rebuild the benchmark schema.
5. If round-trip quality is requested, run the generic translation step again with reversed languages and score the result with Curator `TextQualityMetricStage`.

## Quality guidance

Keep FAITH row filtering disabled while staged rows are being translated. Use attached `faith_*` fields for review, and use round-trip metrics when the translated benchmark will be used as a hard evaluation artifact.
