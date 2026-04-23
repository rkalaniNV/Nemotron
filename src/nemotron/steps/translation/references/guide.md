# Translation Guide

## When to choose this step

Choose this step when the translated output is meant to feed data curation, CPT, or SFT.

## Backend choice

| Backend | Best for | Tradeoff |
|---------|----------|----------|
| `llm` | chat data, nested fields, tool-calling transcripts | highest cost, best formatting control |
| `nmt` | very large plain-text corpora | needs a local translation server |
| `google` | managed cloud translation | external credentials and billing |
| `aws` | managed cloud translation | external credentials and billing |

## Data shape guidance

- Plain text: keep `text_field=text`
- Nested chat: set `text_field=messages.*.content`
- Reconstructed chat output: also set `reconstruct_messages=true`
- Multiple fields: use a list of field paths
- Existing translated rows: set `skip_translated=true` and point `translation_column` at the stored translated field

## FAITH guidance

- Set `faith_eval.enabled=true` when you need quality scoring or threshold-based filtering.
- Set `faith_eval.model_name` when FAITH should run on a different judge model from translation.
- Set `faith_eval.segment_level=true` when long documents would make whole-document FAITH requests too large or too expensive.
- In the current Curator flow, segment-level FAITH scores each exploded segment before reassembly, then aggregates document-level `faith_*` scores after reassembly.
- Filtering still happens after reassembly, so low-quality documents are dropped as documents rather than as individual segments.

## Chunking guidance

The checked-in step uses Curator's reader partitioning knobs:
- `files_per_partition` when the dataset is naturally split into many files
- `blocksize` when Curator's file partitioning is sufficient

If the user has one very large JSONL or Parquet file and needs row-level chunking, the agent should generate a temporary preprocessing or chunking helper for that dataset instead of baking custom pandas loops into this step.

## Single-file fallback

Use a one-off chunking helper when all of the following are true:
- the dataset is mostly a single huge file
- Curator file partitioning is not enough to control memory
- the user still wants the generic translation step, not a custom checked-in runtime

Recommended agent flow:
1. Split the source file into temporary JSONL or Parquet shards while preserving row order and schema.
2. Run the generic translation step over the shard directory.
3. If the user needs one final artifact, merge the translated shards after translation completes.
4. Keep the helper local to the run unless the same pattern becomes common enough to deserve a reusable utility.

For benchmark or schema-sensitive data, preserve stable record identifiers before splitting so reassembly and verification stay deterministic.
