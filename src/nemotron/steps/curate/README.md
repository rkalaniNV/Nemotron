# Nemotron Curation

Use this category to turn raw or third-party JSONL into a filtered corpus that
can feed translation, pretraining prep, or SFT prep.

## Developer Journey

1. Identify the raw source: local JSONL or a Hugging Face snapshot.
2. Run the curate step with all optional filters disabled to verify the
   reader/writer path.
3. Add language, word-count, or domain filters one at a time.
4. Inspect intermediate shards after each filter change — empty output usually
   means a filter is too aggressive.
5. Hand the filtered JSONL to translation or data prep.

## Steps

| Need | Step | Input | Output |
|---|---|---|---|
| Run the whole category from one config | [`curate/flow`](flow/README.md) | `raw_jsonl` | everything below, plus `flow_report` |
| Lightweight JSONL filtering with optional language/word-count/domain gates | [`curate/nemo_curator`](nemo_curator/README.md) | `raw_jsonl` (or HF snapshot) | `filtered_jsonl` |
| Evidence that a curation run did not silently lose records | [`curate/audit`](audit/README.md) | `filtered_jsonl` | `curation_report` |
| What candidate filtering thresholds would do to this corpus | [`curate/profile`](profile/README.md) | `filtered_jsonl` | `profile_report`, `filter_policy` |
| Fixed-size corpora for a filtering ablation, guaranteed to nest | [`curate/subset`](subset/README.md) | `filtered_jsonl` | `filtered_jsonl`, `subset_plan`, `subset_report` |
| Training documents that near-duplicate a held-out split | [`curate/decontamination`](decontamination/README.md) | `filtered_jsonl` | `filtered_jsonl`, `decontamination_report` |

## Data And Artifact Flow

```text
raw_jsonl / HF snapshot
  -> curate/nemo_curator (JsonlReader -> optional filters -> JsonlWriter)
  -> filtered_jsonl
  -> translate/* or data_prep/*

filtered_jsonl (+ the run manifest curate/nemo_curator emits)
  -> curate/audit (readability, row counts, content digest, manifest comparison)
  -> curation_report

filtered_jsonl (typically produced with filters disabled)
  -> curate/profile (signal distributions, retention curves, gate co-occurrence)
  -> profile_report + candidate_policies.yaml

filtered_jsonl (one corpus per policy being compared)
  -> curate/subset (one plan, all budgets, nested by construction)
  -> plan.json + one corpus per budget + subset_report

filtered_jsonl + a held-out split
  -> curate/decontamination (source identity, then MinHash/LSH, then exact Jaccard)
  -> reduced training split + decontamination_report
```

`curate/decontamination` is the only step in this category that requires a GPU,
and the only one whose scope limit is worth stating twice: it detects
whole-document near-duplicates, not a benchmark question embedded inside a long
document. Its reports say "near-duplicate overlap detected and removed", never
"holdout verified clean".

`curate/profile` measures; it does not decide. Its candidate policies carry
`approved: false` and are not executable — a retention curve says what a
threshold removes, not whether what it removes is worth removing. Promoting a
candidate is a separate, recorded act.

Audit runs after a curation step, on the files it produced. It reads from disk
and starts no Ray cluster. Set `emit_manifest` on `curate/nemo_curator` to give
it something to check against — without a manifest it can report counts but
cannot claim the corpus is complete.

`curate/subset` exists because comparing two policies on whatever each one
retained measures filter quality and dataset size at once. It fixes the budget
instead, and guarantees that a smaller tier is contained in every larger one —
which means it cannot also promise to fill the budget. The unmet remainder is
reported as `token_shortfall` rather than made up from another stratum.

## One Config, Or Five

[`curate/flow`](flow/README.md) runs the steps below from a single config with
per-step `enabled` flags. It exists for one reason beyond convenience: two
agreements between these steps fail *silently* when made by hand.
`curate/nemo_curator` ships `emit_manifest: null` and `curate/audit` ships
`declared_manifest: null` — an audit against a producer that emitted no manifest
claims nothing, which reads exactly like a clean result. The flow derives both
from one path, so they cannot disagree.

It does not replace the steps. Each remains independently runnable, and any key
written inside a `steps.<name>` block overrides the flow's derivation.

This category is intentionally lightweight. Deduplication, crawling, and full
web extraction belong in dedicated NeMo Curator recipes, not this step.

## Guardrails

- Don't enable every filter on the first run.
- Inspect intermediate JSONL before tightening filters.
- Split very large input files before reading; OOMs usually come from
  oversized partitions.
