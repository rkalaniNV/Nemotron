---
id: version-sdg-pipeline
title: "Version synthetic data generation as a pipeline"
tags: [sdg, data-quality, reproducibility]
triggers:
  - "Synthetic SFT, tool-use, prompt, or preference data is being generated."
  - "A Data Designer config is moving from preview mode to a larger generation job."
  - "Generated data will feed SFT, DPO, RLVR, RLHF, or downstream data prep."
steps: [sdg/data_designer, prep/sft_packing, prep/rl_prep, sft/automodel, rl/nemo_rl/dpo]
confidence: high
---

## When to apply

Use this whenever synthetic data becomes part of the training corpus. Synthetic data generation is a data pipeline with inputs, prompts, models, projections, and quality checks, not a one-off prompt.

Apply it before scaling from preview output to a larger generated dataset. The quality mistakes that appear in ten preview records usually become expensive at ten thousand records.

It is especially important for preference generation, tool-use data, and domain-specific chat data where schema correctness matters as much as text fluency.

## What to do

Version seed files, prompts, model aliases, inference parameters, validators, projection rules, and generated outputs together.

Start with preview mode or `config/tiny.yaml`. Inspect records before increasing `num_records`.

Keep output schemas explicit. SFT data should project to chat `messages` when the trainer expects messages. DPO data should project to prompt, chosen, and rejected fields.

Use high-quality seed data. Seeds should be representative, licensed for the intended use, free of private data unless approved, and balanced across target domains.

Control diversity deliberately with sampler columns, seed columns, personas, task categories, difficulty fields, and constraints. Do not rely on temperature alone.

Validate generated data for schema, safety, duplication, language, length, refusals, malformed tool calls, and train/test contamination.

Keep a held-out human or task eval set from non-synthetic sources when possible. Synthetic data can broaden coverage while still missing real distribution details.

## Exceptions

For quick API or credential testing, a tiny unversioned preview can be acceptable. Do not promote those records into training data without reconstructing the config and lineage.

If generated records are only illustrative examples for documentation, full training-grade validation is unnecessary.

## References

- Pair with `data-quality-before-quantity` before scaling synthetic data volume.
- Pair with `prepared-data-is-tokenizer-locked` before packing generated chat data for Megatron-Bridge.
- Pair with `validate-rl-rewards-before-scale` when generated preferences feed DPO or RLHF.
