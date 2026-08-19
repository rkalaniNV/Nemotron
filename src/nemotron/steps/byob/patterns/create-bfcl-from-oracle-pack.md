---
id: create-bfcl-from-oracle-pack
title: Create a function-calling benchmark from an oracle pack
tags: [byob, benchmark, function-calling, bfcl]
triggers:
  - Generate a function-calling benchmark from executable tools and fixtures.
  - Validate an oracle pack and produce replay-verified tool-call conversations.
steps: [byob]
confidence: high
---

Use the BFCL family and follow `references/bfcl-oracle-pack.md`. Start from
`bfcl/config/default.yaml`, point `oracle_pack.manifest_path` at the pack, and keep
all domain data, tool names, fixtures, and assertions inside that pack.

Run `stage=prepare` first to inspect `oracle_validation_report.json`. Generation
requires every validation check to pass with `oracle_runtime.worker: process`.
Then run `stage=generate`, or use `stage=all`, to expand templates, render the
conversation, validate the expected trace, replay it twice, and write the parquet
artifacts plus `run_manifest.json`.

Set `exports.bfcl_json` and/or `exports.nemo_evaluator_bundle` to emit optional
compatibility trees from the published parquet. Stage 12 reads them back for
equivalence and writes `exports/export_validation_report.json` before committing
`run_manifest.json`; the NeMo bundle remains input for the W5 native-tool adapter.
