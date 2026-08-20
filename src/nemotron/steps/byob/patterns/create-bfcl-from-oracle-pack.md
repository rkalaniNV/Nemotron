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

To evaluate candidate models on the result, write a separate `eval_config.yaml`
from `bfcl/config/eval.default.yaml`. It points at the run's `run_manifest.json`
rather than the parquet, pins the scoring contract in
`references/bfcl-eval-scoring-contract.md` by content hash, and requires an
immutable commit or weights digest per candidate. Executable mode also pins the
matching pack manifest and concrete backend.py or endpoint config in
`source_oracle`; the lineage label in `run_manifest.json` cannot locate a resource
by itself. Keeping the eval config separate is what stops a new candidate from
changing the benchmark's own lineage hash; `stage=generate` refuses
`eval_config_path` and inline `eval` blocks while the eval runner is unwired.

A valid config only records what was named, so `verify_eval_source()` reads the
source back before any candidate is contacted: the manifest still hashes to what
was resolved, both tables match every hash the publication declares, the
published table is an unmodified selection of the raw table with no held-out row,
the rows decode into a unique task set, and an executable run's oracle pack still
fingerprints to what generation certified. It writes
`source_verification_report.json` into the eval output directory, and
`assert_source_unchanged()` re-pins everything immediately before execution so a
run cannot span two sources.
