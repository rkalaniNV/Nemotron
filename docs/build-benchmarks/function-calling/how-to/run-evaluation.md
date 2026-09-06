<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluate a Candidate Model

Use this guide to score one or more candidate models against a benchmark that has already been published. Evaluation is a separate run with its own configuration, so that changing a candidate can never change the identity of the benchmark it is scored on.

## Before You Start

- A published generation run containing `run_manifest.json`, `benchmark.parquet`, and `benchmark_raw.parquet`. Copying only the Parquet data is not enough: the manifest is the publication commit marker and names the table it published.
- An OpenAI-compatible candidate endpoint that supports tool calling, its served model identifier, and the name of an environment variable holding its credential. A literal key anywhere in the configuration is refused.
- For executable mode, the exact Oracle Pack the benchmark was generated from.
- An output directory outside the generation publication tree.

## Step 1: Resolve Your Own Evaluation Config

Copy the template and fill it in. Every `REPLACE_ME_*` value must be resolved and `config_status` must become `resolved`.

```bash
mkdir -p /srv/bfcl/eval/candidate-a && \
  cp src/nemotron/steps/byob/bfcl/config/eval.default.yaml \
    /srv/bfcl/eval/candidate-a/eval.yaml
```

:::{important}
Nothing in this file falls back to a default. Every scoring gate, runtime limit, and decoding parameter is stated, because each one changes what the resulting number means: a model cut off at two turns did not answer the same question as one given ten. Quoted booleans and numbers are refused rather than coerced, since a `"false"` that became `true` would silently switch off a correctness gate.
:::

```yaml
schema_version: "1.2"
config_status: resolved
source_run_manifest: /srv/bfcl/runs/warehouse-gold/run_manifest.json
```

Relative paths resolve from the evaluation config's own directory. Resolution ends in one `eval_config_hash` taken over the configuration's meaning: referenced files enter as content hashes, candidates are ordered by alias, and absolute paths, output locations, and secret values are excluded. Moving the checkout leaves the hash alone; changing a candidate, a revision, an inference parameter, a limit, the scoring contract, or the source run changes it.

## Step 2: Choose Trace or Trace Plus Executable

```yaml
eval:
  mode: [trace]
```

`[trace]` measures whether the candidate proposed the expected calls, with the expected arguments, grouping, and ordering, releasing the tool results the benchmark recorded; it needs the published benchmark and nothing else. `[trace, executable]` measures all of that and additionally that the candidate's calls ran against a live oracle and satisfied the pack's assertions; it needs `source_oracle` naming the exact pack manifest and its concrete `backend.py` or endpoint config.

For executable mode, add the resource explicitly, because the `oracle.kind` field in `run_manifest.json` is lineage only and does not locate a backend for a later process.

```yaml
source_oracle:
  kind: python
  pack_manifest: /srv/bfcl/packs/warehouse_assets/manifest.yaml
  resource: /srv/bfcl/packs/warehouse_assets/backend.py
```

The two modes are not redundant. A candidate can emit well-formed calls on every attempt and still fail a large share of tasks once the backend and the pack's assertions are in the loop, because a well-formed call is not necessarily the right call against the state the task established. Executable publication additionally requires a Gold-eligible source run, since only Gold rows were validated against a real oracle.

## Step 3: Point at the Candidate

A candidate separates two identities: `provider`, `model`, and `api.base_url` name the route a request takes, while `model_identity` names the weights that answered.

```yaml
candidates:
  - alias: candidate_a
    model: <SERVED_MODEL_ID>
    provider: openai_compatible
    provider_api_version: v1
    api:
      base_url: https://candidate.example.com/v1
      api_key_env: BFCL_CANDIDATE_API_KEY
    model_identity:
      source: huggingface
      model: <ORG>/<MODEL>
      revision: <40_TO_64_HEX_COMMIT>
      weights_digest: null
```

Do not fill the identity block by hand. The resolver writes it for you and records what it actually observed:

```bash
python -m nemotron.steps.byob.scripts.resolve_bfcl_model_identity registry \
  --model <ORG>/<MODEL> \
  --revision main
```

`registry` resolves a reference to the commit it currently names, `local --model <NAME> --weights-dir <DIR>` digests weights on disk, and `provider-managed --source <PROVIDER> --model <MODEL>` records a hosted route that publishes neither. Each subcommand accepts `--output` to write the YAML fragment to a file as well.

A branch-style reference such as `main` is refused, because the same configuration would score different weights next month. Leaving both `revision` and `weights_digest` null is allowed and means what it says: the identity resolves as provider-managed, the run is still scored, and it may not publish.

## Step 4: Understand the Gates That Run First

Two gates run before any candidate token is spent, and both fail closed.

**Source verification** re-reads the publication from disk and holds it to what the configuration recorded. It proves that `run_manifest.json` is a commit marker whose bytes still hash to what the configuration resolved, that both tables hash to what the publication declares in all three places that declare them, that the published table selects raw rows without rewriting truth and ships no held-out row, and that every published row decodes into a unique addressable task. For executable mode it also recomputes the whole pack fingerprint, because a helper module the backend imports changes what the oracle does. It writes `source_verification_report.json` on success and `source_verification_failure.json` on refusal, using different names so no reader can mistake a diagnosis for a pass.

**The contamination gate** then decides who may answer which rows. Source verification records every model that read a published row while it was being built, together with the rows it read. Each candidate is compared against that inventory on the strongest available evidence, and the comparison returns one of three verdicts: `different` records nothing, `match` is a violation, and `unknown` means neither side pinned enough to tell. Under the locked publication setting `on_violation: fail_run`, a match refuses the run. Unknown evidence never shrinks a task set on suspicion and never aborts a debug run, but it always blocks publication. The gate writes `contamination_report.json` or `contamination_failure.json`.

Both are rechecked immediately before the first request, because verification and use are separated in time and that gap is exactly where a source or a plan gets replaced.

## Step 5: Select the CLI Envelope

The evaluation configuration carries scoring semantics; a separate envelope carries operational choices, so the envelope cannot change `eval_config_hash`.

Copy `eval.cli.yaml`, whose `execution_backend: direct` runs the evaluation in this process on this host, beside your resolved configuration and point it at the file. The alternative, `eval.launcher.yaml`, sets `execution_backend: nemo_launcher` and submits the published `nemo_evaluator_bundle` as a native NeMo Evaluator task for exactly one candidate.

```yaml
schema_version: "1.0"
family: bfcl
stage: eval
eval_config_path: ./eval.yaml
execution_backend: direct
output_format: human
probe_oracle: true
dry_run: true
```

`output_format: json` emits stable machine-readable run and artifact locations; `human` renders the same payload one key per line. `dry_run: true` verifies the source and the contamination decision and reports authorized task counts without any candidate inference.

The Launcher envelope is a two-phase materialize-then-submit flow. The first invocation with `launcher.submit: false` verifies the bundle and materializes the adapter configuration, the immutable framework package, the task entry, and the merged Launcher configuration. Install the printed framework package explicitly in the Launcher environment, then rerun the same configuration with `submit: true`. Every orchestration path must lie outside `bundle_root`, because the bundle is verified by exact file set and one extra file would fail its next verification. If the merged configuration names an evaluation container anywhere, `launcher.evaluation_mounts` becomes mandatory and must use identity mappings covering the adapter and evaluation configurations, the verified source, the executable oracle resources, and the output trees.

## Step 6: Run the Preflight, Then the Evaluation

```bash
nemotron steps run byob/bfcl -c /srv/bfcl/eval/candidate-a/eval.cli.yaml
```

With `dry_run: true` this performs no candidate inference and commits no artifacts. Resolve any contamination finding here rather than weakening the gate, then set `dry_run: false` and run the same command again.

:::{tip}
Start with `limits.max_parallel_tasks: 1` and raise it only after confirming the endpoint's concurrency and rate limits. Any such change produces a different `eval_config_hash`, which is correct: it is a different measurement.
:::

## Step 7: Read the Artifacts

A completed run writes an immutable set under `outputs.output_dir`:

```text
artifacts/
├── resolved_eval_config.json
├── source_verification_report.json
├── contamination_report.json
├── candidate_io_cache.jsonl
├── tool_trace_cache.jsonl        # executable mode only
├── eval_report.json
├── eval_task_results.parquet
└── eval_manifest.json
```

`eval_manifest.json` is written last and binds the source, plan, and candidate aggregate identities to the byte hashes of the report, the task table, and the required caches. Read trace and executable numbers separately: a trace aggregate reports one pass rate per gate over tasks, while an executable aggregate reports per-call accuracies, so `arguments_pass_rate` and `argument_accuracy` are deliberately different names for incomparable measurements. Every aggregate declares the scope it measured, and a report whose aggregates mix scopes is refused. A metric no task applied is reported as not applicable with a stable reason code rather than as a vacuous zero, and a failed gate carries a failure class, so an unreachable endpoint reads as infrastructure rather than as weak tool use.

## What Makes a Run Publication-Eligible

With `publication.requested: true` the loader requires every locked gate: `scoring.argument_matching: schema_then_canonical` with call order and grouping respected and `allow_llm_repair: false`; `scoring.task_success: all_applicable_gates`; `contamination.enforce: true` with `on_violation: fail_run` and `comparison_set: common_intersection`; every artifact under `outputs` enabled; pinned weights on every candidate, since a provider-managed identity names a route the provider may re-point; and, for executable mode, a Gold-eligible source run.

Relaxing any of those is allowed only with `publication.requested: false`, and the configuration then reports each weakened field in `non_publication_reasons`.

## Common Failures

| Symptom | What it means |
| --- | --- |
| Refused before any request | Validation reports every independent violation in one pass. Fix the fields named, starting with the first in file order. |
| The endpoint answered 401 or 403 | Every task presents the same credential, so this is a configuration fault and the run stops on the first refusal. No completion is cached, so a rerun with a working key re-contacts the endpoint. |
| The source hash moved | Something wrote into the publication tree during the evaluation, usually a regeneration into the same directory. Verify again from a clean tree. |
| A finished artifact set exists | Immutable results are never overwritten. Use a new output directory. |

Every failure leaves through one published exit status: `2` for a configuration you must edit, `3` for a setup, source, scoring, or aggregation refusal, `4` for contamination or answer-key exposure, `5` for a candidate-endpoint failure, `6` for live oracle or assertion infrastructure, and `7` for an immutable artifact that already holds different evidence.

## Next Steps

- Field-by-field details: {doc}`../reference/eval-config`, and symptom-to-fix entries in {doc}`../reference/troubleshooting`.
- What each gate and metric means: {doc}`../explanation/evaluation`, with the normative contract at `src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md`.
