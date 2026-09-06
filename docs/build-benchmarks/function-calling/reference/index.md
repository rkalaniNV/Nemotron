<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Reference

Field-level and artifact-level detail for `nemotron steps run byob/bfcl` with the `bfcl` family.

```{toctree}
:maxdepth: 1
:hidden:

Generation Config <generate-config>
Evaluation Config <eval-config>
Output Files <output-files>
Troubleshooting <troubleshooting>
```

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Generation config
:link: generate-config
:link-type: doc
Every accepted key in a generation YAML, block by block, with types and defaults.
+++
{bdg-secondary}`YAML`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Evaluation config
:link: eval-config
:link-type: doc
Scoring contract, modes, candidate declarations, and the difference between the three bundled envelopes.
+++
{bdg-secondary}`YAML`
:::

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Output files
:link: output-files
:link-type: doc
Every path written under `output_dir` / `expt_name`, and which artifacts are content-addressed.
+++
{bdg-secondary}`artifacts`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc
Symptom-to-fix index derived from the step's declared error taxonomy.
+++
{bdg-secondary}`errors`
:::

::::

## Bundled Configurations

The step ships runnable configurations under `src/nemotron/steps/byob/bfcl/config/`. Copy the one closest to your intent rather than starting from an empty file.

| File | Purpose |
| --- | --- |
| `tiny.yaml` | Plumbing smoke run against the bundled tiny pack. Not publication-eligible. |
| `default.yaml` | Annotated template. Its pack path is a placeholder, so it cannot publish an example domain by omission. |
| `smoke.example.yaml` | Domain-sized smoke run. Copy it and repoint it at your own pack. |
| `publication.example.yaml` | Publication-scale, template-only Gold profile with a worked budget and balancing targets. |
| `publication.paraphrase.example.yaml` | The same executable cases with an opt-in model-authored surface role. |
| `eval.default.yaml` | Annotated evaluation template to resolve into your own config. |
| `eval.cli.yaml` | Direct evaluation envelope. |
| `eval.launcher.yaml` | Launcher evaluation envelope. |
| `translate.yaml` | Localization of an already published benchmark. |

## Command-line Conventions

The helper commands under `nemotron.steps.byob.scripts` — the pack validator, the bias auditor, the release archiver, the authoring and MCP release commands — share one exit contract, so a wrapper can branch on the status alone:

| Status | Meaning | Output |
| --- | --- | --- |
| `0` | The command ran and the answer was yes: the pack is Gold-eligible, the audit passed, the artifact was written. | The result document on stdout. |
| `1` | The command could not reach an answer. A path was missing, a file would not parse, an invariant was violated. | A JSON failure envelope on stderr with `status`, `error_type`, and `reason`. |
| `2` | The command ran and the answer was no. The pack is not Gold-eligible, the audit found an unexcepted failure, the review packet is blocked. | The full verdict document on stdout, so you can see which check said no. |

The distinction between `1` and `2` is what makes these commands safe to automate: retry on `1`, because a crash may be transient; never retry on `2`, because the verdict will not change until a human changes the inputs.

The evaluator is the exception. `nemotron steps run byob/bfcl` with `stage=eval` publishes a wider taxonomy — `2` through `7` — because an operator needs to know whether to edit a config, fix a candidate endpoint, or investigate a contamination finding. See {doc}`../how-to/run-evaluation`.

## Normative Contracts

These pages describe the operator-facing surface. The normative contracts live in the source tree, beside the code that enforces them:

| Contract | File |
| --- | --- |
| Oracle Pack layout, tiers, and Gold rules | `src/nemotron/steps/byob/references/bfcl-oracle-pack.md` |
| Evaluation scoring | `src/nemotron/steps/byob/references/bfcl-eval-scoring-contract.md` |
| Bias audit dimensions | `src/nemotron/steps/byob/references/bfcl-bias-audit-contract.md` |
| MCP oracle profile | `src/nemotron/steps/byob/references/bfcl-mcp-oracle-contract.md` |
| MCP trust boundaries | `src/nemotron/steps/byob/references/bfcl-mcp-threat-model.md` |
| Supported, experimental, and refused capabilities | `src/nemotron/steps/byob/references/bfcl-authoring-support-matrix.md` |

The step's own declared inputs, outputs, and error taxonomy are in `src/nemotron/steps/byob/bfcl/step.toml`.
