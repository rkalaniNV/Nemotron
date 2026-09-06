<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# How-To Guides

Task-focused guides for `nemotron steps run byob/bfcl` with the `bfcl` family.

Start with {doc}`../getting-started` if you have not produced a `benchmark.parquet` yet.

```{toctree}
:maxdepth: 1
:hidden:

Author a Pack <author-a-pack>
Assisted Authoring <assisted-authoring>
Onboard an MCP Server <mcp-server>
Publish a Release <publish-a-release>
Run an Evaluation <run-evaluation>
```

## Get a Pack

Everything downstream depends on the pack, so start by choosing how you will produce one. All three routes end at the same generation pipeline and the same Gold gate; see {doc}`../explanation/authoring-flows` for the trade-offs.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Author a pack by hand
:link: author-a-pack
:link-type: doc
Scaffold the pack files, write the tools, templates, and assertions, then validate without generating.
+++
{bdg-secondary}`scaffold_oracle_pack`
:::

:::{grid-item-card} {octicon}`copilot;1.5em;sd-mr-1` Draft a pack with model assistance
:link: assisted-authoring
:link-type: doc
Certify a Python package or reviewed HTTP service, then draft, review, approve, and freeze a candidate pack.
+++
{bdg-secondary}`bfcl_author`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Onboard an MCP server
:link: mcp-server
:link-type: doc
Discover a server, expose it through the oracle gateway, and take its evidence into the same authoring flow.
+++
{bdg-secondary}`experimental`
:::

::::

## Publish and Evaluate

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Publish a release
:link: publish-a-release
:link-type: doc
Choose a size target and the balancing mixes, generate at publication scale, then verify the manifest and exports.
+++
{bdg-secondary}`publication`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Run an evaluation
:link: run-evaluation
:link-type: doc
Point an evaluation config at a published benchmark and a candidate endpoint, then read the report.
+++
{bdg-secondary}`stage=eval`
:::

::::

## When Something Fails

The step declares its failure modes with recovery guidance, and {doc}`../reference/troubleshooting` indexes them by symptom. Two habits save the most time:

- Run `stage=prepare` first to validate a pack without paying for generation.
- When generation drops tasks, join the adjacent `stage_cache/` tables on `task_id` to find the stage that dropped them, as described in {doc}`../reference/output-files`.
