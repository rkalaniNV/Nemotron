---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run nemotron steps run curate/nemo_curator on the packaged tiny JSONL file."
topics: ["Curation", "Tutorial", "NeMo Curator"]
tags: ["Tutorial", "Curation"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(getting-started-curate)=
# Getting Started With Curation

This tutorial runs `curate/nemo_curator` on the packaged tiny JSONL fixture.
The run verifies that the Nemotron CLI can load the step, NeMo Curator can read and write JSONL, and Ray can start for the local curation pipeline.

## Prerequisites

- The `uv` tool is available in your shell.
- You are running commands from the Nemotron repository root.
- The curate dependencies are installed:

  ```console
  $ uv sync --extra curate
  ```

For local `uv run` execution with Curator/Ray, export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` so Ray workers reuse the synchronized project environment.

## Inspect the Tiny Input

The packaged fixture lives at `src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl`.
It contains JSONL records with a `text` field.

```{literalinclude} ../../src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl
:language: json
:class: scrollable
```

## Run a Local Tiny Curation Job

The checked-in `tiny.yaml` file is tuned for the Lepton Curator container and uses a container path under `/nemo_run/code`.
For a local run, keep the tiny configuration but override `input_glob` and `output_dir` with host paths.

```console
$ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl" \
    output_dir="${PWD}/output/curate-tiny"
```

The tiny configuration disables optional language, word-count, and domain filters.
That makes the first run a reader, writer, and Ray smoke test.

## Inspect the Output

List the output directory:

```console
$ find output/curate-tiny -type f
```

Open the JSONL shard and confirm that records still contain the configured `text_field`.
The exact shard name is assigned by Curator.

## Run the Same Smoke on Lepton

When you have generated an environment profile that includes `lepton_curate`, the packaged tiny config can run without local path overrides:

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny --batch lepton_curate
```

The `lepton_curate` profile uses the NeMo Curator container and sets CPU resources for a small smoke job.

## Next Steps

- Use {doc}`how-to/run-local-jsonl` to point the step at your own files.
- Use {doc}`how-to/use-huggingface-snapshot` to hydrate a Hugging Face dataset before reading.
- Use {doc}`how-to/enable-filters` to add language, word-count, or domain filters.
