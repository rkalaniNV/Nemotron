---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Explains how training steps declare typed inputs and outputs, what the common training and alignment paths look like, and why tokenizer alignment matters across a pipeline."
topics: ["Training", "Explanation", "Concepts"]
tags: ["Artifact Graph", "Pipeline", "JSONL", "Checkpoint", "Tokenizer"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Artifact Graph

Every training step declares the type of input artifact it consumes and the type of output artifact it produces.
The full set of declarations forms an *artifact graph* that the pipeline uses to check whether two steps can connect directly or whether a conversion step needs to sit between them.

## What the Graph Contains

The artifact graph names every kind of artifact a Nemotron training step can take in or hand off, such as `training_jsonl` for a chat-formatted dataset or `checkpoint_megatron` for a sharded Megatron checkpoint.
Each artifact type has a short description.
Some types also declare connections to related types: a checkpoint type that one library writes can carry an explicit conversion edge to the checkpoint type that a different library expects.

You can inspect what a specific step consumes and produces with the `show` subcommand.

```console
$ nemotron steps show sft/automodel
```

The relevant section of the output names the consumed and produced artifact types.

```text
Consumes
  • training_jsonl — Instruction data in JSONL with a messages field

Produces
  • checkpoint_hf — Hugging Face checkpoint directory (full model or adapter-style PEFT output)
```

When the type a step expects to consume does not match the type the previous step produced, you must insert an explicit conversion step between the two.

The packaged checkpoint conversion steps cover the common format bridges:

- `convert/hf_to_megatron` converts `checkpoint_hf` to `checkpoint_megatron`.
- `convert/megatron_to_hf` converts `checkpoint_megatron` to `checkpoint_hf`.
- `convert/merge_lora` merges `checkpoint_lora` with its original base checkpoint and produces a standalone checkpoint.

## Common Training Paths

The supervised fine-tuning paths in the Nemotron pipeline follow one of the following two chains.

- The Hugging Face line used by `sft/automodel`: `training_jsonl` → `sft/automodel` → `checkpoint_hf`.
- The packed Megatron line used by `sft/megatron_bridge`: `training_jsonl` → packing prep → `packed_parquet` → `sft/megatron_bridge` → `checkpoint_megatron`.

A typical alignment path starts from a `checkpoint_megatron` policy, adds preference or reward-side data, runs one of the `rl/nemo_rl/...` steps, and produces a new `checkpoint_megatron`.

A typical compression path starts from `checkpoint_hf`, runs `optimize/modelopt/quantize`, and produces `checkpoint_megatron`.
Run `convert/megatron_to_hf` after quantization when the next consumer needs a Hugging Face layout again.

## Tokenizer and Chat Template Consistency

Matching artifact types is not enough for correctness.
The tokenizer, the chat template, and the maximum sequence length must stay consistent across every step that tokenizes text or loads weights for the same model line.
A mismatch often appears as a plausible training loss curve with poor downstream quality.

## Related Reading

- [Nemotron Steps Basics](../../steps/basics.md) defines the concepts of *step*, *configuration*, *environment profile*, and *artifact*.
- [Training Basics](basics.md) defines the training-specific terms, including tokenizers and chat templates.
- [Data and Checkpoint Formats](../how-to/data-and-checkpoint-formats.md) describes the on-disk layouts of each artifact type.
- [Convert Checkpoints Between Training Steps](../how-to/convert-checkpoints.md) gives command examples for the conversion steps.
- [Training Libraries](training-libraries.md) describes the ecosystem of libraries that back the steps.
