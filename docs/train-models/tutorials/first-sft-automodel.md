# First SFT Run with AutoModel

This tutorial walks through supervised fine tuning (SFT) by using the `sft/automodel` step. That step trains models in a Hugging Face layout from OpenAI chat-formatted JSON Lines (JSONL). You will point YAML configuration at your data paths, run a tiny job, and confirm that the step completes.

## What You Will Need

- If you override the defaults in `tiny.yaml`, you need instruction data in JSON Lines (JSONL) form where each record includes a `messages` field in chat format.
- A Hugging Face access token if the base model is gated or must be downloaded from the Hugging Face Hub.
- Enough GPU memory for the model you select in YAML. Tiny configurations often use a small model. You should still verify memory before you scale the run.

## Configure Paths

1. Open `src/nemotron/steps/sft/automodel/config/tiny.yaml`. The checked-in defaults already pull small public slices from the Hugging Face Hub. When you move to your own instruction data, set dataset and model fields there and keep tokenizer and chat template aligned with how the JSON Lines (JSONL) was built.
2. Ensure the tokenizer and chat template match how the JSON Lines (JSONL) was built.

## Run on Lepton

This repository’s getting started path assumes DGX Cloud Lepton and the `lepton_sft_automodel` profile from root `env.toml`. See [Getting Started with Training Steps](../getting-started.md) for the environment snippet and prerequisites.

```console
$ uv run nemotron steps run sft/automodel -c tiny -r lepton_sft_automodel
```

See [Execution through NeMo Run](../../nemo_runspec/nemo-run.md) for profile setup and scheduler behavior. Replace `lepton_sft_automodel` with another profile name from your `env.toml` when your team uses a different table key.

## Verify Output

The step manifest declares a `checkpoint_hf` artifact on success. Confirm the output directory you set in training configuration contains checkpoints. Confirm that logs show stable loss for the short tiny run.

## Common Issues

- When you hit CUDA out of memory, reduce batch sizes in YAML or switch to a smaller base model for the tutorial run.
- When the trainer reports a missing chat template, pick a tokenizer that defines a template, or convert data to a format the trainer accepts. The `step.toml` file lists `[[errors]]` entries such as `chat_template_missing` with recovery hints.

## Next Steps

- Read [Choose an SFT Backend](../how-to/choose-sft-backend.md) when you need Megatron Bridge and packed Parquet.
- Read [Data and Checkpoint Formats](../how-to/data-and-checkpoint-formats.md) to learn how JSONL and checkpoints chain with other steps.
