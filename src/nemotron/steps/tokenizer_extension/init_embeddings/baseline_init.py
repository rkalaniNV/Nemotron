#!/usr/bin/env python3
"""Extend a model's embedding matrices to cover an expanded tokenizer.

Three initialization modes for the new token rows:

    hf_default   HuggingFace's own resize, which draws from a multivariate normal
                 fitted to the mean and covariance of the existing embeddings
                 (see https://nlp.stanford.edu/~johnhew/vocab-expansion.html)
    mean_all     the mean of every existing embedding
    mean_hindi   the mean of the existing Devanagari token embeddings only

With --norm-correction the new input embeddings are rescaled so their L2 norm
matches the median norm of the original input embeddings.  Output (LM head)
embeddings are never rescaled: inflating their norm makes the model
over-confidently predict the new tokens and explodes the training loss.  The
flag has no effect in hf_default mode, where initialization is left entirely to
HuggingFace.

Example
    python extend_model.py \
        --extended-tokenizer /path/to/merged_tokenizer \
        --output-dir /path/to/extended_model \
        --mode mean_hindi \
        --norm-correction
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEVANAGARI_RANGE = (0x0900, 0x097F)
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
RULE = "=" * 60


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                       help="Model whose embedding matrices are extended.")
    paths.add_argument("--extended-tokenizer", required=True,
                       help="Tokenizer containing the base vocabulary plus the new tokens.")
    paths.add_argument("--output-dir", required=True,
                       help="Directory to save the extended model and tokenizer to.")
    paths.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16",
                       help="Precision to load the base model in.")

    init = parser.add_argument_group("initialization")
    init.add_argument("--mode", choices=("hf_default", "mean_all", "mean_hindi"),
                      default="hf_default",
                      help="How to initialize the new token embeddings.")
    init.add_argument("--norm-correction", action=argparse.BooleanOptionalAction, default=False,
                      help="Rescale the new input embeddings to the median original norm; "
                           "output embeddings are never rescaled.")
    init.add_argument("--num-samples", type=int, default=10,
                      help="Devanagari tokens to list in mean_hindi mode.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

@contextmanager
def phase(title: str) -> Iterator[None]:
    print("\n" + RULE)
    print(title)
    print(RULE)
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f}s ({elapsed / 60:.2f} min)")


def describe_tensor(norms: torch.Tensor, extremes: bool = True) -> str:
    parts = [f"median: {norms.median().item():.4f}",
             f"mean: {norms.mean().item():.4f}",
             f"std: {norms.std().item():.4f}"]
    if extremes:
        parts += [f"min: {norms.min().item():.4f}", f"max: {norms.max().item():.4f}"]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def is_devanagari(text: str) -> bool:
    lo, hi = DEVANAGARI_RANGE
    return any(lo <= ord(char) <= hi for char in text)


def find_devanagari_token_ids(tokenizer, vocab_size: int) -> List[int]:
    token_ids = []
    for token_id in tqdm(range(vocab_size), desc="Scanning vocabulary for Hindi tokens"):
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue
        if is_devanagari(text):
            token_ids.append(token_id)
    return token_ids


# ---------------------------------------------------------------------------
# Initialization modes
# ---------------------------------------------------------------------------

def initialize_hf_default(model, new_vocab_size: int) -> None:
    print("Resizing with HuggingFace's multivariate-normal initialization...")
    model.resize_token_embeddings(new_vocab_size)


def initialize_with_vector(model, new_vocab_size: int, original_vocab_size: int,
                           source_ids: Optional[torch.Tensor], label: str) -> None:
    """Fill every new row with the mean over `source_ids`, or over the whole vocabulary."""
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings().weight
        output_embeddings = model.get_output_embeddings().weight
        rows = slice(None, original_vocab_size) if source_ids is None else source_ids
        mean_input = input_embeddings[rows].mean(dim=0).clone()
        mean_output = output_embeddings[rows].mean(dim=0).clone()

    print(f"  Mean {label} input norm:  {mean_input.norm().item():.4f}")
    print(f"  Mean {label} output norm: {mean_output.norm().item():.4f}")

    # mean_resizing=False keeps HuggingFace from initializing the rows we overwrite.
    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

    with torch.no_grad():
        model.get_input_embeddings().weight[original_vocab_size:] = mean_input
        model.get_output_embeddings().weight[original_vocab_size:] = mean_output


def initialize_mean_all(model, new_vocab_size: int, original_vocab_size: int) -> None:
    print("Initializing every new token with the mean of all existing embeddings...")
    initialize_with_vector(model, new_vocab_size, original_vocab_size, None, "all-token")


def initialize_mean_hindi(model, tokenizer, new_vocab_size: int, original_vocab_size: int,
                          num_samples: int) -> None:
    print("Initializing every new token with the mean of the Hindi embeddings...")
    hindi_ids = find_devanagari_token_ids(tokenizer, original_vocab_size)
    if not hindi_ids:
        raise SystemExit("No Devanagari tokens in the original vocabulary; "
                         "mean_hindi initialization is not possible.")

    share = 100.0 * len(hindi_ids) / original_vocab_size
    print(f"  Found {len(hindi_ids)} Hindi tokens ({share:.2f}% of the vocabulary)")
    for token_id in hindi_ids[:num_samples]:
        print(f"    ID {token_id}: {tokenizer.decode([token_id])!r}")

    initialize_with_vector(model, new_vocab_size, original_vocab_size,
                           torch.tensor(hindi_ids, dtype=torch.long), "Hindi")


def apply_input_norm_correction(model, original_vocab_size: int) -> None:
    with torch.no_grad():
        weights = model.get_input_embeddings().weight
        original_norms = weights[:original_vocab_size].norm(dim=1)
        target_norm = original_norms.median()
        print(f"  Original input norms - {describe_tensor(original_norms)}")
        print(f"  Target norm (median of all tokens): {target_norm.item():.4f}")

        new_rows = weights[original_vocab_size:]
        before = new_rows.norm(dim=1)
        scale = torch.where(before > 1e-8, target_norm / before, torch.ones_like(before))
        new_rows.mul_(scale.unsqueeze(1))

        print(f"  New input norms before - {describe_tensor(before, extremes=False)}")
        print(f"  New input norms after  - "
              f"{describe_tensor(new_rows.norm(dim=1), extremes=False)}")
        print("  Output embeddings left untouched by design")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    script_start = time.time()

    print(f"Mode:                {args.mode}")
    print(f"Norm correction:     {args.norm_correction}")
    print(f"Base model:          {args.base_model}")
    print(f"Extended tokenizer:  {args.extended_tokenizer}")
    print(f"Output:              {args.output_dir}")

    with phase("Phase 1: Loading model and tokenizers"):
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=DTYPES[args.dtype], trust_remote_code=True,
        )
        extended_tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer,
                                                           fix_mistral_regex=True)
        original_tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                                           fix_mistral_regex=True)

    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(extended_tokenizer)
    if new_vocab_size <= original_vocab_size:
        raise SystemExit(
            f"Extended tokenizer has {new_vocab_size} tokens, which does not exceed the "
            f"base model's {original_vocab_size} embeddings; nothing to initialize."
        )

    print(f"\nOriginal vocab size: {original_vocab_size}")
    print(f"New vocab size:      {new_vocab_size}")
    print(f"New tokens to add:   {new_vocab_size - original_vocab_size}")
    print(f"Input  embeddings:   {tuple(model.get_input_embeddings().weight.shape)}")
    print(f"Output embeddings:   {tuple(model.get_output_embeddings().weight.shape)}")

    with phase(f"Phase 2: Initializing new embeddings ({args.mode})"):
        if args.mode == "hf_default":
            initialize_hf_default(model, new_vocab_size)
        elif args.mode == "mean_all":
            initialize_mean_all(model, new_vocab_size, original_vocab_size)
        else:
            initialize_mean_hindi(model, original_tokenizer, new_vocab_size,
                                  original_vocab_size, args.num_samples)

    if args.norm_correction and args.mode == "hf_default":
        print("\nIgnoring --norm-correction: hf_default leaves initialization to HuggingFace.")
    elif args.norm_correction:
        with phase("Phase 3: Input norm correction"):
            apply_input_norm_correction(model, original_vocab_size)

    print(f"\nNew input  embedding shape: {tuple(model.get_input_embeddings().weight.shape)}")
    print(f"New output embedding shape: {tuple(model.get_output_embeddings().weight.shape)}")

    with phase(f"Phase 4: Saving to {args.output_dir}"):
        from vocab_pad import pad_vocab_to_multiple
        pad_vocab_to_multiple(model, 4)   # TP-safe: divisible by TP (=4); minimal padding across all arms/methods
        model.save_pretrained(args.output_dir)
        extended_tokenizer.save_pretrained(args.output_dir)

    total_elapsed = time.time() - script_start
    print("\n" + RULE)
    print("DONE")
    print(RULE)
    print(f"  Wall-clock time:     {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(f"  Mode:                {args.mode}")
    print(f"  Norm correction:     {args.norm_correction}")
    print(f"  Base model:          {args.base_model}")
    print(f"  Extended tokenizer:  {args.extended_tokenizer}")
    print(f"  Output:              {args.output_dir}")
    print(RULE)


if __name__ == "__main__":
    main()
