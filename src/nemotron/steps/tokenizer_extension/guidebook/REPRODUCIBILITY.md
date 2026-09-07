# Reproducibility — Tokenizer Extension

The data, pipeline and settings behind the results in this guidebook.

Cluster-specific values — scheduler profiles, mount points, output paths — are
site configuration; substitute your own. Everything that affects the result is
given explicitly.

**Contents:** [Quick start](#quick-start) · [Hardware](#hardware) ·
[Data](#data) · [Pipeline](#pipeline) · [Settings](#settings) ·
[Cost](#cost-and-runtime) · [Adapting to a new language](#adapting-to-a-new-language)

---

## Quick start

Build one tokenizer and measure it — +30,000 tokens from 1M documents:

```bash
nemotron steps run tokenizer_extension/extend          -c <extend-config> -b <profile>
nemotron steps run tokenizer_extension/init_embeddings -c <init-config>   -b <profile>
nemotron steps run tokenizer_extension/evaluate        -c <fert-config>   -b <profile>
```

Then compare held-out fertility against the base tokenizer. Two builds — a
budget sweep and a data-size sweep — are enough to locate the knee.

---

## Hardware

| Step | Nodes | GPUs | Host RAM | Notes |
|---|---|---|---|---|
| `extend` | 1 | none used | **~1 TB** | BPE training is CPU and memory bound. We ran it on an 8 × A100 80GB node purely for its host memory; the GPUs sit idle. Reduce `samples` or `max_doc_chars` to fit a smaller machine. |
| `init_embeddings` | 1 | 1 × 80GB (A100 or H100) | 200 GB+ | loads the base model in BF16 to write new rows |
| `evaluate` (fertility) | 1 | none | 64 GB | tokenizer only, no model |
| `eval_init` (BPB) | 1 | 1–8 × 80GB (A100 or H100) | 200 GB+ | scores the initialised model in BF16 |

Serving-throughput figures in §5 were measured on **A100 80GB at TP 4**. That
result is hardware-dependent — re-measure on your deployment shape.

`indic-nlp-library` is required for the Devanagari normalizer.

---

## Data

| Use | Dataset | Selector |
|---|---|---|
| Tokenizer training — Hindi | `ai4bharat/sangraha` | `verified/hin` |
| Tokenizer training — Malayalam | `ai4bharat/sangraha` | `verified/mal` |
| Tokenizer training — Vietnamese | `VTSNLP/vietnamese_curated_dataset` | shards 0–12 |
| Fertility / BPB eval — Indic | `ai4bharat/samanantar` | config `hi`, field `tgt` |
| Fertility eval — Vietnamese | `VTSNLP/vietnamese_curated_dataset` | held-out shards 121–131 |

Training and evaluation text are disjoint: different datasets for Indic,
non-overlapping shard ranges for Vietnamese.

---

## Pipeline

```bash
# train BPE on the target corpus and build the extended tokenizer
nemotron steps run tokenizer_extension/extend          -c <extend-config> -b <profile>

# initialise embedding / output rows for the new tokens
nemotron steps run tokenizer_extension/init_embeddings -c <init-config>   -b <profile>

# fertility (tokens per word) on held-out text
nemotron steps run tokenizer_extension/evaluate        -c <fert-config>   -b <profile>

# bits-per-byte of the initialised model, before any CPT
nemotron steps run tokenizer_extension/eval_init       -c <bpb-config>    -b <profile>
```

---

## Settings

**`extend`**

| Setting | Values used |
|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| `method` | `add`, `replace`, `expand` |
| `extension_size` | 15,000 / 30,000 / 45,000 |
| `corpus.samples` | 10k / 50k / 100k / 500k / 1M |
| `max_doc_chars` / `min_frequency` | 20,000 / 0 |
| `diversify` | `false` — shards are read in order, so 100k ⊂ 500k ⊂ 1M and corpus scale is the only variable across the sweep |

Confirm `tokens_spliced` in `summary.json` matches `extension_size`; fertility
comparisons are only valid at a matched budget.

**`init_embeddings`**

| Setting | Values used |
|---|---|
| Base model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, bfloat16 |
| `arm` | `add` or `replace` — matches `extend` |
| `method` | `subword`, `focus`, `hfdefault`, `meanconst` |
| Averaging (input / output) | `uniform`, `char_weighted`, `max_char`, `bert_weighted` — set independently per side |
| Auxiliary encoder | MuRIL for Indic, XLM-R otherwise — must cover the target language |
| Temperature | 0.1 |
| Norm correction | input on, output off |

**`evaluate` / `eval_init`**

Fertility runs over the full held-out corpus. BPB uses `max_docs: 3000`,
`max_tokens: 1,000,000`, `max_length: 2048` — set both caps so every model is
scored over the same text.

---

## Cost and runtime

| Step | Typical runtime | Resource |
|---|---|---|
| `extend`, 1M docs @ +30k | 2–4 h | CPU + ~1 TB RAM |
| `extend`, 100k docs | 20–40 min | CPU + ~256 GB RAM |
| `init_embeddings` | 20–40 min | 1 × 80GB GPU |
| `evaluate` (fertility) | 10–30 min | CPU |
| `eval_init` (BPB, 4 models) | ~1 h | 1–8 × 80GB GPU |

A full sweep — 5 corpus sizes × 3 budgets × 3 methods — is dominated by
`extend`; budget roughly one CPU-day per language.

---

## Adapting to a new language

1. **Add a language profile** — normalizer, Unicode script range, fastText code
   and auxiliary encoder. This is a data change, not a code change.
2. **Point `extend` at your corpus** and pick a held-out evaluation set that
   does not overlap it.
3. **Sweep budget and corpus size** — +15k / +30k / +45k over 10k–1M documents
   locates the knee for most scripts.
4. **Choose an auxiliary encoder that covers the language** for embedding
   initialization.

Scripts with poor base-vocabulary coverage gain the most; check how many base
tokens fall in your script's Unicode range before committing to a budget.

---

## What to record

Config name, code revision, corpus and sample count, `extension_size`, the
`tokens_spliced` from `summary.json`, and the output path.

---

[← Back to the guidebook](./README.md)
