<a id="top"></a>

# Tokenizer Extension Guidebook

> A practical decision guide for extending a base-model tokenizer to a new language—how much tokenizer data to use, how many vocabulary rows to add, how to initialize them, and when the serving benefit is worth the vocabulary cost.

**Intended use:** a language-agnostic decision process for adapting any supported language<br>
**Observed evidence:** Nemotron-3-Nano-30B-A3B experiments in Hindi, Malayalam, and Vietnamese<br>
**Reading time:** 8–10 minutes

<p align="center"><img src="./assets/guidebook_overview.png" alt="Tokenizer-extension decision path" width="760"></p>

**Jump to:** [Data requirement](#1-how-much-tokenizer-data-is-enough) · [Vocabulary and method](#2-how-many-vocabulary-rowsand-which-extension-type) · [Embedding initialization](#3-how-should-new-embeddings-be-initialized) · [Learning rate](#4-which-learning-rate-policy-is-safe) · [Serving](#5-does-the-tokenizer-pay-at-serving-time)

> [!IMPORTANT]
> The corrected `expand` tokenizer is being rerun. FOCUS and HF-default results that depend on that rerun remain a **current snapshot**, not a final release claim. Stable and provisional findings are labelled separately below.

## How to read this guide

Each section follows the same order:

1. **Decision rule** — the portable recommendation for a new language.
2. **Validation** — what the customer should measure on their own data and deployment.
3. **Observed evidence** — results from Hindi, Malayalam, and Vietnamese that support or limit the rule.

The tested languages are examples, not prerequisites and not universal constants.

## Start here

| Customer question | Short answer | Evidence strength |
|---|---|---|
| How much tokenizer-training data? | Start with **10k–100k documents** and stop when held-out fertility flattens | Established across three languages |
| How many vocabulary rows? | **+30k** is the common tested knee | Established within this model |
| Add, Replace, or naive Expand? | Use **Add** for compatibility; **Replace** when rows matter; treat Expand as a baseline | Add vs Replace established; corrected Expand rerun in progress |
| Which embedding initialization? | Begin with **meanconst**, then compare at the final training horizon | Directional until the corrected rerun closes |
| Which learning-rate policy? | Use **DLR** when the vocabulary change is substantial | Replicated target/English BPB pattern across three languages |
| How should different tokenizers be compared? | Use **bits per byte (BPB)**, not raw per-token perplexity | Required for a fair cross-tokenizer comparison |
| Will extension improve model quality? | Treat it primarily as an **efficiency intervention** | Independent downstream-quality contribution not yet established |
| Will it improve serving? | Usually—if fertility savings exceed the vocabulary-row tax on the intended deployment shape | Matched A100/TP4 evidence across three languages |

### Recommended starting recipe

```text
10k–100k diverse documents
        ↓
measure held-out fertility at +15k / +30k / +45k
        ↓
choose the smallest useful vocabulary knee (often +30k)
        ↓
initialize with meanconst + train with DLR
        ↓
validate BPB, target quality, English retention, and serving throughput
```

## 1. How much tokenizer data is enough?

> **Decision rule:** begin with a diverse 10k–100k-document sample. Increase it only while held-out fertility, token utilization, or script/domain coverage is still improving.

### Observed evidence

<p align="center"><img src="./assets/tokenizer_data_saturation.png" alt="Tokenizer-training data saturation" width="720"></p>

<details>
<summary><strong>Observed values by language</strong></summary>

| Language | Earliest practical plateau | 10k→1M spread | Interpretation |
|---|---:|---:|---|
| Malayalam | **10k documents** | 0.6% | The 10k sample captures essentially the full fertility benefit |
| Hindi | **10k documents** | 0.5% | Small non-monotonicity is consistent with independent resampling noise |
| Vietnamese | **50k documents** | 2.7% from 10k; 50k is within 0.17% of 1M | One extra step is useful, but 1M is unnecessary |

</details>

> **Clean takeaway:** tokenizer construction needs coverage and diversity—not CPT-scale data. Begin small, measure on a held-out slice, and scale only while fertility is still moving.

<details>
<summary><strong>What should be represented in the tokenizer corpus?</strong></summary>

- Native-script and romanized text.
- Formal and conversational text.
- Domain text, tables, OCR, numbers, and named entities.
- Code-switched text if it is part of the product workload.
- Enough examples of each script and normalization pattern to avoid dead tokens.

Document count is only a convenient control variable. For reproducibility, also record bytes, characters, domains, sampling policy, and unique-script coverage.

</details>

[Back to top](#top)

## 2. How many vocabulary rows—and which extension type?

> **Decision rule:** measure at several vocabulary budgets and select the smallest knee that meets the product’s context and serving goals. Choose the construction method from compatibility and row-budget constraints—not from a tiny fertility difference.

### Observed evidence

<p align="center"><img src="./assets/vocabulary_knee_and_method.png" alt="Vocabulary knee and Add-versus-Replace comparison" width="700"></p>

### Vocabulary budget

<details>
<summary><strong>Observed token reduction by language and vocabulary budget</strong></summary>

| Added rows | Hindi reduction | Malayalam reduction | Vietnamese reduction | Reading |
|---:|---:|---:|---:|---|
| +15k | 33.4% | 52.1% | 15.9% | Captures most of the available compression |
| **+30k** | **37.0%** | **56.6%** | **16.6%** | Common tested knee |
| +45k | 38.2% | 58.7% | 16.9% | Smaller marginal return; language-dependent value |

</details>

Every second +15k increment buys only **0.40–0.48×** the gain of the first. Use +30k as a starting hypothesis—not as an unconditional constant.

### Construction choice

| Method | What it does | Use it when | Current evidence |
|---|---|---|---|
| **Continued-BPE Add** | Preserves the base vocabulary and learns compatible new merges/rows | Compatibility and simplicity matter | Wins 11/11 matched Add-vs-Replace fertility pairs, but only by 0.016–0.30% |
| **Prune-and-Replace** | Removes weak target-script rows before inserting new ones | Serving rows or vocabulary size are constrained | Nearly identical fertility with fewer final rows |
| **Naive Expand** | Registers atomic token surfaces without continued merge learning | You need a baseline or a deliberately simple tokenizer | Corrected tokenizer rerun in progress; do not make a final quality claim yet |

> **Clean takeaway:** the extension budget matters far more than Add versus Replace. Choose Add for operational compatibility and Replace when every row matters.

<details>
<summary><strong>Why fertility alone is insufficient</strong></summary>

Before selecting a production tokenizer, also compare:

- Token-ID compatibility with existing data, adapters, and checkpoints.
- Embedding/output parameter growth and memory.
- Tokenizer construction reproducibility and dead-token rate.
- Words/s or characters/s on the intended hardware, tensor-parallel shape, batch size, and request mix.
- Target quality and retained general capability after CPT.

</details>

[Back to top](#top)

## 3. How should new embeddings be initialized?

> **Decision rule:** start with an initialization derived from the base model’s existing token geometry, then compare alternatives at the final training horizon. Do not select an initializer from an early-only checkpoint.

### Observed evidence

<p align="center"><img src="./assets/embedding_initialization_bpb.png" alt="Embedding-initialization BPB comparison" width="720"></p>

**Raw BPB is lower-is-better.** The chart presents reduction versus the matching base so larger bars are better.

| Initialization | Operational interpretation | Current recommendation |
|---|---|---|
| **meanconst** | Builds each new row from the model’s existing subtoken decomposition | **Starting default**: dependency-free and strongest in one current final-horizon observation |
| BERT-family transfer | Uses an aligned auxiliary encoder | Alternative when the dependency is already available and final-horizon validation supports it |
| FOCUS | Cross-tokenizer alignment | Validate per language; ranking is not stable enough for a universal default |
| HF-default/random | Starts new rows without lexical structure | Baseline only for substantial extensions; can be fragile when new rows learn slowly |

### Observed example: why the final checkpoint matters

<details>
<summary><strong>Exact Add+DLR checkpoint values</strong></summary>

| Hindi Add+DLR | BPB @1k | BPB @2.4k | Reduction vs base @2.4k |
|---|---:|---:|---:|
| **meanconst** | **0.30149** | **0.26745** | **22.3%** |
| BERT | 0.30178 | 0.27895 | 19.0% |
| HF-default | 0.30494 | 0.27921 | 18.9% |
| FOCUS | 0.31228 | 0.28538 | 17.1% |

| Malayalam Add+DLR @1.084k | BPB | Reduction vs base |
|---|---:|---:|
| **meanconst** | **0.34204** | **40.2%** |
| BERT | 0.34216 | 40.2% |
| HF-default | 0.35195 | 38.4% |
| FOCUS | 0.35581 | 37.8% |

</details>

The meanconst–BERT difference grows from **0.00029 BPB at 1k** to **0.01150 at 2.4k**. An early-only screen would incorrectly conclude that initialization does not matter.

> [!CAUTION]
> These initialization values are retained for transparency but remain a current snapshot until the corrected tokenizer rerun and dependent refresh complete. Malayalam FOCUS is now included; the current Add+DLR initializer grid has no missing language/method cell.

[Back to top](#top)

## 4. Which learning-rate policy is safe?

> **Decision rule:** keep the backbone learning rate conservative and accelerate the parameters changed by tokenizer extension. Increase the whole-model learning rate only when a matched retention evaluation justifies it.

### Observed evidence

<p align="center"><img src="./assets/learning_rate_bpb_tradeoff.png" alt="Target and English BPB trade-off by learning-rate policy" width="700"></p>

| Regime | Definition | Target result | English result | Recommendation |
|---|---|---|---|---|
| Low LR | Whole model at `1e-5` | Safe but slower target learning | Small English change | Use for a small tokenizer change |
| **DLR** | Backbone at `1e-5`; embeddings and output head at `1e-4` | Adds **+7.5 hi · +2.4 vi · +7.7 ml pp** over Low LR | English BPB stays between **−0.16% and +1.72%** | **Default for a substantial extension** |
| High LR | Whole model at `1e-4` | Learns the target fastest | **+14.99–18.35% English BPB damage** | Do not use as the safe default |

> **Clean takeaway:** train changed lexical capacity faster, not the whole model. DLR is the best tested target/retention exchange rate.

### BPB versus perplexity

Per-token perplexity changes when token boundaries change, so it cannot fairly compare the base and extended tokenizers. BPB scores the same held-out bytes and is therefore the primary tokenizer-independent language-modeling metric.

<details>
<summary><strong>Recommended evaluation bundle</strong></summary>

1. Fertility or characters/token on held-out target text.
2. Target-language BPB and English BPB on identical fixed-byte corpora.
3. Target downstream evaluation such as MILU/VMLU.
4. English/general retention such as ProX-en, ARC-Challenge, HellaSwag, and MMLU-family tasks.
5. Matched serving throughput and memory on the intended deployment shape.

</details>

[Back to top](#top)

## 5. Does the tokenizer pay at serving time?

> **Decision rule:** treat serving value as a deployment measurement. Compare the token-count reduction with the added vocabulary-row cost on the exact hardware, tensor-parallel shape, batch profile, and request mix.

### Observed evidence

<p align="center"><img src="./assets/serving_throughput.png" alt="Matched serving throughput" width="720"></p>

| Language | Fertility gain | Vocabulary tax | Net batched throughput | Net single-request throughput |
|---|---:|---:|---:|---:|
| Malayalam | 2.303× | −3.0% | **2.23×** | **2.29×** |
| Hindi † | 1.588× | ≈−3.0% | **1.54×** | **1.58×** |
| Vietnamese | 1.199× | −3.1% | **1.16×** | **1.19×** |

† Hindi is derived from the matched vocabulary-size tax; Malayalam and Vietnamese were directly benchmarked.

> **Clean takeaway:** vocabulary rows impose a serving tax, but compression wins at matched A100/TP4 for all three languages. Re-measure on the production tensor-parallel shape—this result is not hardware-independent.

## Customer release checklist

- [ ] Held-out fertility is reported by language, script, and domain.
- [ ] Tokenizer corpus size is justified by a saturation curve.
- [ ] Vocabulary size is justified by marginal gain and serving cost.
- [ ] `tokens_spliced` equals the requested extension budget.
- [ ] BPB uses the same held-out byte strings across tokenizers.
- [ ] Target quality and English retention are evaluated after CPT.
- [ ] Throughput is measured on the intended hardware, TP, batch, and request mix.
- [ ] Provisional reruns are clearly distinguished from release-ready findings.

[Back to top](#top)
