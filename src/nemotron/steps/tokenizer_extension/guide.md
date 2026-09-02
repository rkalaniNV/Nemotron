# tokenizer_extension

Extend a base tokenizer with target-language subwords, initialize the model
embeddings for the new vocab, and (optionally) measure fertility — producing a
resized HF checkpoint ready for continued pretraining.

## Steps (typed-artifact pipeline)

| Step | Compute | Consumes → Produces |
|------|---------|---------------------|
| `extend` | CPU | `checkpoint_hf` (tokenizer) + corpus → `tokenizer` (add/ and/or replace/) |
| `init_embeddings` | GPU | `tokenizer` + base `checkpoint_hf` → resized `checkpoint_hf` |
| `evaluate` | CPU | `tokenizer` → `eval_results` (fertility) |
| `eval_init` | GPU | resized `checkpoint_hf` → `eval_results` (BPB / perplexity) |

```
extend ──▶ tokenizer ──▶ init_embeddings ──▶ checkpoint_hf ──▶ pretrain/megatron_bridge (CPT)
   └─────▶ evaluate (fertility)
```

## Add vs Replace
- **Add** keeps the base's residual script tokens and appends new ones (`vocab = V + N`). Safer across a script family (e.g. Hindi + Marathi).
- **Replace** prunes the target script's residual tokens, then splices fresh corpus-optimal ones into the pruned base (`vocab = V − R + N`). Fewer rows; best for a single target language.

The splice is *constructive* (rank-dead-safe): every added token is emittable by the merge table, so the tokenizer never silently byte-fragments (see `extend/README.md`).

## Design notes
- Each step is **self-contained**: `step.py` is the executor and its supporting `.py` files live beside it. The only shared modules are `languages.py` and `script_ranges.py` at the block root, which every sub-step imports. Backend (Slurm / Lepton / DGX) is chosen at run time by the selected `env.toml` profile, not the step.
- **`init_embeddings` is the growth point** for embedding-init research: `embeddings.py` dispatches `method:` (`baseline` | `subword` | `focus`, see `METHODS`) to a separate engine script — `baseline_init.py`, `subword_init.py`, `focus_init.py`, and `replace_init.py` for the Replace arm. Add a technique by adding an engine and a `METHODS` entry.
- Post-CPT model / downstream evaluation is **not** here — use the `eval` catalog.

## Dependencies

Two packages are **not** in the base install and are declared as an extra
(`pyproject.toml` → `[project.optional-dependencies] tokenizer-extension`):

| Package | Needed by | If missing |
|---|---|---|
| `indic-nlp-library` | `extend` with `script_normalizer: devanagari` (i.e. `language:` hindi/marathi/nepali/sanskrit) | **hard error** — it used to fall back to NFKC-only, which silently trained a *different* tokenizer |
| `fasttext-wheel` | `init_embeddings` with `method: focus` | **hard error** at the point of use (import is lazy, so other methods are unaffected) |

Local install:
```bash
uv pip install -e '.[tokenizer-extension]'
```
For a remote runner the container is built from the profile, so add them there —
in your `env.toml`, on the profile used for these steps:
```toml
pip_extras = ["typer", "rich", "pydantic-settings", "indic-nlp-library", "fasttext-wheel"]
```
Neither is installed by the default profile; a run will fail with the exact
install command rather than produce a quietly different tokenizer.

## Run
```bash
# CPU tokenizer build (Lepton cpu profile)
uv run nemotron steps run tokenizer_extension/extend -c default --batch <cpu_profile>
# GPU embedding init -> resized checkpoint
uv run nemotron steps run tokenizer_extension/init_embeddings -c default --batch <gpu_profile>
# CPU fertility check
uv run nemotron steps run tokenizer_extension/evaluate -c default --batch <cpu_profile>
# GPU BPB check (cross-vocabulary comparable; use max_docs, NOT max_tokens)
uv run nemotron steps run tokenizer_extension/eval_init -c default --batch <gpu_profile>
```

Set `language:` on `extend` and `init_embeddings` to target a language other than
Hindi (see `LANGUAGES.md`); everything else follows from the profile.
