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

```
extend ──▶ tokenizer ──▶ init_embeddings ──▶ checkpoint_hf ──▶ pretrain/megatron_bridge (CPT)
   └─────▶ evaluate (fertility)
```

## Add vs Replace
- **Add** keeps the base's residual script tokens and appends new ones (`vocab = V + N`). Safer across a script family (e.g. Hindi + Marathi).
- **Replace** prunes the target script's residual tokens, then splices fresh corpus-optimal ones into the pruned base (`vocab = V − R + N`). Fewer rows; best for a single target language.

The splice is *constructive* (rank-dead-safe): every added token is emittable by the merge table, so the tokenizer never silently byte-fragments (see `extend/README.md`).

## Design notes
- Each step is **self-contained**: `step.py` is the executor and its supporting `.py` files live beside it (no shared package). Backend (Slurm / Lepton / DGX) is chosen at run time by the selected `env.toml` profile, not the step.
- **`init_embeddings` is the growth point** for embedding-init research: register a new strategy in `embeddings.py` (`INIT_STRATEGIES`) and select it via `init_method`.
- Post-CPT model / downstream evaluation is **not** here — use the `eval` catalog.

## Run
```bash
# CPU tokenizer build (Lepton cpu profile)
uv run nemotron steps run tokenizer_extension/extend -c default --batch <cpu_profile>
# GPU embedding init -> resized checkpoint
uv run nemotron steps run tokenizer_extension/init_embeddings -c default --batch <gpu_profile>
# CPU fertility check
uv run nemotron steps run tokenizer_extension/evaluate -c default --batch <cpu_profile>
```
