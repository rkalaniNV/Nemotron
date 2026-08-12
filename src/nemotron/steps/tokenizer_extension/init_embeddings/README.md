# tokenizer_extension/init_embeddings

Attach an extended tokenizer to the base model, initialize the new embedding (and
LM-head) rows, and save a **resized HF checkpoint** for CPT. Needs a GPU node.

## Files
- `step.py` — executor (loads YAML, calls `embeddings.run_init`)
- `embeddings.py` — the init **strategy registry** + Add/Replace embedding surgery

## Adding a new init technique (the whole point)
Register a function; both the Add and Replace paths pick it up automatically:
```python
@register("random")
def _init_random(surface: str, ctx: InitContext):
    std = ctx.old_in.std().item()
    vout = torch.randn_like(ctx.global_mean_out) * std if ctx.global_mean_out is not None else None
    return torch.randn_like(ctx.global_mean_in) * std, vout
```
`InitContext` gives you the base tokenizer, the original input/output embedding
snapshots, the global means, and `old_n`. Select with `init_method: random`.

Shipped strategies: `mean`, `mean_of_constituents`. Planned: `focus`, `wechsel`,
`random` (stubs shown in `embeddings.py`).

## Add vs Replace surgery
- **add** → resize, keep base rows, init appended rows `[old_n, new_n)`.
- **replace** → resize, permute survivor rows via `id_remap.json`, init new rows
  `[pruned_size, final_n)`. (`tokenizer_path` must be extend's `replace/` output.)

## Run
```bash
uv run nemotron steps run tokenizer_extension/init_embeddings -c default --batch <gpu_profile>
```
Output: `output_dir/` = resized HF checkpoint (weights + tokenizer) → set as
`pretrain/megatron_bridge` `hf_model_path`.
