# tokenizer_extension/init_embeddings

Attach an extended tokenizer to the base model, initialize the new embedding (and
LM-head) rows, and save a **resized HF checkpoint** for CPT.

The base model is embedding surgery only (no forward pass) and is loaded
**host-resident**, not on GPU. GPUs are used only by the auxiliary encoders:
`bert_weighted` (MuRIL-class) and `gemma_weighted` (Gemma, sharded via
`device_map='auto'`). The step requests 8 GPUs, which only the gemma path uses.

## Files
- `step.py` — executor (loads YAML, calls `embeddings.run_init`)
- `embeddings.py` — the `method:` dispatcher + Add/Replace argv construction

## Adding a new init technique (the whole point)
`embeddings.py` does not hold the strategies itself; it selects an **engine
script** from `METHODS = ("baseline", "subword", "focus")` and builds its argv.
The Replace arm routes through `replace_init.py`, which additionally copies
survivor rows via `id_remap.json`.

To add a technique:
1. write `<name>_init.py` next to the others, taking `--base-model`,
   `--extended-tokenizer`, `--output-dir`, `--dtype` and `--language`;
2. add `"<name>"` to `METHODS`;
3. add a `_<name>_argv(cfg)` builder and a dispatch branch in `embeddings.py`
   (and in `_replace_argv` if it should support the Replace arm).

Select it with `method: <name>`.

Shipped: `baseline` (`hf_default` | `mean_all` | `mean_target`), `subword`
(`input_averaging`: `uniform` = mean-of-constituents, `char_weighted`,
`max_char`, `bert_weighted`, `gemma_weighted`), and `focus` (fastText +
sparsemax). Set `language:` so the auxiliary encoder and fastText vectors are
chosen for your target — passing `subword.bert_model` explicitly overrides it.

## Add vs Replace surgery
- **add** → resize, keep base rows, init appended rows `[old_n, new_n)`.
- **replace** → resize, permute survivor rows via `id_remap.json`, init new rows
  `[pruned_size, final_n)`. (`tokenizer_path` must be extend's `replace/` output.)

## Dependencies
`method: focus` needs `fasttext-wheel`
(`uv pip install -e '.[tokenizer-extension]'`, or add it to the profile's
`pip_extras`). The import is lazy, so the other methods run without it.
See `../guide.md`.

## Run
```bash
uv run nemotron steps run tokenizer_extension/init_embeddings \
  -b lepton_tokenizer_init_embeddings -c default \
  language=<your-language> arm=add extended_tokenizer=./output/tokenizer_extension/add
```
Output: `output_dir/` = resized HF checkpoint (weights + tokenizer) → set as
`pretrain/megatron_bridge` `hf_model_path`.
