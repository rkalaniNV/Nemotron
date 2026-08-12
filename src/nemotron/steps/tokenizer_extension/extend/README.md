# tokenizer_extension/extend

Train one BPE on a corpus and splice it into the base tokenizer as **Add** and/or
**Replace**. CPU-only.

## Files
- `step.py` — executor (thin; loads YAML, calls `extension.run_extension`)
- `extension.py` — corpus reader (HF / local parquet / jsonl) + the one-BPE→both-arms orchestration
- `continued_bpe.py` — shared core: streaming, merge-diff, **constructive splice**, rank-dead check
- `replace_bpe.py` — prune + dense re-index + NFKC-preserving wrap (Replace path)

## Corpus input (config `corpus:`)
Provide **either** an HF dataset **or** a local path, plus the text column:
```yaml
corpus:
  hf_dataset: ai4bharat/sangraha   # streamed;  OR set path: below
  hf_config: verified
  hf_split: hin_Deva
  path: null                       # local parquet dir/glob or jsonl
  glob: "*.parquet"
  text_field: text                 # the text column
  samples: 1000000
  max_doc_chars: 20000
  min_frequency: 3
```

## Add vs Replace
`method: add | replace | both`. Add keeps the base residual script tokens; Replace
prunes them (`remove_script`) and refills with corpus-optimal tokens. See the
category `guide.md`.

## Why the splice is safe
Naively appending trained merges below the base merges can leave tokens in the
vocab that greedy BPE can never emit (they get shadowed by a higher-priority base
rule) — which silently inflates fertility while every graph-reachability check
still passes. `continued_bpe._apply_bpe_extension_backend` instead builds each new
token from the pieces the *current* merge table yields, so the appended rule is the
one that fires. `find_rank_dead_tokens` asserts zero unemittable tokens; the build
**hard-fails** otherwise.

## Run
```bash
uv run nemotron steps run tokenizer_extension/extend -c default --batch <cpu_profile>
```
Output: `output_dir/{add,replace}/` (tokenizers) + `summary.json`.
