# SKILL: BYOB MCQ (this package)

## Purpose

**Executable workflow** for building a domain multiple-choice benchmark from custom text, using the code in this directory. Use with `STEP.md` (what each stage does), `types.toml` (how artifacts connect), `patterns/*.md` (optional-stage heuristics), and `context/*.txt` (deeper reference).

## When to use

- You are in `nemotron.data_prep.stages.byob` to generate or harden an MCQ dataset from corpora and optional MMLU-style few-shot data.
- You have or will create per-subject `.txt` data and a `default.yaml` (or equivalent) describing paths, models, and feature flags.

## Main loop (human or agent)

1. **Prepare data** — `ByobConfig.from_yaml`, `make_from_config`, `sample_and_dump` (see the use-case notebook below).
2. **Generate** — `generate_questions` + postprocess; writes `generated_questions.parquet` under `output_dir/…/stage_cache/`.
3. **Judge** — `judge_questions` → `judged_questions.parquet`.
4. **Deduplicate** — `TextSemanticDeduplicationMCQ` → `semantic_deduplicated_questions.parquet`.
5. **Optional** — Distractor expansion and coverage (YAML-gated) → their parquets; otherwise chain stays on the previous path.
6. **Validity, outliers, filter** — as in `build_mcq_benchmark.ipynb` Section 2.
7. **Export** — `benchmark.parquet` / `benchmark_raw.parquet` at the experiment output root.

**Primary runnable reference:**  
`Nemotron/use-case-examples/build-your-own-benchmark/build_mcq_benchmark.ipynb`  
**Config next to the notebook:** `use-case-examples/build-your-own-benchmark/default.yaml`  

**Customization recipe (higher-level narrative):**  
`src/nemotron/customization_recipes/nemotron/stage4_byob/SKILL.md`

## If something fails

- Check `input_dir` and that `NEMOTRON` `src` is on `sys.path` (see the example notebook’s first cell).
- Confirm optional stages match YAML; see `STEP.md` and `patterns/`.

## Eval

Golden shapes for stages live under `eval/`. Add cases as your pipeline stabilizes.
