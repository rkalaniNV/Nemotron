---
# Structural library: what BYOB (MCQ) can do
id: byob-mcq
version: 1
contract: six-block
stages:
  - id: prepare
    name: prepare_data
    outputs: [seed_artifacts, parquet]
  - id: generate
    name: generate_questions
    requires: [ByobConfig, generation seed]
  - id: judge
    name: judge_questions
  - id: deduplicate
    name: TextSemanticDeduplicationMCQ
  - id: expand_distractors
    name: expand_distractors
    optional: do_distractor_expansion
  - id: coverage
    name: TextCoverageMCQ
    optional: do_coverage_check
  - id: distractor_validity
    name: check_distractor_validity
  - id: outlier
    name: TextSemanticOutlierDetectionMCQ
  - id: filter
    name: filter_questions
  - id: export
    name: final_benchmark
    outputs: [benchmark.parquet]
---

# BYOB step library (MCQ)

Nemotron’s **Build Your Own Benchmark (BYOM)** code here turns domain text and optional few-shot sources into a scored MCQ `benchmark.parquet`. Each *step* is implemented as a Python entry point under `mcq/` (see `mcq/stages.py`, `config.py`).

**Typical errors**

- Missing or wrong `input_dir` (no per-subject `.txt` layout expected by the YAML). Fix paths or run the Wikipedia example notebook in `use-case-examples/build-your-own-benchmark/`.
- `NEMOTRON_SRC` not on `sys.path` or wrong kernel working directory; the example notebook sets repo `src` from `cwd`.
- NeMo Data Designer / NIM or model config mismatch: see `ByobConfig` in `config.py` and the recipe `customization_recipes/nemotron/stage4_byob/`.

**Strategies**

- Disable heavy optional steps in YAML when iterating (`do_distractor_expansion`, `do_coverage_check`).
- Tune `easiness_threshold` and `hallucination_threshold` before re-running filtering.

For end-to-end usage from an agent, start with `SKILL.md` in this directory, then this file for stage names and errors.
