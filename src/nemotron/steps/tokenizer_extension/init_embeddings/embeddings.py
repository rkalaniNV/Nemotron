#!/usr/bin/env python3
"""Config-driven embedding initialization -> resized HF checkpoint for CPT.

This is a thin dispatcher over three vendored, self-contained init engines
(Ravi Rajaj's tokeniser-extend reference), kept verbatim so their tested
norm-correction / validation logic is preserved:

  * baseline_init.py  (method: baseline)
        hf_default | mean_all | mean_hindi   [+ optional input norm-correction]
  * subword_init.py   (method: subword)
        decompose every new token into base subwords and average their rows;
        averaging = uniform | char_weighted | max_char | bert_weighted |
        gemma_weighted, chosen independently for the input and output side,
        with per-side (optionally Hindi-only) norm correction.
  * focus_init.py     (method: focus)
        FOCUS (Dobler & de Melo 2023): sparsemax-weighted combination of the
        base tokens closest to each new token in a fastText auxiliary space.

Each engine exposes ``main(argv)`` (argparse). ``run_init(cfg)`` translates the
YAML config into that engine's argv and invokes it. The engines are imported
LAZILY inside each branch so that e.g. selecting `subword` never requires
`fasttext` (only `focus` does).

Domain note — ADD vs REPLACE:
  These engines assume an *append-style* extended tokenizer: rows [:base_vocab]
  are the untouched base embeddings and [base_vocab:] are the appended new
  tokens (`new_token_ids = range(base_vocab, new_vocab)`). That is exactly the
  ADD arm. The REPLACE arm densely re-indexes surviving ids via id_remap.json,
  which violates that assumption, so REPLACE needs a remap-aware wrapper (it
  must permute survivor rows before the append-style init runs). `run_init`
  refuses a replace tokenizer rather than silently producing a wrong checkpoint.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

METHODS = ("baseline", "subword", "focus")


def _flag(name: str, value: bool) -> str:
    """argparse BooleanOptionalAction: True -> --name, False -> --no-name."""
    return f"--{name}" if value else f"--no-{name}"


def _common_argv(cfg: dict) -> list[str]:
    return [
        "--base-model", str(cfg["base_model"]),
        "--extended-tokenizer", str(cfg["extended_tokenizer"]),
        "--output-dir", str(cfg["output_dir"]),
        "--dtype", str(cfg.get("dtype", "bfloat16")),
    ]


def _baseline_argv(cfg: dict) -> list[str]:
    b = cfg.get("baseline", {}) or {}
    argv = _common_argv(cfg)
    argv += ["--mode", str(b.get("mode", "mean_hindi"))]
    argv += [_flag("norm-correction", bool(b.get("norm_correction", True)))]
    argv += ["--num-samples", str(int(b.get("num_samples", 10)))]
    return argv


def _subword_argv(cfg: dict) -> list[str]:
    s = cfg.get("subword", {}) or {}
    argv = _common_argv(cfg)
    argv += ["--input-averaging", str(s.get("input_averaging", "uniform"))]
    argv += ["--output-averaging", str(s.get("output_averaging", "uniform"))]
    argv += [_flag("input-norm-correction", bool(s.get("input_norm_correction", True)))]
    argv += [_flag("output-norm-correction", bool(s.get("output_norm_correction", False)))]
    argv += [_flag("input-hindi-norm", bool(s.get("input_hindi_norm", True)))]
    argv += [_flag("output-hindi-norm", bool(s.get("output_hindi_norm", False)))]
    argv += ["--temperature", str(float(s.get("temperature", 0.1)))]
    argv += ["--bert-model", str(s.get("bert_model", "google/muril-base-cased"))]
    argv += ["--gemma-model", str(s.get("gemma_model", "google/gemma-2-27b"))]
    argv += ["--num-samples", str(int(s.get("num_samples", 10)))]
    return argv


def _focus_argv(cfg: dict) -> list[str]:
    f = cfg.get("focus", {}) or {}
    if not f.get("fasttext_model"):
        raise ValueError("method=focus requires focus.fasttext_model (a fastText .bin).")
    argv = _common_argv(cfg)
    argv += ["--fasttext-model", str(f["fasttext_model"])]
    if f.get("fasttext_url"):
        argv += ["--fasttext-url", str(f["fasttext_url"])]
    argv += ["--candidate-pool", str(f.get("candidate_pool", "hindi"))]
    argv += ["--sparsemax-temperature", str(float(f.get("sparsemax_temperature", 0.05)))]
    argv += ["--num-samples", str(int(f.get("num_samples", 10)))]
    argv += ["--top-contributors", str(int(f.get("top_contributors", 5)))]
    return argv


def _replace_argv(cfg: dict) -> list[str]:
    """Map the add-style config (method + sub-blocks) to replace_init's --method dispatch.

    baseline(hf_default|mean_all) -> --method hf_default|mean_all
    focus                         -> --method focus (+ fastText args)
    subword(bert_weighted)        -> --method bert  (+ MuRIL args)
    subword(uniform|char|max)     -> --method subword (length-based)
    """
    method = str(cfg.get("method", "subword")).lower()
    argv = _common_argv(cfg)
    if method == "baseline":
        mode = str((cfg.get("baseline") or {}).get("mode", "mean_all"))
        if mode not in ("hf_default", "mean_all"):
            raise ValueError(f"arm=replace baseline supports mode hf_default|mean_all, got {mode!r}")
        argv += ["--method", mode]
    elif method == "focus":
        f = cfg.get("focus") or {}
        if not f.get("fasttext_model"):
            raise ValueError("arm=replace method=focus requires focus.fasttext_model")
        argv += ["--method", "focus", "--fasttext-model", str(f["fasttext_model"]),
                 "--candidate-pool", str(f.get("candidate_pool", "hindi")),
                 "--sparsemax-temperature", str(float(f.get("sparsemax_temperature", 0.05)))]
        if f.get("fasttext_url"):
            argv += ["--fasttext-url", str(f["fasttext_url"])]
    else:  # subword family
        s = cfg.get("subword") or {}
        ia = str(s.get("input_averaging", "uniform"))
        if ia in ("bert_weighted", "gemma_weighted"):
            argv += ["--method", "bert", "--bert-model", str(s.get("bert_model", "google/muril-base-cased")),
                     "--temperature", str(float(s.get("temperature", 0.1)))]
        else:
            argv += ["--method", "subword", "--input-averaging", ia,
                     "--output-averaging", str(s.get("output_averaging", "uniform"))]
        argv += [_flag("input-norm-correction", bool(s.get("input_norm_correction", True)))]
        argv += [_flag("output-norm-correction", bool(s.get("output_norm_correction", False)))]
        argv += [_flag("input-hindi-norm", bool(s.get("input_hindi_norm", True)))]
        argv += [_flag("output-hindi-norm", bool(s.get("output_hindi_norm", False)))]
    return argv


def _guard_add(cfg: dict) -> None:
    """On the add (append-style) path, refuse a replace tokenizer (has id_remap.json)."""
    tok = Path(str(cfg["extended_tokenizer"]))
    if (tok / "id_remap.json").exists():
        raise ValueError(
            f"{tok} contains id_remap.json (a replace tokenizer) but arm=add. Append-style "
            "init would mis-place survivor rows. Set arm=replace for this tokenizer."
        )


def run_init(cfg: dict) -> None:
    method = str(cfg.get("method", "subword")).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    for key in ("base_model", "extended_tokenizer", "output_dir"):
        if not cfg.get(key):
            raise ValueError(f"config is missing required key: {key}")

    arm = str(cfg.get("arm", "add")).lower()
    if arm not in ("add", "replace"):
        raise ValueError(f"arm must be 'add' or 'replace', got {arm!r}")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # REPLACE: survivor rows are re-indexed (id_remap.json), so replace_init copies
    # them exactly and inits the fresh tokens by the chosen method (hf_default /
    # mean_all / focus / bert / subword), then pads the vocab for TP divisibility.
    if arm == "replace":
        argv = _replace_argv(cfg)
        log.info("MILESTONE: REPLACE init (survivor remap, method=%s) | argv=%s", method, " ".join(argv))
        import replace_init as engine
        engine.main(argv)
        log.info("MILESTONE: DONE — resized HF checkpoint -> %s", cfg["output_dir"])
        return

    # ADD (append-style): the vendored engines assume base rows are unchanged.
    _guard_add(cfg)
    if method == "baseline":
        argv = _baseline_argv(cfg)
        log.info("MILESTONE: baseline init | argv=%s", " ".join(argv))
        import baseline_init as engine
    elif method == "subword":
        argv = _subword_argv(cfg)
        log.info("MILESTONE: subword init | argv=%s", " ".join(argv))
        import subword_init as engine
    else:  # focus
        argv = _focus_argv(cfg)
        log.info("MILESTONE: FOCUS init | argv=%s", " ".join(argv))
        import focus_init as engine

    engine.main(argv)
    log.info("MILESTONE: DONE — resized HF checkpoint -> %s", cfg["output_dir"])
