#!/usr/bin/env python3
"""Runner for the agentic-RAG deep-research SDG step.

Loads the outer YAML, wires Data Designer (models + samplers + the plugin
column), and runs preview/create. Mirrors the reference notebook's
``DataDesignerConfigBuilder`` flow so the pipeline is fully config-driven:
this file contains NO domain logic — it just translates the YAML into DD calls.

    python step.py --config config/indian_legal.yaml --mode preview
    python step.py --config config/indian_legal.yaml --mode create

Prerequisites (corpus side, run once — see data_prep/):
  chunk_document.py  -> data/constitution_chunks.jsonl
  retriever.py build -> data/index
  bundle_builder.py  -> data/bundles.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent

# Fallback personas (used only when the Nemotron-Personas managed asset for the
# configured locale is not installed locally — keeps the pipeline runnable).
_FALLBACK_PERSONAS = [
    {"first_name": "Asha", "last_name": "Rao", "age": 34, "sex": "female",
     "occupation": "schoolteacher", "city": "Pune", "state": "Maharashtra",
     "persona": "A practical person seeking clear answers about her rights."},
    {"first_name": "Vikram", "last_name": "Singh", "age": 45, "sex": "male",
     "occupation": "small business owner", "city": "Jaipur", "state": "Rajasthan",
     "persona": "Detail-oriented; wants precise legal provisions and cross-references."},
    {"first_name": "Meera", "last_name": "Nair", "age": 27, "sex": "female",
     "occupation": "law student", "city": "Kochi", "state": "Kerala",
     "persona": "A curious learner who asks comparative, conceptual questions."},
    {"first_name": "Arjun", "last_name": "Das", "age": 52, "sex": "male",
     "occupation": "journalist", "city": "Kolkata", "state": "West Bengal",
     "persona": "Skeptical; probes edge cases, exceptions, and limits."},
]


def _persona_asset_available(locale: str) -> bool:
    return (Path.home() / ".data-designer" / "managed-assets" / "datasets" / f"{locale}.parquet").exists()


# ── seed enrichment: attach anchor text + the constant tool catalog ──────────
def _resolve(base: Path, p: str) -> Path:
    """Resolve a YAML path relative to the config file's directory."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def enrich_seed(cfg: Dict[str, Any], base: Path, out_path: Path, include_persona: bool) -> Path:
    """Bundles carry chunk ids; the generator needs the anchor's text and the
    tool schemas per row. Join them here so the seed is self-contained. When the
    persona managed-asset is unavailable, attach a fallback persona per row."""
    bundles = [json.loads(l) for l in _resolve(base, cfg["seed_dataset"]["path"]).open() if l.strip()]
    corpus_path = _plugin_spec(cfg).get("corpus_path")
    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in _resolve(base, corpus_path).open() if l.strip())}
    tools_json = json.dumps(cfg["tools"], ensure_ascii=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, b in enumerate(bundles):
            anchor = chunks.get(b.get("anchor_id"), {})
            b["anchor_text"] = anchor.get("text", "")
            row = {"bundle": json.dumps(b, ensure_ascii=False), "tools": tools_json}
            if include_persona:
                row["persona"] = json.dumps(_FALLBACK_PERSONAS[i % len(_FALLBACK_PERSONAS)], ensure_ascii=False)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path


def _plugin_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    for col in cfg["columns"]:
        if col.get("type") == "deep-research-simulator":
            return col
    raise ValueError("no deep-research-simulator column in config")


# ── DD wiring (lazy imports so the file is importable without DD) ────────────
def build_config_builder(cfg: Dict[str, Any], seed_path: Path, base: Path, use_persona_sampler: bool):
    import data_designer.config as dd
    from agentic_rag.config import DeepResearchSimulatorConfig

    model_configs = [
        dd.ModelConfig(
            alias=m["alias"], model=m["model"], provider=m.get("provider", "nvidia"),
            inference_parameters=dd.ChatCompletionInferenceParams(
                **{k: v for k, v in m.get("inference_parameters", {}).items()}),
        )
        for m in cfg["models"]
    ]

    cb = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    cb.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)),
                         sampling_strategy=dd.SamplingStrategy.SHUFFLE)

    # persona: real Nemotron-Personas sampler when the asset is installed,
    # otherwise personas are provided by the enriched seed (fallback).
    if use_persona_sampler:
        locale = cfg.get("persona", {}).get("locale", "en_IN")
        cb.add_column(dd.SamplerColumnConfig(
            name="persona", drop=True, sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(locale=locale, with_synthetic_personas=True)))

    # declarative sampler columns from YAML (archetype / outcome / depth / theme / ...)
    for col in cfg["columns"]:
        t = col.get("type")
        if t == "category":
            cb.add_column(dd.SamplerColumnConfig(
                name=col["name"], sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=col["values"])))
        elif t == "seed":
            continue  # seed fields are surfaced automatically
        elif t == "deep-research-simulator":
            knobs = {k: v for k, v in col.items() if k not in ("type", "name")}
            # resolve corpus/index paths to absolute so the generator finds them at runtime
            for pk in ("corpus_path", "index_dir"):
                if knobs.get(pk):
                    knobs[pk] = str(_resolve(base, knobs[pk]))
            cb.add_column(DeepResearchSimulatorConfig(name=col["name"], **knobs))
    return cb


# ── output projection: OpenAI tool-calling trajectory for SFT ────────────────
def project_and_write(records: List[Dict[str, Any]], cfg: Dict[str, Any], out_path: Path) -> int:
    proj = cfg["output_projection"]
    src = proj["source_field"]
    meta_fields = proj.get("metadata_fields", [])
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            if not r.get("conversation_status"):
                continue  # keep only successful / salvaged trajectories
            try:
                messages = json.loads(r[src]) if isinstance(r[src], str) else r[src]
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            row = {"messages": messages, "tools": json.loads(cfg_tools(cfg))}
            for mf in meta_fields:
                if mf in r:
                    row[mf] = r[mf]
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            kept += 1
    return kept


def cfg_tools(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg["tools"], ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agentic-RAG deep-research SDG runner.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["preview", "create"], default="preview")
    ap.add_argument("--num-records", type=int, default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    base = args.config.resolve().parent
    locale = cfg.get("persona", {}).get("locale", "en_IN")
    use_persona_sampler = _persona_asset_available(locale)
    if not use_persona_sampler:
        print(f"[agentic-rag] persona asset '{locale}.parquet' not found — using fallback seed personas.")
    seed_path = enrich_seed(cfg, base, HERE / "data" / "_enriched_seed.jsonl",
                            include_persona=not use_persona_sampler)
    cb = build_config_builder(cfg, seed_path, base, use_persona_sampler)

    import data_designer.config as dd
    from data_designer.interface import DataDesigner
    providers = [dd.ModelProvider(**p) for p in cfg.get("providers", [])]
    client = DataDesigner(model_providers=providers) if providers else DataDesigner()

    if args.mode == "preview":
        result = client.preview(cb)
    else:
        n = args.num_records or cfg.get("num_records", 10)
        result = client.create(cb, num_records=n)

    records = _records(result)
    out = Path(cfg["output_path"])

    # raw dump: every generated row (incl. failed/salvaged) for inspection/debug
    raw = out.with_suffix(".raw.jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    kept = project_and_write(records, cfg, out)
    print(f"[agentic-rag] {args.mode}: {len(records)} generated, {kept} kept")
    print(f"  training file (status=ok): {out}")
    print(f"  raw dump (all rows):       {raw}")


def _records(result) -> List[Dict[str, Any]]:
    for attr in ("dataset", "records", "data"):
        obj = getattr(result, attr, None)
        if obj is not None:
            return obj.to_dict("records") if hasattr(obj, "to_dict") else list(obj)
    return list(result)


if __name__ == "__main__":
    main()
