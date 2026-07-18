"""OPTIONAL step 1: extract text from PDFs using NeMo Retriever, directly.

Thin, resumable driver around ``nemo_retriever.pdf.extract.pdf_extraction`` — no
wrapper logic; it only does file I/O, checkpointing, and calls the NeMo function.
Reads a directory of PDFs and writes one extracted page-record per line to a JSONL
that ``chunk.py`` then splits.

  INPUT_PDF_DIR  → extract.py → EXTRACT_OUTPUT_JSONL → chunk.py → chunks.jsonl → ingest

Env knobs: INPUT_PDF_DIR, EXTRACT_OUTPUT_JSONL, and EXTRACT_TABLES/EXTRACT_CHARTS/
EXTRACT_INFOGRAPHICS ("1"/"0"), EXTRACT_TEXT_METHOD (default pdfium_hybrid).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nemo_retriever.pdf.extract import pdf_extraction

INPUT_PDF_DIR = Path(os.getenv("INPUT_PDF_DIR", "./data/pdfs"))
OUTPUT_JSONL = Path(os.getenv("EXTRACT_OUTPUT_JSONL", "./data/processed/extraction/extracted.jsonl"))
TEXT_METHOD = os.getenv("EXTRACT_TEXT_METHOD", "pdfium_hybrid")


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


STATE = OUTPUT_JSONL.with_suffix(".state.json")
MARKER = OUTPUT_JSONL.with_suffix(".complete")


def _pdf_paths() -> list[Path]:
    return sorted(p for p in INPUT_PDF_DIR.rglob("*") if p.suffix.lower() == ".pdf")


def load_state() -> int:
    return int(json.loads(STATE.read_text())["processed"]) if STATE.exists() else 0


def save_state(processed: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed": processed}) + "\n")
    tmp.replace(STATE)


def main() -> None:
    if MARKER.exists():
        print(f"extraction already complete: {MARKER}", flush=True)
        return
    pdfs = _pdf_paths()
    processed = load_state()
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if processed else "w"
    with OUTPUT_JSONL.open(mode, encoding="utf-8") as out:
        for i in range(processed, len(pdfs)):
            path = pdfs[i]
            records = pdf_extraction(
                path.read_bytes(),
                extract_text=True,
                extract_tables=_flag("EXTRACT_TABLES"),
                extract_charts=_flag("EXTRACT_CHARTS"),
                extract_infographics=_flag("EXTRACT_INFOGRAPHICS"),
                text_extraction_method=TEXT_METHOD,
            )
            for rec in records:
                # pdf_extraction gets bytes, so stamp the real path for provenance
                rec.setdefault("source_path", str(path))
                rec.setdefault("path", str(path))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            save_state(i + 1)
            if (i + 1) % 50 == 0:
                print(f"extracted {i + 1}/{len(pdfs)} pdfs", flush=True)
    MARKER.write_text(json.dumps({"pdfs": len(pdfs)}) + "\n")
    print(f"complete: extracted {len(pdfs)} pdfs -> {OUTPUT_JSONL}", flush=True)


if __name__ == "__main__":
    main()
