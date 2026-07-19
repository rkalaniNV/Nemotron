# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Step 2: chunk extracted text into passages using NeMo Retriever, directly.

Thin, resumable driver around ``nemo_retriever.txt.split.split_df`` — no wrapper
logic; only I/O, checkpointing, and a stable ``chunk_id`` stamp. Reads the extracted
JSONL (from extract.py, or any JSONL with a ``text`` field) and writes ``chunks.jsonl``
in the schema ``ingest_processed_chunks.py`` consumes.

  extracted.jsonl → chunk.py → chunks.jsonl → ingest_processed_chunks.py

Env knobs: CHUNK_INPUT_JSONL, CHUNK_OUTPUT_JSONL, CHUNK_MAX_TOKENS (1024),
CHUNK_OVERLAP_TOKENS (0), CHUNK_TOKENIZER_MODEL_ID (HF id; blank = NeMo default),
CHUNK_BATCH_SIZE (input rows per split_df call).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from nemo_retriever.txt.split import split_df

INPUT_JSONL = Path(os.getenv("CHUNK_INPUT_JSONL", "./data/processed/extraction/extracted.jsonl"))
OUTPUT_JSONL = Path(os.getenv("CHUNK_OUTPUT_JSONL", "./data/processed/extraction/chunks.jsonl"))
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))
OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "0"))
TOKENIZER_MODEL_ID = os.getenv("CHUNK_TOKENIZER_MODEL_ID", "") or None
BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "2000"))

STATE = OUTPUT_JSONL.with_suffix(".chunk-state.json")
MARKER = OUTPUT_JSONL.with_suffix(".chunk-complete")


def load_state() -> int:
    return int(json.loads(STATE.read_text())["processed"]) if STATE.exists() else 0


def save_state(processed: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed": processed}) + "\n")
    tmp.replace(STATE)


def _chunk_id(path: str, page: int, text: str) -> str:
    return hashlib.sha1(f"{path}|{page}|{text}".encode("utf-8")).hexdigest()


def _split(batch: list[dict]) -> pd.DataFrame:
    # split_df overwrites page_number with the chunk ordinal, so stash the REAL page in
    # metadata first (restored in _write). Assumes split_df carries input metadata through.
    for rec in batch:
        md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        md.setdefault("source_page", int(rec.get("page_number") or -1))
        rec["metadata"] = md
    return split_df(pd.DataFrame(batch), max_tokens=MAX_TOKENS,
                    overlap_tokens=OVERLAP_TOKENS, tokenizer_model_id=TOKENIZER_MODEL_ID)


def _write(rows: pd.DataFrame, out) -> None:
    for r in rows.to_dict(orient="records"):
        text = str(r.get("text") or "")
        path = str(r.get("path") or r.get("source_path") or "")
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        page = int(meta.get("source_page", r.get("page_number") or -1))   # real PDF page, restored
        cid = _chunk_id(path, page, text)
        # source_id = document path (shared across a doc's chunks) so cross_doc grouping works
        out.write(json.dumps({
            "text": text, "source_id": path, "source_path": path, "path": path,
            "page_number": page, "chunk_id": cid,
            "metadata": {**meta, "source_path": path, "chunk_id": cid},
        }, ensure_ascii=False) + "\n")


def main() -> None:
    if MARKER.exists():
        print(f"chunking already complete: {MARKER}", flush=True)
        return
    processed = load_state()
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if processed else "w"
    total = 0
    with INPUT_JSONL.open(encoding="utf-8") as src, OUTPUT_JSONL.open(mode, encoding="utf-8") as out:
        for _ in range(processed):                 # resume: skip already-chunked input rows
            if not src.readline():
                break
        batch: list[dict] = []
        for line in src:
            line = line.strip()
            if line:
                batch.append(json.loads(line))
            if len(batch) < BATCH_SIZE:
                continue
            _write(_split(batch), out)
            processed += len(batch); total += len(batch); batch.clear()
            save_state(processed)
            print(f"chunked {processed} input rows", flush=True)
        if batch:
            _write(_split(batch), out)
            processed += len(batch); save_state(processed)
    MARKER.write_text(json.dumps({"input_rows": processed}) + "\n")
    print(f"complete: chunked {processed} input rows -> {OUTPUT_JSONL}", flush=True)


if __name__ == "__main__":
    main()
