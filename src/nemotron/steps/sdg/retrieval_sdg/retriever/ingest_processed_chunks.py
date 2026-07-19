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

"""Step 3: embed chunks.jsonl and build the LanceDB index — NeMo Retriever, directly.

Resumable bulk ingest: each row needs `text`; rows are embedded with the retriever's
embedding NIM and appended as LanceDB rows in the NeMo schema, then the vector index is
built. Checkpoints after every batch (.ingest-state.json) and writes .ingest-complete
once the index exists. Env knobs (INPUT_JSONL, LANCEDB_URI, TABLE_NAME, MODEL_NAME,
EMBEDDING_ENDPOINT, INGEST_/EMBED_/INDEX_* sizes) are documented in README.md.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lancedb
import pandas as pd

from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
from nemo_retriever.vdb.lancedb_bulk import LanceDBConfig, create_lancedb_index
from nemo_retriever.vdb.lancedb_schema import build_lancedb_rows, lancedb_schema

env = os.environ.get
INPUT = Path(env("INPUT_JSONL", "./data/chunks.jsonl"))
DB_URI = env("LANCEDB_URI", "./lancedb")
TABLE = env("TABLE_NAME", "hindi-legal-judgments")
MODEL = env("MODEL_NAME", "nvidia/Nemotron-3-Embed-1B-BF16")
EMBED_URL = env("EMBEDDING_ENDPOINT", "http://127.0.0.1:8001/v1/embeddings")
BATCH = int(env("INGEST_BATCH_SIZE", "1024"))
STATE = Path(DB_URI) / ".ingest-state.json"
MARKER = Path(DB_URI) / ".ingest-complete"


def _load_processed() -> int:
    return int(json.loads(STATE.read_text())["processed"]) if STATE.exists() else 0


def _save_processed(n: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")            # atomic checkpoint
    tmp.write_text(json.dumps({"processed": n}) + "\n")
    tmp.replace(STATE)


def _embed(rows: list[dict]) -> pd.DataFrame:
    def path(r): return str(r.get("source_path") or r.get("path") or "")
    df = pd.DataFrame({
        "text": [str(r.get("text") or "") for r in rows],
        "path": [path(r) for r in rows],
        "page_number": [int(r.get("page_number") or -1) for r in rows],
        "_page_number": [int(r.get("page_number") or -1) for r in rows],
        "_content_type": ["text"] * len(rows),
        "metadata": [{**(r.get("metadata") if isinstance(r.get("metadata"), dict) else {}),
                      "source_path": path(r), "chunk_id": r.get("chunk_id")} for r in rows],
    })
    return embed_text_main_text_embed(
        df, model_name=MODEL, embedding_endpoint=EMBED_URL, input_type="passage",
        inference_batch_size=int(env("EMBED_BATCH_SIZE", "32")),
        nim_http_max_concurrent=int(env("EMBED_CONCURRENCY", "32")), request_timeout_s=600)


def _ingest(rows: list[dict], processed: int) -> int:
    lance_rows = build_lancedb_rows(_embed(rows))
    db = lancedb.connect(DB_URI)
    if processed == 0:
        db.create_table(TABLE, data=lance_rows, schema=lancedb_schema(vector_dim=2048), mode="overwrite")
    else:
        db.open_table(TABLE).add(lance_rows)
    processed += len(rows)
    _save_processed(processed)
    return processed


def main() -> None:
    if MARKER.exists():
        print(f"index already complete: {MARKER}", flush=True)
        return

    processed = _load_processed()
    started = time.monotonic()
    batch: list[dict] = []
    with INPUT.open(encoding="utf-8") as src:
        for _ in range(processed):             # resume: skip already-ingested rows
            if not src.readline():
                break
        for line in src:
            batch.append(json.loads(line))
            if len(batch) >= BATCH:
                processed = _ingest(batch, processed)
                batch.clear()
                print(f"processed={processed} rate={processed / max(time.monotonic() - started, 1e-3):.1f}/s", flush=True)
        if batch:
            processed = _ingest(batch, processed)

    table = lancedb.connect(DB_URI).open_table(TABLE)
    print(f"building vector index for {table.count_rows()} rows", flush=True)
    create_lancedb_index(table, cfg=LanceDBConfig(
        uri=DB_URI, table_name=TABLE, overwrite=False, create_index=True,
        num_partitions=int(env("INDEX_PARTITIONS", "256")),
        num_sub_vectors=int(env("INDEX_SUB_VECTORS", "256"))))
    MARKER.write_text(json.dumps({"rows": table.count_rows()}) + "\n")
    print(f"complete rows={table.count_rows()}", flush=True)


if __name__ == "__main__":
    main()
