from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lancedb
import pandas as pd

from nemo_retriever.text_embed.runtime import embed_text_main_text_embed
from nemo_retriever.vdb.lancedb_bulk import (
    LanceDBConfig,
    create_lancedb_index,
)
from nemo_retriever.vdb.lancedb_schema import build_lancedb_rows, lancedb_schema


INPUT = Path(
    os.getenv(
        "INPUT_JSONL",
        "/localhome/local-rkalani/hindi-legal-agent/data/processed/extraction/chunks.jsonl",
    )
)
DB_URI = os.getenv(
    "LANCEDB_URI", "/localhome/local-rkalani/nemo-retriever/lancedb-hindi-legal"
)
TABLE_NAME = os.getenv("TABLE_NAME", "hindi-legal-judgments")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-nemotron-embed-1b-v2")
EMBEDDING_ENDPOINT = os.getenv(
    "EMBEDDING_ENDPOINT", "http://127.0.0.1:8001/v1/embeddings"
)
BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "32"))
INDEX_PARTITIONS = int(os.getenv("INDEX_PARTITIONS", "256"))
INDEX_SUB_VECTORS = int(os.getenv("INDEX_SUB_VECTORS", "256"))
STATE = Path(DB_URI) / ".ingest-state.json"
MARKER = Path(DB_URI) / ".ingest-complete"


def load_state() -> int:
    if not STATE.exists():
        return 0
    return int(json.loads(STATE.read_text())["processed"])


def save_state(processed: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"processed": processed}) + "\n")
    temporary.replace(STATE)


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [str(row.get("text") or "") for row in rows],
            "path": [str(row.get("source_path") or row.get("path") or "") for row in rows],
            "page_number": [int(row.get("page_number") or -1) for row in rows],
            "_page_number": [int(row.get("page_number") or -1) for row in rows],
            "_content_type": ["text"] * len(rows),
            "metadata": [
                {
                    **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
                    "source_path": str(row.get("source_path") or row.get("path") or ""),
                    "chunk_id": row.get("chunk_id"),
                }
                for row in rows
            ],
        }
    )


def embed(df: pd.DataFrame) -> pd.DataFrame:
    return embed_text_main_text_embed(
        df,
        model_name=MODEL_NAME,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        input_type="passage",
        inference_batch_size=EMBED_BATCH_SIZE,
        nim_http_max_concurrent=EMBED_CONCURRENCY,
        request_timeout_s=600,
    )


def process_batch(rows: list[dict], processed: int) -> int:
    embedded = embed(rows_to_dataframe(rows))
    lance_rows = build_lancedb_rows(embedded)
    db = lancedb.connect(DB_URI)
    if processed == 0:
        db.create_table(
            TABLE_NAME,
            data=lance_rows,
            schema=lancedb_schema(vector_dim=2048),
            mode="overwrite",
        )
    else:
        db.open_table(TABLE_NAME).add(lance_rows)
    processed += len(rows)
    save_state(processed)
    return processed


def main() -> None:
    if MARKER.exists():
        print(f"Hindi legal index is already complete: {MARKER}", flush=True)
        return

    processed = load_state()
    started = time.monotonic()
    batch: list[dict] = []

    with INPUT.open(encoding="utf-8") as source:
        for _ in range(processed):
            if not source.readline():
                break

        for line in source:
            batch.append(json.loads(line))
            if len(batch) < BATCH_SIZE:
                continue
            processed = process_batch(batch, processed)
            batch.clear()
            elapsed = time.monotonic() - started
            print(
                f"processed={processed} rate={processed / max(elapsed, 0.001):.2f} chunks/s",
                flush=True,
            )

        if batch:
            processed = process_batch(batch, processed)

    db = lancedb.connect(DB_URI)
    table = db.open_table(TABLE_NAME)
    cfg = LanceDBConfig(
        uri=DB_URI,
        table_name=TABLE_NAME,
        overwrite=False,
        create_index=True,
        num_partitions=INDEX_PARTITIONS,
        num_sub_vectors=INDEX_SUB_VECTORS,
    )
    print(f"building vector index for {table.count_rows()} rows", flush=True)
    create_lancedb_index(table, cfg=cfg)
    MARKER.write_text(json.dumps({"rows": table.count_rows()}) + "\n")
    print(f"complete rows={table.count_rows()}", flush=True)


if __name__ == "__main__":
    main()
