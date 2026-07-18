# retriever

Stand up NeMo Retriever over your own corpus and serve it to `retrieval_sdg`. Every
step is a thin script that calls `nemo_retriever` functions directly — the serving and
ingestion glue NeMo does not ship.

## What it does

1. *(optional)* Extracts text from PDFs.
2. Chunks text into token-sized passages.
3. Embeds the chunks and builds a LanceDB vector index.
4. Serves retrieval over HTTP: `POST /query {"query": "...", "num_chunks": 5}` (`top_k` aliases `num_chunks`).

```
your files → [extract.py] → [chunk.py] → chunks.jsonl → [ingest_processed_chunks.py] → LanceDB → [server.py] :8002
```

If your corpus is already chunked (a JSONL with a `text` field per row), skip
`extract.py`/`chunk.py`.

## Files

| File | Purpose | NeMo call |
|---|---|---|
| `extract.py` | *(optional)* PDFs → extracted text JSONL | `pdf.extract.pdf_extraction` |
| `chunk.py` | text → `chunks.jsonl` (stable `chunk_id`) | `txt.split.split_df` |
| `ingest_processed_chunks.py` | embed + build the LanceDB index | `text_embed.runtime.embed_text_main_text_embed` + `vdb.*` |
| `server.py` | FastAPI query API | `retriever.Retriever` |
| `requirements.txt`, `systemd/*`, `compat/*` | pinned deps + repeatable deployment | — |

Each `chunks.jsonl` row carries `text`, `source_path`, `page_number`, `chunk_id`, and a
`metadata` object — the schema the ingester and `retrieval_sdg/query_gen` expect. The
prep scripts are env-driven and resumable (checkpoint + `.complete` marker).

## Requirements

- Ubuntu, NVIDIA driver, Python 3.12
- `nemo-retriever[local]==26.5.0`, vLLM, LanceDB (see `requirements.txt`)
- A GPU for the embedding service and indexing

## Quick start

Prepare a corpus:

```bash
# (optional) PDFs -> text
INPUT_PDF_DIR=/path/to/pdfs EXTRACT_OUTPUT_JSONL=/path/to/extracted.jsonl \
  .venv/bin/python extract.py

# text -> chunks (default 1024 tokens, no overlap)
CHUNK_INPUT_JSONL=/path/to/extracted.jsonl CHUNK_OUTPUT_JSONL=/path/to/chunks.jsonl \
CHUNK_MAX_TOKENS=1024 .venv/bin/python chunk.py
```

Ingest + serve (via systemd; env knobs override defaults):

```bash
# 1. embedding model (must be up before indexing / querying)
sudo systemctl enable --now nemo-retriever-embed.service      # :8001

# 2. embed chunks + build the index (resumable). Point the ingest unit at your corpus
#    via its Environment= lines (systemd does NOT inherit env from this shell):
#      sudo systemctl edit --full nemo-retriever-hindi-ingest.service
#      # set Environment=INPUT_JSONL=... LANCEDB_URI=... TABLE_NAME=...
sudo systemctl start --no-block nemo-retriever-hindi-ingest.service

# 3. query API
sudo systemctl enable --now nemo-retriever-hindi-api.service  # :8002
curl -sf localhost:8002/health
```

Point `retrieval_sdg/config/pipeline.yaml` at the endpoint (and, optionally, at the
LanceDB path so `query_gen` can cluster on the stored vectors):

```yaml
retrieval:
  endpoint: http://<host>:8002/query
query_gen:
  lancedb: { uri: /path/to/lancedb, table: my-table }
```

## Reference deployment

| Setting | Value |
|---|---|
| Host / API | `10.117.9.203:8002` (4 Uvicorn workers) |
| Embedding API | `127.0.0.1:8001` |
| Model | `nvidia/llama-nemotron-embed-1b-v2` (2048-dim, `input_type=passage`) |
| LanceDB | `/localhome/local-rkalani/nemo-retriever/lancedb-hindi-legal`, table `hindi-legal-judgments` |
| Corpus | 291,686 already-extracted page chunks |

Notes: PDF extraction is not part of the server — chunk sizes come from the input JSONL.
The API has no auth and binds `0.0.0.0`; firewall it or front it with an authenticated
proxy before exposing. A `502` usually means the embedding service (`:8001`) is down.
Tuning knobs: `--workers` (API), `MAX_IN_FLIGHT`, `--max-num-seqs` (vLLM),
`--gpu-memory-utilization`.
