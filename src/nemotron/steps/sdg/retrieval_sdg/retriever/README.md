# NeMo Retriever server

This directory reproduces the NeMo Retriever deployment used by
`retrieval_sdg`. It serves the already-extracted Hindi legal chunks on port
`8002` and uses a persistent vLLM embedding service on localhost port `8001`.

The public API is:

```http
POST /query
Content-Type: application/json

{"query":"...", "num_chunks":5}
```

`top_k` is accepted as an alias for `num_chunks`.

## Files required

Only two project Python files are needed:

| File | Purpose |
|---|---|
| `server.py` | FastAPI wrapper around `nemo_retriever.retriever.Retriever` |
| `ingest_processed_chunks.py` | Embeds an extracted JSONL corpus and builds the LanceDB vector index |

The remaining files make the deployment repeatable:

| File | Purpose |
|---|---|
| `requirements.txt` | Versions running on the current machine |
| `ingest-hindi-legal.sh` | Ingestion launcher used by systemd |
| `systemd/*.service` | Persistent embedding, indexing, and API processes |
| `compat/prometheus_routing.py` | Compatibility fix required by this tested FastAPI/vLLM dependency set |

PDF extraction is intentionally not part of this server. The Hindi legal corpus
is already extracted at:

```text
/localhome/local-rkalani/hindi-legal-agent/data/processed/extraction/chunks.jsonl
```

Each JSONL row must contain `text`. The ingester also preserves `source_path`,
`page_number`, `chunk_id`, and the `metadata` object when present.

## Current deployment

| Setting | Value |
|---|---|
| Host | `10.117.9.203` |
| Query API | `0.0.0.0:8002`, four Uvicorn workers |
| Embedding API | `127.0.0.1:8001` |
| Model | `nvidia/llama-nemotron-embed-1b-v2` |
| LanceDB | `/localhome/local-rkalani/nemo-retriever/lancedb-hindi-legal` |
| Table | `hindi-legal-judgments` |
| Corpus | 291,686 already-extracted page chunks |

The chunks are not fixed at 1,024 tokens. Chunk boundaries come from the input
JSONL. For this corpus a sample measured a median of about 759 tokens, p95 of
about 965, and some pages above 1,024. Change chunk size in the extraction step,
then rebuild the index; it is not a query-time setting.

## Install once

These commands assume Ubuntu, an NVIDIA driver, and Python 3.12. Run them as
`local-rkalani`:

```bash
REPO=/localhome/local-rkalani/Nemotron/src/nemotron/steps/sdg/retrieval_sdg/retriever
DEPLOY=/localhome/local-rkalani/nemo-retriever

install -d "$DEPLOY/compat"
install -m 0644 "$REPO/server.py" "$REPO/ingest_processed_chunks.py" \
  "$REPO/requirements.txt" "$DEPLOY/"
install -m 0755 "$REPO/ingest-hindi-legal.sh" "$DEPLOY/"
install -m 0644 "$REPO/compat/prometheus_routing.py" "$DEPLOY/compat/"
cd "$DEPLOY"

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The tested environment uses `vllm==0.20.0` and CUDA-enabled
`torch==2.11.0`. Verify CUDA before indexing:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

With the tested dependency versions, install the included metrics routing
compatibility file after the dependencies:

```bash
SITE=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
install -m 0644 compat/prometheus_routing.py \
  "$SITE/prometheus_fastapi_instrumentator/routing.py"
```

Copy the three unit files to `/etc/systemd/system/`, then reload systemd:

```bash
sudo install -m 0644 "$REPO"/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
```

The checked-in units use `/localhome/local-rkalani/nemo-retriever` as their
working directory.

## Build or resume the index

Start the embedding model first:

```bash
sudo systemctl enable --now nemo-retriever-embed.service
curl --fail http://127.0.0.1:8001/health
```

Then start indexing:

```bash
sudo systemctl enable nemo-retriever-hindi-ingest.service
sudo systemctl start --no-block nemo-retriever-hindi-ingest.service
journalctl -fu nemo-retriever-hindi-ingest.service
```

Ingestion checkpoints after every batch and resumes from
`.ingest-state.json`. It writes `.ingest-complete` only after creating the
vector index. Do not delete either file unless intentionally rebuilding the
database. To use a different already-extracted corpus, set these environment
variables in the ingest unit:

```text
INPUT_JSONL=/path/to/chunks.jsonl
LANCEDB_URI=/path/to/lancedb
TABLE_NAME=my-table
```

Optional ingestion knobs are `INGEST_BATCH_SIZE`, `EMBED_BATCH_SIZE`,
`EMBED_CONCURRENCY`, `INDEX_PARTITIONS`, and `INDEX_SUB_VECTORS`.

## Start the query server

After the index exists:

```bash
sudo systemctl enable --now nemo-retriever-hindi-api.service
curl --fail http://127.0.0.1:8002/health
```

Query one chunk from another machine on the reachable network:

```bash
curl --fail --silent --show-error \
  -X POST http://10.117.9.203:8002/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are the requirements for anticipatory bail?","num_chunks":1}'
```

The SDG pipeline already maps this schema in `config/pipeline.yaml`:

```yaml
retrieval:
  endpoint: http://10.117.9.203:8002/query
```

## Stop, restart, and inspect

```bash
sudo systemctl restart nemo-retriever-hindi-api.service
sudo systemctl restart nemo-retriever-embed.service
sudo systemctl stop nemo-retriever-hindi-api.service

systemctl status nemo-retriever-hindi-api.service --no-pager
systemctl status nemo-retriever-embed.service --no-pager
journalctl -u nemo-retriever-hindi-api.service -n 100 --no-pager
journalctl -u nemo-retriever-embed.service -n 100 --no-pager
```

If the client reports `502 Bad Gateway`, check both services. The API depends on
the embedding service and will fail queries when port `8001` is unhealthy. The
API unit also sets `LimitNOFILE=65536`; keep that setting for concurrent load.

## Throughput controls

- `--workers 4` in the API unit controls Uvicorn worker processes.
- `MAX_IN_FLIGHT=32` limits concurrent retrievals per worker.
- `--max-num-seqs 256` controls vLLM embedding concurrency.
- `--gpu-memory-utilization 0.45` caps this deployment's vLLM GPU allocation.

Increase one setting at a time and benchmark p95 latency and error rate. More API
workers do not add GPU capacity; they only increase pressure on the shared vLLM
process.

## Network exposure

The unit binds the query API to `0.0.0.0` and the app has no authentication.
Anyone who can route to port `8002` can query it without the SSH password. For a
production or broadly reachable host, restrict the firewall or put the API
behind an authenticated reverse proxy before exposing the port.
