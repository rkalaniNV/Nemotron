# `nv_retrieval` — NVIDIA embed + rerank + retrieval utility

Fast, modular retrieval layer for the agentic-RAG SDG pipeline. Three swappable
pieces so **clustering (Stage 1)** and the **retrieval tool (Stage 3/5)** share one
embedder while the backend stays replaceable.

| Piece | Class | NVIDIA backend (hosted, verified live 2026-07) | Offline fallback |
|-------|-------|----------------|------------------|
| Embedder | `NIMEmbedder` | `nvidia/llama-nemotron-embed-1b-v2` → `POST https://integrate.api.nvidia.com/v1/embeddings` | `HashEmbedder` |
| Reranker | `NIMReranker` | `nvidia/rerank-qa-mistral-4b` → `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking` | `LexicalReranker` |
| Retriever | `Retriever` / `ClusterIndex` | (uses the above) | — |

`make_embedder("nim")` / `make_reranker("nim")` auto-fall-back to the offline
implementations when `NVIDIA_API_KEY` (or `httpx`) is missing, so it runs anywhere.

> The older `llama-3.2-nv-embedqa-1b-v2` embedding model and the `nv-rerankqa-1b-v2`
> rerank endpoint both reached **end-of-life (2026-05-18)** on the hosted API; the
> defaults above are the current live replacements. The rerank endpoint is on a
> **different host** (`ai.api.nvidia.com`) than embeddings.

## Config

```bash
export NVIDIA_API_KEY=nvapi-...        # build.nvidia.com, INFERENCE access
# optional overrides (defaults are the live hosted endpoints above):
export NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1                 # embeddings host (or self-hosted NIM)
export NVIDIA_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
export NVIDIA_RERANK_URL=https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking  # or http://host:port/v1/ranking
export NVIDIA_RERANK_MODEL=nvidia/rerank-qa-mistral-4b
```

## The integration seam with clustering (Stage 1)

Clustering owns *grouping*; this module owns *embedding*. One embedder serves both
granularities — whole documents (to cluster) and chunks (to index):

```python
from nv_retrieval import make_embedder, build_cluster_indexes, Retriever, make_reranker

embedder = make_embedder("nim", dimensions=768)   # smaller dim = faster clustering/search

# Stage 1a — your teammate's clustering calls THIS to embed whole documents:
doc_vectors = embedder.embed_documents(documents)   # (n_docs, dim); long docs mean-pooled
# ... teammate clusters doc_vectors -> {cluster_id: [doc, ...]} -> chunks per cluster ...

# Stage 1b — build one independent index per cluster:
#   clusters = {cluster_id: [ {chunk_id, text, **metadata}, ... ]}
indexes = build_cluster_indexes(clusters, embedder, out_dir="data/index")

# Stage 3/5 — retrieval tool, per cluster:
retr = Retriever(indexes[cluster_id], reranker=make_reranker("nim"))
hits = retr.retrieve(query, n=4, candidate_multiplier=2, rerank=False, seed=None)
tool_response = [h.to_tool_payload() for h in hits]
```

`embed_documents` covers the whole document despite the 8192-token cap: over-long
docs are windowed and the windows are **mean-pooled** into one vector, so the
"cluster whole documents, no chunking" interface holds without silent truncation.

## Retrieval policy (the SDG knob)

`retrieve()` implements: **semantic top `candidate_multiplier * n` (default 2n)
→ [optional] rerank → random-sample down to `n`**. The random sample injects
diversity so the agent must search more / go deeper in multi-step. Knobs:

- `n` — chunks returned (`number_of_chunks_to_retrieve`)
- `candidate_multiplier` — the `2` in "top 2n" (bigger = more exploration)
- `rerank` — sharpen the pool before sampling (default off; opposite pull to sampling)
- `seed` — reproducibility (leave `None` for run-to-run diversity)

## Speed

Embedding HTTP is the bottleneck: `NIMEmbedder` batches inputs (`batch_size`) and
fires batches concurrently over a thread pool (`concurrency`). Use a smaller
Matryoshka `dimensions` (384/512/768) for cheaper clustering + search. Per-cluster
indexes are cached to disk (`embeddings.npy` + `chunks.jsonl` + `meta.json`); brute-
force numpy cosine is fine per cluster — add FAISS/cuVS only if one cluster > ~100k
chunks.

## Test

```bash
cd data_prep && python -m pytest test_nv_retrieval.py -q   # 18 tests, no key/network needed
python nv_retrieval.py                                      # offline demo
```
