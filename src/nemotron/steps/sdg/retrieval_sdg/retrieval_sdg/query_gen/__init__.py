"""Query generation (Module B) — synthesize diverse seed queries FROM the corpus.

Reads the on-disk chunk corpus directly (streaming), samples a bounded pool,
clusters it in embedding space, then generates end-user questions across topic
clusters x question kinds (multi-hop = related same-cluster chunks). Optionally
validates each query is answerable by the live retriever. Output feeds query_prep.

    corpus.jsonl -> reservoir_sample -> embed+cluster -> units(kind x chunks)
                 -> generate (LLM) -> validate -> queries.jsonl
"""

from .corpus import Chunk, count_chunks, reservoir_sample, stream_chunks
from .lancedb_source import read_lancedb, rows_to_pool
from .run import run_query_gen
from .sizing import plan_sizes

__all__ = ["Chunk", "stream_chunks", "reservoir_sample", "count_chunks",
           "read_lancedb", "rows_to_pool", "run_query_gen", "plan_sizes"]
