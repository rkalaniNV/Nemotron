#!/usr/bin/env python3
"""Offline retriever CLI — thin shim over the canonical ``agentic_rag.retrieval``.

The real implementation (backends, subsample, gold-rank) lives in the installed
plugin package so the runtime and the offline tooling share ONE retrieval module
(swap a backend once, everywhere). This file only exposes the CLI and re-exports
the public names so existing imports (`from retriever import EmbeddingRetriever`)
keep working.

Usage:
    # single index
    python retriever.py build --chunks ../data/constitution_chunks.jsonl \
        --index ../data/index --backend embedding
    # one index PER cluster (streaming pipeline layout)
    python retriever.py build --clusters-root ../data/clusters --backend embedding
    # smoke-test a query
    python retriever.py query --index ../data/index --q "..." --gold 32
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer the LOCAL sibling package over any (possibly stale) editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_rag.retrieval import (  # noqa: F401,E402
    EMBED_MODEL, EmbeddingRetriever, LexicalRetriever, RetrievedChunk,
    Retriever, gold_rank, load_corpus, main, make_retriever,
)


if __name__ == "__main__":
    main()
