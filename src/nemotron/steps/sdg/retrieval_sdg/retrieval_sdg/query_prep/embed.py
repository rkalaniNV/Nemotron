"""Lightweight query embedding — MiniLM by default. Shared by dedup + cluster."""

from __future__ import annotations

from typing import List

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: List[str], model_name: str = EMBED_MODEL):
    """One L2-normalized vector per text (numpy float32)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name or EMBED_MODEL)
    emb = model.encode([t or " " for t in texts], batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")
