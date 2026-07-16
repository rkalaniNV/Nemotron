"""Cluster the deduped queries into topical groups for diverse sampling.

Operates over QUERY embeddings (no documents, no chunking). Backends mirror the
proven agentic_rag clustering: kmeans / agglomerative / hdbscan (discovers k).
Returns a 0..C-1 label per query.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


def _resolve_k(k: Optional[int], n: int) -> int:
    if k:
        return max(1, min(k, n))
    return max(2, min(n, round((n / 2) ** 0.5)))  # heuristic when k unspecified


def _absorb_noise(emb, labels):
    """Reassign HDBSCAN outliers (-1) to the nearest cluster centroid."""
    import numpy as np
    labels = np.asarray(labels)
    clustered = labels >= 0
    if not clustered.any():
        return np.arange(len(labels))
    ids = sorted(set(labels[clustered].tolist()))
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in ids])
    out = labels.copy()
    for i in np.where(~clustered)[0]:
        d = np.linalg.norm(centroids - emb[i], axis=1)
        out[i] = ids[int(d.argmin())]
    return out


def cluster_embeddings(emb, algo: str = "kmeans", k: Optional[int] = None,
                       min_cluster_size: int = 5) -> List[int]:
    import numpy as np
    n = len(emb)
    if n <= 1:
        return [0] * n
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=_resolve_k(k, n), n_init=10, random_state=7).fit_predict(emb).tolist()
    if algo == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        if k:
            return AgglomerativeClustering(n_clusters=min(k, n)).fit_predict(emb).tolist()
        return AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric="cosine",
                                       linkage="average").fit_predict(emb).tolist()
    from sklearn.cluster import HDBSCAN
    mcs = max(2, min(min_cluster_size, n))
    labels = HDBSCAN(min_cluster_size=mcs, metric="euclidean").fit_predict(emb)
    return _absorb_noise(np.asarray(emb), labels).tolist()


def _remap(labels: List[int]) -> List[int]:
    order = {old: new for new, old in enumerate(sorted(set(labels)))}
    return [order[l] for l in labels]


def cluster_queries(queries: List[Any], *, algo: str = "kmeans", k: Optional[int] = None,
                    min_cluster_size: int = 5, model_name: str = "",
                    embed_fn: Optional[Callable[[List[str]], Any]] = None,
                    query_field: str = "query") -> List[int]:
    """Return a contiguous 0..C-1 cluster label per query (input order)."""
    import numpy as np
    texts = [str(q.get(query_field, "")) if isinstance(q, dict) else str(q) for q in queries]
    if embed_fn is None:
        from .embed import embed_texts
        embed_fn = lambda ts: embed_texts(ts, model_name)  # noqa: E731
    emb = np.asarray(embed_fn(texts), dtype="float32")
    return _remap(cluster_embeddings(emb, algo, k, min_cluster_size))
