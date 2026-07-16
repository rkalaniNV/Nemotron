"""Sample down to the target conversation count with cross-cluster diversity.

Round-robins across clusters (largest-first within a round) so the sampled set
spans topics rather than clumping in the biggest cluster. Deterministic given a
seed, for reproducible runs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional


def sample(queries: List[Dict[str, Any]], n_target: Optional[int], *,
           cluster_field: str = "cluster_id", seed: int = 7) -> List[Dict[str, Any]]:
    """Pick ``n_target`` records, round-robin across clusters. n_target=None => all."""
    if not n_target or n_target >= len(queries):
        return list(queries)
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for q in queries:
        buckets[q.get(cluster_field, 0)].append(q)
    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets, key=lambda c: (-len(buckets[c]), str(c)))  # big clusters first, stable
    out: List[Dict[str, Any]] = []
    while len(out) < n_target and any(buckets[c] for c in order):
        for c in order:
            if buckets[c] and len(out) < n_target:
                out.append(buckets[c].pop())
    return out
