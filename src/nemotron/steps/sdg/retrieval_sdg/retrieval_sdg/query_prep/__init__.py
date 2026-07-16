"""Input-side stages: dedup -> cluster -> sample the incoming queries.

The query source hands us a JSONL list of queries. These offline stages turn that
raw list into a diverse, deduplicated seed set for the conversation engine.
Trajectory shape is sampled per-row by the planner (no separate classification pass).
"""

from .dedup import dedup, normalize
from .cluster import cluster_queries
from .sample import sample

__all__ = ["dedup", "normalize", "cluster_queries", "sample"]
