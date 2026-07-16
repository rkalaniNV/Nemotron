"""retrieval_sdg — the multi-turn retrieval-grounded conversation-generation engine.

Given a JSONL list of queries and a retrieval service, this package deduplicates,
clusters, and samples the queries, then generates grounded multi-turn / multi-step
tool-calling conversations (trajectory shape sampled per row) with an embedded
judge — ready for SFT. The retrieval service and the upstream query producer are
external; this package wraps the retrieval service thinly and consumes the queries.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
