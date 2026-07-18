"""Public boundary for seed-to-conversation generation.

Query generation intentionally depends only on the shared seed schema. This
module is the inverse boundary: it consumes seed JSONL without knowing how the
queries were authored.
"""

from __future__ import annotations

from .config import PipelineConfig


def prepare_conversation_seeds(cfg: PipelineConfig) -> int:
    from .seeds import prepare_seed_file

    return prepare_seed_file(cfg)


def generate_conversations(cfg: PipelineConfig) -> int:
    from .pipeline import generate

    return generate(cfg)


__all__ = ["generate_conversations", "prepare_conversation_seeds"]
