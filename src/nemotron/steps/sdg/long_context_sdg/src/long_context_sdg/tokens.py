"""Token accounting and active-context compaction triggers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

_ENCODER = None
_TRIED = False


def _encoder():
    global _ENCODER, _TRIED
    if _TRIED:
        return _ENCODER
    _TRIED = True
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return int(len(text) / 3.5) + len(text.split()) // 2 + 1


def message_tokens(message: Any) -> int:
    m = message.to_openai() if hasattr(message, "to_openai") else dict(message)
    total = 4
    for key in ("content", "reasoning_content", "tool_calls", "name"):
        value = m.get(key)
        if value:
            total += count_tokens(
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
    return total


def context_tokens(messages: Iterable[Any]) -> int:
    return sum(message_tokens(m) for m in messages)


class ContextMeter:
    def __init__(self, threshold: int, min_turns_between: int):
        self.threshold = threshold
        self.min_turns_between = min_turns_between
        self.active_tokens = 0
        self.last_compression_turn = 0
        self.history = []

    def add_all(self, messages: Iterable[Any]) -> None:
        self.active_tokens += context_tokens(messages)

    def should_compress(self, turn: int) -> bool:
        return (
            self.active_tokens >= self.threshold
            and turn - self.last_compression_turn >= self.min_turns_between
        )

    def reset(self, turn: int, summary: str, recent_messages: Iterable[Any]) -> None:
        self.history.append({"turn": turn, "tokens_before": self.active_tokens})
        self.active_tokens = count_tokens(summary) + context_tokens(recent_messages) + 4
        self.last_compression_turn = turn
