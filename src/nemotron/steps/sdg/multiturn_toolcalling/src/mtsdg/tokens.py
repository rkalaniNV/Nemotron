"""Token accounting for the context-compression simulation.

In production the application compresses the running conversation when its token
count crosses a threshold (32k). This module reproduces that trigger during SDG
so the synthetic trajectories learn *when* to compress from a realistic signal
rather than fixed turn indices.

``count_tokens`` uses tiktoken when available (real BPE) and falls back to a
conservative char heuristic. ``context_tokens`` measures the *active* context the
way a serving stack would: system prompt + the messages currently in the window
(after the last compression), including tool results (retrieved chunks dominate
the budget) and reasoning traces.

The threshold is configurable: pass the true 32000 for production-faithful
behaviour, or a smaller simulation threshold so short synthetic turns still
trigger 2-3 compressions within a 20-25 turn episode.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

_DEFAULT_ENCODING = "cl100k_base"

# Cache the encoder so we do not re-init tiktoken per call.
_ENC = None
_ENC_TRIED = False


def _encoder():
    global _ENC, _ENC_TRIED
    if _ENC_TRIED:
        return _ENC
    _ENC_TRIED = True
    try:
        import tiktoken

        try:
            _ENC = tiktoken.get_encoding(_DEFAULT_ENCODING)
        except Exception:
            _ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENC = None
    return _ENC


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken when available; else a conservative heuristic.

    The heuristic over-counts slightly (~1 token / 3.5 chars + word count) so a
    missing tokenizer never lets an over-budget context slip past the trigger.
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return int(len(text) / 3.5) + len(text.split()) // 2 + 1


#: Per-message overhead a chat template adds (role tags, delimiters). Rough but
#: consistent with OpenAI's ~3-4 tokens/message accounting.
_MESSAGE_OVERHEAD = 4


def message_tokens(message: Any) -> int:
    """Token cost of one message (dict or object with ``to_openai``).

    Counts content, serialized tool_calls, and reasoning_content — everything that
    actually occupies the model's context window.
    """
    m = message.to_openai() if hasattr(message, "to_openai") else dict(message)
    total = _MESSAGE_OVERHEAD
    content = m.get("content")
    if isinstance(content, str):
        total += count_tokens(content)
    elif content is not None:
        total += count_tokens(json.dumps(content, ensure_ascii=False))
    if m.get("tool_calls"):
        total += count_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    rc = m.get("reasoning_content")
    if rc:
        total += count_tokens(rc if isinstance(rc, str) else json.dumps(rc, ensure_ascii=False))
    if m.get("name"):
        total += count_tokens(str(m["name"]))
    return total


def context_tokens(messages: Iterable[Any]) -> int:
    """Total token cost of an ordered message window (the active context)."""
    return sum(message_tokens(m) for m in messages)


class ContextMeter:
    """Tracks the active-context token count across a trajectory and decides when
    automatic context compaction should fire.

    After a compaction the meter only counts the summary + subsequent messages,
    exactly like a serving stack that dropped the compacted prefix.
    """

    def __init__(self, threshold: int = 32000, *, min_turns_between: int = 3):
        self.threshold = threshold
        self.min_turns_between = min_turns_between
        self._running = 0
        self._last_compression_turn = 0
        self.history: List[Dict[str, int]] = []

    def add(self, message: Any) -> None:
        self._running += message_tokens(message)

    def add_all(self, messages: Iterable[Any]) -> None:
        for m in messages:
            self.add(m)

    @property
    def active_tokens(self) -> int:
        return self._running

    def should_compress(self, current_turn: int) -> bool:
        """True when the active context has crossed the threshold and enough turns
        have elapsed since the last compression (avoids back-to-back compressions).
        """
        if self._running < self.threshold:
            return False
        if current_turn - self._last_compression_turn < self.min_turns_between:
            return False
        return True

    def reset_after_compression(self, current_turn: int, summary_text: str) -> None:
        """Collapse the active window to just the summary's token cost.

        Mirrors the app dropping the compressed prefix and keeping only the rolling
        summary (recent raw turns are re-added by the generator as it proceeds).
        """
        self.history.append(
            {"turn": current_turn, "tokens_before": self._running, "threshold": self.threshold}
        )
        self._running = count_tokens(summary_text) + _MESSAGE_OVERHEAD
        self._last_compression_turn = current_turn
