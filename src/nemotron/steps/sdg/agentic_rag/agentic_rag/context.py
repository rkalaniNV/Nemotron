"""Context management: sliding window + running scratchpad.

Solves the long-context problem without one-shot generation. The trajectory is
*assembled* turn-by-turn; this module bounds what each LLM call must *read*:

  - keep the last ``window_k`` tool responses RAW,
  - compact older tool responses to a short reference (id + snippet) or summary,
  - carry a running "findings so far" scratchpad the assistant reads instead of
    re-reading every raw chunk.

Key decoupling (config ``store_full_trace``): the *generation view* is compact,
but the *stored training trajectory* keeps full chunks — the model still learns
to read real retrieved text. Everything here is driven by the config knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScratchpadEntry:
    hop: int
    tool: str
    query: str
    finding: str          # distilled note (not the raw chunk)
    source_ids: List[str] = field(default_factory=list)


@dataclass
class ResearchMemory:
    """Running state carried across hops within one research turn."""
    plan: str = ""
    entries: List[ScratchpadEntry] = field(default_factory=list)

    def add(self, hop: int, tool: str, query: str, finding: str, source_ids: List[str]) -> None:
        self.entries.append(ScratchpadEntry(hop, tool, query, finding, source_ids))

    def render(self) -> str:
        if not self.entries:
            return "(no findings yet)"
        lines = []
        for e in self.entries:
            src = f" [{', '.join(e.source_ids)}]" if e.source_ids else ""
            lines.append(f"- (hop {e.hop}) {e.finding}{src}")
        return "\n".join(lines)

    def covered_sources(self) -> List[str]:
        out: List[str] = []
        for e in self.entries:
            out.extend(e.source_ids)
        return out


def _estimate_tokens(msgs: List[Dict[str, Any]]) -> int:
    chars = sum(len(m.get("content", "") or "") + len(m.get("reasoning_content", "") or "") for m in msgs)
    return chars // 4


def _compact_tool_message(msg: Dict[str, Any], mode: str, snippet_chars: int = 160) -> Dict[str, Any]:
    """Reduce a tool response per compaction mode. NEVER returns None / drops the
    message: the tool_call_id must stay so the assistant tool_call it answers is
    not orphaned (that would make the messages invalid for the API)."""
    content = msg.get("content", "") or ""
    cid = msg.get("_chunk_id") or msg.get("tool_call_id", "?")
    if mode == "drop":
        content = f"[retrieved {cid}: omitted]"
    elif mode == "summary":
        content = msg.get("_summary") or (content[:snippet_chars] + "…" if len(content) > snippet_chars else content)
    else:  # reference: id + short snippet
        snip = content[:snippet_chars].replace("\n", " ")
        content = f"[retrieved {cid}: {snip}…]" if len(content) > snippet_chars else f"[retrieved {cid}: {snip}]"
    out = dict(msg)
    out["content"] = content
    return out


def _shrink_middle_further(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Second-pass shrink when the middle is still over budget: drop assistant
    reasoning traces and hard-cap any remaining content. Structure (roles,
    tool_calls, tool_call_ids) is preserved so the message list stays valid."""
    out = dict(msg)
    if out.get("role") == "assistant" and out.get("reasoning_content"):
        out["reasoning_content"] = ""
    c = out.get("content", "") or ""
    if len(c) > 80:
        out["content"] = c[:80].replace("\n", " ") + "…"
    return out


def build_assistant_view(
    messages: List[Dict[str, Any]],
    memory: Optional[ResearchMemory],
    *,
    window_k: int,
    compaction_mode: str,
    compression_token_limit: int = 2000,
    preserve_last_user_turn: bool = True,
    use_scratchpad: bool = True,
) -> List[Dict[str, Any]]:
    """Outer compression (Stage 5a).

    The HEAD (leading system prompt) and the TAIL (everything from the last user
    turn to the end) are preserved verbatim. Only the MIDDLE span between them is
    compressed, and only once its estimated size exceeds
    ``compression_token_limit``. Messages are never removed — only their content
    is shrunk — so assistant tool_calls always keep their matching tool
    responses. A scratchpad note (running findings) is inserted after the head so
    the assistant reasons over distilled notes instead of re-reading raw chunks.
    """
    n = len(messages)
    head_end = 0
    while head_end < n and messages[head_end].get("role") == "system":
        head_end += 1

    tail_start = n
    if preserve_last_user_turn:
        for i in range(n - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                tail_start = i
                break

    head = list(messages[:head_end])
    middle = list(messages[head_end:tail_start])
    tail = list(messages[tail_start:])

    compressed_middle = _compress_span(middle, window_k, compaction_mode, compression_token_limit)
    view = head + compressed_middle + tail

    if use_scratchpad and memory is not None:
        note = "Research plan:\n" + (memory.plan or "(none)") + "\n\nFindings so far:\n" + memory.render()
        view.insert(head_end, {"role": "system", "content": note})

    return view


def _compress_span(middle: List[Dict[str, Any]], window_k: int, mode: str, budget: int) -> List[Dict[str, Any]]:
    """Compress the middle span down toward ``budget`` estimated tokens, keeping
    the last ``window_k`` tool responses raw and never dropping any message."""
    if _estimate_tokens(middle) <= budget:
        return middle
    tool_positions = [i for i, m in enumerate(middle) if m.get("role") == "tool"]
    keep_raw = set(tool_positions[-window_k:]) if window_k > 0 else set()

    out = [
        _compact_tool_message(m, mode) if (m.get("role") == "tool" and i not in keep_raw) else m
        for i, m in enumerate(middle)
    ]
    if _estimate_tokens(out) <= budget:
        return out
    # still over budget: shrink everything outside the raw window further
    return [m if i in keep_raw else _shrink_middle_further(m) for i, m in enumerate(out)]


def build_judge_view(
    messages: List[Dict[str, Any]],
    *,
    window_k: int,
    compaction_mode: str,
    compression_token_limit: int = 2000,
) -> List[Dict[str, Any]]:
    """Compacted view for judges (no scratchpad; judges see the conversation)."""
    return build_assistant_view(
        messages, None, window_k=window_k, compaction_mode=compaction_mode,
        compression_token_limit=compression_token_limit, preserve_last_user_turn=True,
        use_scratchpad=False,
    )
