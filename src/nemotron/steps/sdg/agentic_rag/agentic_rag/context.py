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


def _compact_tool_message(msg: Dict[str, Any], mode: str, snippet_chars: int = 160) -> Optional[Dict[str, Any]]:
    """Reduce an out-of-window tool response per compaction mode."""
    if mode == "drop":
        return None
    content = msg.get("content", "") or ""
    if mode == "summary":
        # summary is expected to be pre-written into metadata by the loop;
        # fall back to a truncation if absent.
        content = msg.get("_summary") or (content[:snippet_chars] + "…" if len(content) > snippet_chars else content)
    else:  # reference: id + short snippet
        cid = msg.get("_chunk_id") or msg.get("tool_call_id", "?")
        snip = content[:snippet_chars].replace("\n", " ")
        content = f"[retrieved {cid}: {snip}…]" if len(content) > snippet_chars else f"[retrieved {cid}: {snip}]"
    out = dict(msg)
    out["content"] = content
    return out


def build_assistant_view(
    messages: List[Dict[str, Any]],
    memory: Optional[ResearchMemory],
    *,
    window_k: int,
    compaction_mode: str,
    use_scratchpad: bool,
) -> List[Dict[str, Any]]:
    """Return a compacted message list for the assistant's next generation.

    System + user turns and all assistant turns are preserved verbatim. Only tool
    responses are windowed: the last ``window_k`` stay raw, older ones are
    compacted. If ``use_scratchpad``, a synthetic system note carries the running
    findings so the assistant can reason over notes instead of raw chunks.
    """
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep_raw = set(tool_positions[-window_k:]) if window_k > 0 else set()

    view: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if m.get("role") == "tool" and i not in keep_raw:
            compacted = _compact_tool_message(m, compaction_mode)
            if compacted is not None:
                view.append(compacted)
        else:
            view.append(m)

    if use_scratchpad and memory is not None:
        note = "Research plan:\n" + (memory.plan or "(none)") + "\n\nFindings so far:\n" + memory.render()
        # insert right after the leading system message
        insert_at = 1 if view and view[0].get("role") == "system" else 0
        view.insert(insert_at, {"role": "system", "content": note})

    return view


def build_judge_view(
    messages: List[Dict[str, Any]],
    *,
    window_k: int,
    compaction_mode: str,
) -> List[Dict[str, Any]]:
    """Compacted view for judges (no scratchpad; judges see the conversation)."""
    return build_assistant_view(
        messages, None, window_k=window_k, compaction_mode=compaction_mode, use_scratchpad=False,
    )
