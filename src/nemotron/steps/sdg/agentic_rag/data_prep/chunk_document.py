#!/usr/bin/env python3
"""Generic document chunker for the agentic-RAG SDG pipeline.

Turns one long text document into a JSONL corpus of clean, metadata-tagged
chunks that become (a) the retriever's search index and (b) the seed pool for
document-level query generation — *before* Nemotron Data Designer runs.

The core is domain-agnostic:

  1. (optional) drop noise lines            -- e.g. page headers, rule lines
  2. (optional) split into sections         -- on a `section_pattern` regex
     while tracking running `heading_patterns` context (e.g. chapters/parts)
  3. size-bound every unit                  -- recursive separator split + overlap
  4. (optional) attach metadata             -- via `metadata_extractors` regexes

Everything domain-specific lives in a *profile* (see PROFILES at the bottom), not
in the algorithm. `--profile plain` is pure size-based chunking that works on any
text; other profiles just supply regexes.

Usage:
    python chunk_document.py --input DOC.txt --output chunks.jsonl \
        --profile plain --max-chars 1200 --overlap 150
    python chunk_document.py --input constitution.txt --output chunks.jsonl \
        --profile indian_statute
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class ChunkProfile:
    """All domain-specific configuration lives here; the algorithm stays generic."""
    name: str = "plain"
    max_chars: int = 1200
    overlap: int = 150
    separators: Sequence[str] = tuple(DEFAULT_SEPARATORS)
    start_marker: Optional[str] = None                 # begin parsing at first match
    stop_marker: Optional[str] = None                  # stop parsing at first match (after start)
    section_pattern: Optional[str] = None              # a match starts a new section
    heading_patterns: Tuple[Tuple[str, str], ...] = () # (level_name, regex) running context
    drop_line_patterns: Tuple[str, ...] = ()           # noise lines removed before chunking
    metadata_extractors: Tuple[Tuple[str, str], ...] = ()  # (field, regex) -> first group(s)
    reject_section_pattern: Optional[str] = None       # section candidates matching this (full line) are skipped
    footnote_rule_pattern: Optional[str] = None        # a match opens a footnote block; lines dropped until blank


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str = ""                     # source document id (cluster/pipeline provenance)
    section_id: str = ""                 # e.g. article number, heading slug, or running index
    section_title: str = ""
    headings: Dict[str, str] = field(default_factory=dict)   # running context (part/chapter/...)
    metadata: Dict[str, List[str]] = field(default_factory=dict)
    sub_index: int = 0
    char_len: int = 0
    token_estimate: int = 0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── generic recursive size splitter (LangChain-style, dependency-free) ───────
def recursive_split(text: str, max_chars: int, separators: Sequence[str]) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sep, rest = "", []
    for i, s in enumerate(separators):
        if s == "":
            sep, rest = "", []
            break
        if s in text:
            sep, rest = s, list(separators[i + 1:])
            break
    if sep == "":
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    out: List[str] = []
    buf = ""
    for part in text.split(sep):
        candidate = part if not buf else buf + sep + part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
                buf = ""
            if len(part) > max_chars:
                out.extend(recursive_split(part, max_chars, rest))
            else:
                buf = part
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        out.append((tail + " " + cur).strip())
    return out


def size_chunks(text: str, prof: ChunkProfile) -> List[str]:
    pieces = recursive_split(text, prof.max_chars, prof.separators)
    return _add_overlap(pieces, prof.overlap)


# ── optional structure pass ──────────────────────────────────────────────────
def _compiled(p: Optional[str]) -> Optional[Pattern]:
    return re.compile(p, re.MULTILINE) if p else None


def _slice_body(text: str, prof: ChunkProfile) -> str:
    if prof.start_marker:
        m = re.search(prof.start_marker, text)
        if m:
            text = text[m.start():]
    if prof.stop_marker:
        m = re.search(prof.stop_marker, text, re.MULTILINE)
        if m:
            text = text[:m.start()]
    return text


def _extract_metadata(text: str, prof: ChunkProfile) -> Dict[str, List[str]]:
    meta: Dict[str, List[str]] = {}
    for field_name, pat in prof.metadata_extractors:
        vals = sorted({m.group(1) for m in re.finditer(pat, text)})
        if vals:
            meta[field_name] = vals
    return meta


def _sectionize(body: str, prof: ChunkProfile) -> List[Tuple[str, str, Dict[str, str], str]]:
    """Return [(section_id, section_title, headings, section_text)] using the
    section_pattern; tracks running heading context. If no section_pattern,
    the whole body is one section."""
    if not prof.section_pattern:
        return [("0", "", {}, body)]

    sec_re = re.compile(prof.section_pattern)
    reject_re = _compiled(prof.reject_section_pattern)
    footnote_re = _compiled(prof.footnote_rule_pattern)
    heading_res = [(name, re.compile(pat)) for name, pat in prof.heading_patterns]
    drop_res = [re.compile(p) for p in prof.drop_line_patterns]
    in_footnote = False

    sections: List[Tuple[str, str, Dict[str, str], str]] = []
    running: Dict[str, str] = {}
    cur_id: Optional[str] = None
    cur_title = ""
    cur_headings: Dict[str, str] = {}
    buf: List[str] = []

    def flush():
        if cur_id is not None:
            body_text = "\n".join(buf).strip()
            if body_text:
                sections.append((cur_id, cur_title, dict(cur_headings), body_text))

    for ln in body.splitlines():
        # footnote block: opened by a rule line, closed by a blank line
        if footnote_re and footnote_re.match(ln):
            in_footnote = True
            continue
        if not ln.strip():
            in_footnote = False
            continue
        if in_footnote:
            continue
        if any(r.match(ln) for r in drop_res):
            continue
        # running heading context (chapters/parts/...)
        heading_hit = False
        for name, hre in heading_res:
            hm = hre.match(ln)
            if hm:
                running[name] = (hm.group(1) if hm.groups() else ln.strip())
                heading_hit = True
                break
        if heading_hit:
            continue
        # section boundary (reject footnote-style candidates matched on the full line)
        sm = sec_re.match(ln)
        if sm and not (reject_re and reject_re.search(ln)):
            flush()
            cur_id = sm.group("id") if "id" in sm.groupdict() else (sm.group(1) if sm.groups() else str(len(sections)))
            cur_title = (sm.group("title") if "title" in sm.groupdict() else "").strip()
            cur_headings = dict(running)
            buf = [ln]
            continue
        if cur_id is not None:
            buf.append(ln)
        # text before the first section is ignored (front-matter)

    flush()
    return sections or [("0", "", {}, body)]


# ── driver ───────────────────────────────────────────────────────────────────
def build_chunks(raw: str, prof: ChunkProfile, doc_id: str = "", id_prefix: str = "") -> List[Chunk]:
    body = _slice_body(raw, prof)
    chunks: List[Chunk] = []
    for sec_id, sec_title, headings, sec_text in _sectionize(body, prof):
        pieces = size_chunks(sec_text, prof)
        for si, piece in enumerate(pieces):
            base = f"sec_{sec_id}" if prof.section_pattern else f"chunk_{len(chunks)}"
            cid = f"{id_prefix}{base}" if len(pieces) == 1 else f"{id_prefix}{base}__{si}"
            chunks.append(Chunk(
                chunk_id=cid,
                text=piece,
                doc_id=doc_id,
                section_id=sec_id,
                section_title=sec_title,
                headings=headings,
                metadata=_extract_metadata(piece, prof),
                sub_index=si,
                char_len=len(piece),
                token_estimate=_estimate_tokens(piece),
            ))
    return chunks


# ── profiles: the ONLY place domain knowledge lives ──────────────────────────
PROFILES: Dict[str, ChunkProfile] = {
    # pure size-based; works on any document
    "plain": ChunkProfile(name="plain", max_chars=1200, overlap=150),

    # example structural profile for Indian bare-acts / the Constitution.
    # Provided as configuration — the algorithm above knows nothing about it.
    "indian_statute": ChunkProfile(
        name="indian_statute",
        max_chars=2000,
        overlap=0,
        start_marker=r"WE,\s+THE\s+PEOPLE",
        stop_marker=r"^\s*\[?\s*(?:THE\s+)?FIRST\s+SCHEDULE\b",
        # a section starts at:  "<no>. <Title>..."  (title may wrap before the dash)
        section_pattern=r"^\s*\[?\s*(?P<id>\d+[A-Z]?)\.\s+(?P<title>.+?)(?:[—–-]|$)",
        heading_patterns=(("part", r"^\s*PART\s+((?=[IVXLC])(?:X{0,3})(?:IX|IV|V?I{0,3})[AB]?)\s*$"),),
        drop_line_patterns=(
            r"^\s*THE CONSTITUTION OF INDIA\s*$",
            r"^\s*\(Part[^)]*\)\s*$",
            r"^\s*\d{1,4}\s*$",
            r"^\s*_{5,}\s*$",
            r"^\s*ARTICLES\s*$",
        ),
        # reject footnote markers ("1. Subs. by ...", "1. The words ... omitted")
        reject_section_pattern=r"^\s*\d+[A-Z]?\.\s+(Subs|Ins|Added|Omitted|Rep|Cl|Sub|The words|The word|Certain words|w\.e\.f)\b",
        # amendment footnotes follow an underscore rule line and run until a blank line
        footnote_rule_pattern=r"^\s*_{5,}\s*$",
        metadata_extractors=(
            ("refs_article", r"\barticles?\s+(\d+[A-Z]?)"),
            ("refs_part", r"\bPart\s+([IVXLC]+[AB]?)\b"),
        ),
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic document chunker (profile-driven).")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--profile", default="plain", choices=sorted(PROFILES))
    ap.add_argument("--max-chars", type=int, default=None, help="Override profile max_chars.")
    ap.add_argument("--overlap", type=int, default=None, help="Override profile overlap.")
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    if args.max_chars is not None:
        prof.max_chars = args.max_chars
    if args.overlap is not None:
        prof.overlap = args.overlap

    raw = args.input.read_text(encoding="utf-8", errors="replace")
    chunks = build_chunks(raw, prof)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    toks = [c.token_estimate for c in chunks]
    n_sec = len({c.section_id for c in chunks})
    print(f"[{prof.name}] wrote {len(chunks)} chunks / {n_sec} sections -> {args.output}")
    if toks:
        print(f"avg tokens/chunk: {sum(toks)/len(toks):.0f} | max: {max(toks)} | >max_chars/4: {sum(t> prof.max_chars//4 for t in toks)}")


if __name__ == "__main__":
    main()
