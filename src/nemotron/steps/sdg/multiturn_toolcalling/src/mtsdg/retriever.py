"""Client for the live NeMo Retriever running on the box.

    POST http://<host>:8000/query  {"query": str, "num_chunks": int}
    -> {"chunks": [{"rank", "text", "source", "page_number", "distance", ...}]}

Called live during generation: the assistant issues the actual query and the
retriever answers from the real document index. A deliberately vague query surfaces
weak/adjacent chunks; a precise rewrite surfaces the authoritative ones, so the
retrieve -> assess -> rewrite loop is genuine and grounded in real documents.
"""

from __future__ import annotations

import os
import re
from typing import List

from mtsdg.schemas import RetrievalChunk

DEFAULT_RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://localhost:8000")


def _chunk_id(source: str, page: int, rank: int) -> str:
    base = os.path.basename(source or "doc").replace(".txt", "")
    return f"{base}_p{page}_r{rank}"


def _title(source: str, page: int) -> str:
    base = os.path.basename(source or "doc").replace(".txt", "")
    return f"Constitution of India — {base}, p.{page}"


def _clean(text: str) -> str:
    # Collapse the noisy page headers/footers a little; keep legal text intact.
    text = re.sub(r"\n{3,}", "\n\n", text or "")
    return text.strip()


class RetrieverClient:
    """Thin HTTP client over the /query endpoint."""

    def __init__(self, base_url: str = DEFAULT_RETRIEVER_URL, *, timeout: int = 45, retries: int = 4):
        self.url = base_url.rstrip("/") + "/query"
        self.timeout = timeout
        self.retries = retries

    def query(self, text: str, num_chunks: int = 3) -> List[RetrievalChunk]:
        import time as _time

        import httpx

        # The retriever's vLLM backend can briefly stall under burst load; retry
        # with backoff so a transient hang doesn't drop a turn's grounding.
        last_exc: Exception | None = None
        data = None
        for attempt in range(self.retries):
            try:
                resp = httpx.post(
                    self.url, json={"query": text, "num_chunks": num_chunks}, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:  # timeout / connection / 5xx
                last_exc = exc
                _time.sleep(min(2 ** attempt, 15))
        if data is None:
            raise RuntimeError(f"retriever unavailable after {self.retries} attempts: {last_exc}")
        out: List[RetrievalChunk] = []
        for ch in data.get("chunks", []):
            page = int(ch.get("page_number", 0) or 0)
            rank = int(ch.get("rank", len(out) + 1) or (len(out) + 1))
            source = ch.get("source", "doc")
            out.append(
                RetrievalChunk(
                    chunk_id=_chunk_id(source, page, rank),
                    title=_title(source, page),
                    content=_clean(ch.get("text", "")),
                    source=os.path.basename(source or "doc"),
                    url=None,
                    date=None,
                )
            )
        return out

    def health(self) -> bool:
        import httpx

        try:
            r = httpx.get(self.url.replace("/query", "/health"), timeout=10)
            return r.status_code == 200
        except Exception:
            return False
