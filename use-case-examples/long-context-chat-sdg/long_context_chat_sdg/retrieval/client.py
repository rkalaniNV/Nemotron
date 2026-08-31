# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HttpRetrievalClient — the thin wrapper over the external retrieval service.

The retrieval service is an external POST API that returns retrieved chunks for a
query. This wrapper adds exactly two behaviours and nothing else:

  1. OVERSAMPLE — ask the service for ``k * oversample_factor`` chunks.
  2. RANDOMIZE  — randomly subsample back down to ``k`` (deterministic given rng),
     so a single search is deliberately lossy and the agent must take more hops.

The request/response JSON shapes are configurable via ``field_map`` so the
wrapper adapts to the service's actual schema without code changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# request/response field names — override any subset via the config's field_map.
_DEFAULT_FIELD_MAP: Dict[str, Any] = {
    "query_field": "query",       # request: the query string
    "top_k_field": "top_k",       # request: how many chunks to ask for
    "results_path": "chunks",     # response: key holding the list (""/None => body is the list)
    "id_field": "id",             # response item: unique chunk id
    "text_field": "text",         # response item: chunk text
    "score_field": "score",       # response item: relevance score
    "doc_id_field": "doc_id",     # response item: source document id
    "extra_body": {},             # static fields merged into every request body
}


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0
    doc_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """What the assistant sees as the tool response (domain-agnostic keys)."""
        p = {"id": self.id, "text": self.text, "score": round(self.score, 4)}
        if self.doc_id:
            p["doc_id"] = self.doc_id
        return p


class HttpRetrievalClient:
    def __init__(self, endpoint: str, *, oversample_factor: int = 2, timeout: int = 30,
                 field_map: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
                 max_retries: int = 2, backoff: float = 1.0,
                 post_fn: Optional[Callable[..., Any]] = None):
        self.endpoint = endpoint
        self.oversample_factor = max(1, int(oversample_factor))
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))     # extra attempts on transient endpoint errors
        self.backoff = float(backoff)                   # base seconds between attempts (linear)
        self.fm = {**_DEFAULT_FIELD_MAP, **(field_map or {})}
        self.headers = headers or {"Content-Type": "application/json"}
        self._post_fn = post_fn  # injectable for tests; defaults to requests.post

    # ── HTTP ──────────────────────────────────────────────────────────────────
    def _post(self, query: str, n: int) -> Any:
        """POST with bounded retries; raises the last error only after all attempts fail."""
        import time
        body = {self.fm["query_field"]: query, self.fm["top_k_field"]: n, **self.fm.get("extra_body", {})}
        post = self._post_fn
        if post is None:
            import requests
            post = requests.post
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = post(self.endpoint, json=body, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:                     # timeout / 5xx / conn reset / bad JSON
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (attempt + 1))
        raise last_exc if last_exc else RuntimeError("retrieval POST failed")

    def _parse(self, payload: Any) -> List[Chunk]:
        path = self.fm.get("results_path")
        items = payload.get(path, []) if (path and isinstance(payload, dict)) else payload
        if not isinstance(items, list):
            return []
        chunks: List[Chunk] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = str(it.get(self.fm["text_field"], ""))
            # prefer a real id; else a stable content hash (so chunks remain citable
            # even when the service returns no id).
            cid = it.get(self.fm["id_field"]) or it.get("chunk_id")
            cid = str(cid) if cid is not None else "h" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            reserved = {self.fm["id_field"], self.fm["text_field"], self.fm["score_field"],
                        self.fm["doc_id_field"]}
            chunks.append(Chunk(
                id=cid,
                text=text,
                score=float(it.get(self.fm["score_field"], 0.0) or 0.0),
                doc_id=str(it.get(self.fm["doc_id_field"], "") or ""),
                meta={k: v for k, v in it.items() if k not in reserved}))
        return chunks

    # ── the one method the generator calls ────────────────────────────────────
    def retrieve(self, query: str, k: int, *, rng) -> List[Chunk]:
        """Oversample ``k * oversample_factor``, then randomly keep ``k`` (deterministic
        given ``rng``). On persistent failure return [] (empty hop) rather than raise."""
        try:
            payload = self._post(query, k * self.oversample_factor)
        except Exception:
            return []
        pool = self._parse(payload)
        if len(pool) > k:
            idx = sorted(rng.sample(range(len(pool)), k))  # random subset, original (score) order preserved
            pool = [pool[i] for i in idx]
        return pool
