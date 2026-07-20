"""Configurable, retrying HTTP retrieval adapter."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable
from typing import Any

import httpx

from .schemas import RetrievalChunk
from .service_config import RetrieverConfig


def _get_path(value: Any, path: str, default: Any = None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _stable_chunk_id(item: dict[str, Any], text: str) -> str:
    raw = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) if item else text
    return "h-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class RetrieverClient:
    def __init__(self, config: RetrieverConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._client = httpx.Client(
            transport=transport,
            timeout=config.timeout_seconds,
            headers=config.headers,
        )

    def close(self) -> None:
        self._client.close()

    def query(self, text: str, *, top_k: int | None = None) -> list[RetrievalChunk]:
        query = text.strip()
        if not query:
            raise ValueError("retrieval query must be non-empty")
        k = top_k if isinstance(top_k, int) and top_k > 0 else self.config.top_k
        body = {
            **self.config.extra_body,
            self.config.query_field: query,
            self.config.top_k_field: k,
        }
        payload = self._request(body)
        raw_items = _get_path(payload, self.config.results_path, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"retrieval results_path `{self.config.results_path}` is not a list")
        chunks = [self._chunk(item) for item in raw_items if isinstance(item, dict)]
        chunks = [c for c in chunks if c.content]
        return self._select(chunks, k, query)

    def _request(self, body: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.retries):
            try:
                if self.config.method == "GET":
                    response = self._client.get(self.config.endpoint, params=body)
                else:
                    response = self._client.post(self.config.endpoint, json=body)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.config.retries and self.config.backoff_seconds:
                    time.sleep(min(self.config.backoff_seconds * (2**attempt), 15))
        raise RuntimeError(f"retriever unavailable after {self.config.retries} attempt(s): {last_error}")

    def _chunk(self, item: dict[str, Any]) -> RetrievalChunk:
        f = self.config.fields
        text = str(_get_path(item, f.text, "") or "").strip()
        chunk_id = str(_get_path(item, f.id, "") or _stable_chunk_id(item, text))
        score = _get_path(item, f.score)
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        consumed = {x.split(".")[0] for x in f.model_dump().values() if x}
        return RetrievalChunk(
            chunk_id=chunk_id,
            content=text,
            title=str(_get_path(item, f.title, "") or ""),
            source=str(_get_path(item, f.source, "") or ""),
            score=score,
            url=_optional_str(_get_path(item, f.url)),
            date=_optional_str(_get_path(item, f.date)),
            metadata={k: v for k, v in item.items() if k not in consumed},
        )

    def _select(self, chunks: list[RetrievalChunk], k: int, query: str) -> list[RetrievalChunk]:
        if len(chunks) <= k:
            return chunks
        if self.config.selection == "sampled":
            seed = int(hashlib.sha256(query.encode()).hexdigest()[:16], 16)
            indices = sorted(random.Random(seed).sample(range(len(chunks)), k))
            return [chunks[i] for i in indices]
        if self.config.selection == "diverse":
            return _diverse(chunks, k)
        return chunks[:k]


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _diverse(chunks: Iterable[RetrievalChunk], k: int) -> list[RetrievalChunk]:
    pending = list(chunks)
    chosen: list[RetrievalChunk] = []
    seen_sources = set()
    for chunk in pending:
        key = chunk.source or chunk.title or chunk.chunk_id
        if key not in seen_sources:
            chosen.append(chunk)
            seen_sources.add(key)
        if len(chosen) == k:
            return chosen
    chosen_ids = {c.chunk_id for c in chosen}
    chosen.extend(c for c in pending if c.chunk_id not in chosen_ids)
    return chosen[:k]
