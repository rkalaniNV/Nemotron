"""OpenAI-compatible client for the local vLLM server.

Every model call is cached on disk, keyed by a hash of everything that determines the
response. Three reasons, all of them about the ablation rather than about speed:

  reproducible   an arm re-run from the same cache produces the same benchmark, so a
                 later comparison measures the change under test and not sampling noise
  auditable      the cache is the record of what the model was actually asked and what
                 it actually said, which is the lineage an LLM-authored arm owes
  cheap          A2 and A3 are re-run many times while their harnesses are debugged

`gpt-oss-120b` is a reasoning model: the answer arrives in `message.content` while the
chain of thought arrives in `message.reasoning`. A small `max_tokens` is silently
consumed by reasoning and yields `content: null`, so the floor here is deliberately
generous.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_BASE_URL = os.environ.get("BFCL_ABLATION_LLM_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("BFCL_ABLATION_LLM_MODEL", "openai/gpt-oss-120b")

# Reasoning tokens are drawn from the same budget as the answer, so a ceiling tuned
# for a non-reasoning model returns an empty completion rather than a short one.
MIN_MAX_TOKENS = 512


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or will not answer usably."""


@dataclass
class LLMClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "_generated" / "llm_cache")
    timeout_s: float = 600.0
    max_retries: int = 3
    concurrency: int = 8

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0
        self.cache_hits = 0

    # -- cache ------------------------------------------------------------------

    def _key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps({"url": self.base_url, **payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def _store(self, key: str, record: dict[str, Any]) -> None:
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- transport --------------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = 0,
    ) -> str:
        """Return the model's answer text, from cache when the same call was made before."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max(max_tokens, MIN_MAX_TOKENS),
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed

        key = self._key(payload)
        cached = self._cached(key)
        if cached is not None:
            self.cache_hits += 1
            return cached["content"]

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = self._post(payload)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(2 * (attempt + 1))
                continue
            message = (response.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content")
            if content:
                self.calls_made += 1
                self._store(
                    key,
                    {
                        "model": self.model,
                        "system": system,
                        "user": user,
                        "content": content,
                        "reasoning": message.get("reasoning"),
                        "usage": response.get("usage"),
                        # Encodes the serving stack (vllm version, tensor-parallel
                        # layout, build hash). Without it the cache records what was
                        # asked but not what answered, and a re-run on a rebuilt server
                        # would look reproducible when it is not.
                        "system_fingerprint": response.get("system_fingerprint"),
                    },
                )
                return content
            # An empty answer means reasoning consumed the budget; buy more of it.
            payload["max_tokens"] = int(payload["max_tokens"] * 2)
            last_error = "model returned empty content (reasoning consumed max_tokens)"

        raise LLMError(f"no usable completion after {self.max_retries} attempts: {last_error}")

    # -- structured output ------------------------------------------------------

    def json_object(
        self,
        *,
        system: str,
        user: str,
        validate: Callable[[Any], Any] | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        seed: int | None = 0,
        attempts: int = 3,
    ) -> Any:
        """Ask for JSON and keep asking until it parses and validates.

        The retry varies the seed rather than the prompt, so a failure to parse does
        not quietly become a different question.
        """
        instruction = f"{system}\n\nReply with JSON only. No prose, no markdown fence."
        last_error = ""
        for attempt in range(attempts):
            raw = self.complete(
                system=instruction,
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None if seed is None else seed + attempt,
            )
            parsed = _extract_json(raw)
            if parsed is None:
                last_error = f"unparseable JSON: {raw[:200]}"
                continue
            if validate is None:
                return parsed
            try:
                return validate(parsed)
            except (ValueError, TypeError, KeyError) as error:
                last_error = f"{type(error).__name__}: {error}"
        raise LLMError(f"no valid JSON after {attempts} attempts: {last_error}")

    # -- batching ---------------------------------------------------------------

    def map(self, jobs: Sequence[Callable[[], Any]]) -> list[Any]:
        """Run independent calls concurrently, preserving order.

        A job that raises resolves to None rather than killing the batch: one refused
        paraphrase should cost one row, not the arm.
        """
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(job) for job in jobs]
            results: list[Any] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception:  # noqa: BLE001 - reported by the caller as a null row
                    results.append(None)
        return results

    def stats(self) -> dict[str, int]:
        return {"calls_made": self.calls_made, "cache_hits": self.cache_hits}


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(raw: str) -> Any | None:
    """Recover a JSON value from a reply that may be fenced or have a preamble."""
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    return None


def probe(client: LLMClient | None = None) -> dict[str, Any]:
    """Confirm the endpoint answers before an arm spends an hour discovering it does not."""
    client = client or LLMClient()
    request = urllib.request.Request(f"{client.base_url}/models", method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        served = [entry["id"] for entry in json.loads(response.read().decode("utf-8"))["data"]]
    if client.model not in served:
        raise LLMError(f"model {client.model!r} is not served; available: {', '.join(served)}")
    answer = client.complete(system="Answer in one word.", user="Capital of France?")
    return {"base_url": client.base_url, "model": client.model, "served": served, "smoke": answer.strip()}
