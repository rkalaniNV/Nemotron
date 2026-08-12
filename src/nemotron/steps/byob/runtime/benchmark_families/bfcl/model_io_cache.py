"""Content-addressed, append-only cache for deterministic model stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

CACHE_SCHEMA_VERSION = "1.1"


def request_hash(
    *,
    model_canonical: str,
    prompt_hash: str,
    model_input: Any,
    inference_parameters: dict[str, Any],
    output_schema: dict[str, Any],
    seed: int,
) -> str:
    """Identify every input that can affect a model response."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_canonical": model_canonical.strip().lower(),
        "prompt_hash": prompt_hash,
        "input": model_input,
        "inference_parameters": inference_parameters,
        "output_schema": output_schema,
        "seed": seed,
    }
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


class ImmutableModelIOCache:
    """Read and append model responses without replacing prior observations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries = self._read_entries()

    def _read_entries(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid model I/O cache JSON at {self.path}:{line_number}") from exc
            key = entry.get("request_hash") if isinstance(entry, dict) else None
            if not isinstance(key, str):
                raise ValueError(f"model I/O cache entry {self.path}:{line_number} has no request_hash")
            if "response" not in entry:
                raise ValueError(f"model I/O cache entry {self.path}:{line_number} has no response")
            response_hash = entry.get("response_hash")
            actual_response_hash = (
                "sha256:"
                + hashlib.sha256(canonical_json(entry["response"]).encode()).hexdigest()
            )
            if response_hash != actual_response_hash:
                raise ValueError(
                    f"model I/O cache entry {self.path}:{line_number} has an invalid response_hash"
                )
            if key in entries and entries[key] != entry:
                raise ValueError(f"model I/O cache contains conflicting entries for {key}")
            entries[key] = entry
        return entries

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        return None if entry is None else entry["response"]

    def put(
        self,
        key: str,
        response: Any,
        *,
        model_canonical: str,
        input_hash: str,
    ) -> dict[str, Any]:
        response_json = canonical_json(response)
        entry = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "request_hash": key,
            "model_canonical": model_canonical.strip().lower(),
            "input_hash": input_hash,
            "response_hash": (f"sha256:{hashlib.sha256(response_json.encode()).hexdigest()}"),
            "response": response,
        }
        existing = self._entries.get(key)
        if existing is not None:
            if existing != entry:
                raise ValueError(f"refusing to replace immutable model I/O cache entry {key}")
            return existing

        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(descriptor)
        self._entries[key] = entry
        return entry
