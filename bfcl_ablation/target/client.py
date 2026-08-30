"""Tool-calling client for the target model.

`llm.py` posts to `/chat/completions` and reads `message.content`, which is right for
every generative arm but cannot be reused here for two reasons:

  1. it never sends `tools` and never reads `tool_calls`, and a tool-calling reply has
     `content: null` — the existing retry loop would read that as an empty completion,
     double `max_tokens` and eventually raise;
  2. the local vLLM is served without `--enable-auto-tool-choice --tool-call-parser`,
     so `/chat/completions` *silently drops* the call it generated. Sending `tools`
     there returns `tool_calls: null` even when the model called a function, which
     would score every task as "emitted no call" and look like a model result.

`/v1/responses` parses harmony natively on the same server and returns the call as a
`function_call` item, so that is the endpoint. Caching follows `llm.py` exactly —
keyed on everything that determines the reply, storing the server fingerprint so a
rebuilt server cannot masquerade as a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bfcl_ablation.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError

CACHE_DIR = Path(__file__).resolve().parent.parent / "_generated" / "target_cache"

# Reasoning shares the answer budget on gpt-oss; too low a ceiling returns a reasoning
# item and no function_call, which is indistinguishable from a refusal to call.
MIN_OUTPUT_TOKENS = 2048


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    """One model turn: the calls it made, and the text it said if it made none."""

    calls: list[ToolCall]
    text: str
    raw: list[dict[str, Any]]


@dataclass
class TargetClient:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    timeout_s: float = 600.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0
        self.cache_hits = 0

    def _key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps({"url": self.base_url, **payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def respond(
        self,
        *,
        instructions: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        max_output_tokens: int = MIN_OUTPUT_TOKENS,
    ) -> Reply:
        """One model turn against the Responses API, cached on disk."""
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": conversation,
            "tools": tools,
            "temperature": temperature,
            "max_output_tokens": max(max_output_tokens, MIN_OUTPUT_TOKENS),
        }
        key = self._key(payload)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                record = None
            if record is not None:
                self.cache_hits += 1
                return _reply(record["output"])

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = self._post(payload)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
                time.sleep(2 * (attempt + 1))
                continue
            output = response.get("output") or []
            self.calls_made += 1
            path.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "instructions": instructions,
                        "input": conversation,
                        "tools": tools,
                        "output": output,
                        "usage": response.get("usage"),
                        "system_fingerprint": response.get("system_fingerprint"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return _reply(output)

        raise LLMError(f"no response after {self.max_retries} attempts: {last_error}")

    def stats(self) -> dict[str, int]:
        return {"calls_made": self.calls_made, "cache_hits": self.cache_hits}


def _reply(output: list[dict[str, Any]]) -> Reply:
    """Split a Responses `output` list into calls and prose.

    A malformed `arguments` string is kept as a call with empty arguments rather than
    dropped: "called the right tool with unparseable arguments" and "called nothing"
    are different failures and must not collapse into one.
    """
    calls: list[ToolCall] = []
    chunks: list[str] = []
    for item in output:
        kind = item.get("type")
        if kind == "function_call":
            raw = item.get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except ValueError:
                parsed = {}
            calls.append(ToolCall(name=str(item.get("name") or ""), arguments=parsed if isinstance(parsed, dict) else {}))
        elif kind == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    chunks.append(str(part["text"]))
    return Reply(calls=calls, text="\n".join(chunks).strip(), raw=output)


def to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten OpenAI chat-style tool schemas into the Responses shape.

    The pack's `tools.json` carries `x-mutates` / `x-requires-confirmation` alongside
    the schema. Those are oracle metadata, not part of the contract shown to a model,
    and passing them through would leak which tools change state.
    """
    flattened: list[dict[str, Any]] = []
    for tool in tools:
        body = tool.get("function") if "function" in tool else tool
        flattened.append(
            {
                "type": "function",
                "name": body.get("name"),
                "description": body.get("description") or "",
                "parameters": body.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return flattened
