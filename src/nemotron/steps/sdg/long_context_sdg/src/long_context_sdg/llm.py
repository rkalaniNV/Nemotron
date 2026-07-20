"""Small sync/async-tolerant LLM facade helpers for Data Designer models."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
_LOOP = None
_LOOP_LOCK = threading.Lock()


def _global_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True, name="long-context-sdg-llm").start()
            _LOOP = loop
    return _LOOP


def _chat_messages(messages: list[dict[str, Any]]):
    try:
        from data_designer.engine.models.utils import ChatMessage
    except Exception:
        return messages
    out = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content") or ""
        if role == "system":
            out.append(ChatMessage.as_system(content))
        elif role == "assistant":
            out.append(
                ChatMessage.as_assistant(
                    content=content,
                    reasoning_content=m.get("reasoning_content"),
                    tool_calls=m.get("tool_calls"),
                )
            )
        elif role == "tool":
            out.append(ChatMessage.as_tool(content, m.get("tool_call_id") or ""))
        else:
            out.append(ChatMessage.as_user(content))
    return out


def _completion(facade: Any, messages, **kwargs):
    try:
        return facade.completion(messages, **kwargs)
    except Exception as exc:
        if not callable(getattr(facade, "acompletion", None)):
            raise
        future = asyncio.run_coroutine_threadsafe(facade.acompletion(messages, **kwargs), _global_loop())
        try:
            return future.result()
        except Exception:
            raise exc


def _as_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        if "choices" in response:
            return _as_dict(response["choices"][0])
        if "message" in response:
            return _as_dict(response["message"])
        return response
    if hasattr(response, "choices"):
        return _as_dict(response.choices[0])
    if hasattr(response, "message"):
        return _as_dict(response.message)
    content = getattr(response, "content", "") or ""
    out = {"role": getattr(response, "role", "assistant"), "content": content}
    reasoning = getattr(response, "reasoning_content", None) or getattr(response, "reasoning", None)
    if reasoning:
        out["reasoning_content"] = reasoning
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [tc if isinstance(tc, dict) else tc.model_dump() for tc in tool_calls]
    return out


def call_llm(
    models: dict[str, Any],
    alias: str,
    messages: list[dict[str, Any]],
    *,
    retries: int = 8,
    **kwargs: Any,
) -> dict[str, Any]:
    facade = models.get(alias)
    if facade is None:
        raise ValueError(f"model alias `{alias}` not available")
    last_error = None
    for attempt in range(retries):
        try:
            return _as_dict(_completion(facade, _chat_messages(messages), **kwargs))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 5))
    raise RuntimeError(f"model `{alias}` failed after {retries} attempt(s): {last_error}")


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            return value
        raise


def call_structured(
    models: dict[str, Any],
    alias: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    *,
    attempts: int = 3,
) -> T:
    current = list(messages)
    last_error: Exception | None = None
    for _ in range(attempts):
        response = call_llm(models, alias, current)
        try:
            return schema.model_validate(_extract_json(response.get("content", "")))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
            current = current + [
                {
                    "role": "user",
                    "content": "Return only one valid JSON object matching this schema: "
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False),
                }
            ]
    raise ValueError(f"model `{alias}` did not produce valid {schema.__name__}: {last_error}")
