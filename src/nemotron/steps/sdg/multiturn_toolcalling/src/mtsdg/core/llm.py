"""Self-contained LLM utilities for the multi-turn long-context pipeline.

Depends only on Data Designer (``ChatMessage``) and pydantic. Provides:

- message conversion + response parsing,
- a sync/async-tolerant completion shim (DD's async engine constructs the model
  client in ASYNC mode, which refuses sync ``completion`` even from the worker
  thread that runs a cell-by-cell generator),
- Pydantic-validated structured completion,
- majority voting over structured assistant turns,
- an inline ``<explanation>/<rating>`` judge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Tuple, Type, TypeVar

from pydantic import BaseModel

from data_designer.engine.models.utils import ChatMessage

T = TypeVar("T", bound=BaseModel)

_log = logging.getLogger("mtsdg")

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
_BRACES = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


# --------------------------------------------------------------------------- #
# message conversion + response parsing (vendored, DD-only)
# --------------------------------------------------------------------------- #


def _dicts_to_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    out: List[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            out.append(ChatMessage.as_system(content))
        elif role == "assistant":
            out.append(
                ChatMessage.as_assistant(
                    content=content,
                    reasoning_content=msg.get("reasoning_content"),
                    tool_calls=msg.get("tool_calls") or None,
                )
            )
        elif role == "tool":
            out.append(ChatMessage.as_tool(content, msg.get("tool_call_id", "")))
        else:
            out.append(ChatMessage.as_user(content))
    return out


def _response_choice_to_dict(choice: Any) -> Dict[str, Any]:
    msg = choice.message
    result: Dict[str, Any] = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", "") or "",
    }
    if getattr(msg, "reasoning_content", None):
        result["reasoning_content"] = msg.reasoning_content
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        if not isinstance(tcs, (list, tuple)):
            tcs = [tcs]  # some providers return a single tool-call object
        out: List[Dict[str, Any]] = []
        for tc in tcs:
            if hasattr(tc, "model_dump"):
                out.append(tc.model_dump())
            elif isinstance(tc, dict):
                out.append(tc)
            else:
                fn = getattr(tc, "function", None)
                out.append({
                    "id": getattr(tc, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(fn, "name", None) if fn is not None else None,
                        "arguments": getattr(fn, "arguments", "{}") if fn is not None else "{}",
                    },
                })
        result["tool_calls"] = out
    return result


# --------------------------------------------------------------------------- #
# sync/async-tolerant completion
# --------------------------------------------------------------------------- #

# A SINGLE dedicated event loop, running in one background thread, services every
# async completion. DD's async engine runs cell generators across many worker
# threads; if each used its own loop, the model facade's shared async HTTP client
# (whose internal asyncio primitives bind to the loop that first awaited it) would
# be awaited across different loops -> "Event is bound to a different event loop"
# at high concurrency. Routing all acompletion calls through one loop keeps the
# client's I/O on a single loop while still running many calls concurrently.
_GLOBAL_LOOP: "asyncio.AbstractEventLoop | None" = None
_GLOBAL_LOOP_LOCK = threading.Lock()


def _global_loop() -> asyncio.AbstractEventLoop:
    global _GLOBAL_LOOP
    if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
        with _GLOBAL_LOOP_LOCK:
            if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, daemon=True, name="mtsdg-llm-loop").start()
                _GLOBAL_LOOP = loop
    return _GLOBAL_LOOP


def _completion(facade: Any, chat_messages, **kwargs):
    try:
        return facade.completion(chat_messages, **kwargs)
    except Exception as exc:  # SyncClientUnavailableError under DD's async engine
        if "Sync methods are not available" in str(exc) or type(exc).__name__ == "SyncClientUnavailableError":
            fut = asyncio.run_coroutine_threadsafe(
                facade.acompletion(chat_messages, **kwargs), _global_loop()
            )
            return fut.result()
        raise


#: Substrings that mark a transient, retryable endpoint error (rate limits on a
#: shared/saturated model, timeouts, 5xx, overload). Retried with backoff so a
#: single 429 doesn't kill a whole multi-turn episode.
_TRANSIENT_MARKERS = (
    "rate limit", "ratelimit", "429", "saturation", "too many requests",
    "timeout", "timed out", "overloaded", "temporarily", "503", "502", "500",
    "connection", "reset by peer", "service unavailable",
)
_MAX_LLM_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
#: Rate-limit / quota exhaustion needs to ride out per-minute reset windows, so it
#: gets more attempts and longer backoff than ordinary transient errors.
_MAX_RATELIMIT_RETRIES = int(os.environ.get("LLM_MAX_RATELIMIT_RETRIES", "8"))
_RATELIMIT_MARKERS = ("rate limit", "ratelimit", "429", "resource_exhausted", "exhausted", "quota")


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


def _is_ratelimit(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _RATELIMIT_MARKERS)


def call_llm(
    models: Dict[str, Any], alias: str, messages: List[Dict[str, Any]], **kwargs: Any
) -> Dict[str, Any] | List[Dict[str, Any]]:
    facade = models.get(alias)
    if facade is None:
        raise ValueError(f"Model alias '{alias}' not found in models dict")
    chat_messages = _dicts_to_chat_messages(messages)

    response = None
    last_exc: Exception | None = None
    attempt = 0
    while True:
        try:
            response = _completion(facade, chat_messages, **kwargs)
            break
        except Exception as exc:
            last_exc = exc
            ratelimit = _is_ratelimit(exc)
            transient = ratelimit or _is_transient(exc)
            max_attempts = _MAX_RATELIMIT_RETRIES if ratelimit else _MAX_LLM_RETRIES
            if not transient or attempt >= max_attempts - 1:
                raise
            # Rate-limit/quota: longer backoff (rides per-minute resets); other
            # transient: shorter. Jitter avoids thundering-herd re-tries.
            if ratelimit:
                delay = min(5.0 * (1.6 ** attempt), 45.0) + (attempt % 5)
            else:
                delay = min(1.5 * (2 ** attempt), 30.0) + (0.1 * attempt)
            _log.warning(
                "LLM transient error (alias=%s, attempt %d/%d): %s — retrying in %.1fs",
                alias, attempt + 1, max_attempts, str(exc)[:180], delay,
            )
            time.sleep(delay)
            attempt += 1
    if response is None:  # pragma: no cover - defensive
        raise last_exc if last_exc else RuntimeError("call_llm: no response")

    n = kwargs.get("n", 1)
    if n > 1 and len(response.choices) >= n:
        return [_response_choice_to_dict(c) for c in response.choices]
    return _response_choice_to_dict(response.choices[0])


# --------------------------------------------------------------------------- #
# structured completion
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Any:
    text = text.strip()
    for pat in (_JSON_FENCE, _BRACES):
        m = pat.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def call_structured(
    models: Dict[str, Any],
    alias: str,
    messages: List[Dict[str, Any]],
    schema: Type[T],
    *,
    max_repair_attempts: int = 2,
    **kwargs: Any,
) -> T:
    request_kwargs = dict(kwargs)
    request_kwargs.setdefault("response_format", {"type": "json_object"})
    convo = list(messages)
    last_err: Exception | None = None
    for _ in range(max_repair_attempts + 1):
        try:
            resp = call_llm(models, alias, convo, **request_kwargs)
        except Exception:
            request_kwargs.pop("response_format", None)
            resp = call_llm(models, alias, convo, **request_kwargs)
        text = resp.get("content", "") if isinstance(resp, dict) else ""
        try:
            return schema.model_validate(_extract_json(text))
        except Exception as exc:
            last_err = exc
            convo = list(messages) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required JSON schema "
                        f"({exc}). Return ONLY a single valid JSON object conforming to the schema "
                        "below. Do NOT apologize or add any commentary, and keep the real "
                        "user-facing answer inside the `content` field.\n"
                        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                    ),
                },
            ]
    raise ValueError(f"call_structured failed to produce valid {schema.__name__}: {last_err}")


def call_structured_n(
    models: Dict[str, Any],
    alias: str,
    messages: List[Dict[str, Any]],
    schema: Type[T],
    n: int,
    **kwargs: Any,
) -> List[T]:
    """Draw ``n`` independent structured samples (for majority voting).

    Uses ``n`` separate calls rather than the ``n`` request param, since the
    proxy/GPT-5-family endpoint does not reliably return multiple choices.
    """
    out: List[T] = []
    for _ in range(max(1, n)):
        try:
            out.append(call_structured(models, alias, messages, schema, **kwargs))
        except Exception as exc:
            # Do not silently swallow: a stalled/timing-out backend showed up here
            # as an invisible empty result. Log it so the failure mode is visible.
            _log.warning("structured sample failed (alias=%s, schema=%s): %s",
                         alias, schema.__name__, str(exc)[:180])
            continue
    return out


# --------------------------------------------------------------------------- #
# majority voting over assistant turns
# --------------------------------------------------------------------------- #


def _tool_pattern(tool_calls: List[Dict[str, Any]] | None) -> Tuple[str, ...]:
    return tuple((tc.get("function", {}) or {}).get("name") for tc in (tool_calls or []))


def majority_vote_tool_calls(candidates: List[Any]) -> Any:
    """Pick the consensus assistant turn from structured candidates.

    ``candidates`` are objects with ``.tool_calls`` (list of OpenAI tool-call
    dicts), ``.content``, ``.reasoning``. If a tool-call name-pattern is shared by
    a majority, return a candidate with that pattern (args majority-voted per
    call); otherwise return the first final-answer (no-tool) candidate.
    """
    if not candidates:
        raise ValueError("no candidates to vote on")
    patterns = [_tool_pattern(getattr(c, "tool_calls", None)) for c in candidates]
    non_empty = [p for p in patterns if p]
    threshold = max(2, (len(candidates) + 1) // 2)
    if non_empty:
        winner, count = Counter(non_empty).most_common(1)[0]
        if count >= threshold:
            matching = [c for c, p in zip(candidates, patterns) if p == winner]
            return _merge_tool_call_args(matching)
    finals = [c for c, p in zip(candidates, patterns) if not p]
    return finals[0] if finals else candidates[0]


def _merge_tool_call_args(matching: List[Any]) -> Any:
    """Majority-vote scalar arguments across candidates sharing a tool pattern."""
    base = matching[0]
    if len(matching) == 1 or not base.tool_calls:
        return base
    voted_calls = []
    for i, tc in enumerate(base.tool_calls):
        arg_sets = []
        for c in matching:
            if i < len(c.tool_calls):
                fn = c.tool_calls[i].get("function", {}) or {}
                raw = fn.get("arguments", "{}")
                try:
                    arg_sets.append(json.loads(raw) if isinstance(raw, str) else dict(raw))
                except (json.JSONDecodeError, TypeError):
                    pass
        merged: Dict[str, Any] = {}
        for key in {k for a in arg_sets for k in a}:
            vals = [json.dumps(a[key], sort_keys=True, ensure_ascii=False) for a in arg_sets if key in a]
            if vals:
                winner = Counter(vals).most_common(1)[0][0]
                merged[key] = json.loads(winner)
        fn = tc.get("function", {}) or {}
        voted_calls.append({
            "id": tc.get("id", f"call_{i}"),
            "type": "function",
            "function": {"name": fn.get("name"), "arguments": json.dumps(merged, ensure_ascii=False)},
        })
    base.tool_calls = voted_calls
    return base


# --------------------------------------------------------------------------- #
# inline judge
# --------------------------------------------------------------------------- #

JUDGE_FOLLOWUP_PROMPT = """Please reformat your previous response to strictly follow this format:

```
<explanation>
[Your detailed explanation and justification for the rating.]
</explanation>
<rating>
[Must be either 'success' or 'failure' strictly]
</rating>
```"""


def _parse_judge_response(text: str) -> Tuple[str, str, bool]:
    explanation = None
    rating = None
    exp = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    rat = re.search(r"<rating>(.*?)</rating>", text, re.DOTALL)
    if exp:
        explanation = exp.group(1).strip()
    if rat:
        rt = rat.group(1).strip().lower()
        if "success" in rt:
            rating = "success"
        elif "failure" in rt:
            rating = "failure"
    return explanation or "", rating or "failure", rating == "success"


def run_inline_judge(models: Dict[str, Any], alias: str, prompt_text: str) -> Tuple[str, str, bool]:
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": prompt_text}]
    resp = call_llm(models, alias, msgs)
    text = resp.get("content", "") if isinstance(resp, dict) else ""
    explanation, rating, success = _parse_judge_response(text)
    if rating not in ("success", "failure"):
        msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": JUDGE_FOLLOWUP_PROMPT}]
        resp2 = call_llm(models, alias, msgs)
        text2 = resp2.get("content", "") if isinstance(resp2, dict) else ""
        explanation, rating, success = _parse_judge_response(text2)
    return explanation, rating, success
