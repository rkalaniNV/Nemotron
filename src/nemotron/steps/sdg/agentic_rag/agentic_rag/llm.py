"""LLM call helpers + majority voting. Reused from the tool-calling reference."""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from contextvars import ContextVar
from typing import Any, Dict, List

from data_designer.engine.models.utils import ChatMessage

# ── direct OpenAI-compatible clients (bypass DD's facade) ─────────────────────
# DD's facade drops tool-call function names for this endpoint (every model), so
# tool-calling is routed through a direct OpenAI client per alias instead. The
# client config (endpoint / key env / model / params) is injected from the outer
# YAML via the plugin config and registered here once per process.
_DIRECT_CLIENTS: Dict[str, Any] = {}


def configure_direct_clients(model_clients: Dict[str, Any]) -> None:
    if not model_clients:
        return
    from openai import OpenAI
    for alias, c in model_clients.items():
        if alias in _DIRECT_CLIENTS or not c.get("base_url"):
            continue
        client = OpenAI(base_url=c["base_url"],
                        api_key=os.environ.get(c.get("api_key_env", ""), ""),
                        max_retries=4, timeout=180)
        _DIRECT_CLIENTS[alias] = {"client": client, "model": c["model"],
                                  "params": c.get("params", {})}


def _to_openai_message(m: Dict[str, Any]) -> Dict[str, Any]:
    role = m.get("role", "user")
    # some NIM endpoints reject empty string content (min_length 1), even on an
    # assistant turn that only carries tool_calls — send a single space instead.
    content = (m.get("content") or "").strip() or " "
    out: Dict[str, Any] = {"role": role, "content": content}
    if role == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = _clean_tool_calls(m.get("tool_calls"))
    if role == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or ""
    return out


def _direct_completion(alias: str, messages: List[Dict[str, Any]], **kwargs):
    reg = _DIRECT_CLIENTS[alias]
    n = kwargs.get("n", 1)
    req: Dict[str, Any] = {"model": reg["model"],
                           "messages": [_to_openai_message(m) for m in messages],
                           **reg["params"]}
    if kwargs.get("tools"):
        req["tools"] = kwargs["tools"]
        req["tool_choice"] = kwargs.get("tool_choice", "auto")
    if n > 1:
        req["n"] = n
    resp = reg["client"].chat.completions.create(**req)
    if n > 1 and len(resp.choices) >= n:
        return [_choice_to_dict(c) for c in resp.choices]
    return _choice_to_dict(resp.choices[0])

# When DD runs the client in async mode, the sync loop runs in a worker thread
# and this holds DD's main event loop so we can bridge `acompletion` back to it.
# Set by the generator's `agenerate` wrapper; propagated into the worker thread
# via contextvars (asyncio.to_thread copies the context).
DD_EVENT_LOOP: ContextVar = ContextVar("dd_event_loop", default=None)


def _facade_completion(facade: Any, chat_messages, **kwargs):
    """Call the facade, bridging to async when the client is in async mode."""
    loop = DD_EVENT_LOOP.get()
    if loop is not None:
        fut = asyncio.run_coroutine_threadsafe(facade.acompletion(chat_messages, **kwargs), loop)
        return fut.result()
    return facade.completion(chat_messages, **kwargs)


def _clean_tool_calls(tcs: Any) -> list | None:
    """Ensure every outgoing tool call has string id/name/arguments (no nulls).

    The strict server-side deserializer rejects null where a string is expected.
    """
    if not tcs:
        return None
    cleaned = []
    for i, tc in enumerate(tcs):
        d = _tc_to_dict(tc)
        fn = d.get("function") or {}
        args = fn.get("arguments")
        cleaned.append({
            "id": d.get("id") or f"call_{i}",
            "type": d.get("type") or "function",
            "function": {
                "name": fn.get("name") or "",
                "arguments": args if isinstance(args, str) else json.dumps(args or {}),
            },
        })
    return cleaned


def _dicts_to_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    out: List[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""          # never null
        if role == "system":
            out.append(ChatMessage.as_system(content))
        elif role == "assistant":
            rc = msg.get("reasoning_content")
            out.append(ChatMessage.as_assistant(
                content=content,
                reasoning_content=rc if rc else None,   # omit empty/null reasoning
                tool_calls=_clean_tool_calls(msg.get("tool_calls")),
            ))
        elif role == "tool":
            out.append(ChatMessage.as_tool(content, msg.get("tool_call_id") or ""))
        else:
            out.append(ChatMessage.as_user(content))
    return out


def _tc_to_dict(tc: Any) -> Dict[str, Any]:
    """Normalise a tool-call (dict or litellm object) to a plain OpenAI dict."""
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, "model_dump"):
        try:
            return tc.model_dump()
        except Exception:
            pass
    fn = getattr(tc, "function", None)
    return {
        "id": getattr(tc, "id", None),
        "type": getattr(tc, "type", "function") or "function",
        "function": {
            "name": getattr(fn, "name", None) if fn is not None else None,
            "arguments": getattr(fn, "arguments", "{}") if fn is not None else "{}",
        },
    }


def _choice_to_dict(choice: Any) -> Dict[str, Any]:
    msg = choice.message
    result: Dict[str, Any] = {"role": getattr(msg, "role", "assistant"),
                              "content": getattr(msg, "content", "") or ""}
    if getattr(msg, "reasoning_content", None):
        result["reasoning_content"] = msg.reasoning_content
    if getattr(msg, "tool_calls", None):
        result["tool_calls"] = [_tc_to_dict(tc) for tc in msg.tool_calls]
    return result


def call_llm(models: Dict[str, Any], alias: str, messages: List[Dict[str, Any]], **kwargs: Any):
    # prefer a direct OpenAI client (correct tool-call parsing); fall back to DD.
    if alias in _DIRECT_CLIENTS:
        return _direct_completion(alias, messages, **kwargs)
    facade = models.get(alias)
    if facade is None:
        raise ValueError(f"Model alias '{alias}' not found")
    response = _facade_completion(facade, _dicts_to_chat_messages(messages), **kwargs)
    # DD ≥0.7 returns an OpenAI-style response (.choices); DD 0.6.x returns a
    # normalized ChatCompletionResponse with a single .message (AssistantMessage).
    if hasattr(response, "choices"):
        n = kwargs.get("n", 1)
        if n > 1 and len(response.choices) >= n:
            return [_choice_to_dict(c) for c in response.choices]
        return _choice_to_dict(response.choices[0])
    return _message_to_dict(response.message)


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """Normalize DD 0.6.x AssistantMessage (content/reasoning_content/tool_calls
    of ToolCall(id,name,arguments_json)) to our internal message dict."""
    out: Dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", "") or ""}
    if getattr(msg, "reasoning_content", None):
        out["reasoning_content"] = msg.reasoning_content
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        out["tool_calls"] = [{
            "id": getattr(t, "id", None) or f"call_{i}", "type": "function",
            "function": {"name": getattr(t, "name", None),
                         "arguments": getattr(t, "arguments_json", None) or "{}"},
        } for i, t in enumerate(tcs)]
    return out


def call_llm_with_majority_vote(models, alias, messages, tools, n: int = 4,
                                tool_choice: str = "auto") -> Dict[str, Any]:
    openai_tools = []
    for td in tools:
        openai_tools.append({"type": "function", "function": td["tool"]["function"]} if "tool" in td else td)
    result = call_llm(models, alias, messages, tools=openai_tools, n=n, tool_choice=tool_choice)
    if isinstance(result, list) and len(result) >= n:
        return _apply_majority_vote(result)
    return result[0] if isinstance(result, list) else result


def _apply_majority_vote(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    tc = [r for r in responses if r.get("tool_calls")]
    non_tc = [r for r in responses if not r.get("tool_calls")]
    if len(tc) >= 3:
        return _majority_vote_tool_calls(tc)
    if non_tc:
        return non_tc[0]
    return responses[0] if responses else {}


def _majority_vote_tool_calls(tcs: List[Dict[str, Any]]) -> Dict[str, Any]:
    patterns = [tuple(t.get("function", {}).get("name") for t in (r.get("tool_calls") or [])
                      if t.get("type") == "function") for r in tcs]
    common = Counter(patterns).most_common(1)[0][0] if patterns else ()
    if not common:
        return tcs[0]
    matching = [r for r, p in zip(tcs, patterns) if p == common]
    voted = []
    for i, name in enumerate(common):
        arg_sets, ids = [], []
        for resp in matching:
            calls = resp.get("tool_calls") or []
            if i < len(calls) and calls[i].get("type") == "function":
                fn = calls[i].get("function", {})
                if fn.get("name") == name:
                    try:
                        arg_sets.append(json.loads(fn.get("arguments", "{}")))
                        ids.append(calls[i].get("id"))
                    except json.JSONDecodeError:
                        pass
        if arg_sets:
            best_id = Counter(ids).most_common(1)[0][0] if ids else f"call_{i}"
            voted.append({"id": best_id, "type": "function",
                          "function": {"name": name, "arguments": json.dumps(_vote_args(arg_sets))}})
    return {"role": "assistant", "content": matching[0].get("content", ""), "tool_calls": voted}


def _vote_args(arg_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
    params = set().union(*[a.keys() for a in arg_sets]) if arg_sets else set()
    voted: Dict[str, Any] = {}
    for p in params:
        vals = [json.dumps(a[p], sort_keys=True) if isinstance(a[p], (dict, list)) else a[p]
                for a in arg_sets if p in a]
        if vals:
            w = Counter(vals).most_common(1)[0][0]
            try:
                voted[p] = json.loads(w) if isinstance(w, str) and w[:1] in "[{" else w
            except (json.JSONDecodeError, TypeError):
                voted[p] = w
    return voted
