"""ToolEnvironment — routes each tool call to the retrieval service or an LLM sim.

- Retrieval tools (``cfg.retrieval_tools``) call the HTTP client, which oversamples
  and randomly subsamples to top_k. The retrieval log is kept for audit.
- Every other (user-defined) tool is simulated by an LLM producing a realistic
  JSON response — so the customer's own tool list works with no backend.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm import call_llm
from .prompts import AUX_TOOL_SIM_SYSTEM, AUX_TOOL_SIM_TURN

AUX_MODEL = "aux_response_model"


def _fn(td: Dict[str, Any]) -> Dict[str, Any]:
    return (td["tool"] if "tool" in td else td).get("function", {})


def sample_tools(all_tools: List[Dict[str, Any]], max_tools: int, must_include: List[str],
                 rng: random.Random) -> List[Dict[str, Any]]:
    """Always keep the retrieval tools; shuffle the rest; cap to ``max_tools``."""
    keep = [t for t in all_tools if _fn(t).get("name") in must_include]
    rest = [t for t in all_tools if _fn(t).get("name") not in must_include]
    rng.shuffle(rest)
    room = max(0, max_tools - len(keep))
    return keep + rest[:room]


class ToolEnvironment:
    def __init__(self, cfg, client, tools: List[Dict[str, Any]]):
        self.cfg = cfg
        self.client = client
        self.tools = tools
        self.retrieval_log: List[Dict[str, Any]] = []

    def is_retrieval(self, name: str) -> bool:
        return name in self.cfg.retrieval_tools

    def _first(self, args: Dict[str, Any], names: List[str], default: Any) -> Any:
        for n in names:
            if args.get(n) not in (None, ""):
                return args[n]
        return default

    def respond(self, tool_call: Dict[str, Any], models: Dict[str, Any], user_query: str,
                rng: random.Random) -> Tuple[str, bool]:
        """Return (tool_response_content, was_retrieval)."""
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        if self.is_retrieval(name) and self.client is not None:
            return self._retrieve(args, user_query, rng), True
        return self._simulate(name, args, models, user_query), False

    def _retrieve(self, args: Dict[str, Any], user_query: str, rng: random.Random) -> str:
        query = str(self._first(args, self.cfg.query_arg_names, user_query))
        # top_k is governed by the CONFIG KNOB, not the model: the client requests
        # k*oversample_factor from the service and randomly keeps k. Any top_k the
        # model puts in the tool call is ignored on purpose (kept constant).
        k = max(1, int(self.cfg.top_k))
        chunks = self.client.retrieve(query, k, rng=rng)
        self.retrieval_log.append({"query": query, "ids": [c.id for c in chunks]})
        return json.dumps({"results": [c.to_payload() for c in chunks]}, ensure_ascii=False)

    def _simulate(self, name: str, args: Dict[str, Any], models: Dict[str, Any], user_query: str) -> str:
        spec = next((t for t in self.tools if _fn(t).get("name") == name), {})
        tool_json = json.dumps(_fn(spec) or {"name": name}, ensure_ascii=False)
        resp = call_llm(models, AUX_MODEL, [
            {"role": "system", "content": AUX_TOOL_SIM_SYSTEM},
            {"role": "user", "content": AUX_TOOL_SIM_TURN.format(
                tool=tool_json, arguments=json.dumps(args, ensure_ascii=False), user_query=user_query)}])
        return (resp.get("content") if isinstance(resp, dict) else str(resp)) or "{}"
