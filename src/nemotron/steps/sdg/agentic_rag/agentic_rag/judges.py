"""Inline judge invocation + XML-tag parsing. Reused from the reference."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .llm import call_llm

JUDGE_FOLLOWUP_PROMPT = """Reformat your previous response to strictly follow:
<explanation>
[justification]
</explanation>
<rating>
[success or failure]
</rating>"""


def parse_judge_response(text: str) -> Tuple[str, str, bool]:
    exp = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    rat = re.search(r"<rating>(.*?)</rating>", text, re.DOTALL)
    explanation = exp.group(1).strip() if exp else ""
    rating = None
    if rat:
        rt = rat.group(1).strip().lower()
        rating = "success" if "success" in rt else ("failure" if "failure" in rt else None)
    if rating is None:
        # Fallback for models that don't wrap the verdict in <rating> tags:
        # take the LAST clear verdict word anywhere in the response.
        low = text.lower()
        s, f = low.rfind("success"), low.rfind("fail")
        if s >= 0 or f >= 0:
            rating = "success" if s > f else "failure"
    return explanation, rating or "failure", rating == "success"


def run_inline_judge(models: Dict[str, Any], alias: str, prompt_text: str) -> Tuple[str, str, bool]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": ""},
                                  {"role": "user", "content": prompt_text}]
    resp = call_llm(models, alias, msgs)
    text = resp.get("content", "") if isinstance(resp, dict) else ""
    explanation, rating, success = parse_judge_response(text)
    if rating not in ("success", "failure") or not text:
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": JUDGE_FOLLOWUP_PROMPT}]
        resp2 = call_llm(models, alias, msgs)
        text2 = resp2.get("content", "") if isinstance(resp2, dict) else ""
        explanation, rating, success = parse_judge_response(text2)
    return explanation, rating, success
