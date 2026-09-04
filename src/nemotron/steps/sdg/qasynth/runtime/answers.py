# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-resumable async answer generation for QASynth teacher models."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from nemotron.steps.sdg.qasynth.runtime.io import read_jsonl

ANSWER_RE = re.compile(
    r"(?:Answer|उत्तर(?:\s*है)?)\s*[:：]?\s*[$*_{}\\\s]*(?:\\?text\{)?\s*\(?\s*([ABCD])\b",
    re.IGNORECASE,
)

INSTRUCTIONS = {
    "english": (
        "Answer the following multiple choice question. The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of ABCD. "
        "Think step by step before answering."
    ),
    "hindi": (
        "निम्नलिखित बहुविकल्पीय प्रश्न का उत्तर दें। आपकी प्रतिक्रिया की अंतिम पंक्ति निम्न प्रारूप में होनी "
        "चाहिए: 'उत्तर: $LETTER' (बिना उद्धरण चिह्न के) जहाँ $LETTER, A/B/C/D में से एक है। उत्तर देने से पहले "
        "चरण-दर-चरण सोचें।"
    ),
}


def answer_prompt(record: dict[str, Any], language: str) -> str:
    options = "\n".join(f"{chr(65 + index)}) {choice}" for index, choice in enumerate(record["choices"]))
    return f"{INSTRUCTIONS[language]}\n\n{record['question']}\n\n{options}"


def parse_answer_letter(content: str) -> str | None:
    match = ANSWER_RE.search(content.strip()[-300:])
    return match.group(1).upper() if match else None


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record["query_id"]) for record in read_jsonl(path) if record.get("query_id")}


async def _request(
    client: Any,
    record: dict[str, Any],
    language: str,
    model: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    import httpx

    body = {
        "model": model["model"],
        "messages": [{"role": "user", "content": answer_prompt(record, language)}],
        "temperature": model.get("temperature", 1.0),
        "top_p": model.get("top_p", 1.0),
        "max_tokens": model.get("max_tokens", 16384),
        **(model.get("extra_body") or {}),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.post("chat/completions", json=body)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError("retryable model response", request=response.request, response=response)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            return {
                **record,
                "answer_model": model["model"],
                "answer": content,
                "reasoning": reasoning,
                "parsed_letter": parse_answer_letter(content),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(2**attempt, 30))
    raise RuntimeError(f"answer request failed after {max_retries} attempts: {last_error}")


async def generate_answers(
    records: list[dict[str, Any]],
    *,
    language: str,
    model: dict[str, Any],
    output_path: Path,
    failure_path: Path,
    resume: bool,
    max_parallel: int,
    timeout: float,
    max_retries: int,
) -> dict[str, int]:
    import httpx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(output_path) if resume else set()
    pending = [record for record in records if str(record["query_id"]) not in done]
    semaphore = asyncio.Semaphore(max_parallel)
    key_name = model.get("api_key_env", "NVIDIA_API_KEY")
    token = os.environ.get(key_name, model.get("api_key", ""))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    limits = httpx.Limits(max_connections=max_parallel + 20, max_keepalive_connections=max_parallel)
    lock = asyncio.Lock()
    stats = {"pending": len(pending), "answered": 0, "failed": 0, "unparsed": 0}

    async with httpx.AsyncClient(
        base_url=model["endpoint"].rstrip("/") + "/",
        timeout=timeout,
        headers=headers,
        limits=limits,
    ) as client:
        mode = "a" if resume else "w"
        with output_path.open(mode, encoding="utf-8") as output, failure_path.open(mode, encoding="utf-8") as failures:
            async def run_one(record: dict[str, Any]) -> None:
                async with semaphore:
                    try:
                        result = await _request(client, record, language, model, max_retries)
                    except Exception as exc:  # noqa: BLE001 - persist per-row failure and continue.
                        async with lock:
                            failures.write(json.dumps({"query_id": record["query_id"], "error": str(exc)}) + "\n")
                            failures.flush()
                            stats["failed"] += 1
                        return
                    async with lock:
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        stats["answered"] += 1
                        stats["unparsed"] += int(result["parsed_letter"] is None)

            await asyncio.gather(*(run_one(record) for record in pending))
            os.fsync(output.fileno())
            os.fsync(failures.fileno())
    return stats
