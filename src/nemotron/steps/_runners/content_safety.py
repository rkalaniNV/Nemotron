# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nemotron 3.5 Content Safety dataset factory for the AutoModel steps.

Used by ``sft/automodel`` and ``peft/automodel`` through the ``content_safety``
config preset (``dataset._target_``).
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

Transform = Callable[[dict], "dict | None"]

NEMOTRON_CONTENT_SAFETY_DATASET = "nvidia/Nemotron-3.5-Content-Safety-Dataset"
NEMOTRON_CONTENT_SAFETY_DATASET_REVISION = "841f5023b8db12b484180c6121bb10fcf400d1c3"
_SAFETY_CHAT_KWARGS = {"request_categories": "/categories", "enable_thinking": False}
_SAFETY_CATEGORIES = {
    "Violence", "Sexual", "Criminal Planning/Confessions", "Guns and Illegal Weapons",
    "Controlled/Regulated Substances", "Suicide and Self Harm", "Sexual (minor)",
    "Hate/Identity Hate", "PII/Privacy", "Harassment", "Threat", "Profanity",
    "Needs Caution", "Other", "Manipulation", "Fraud/Deception", "Malware",
    "High Risk Gov Decision Making", "Political/Misinformation/Conspiracy",
    "Copyright/Trademark/Plagiarism", "Unauthorized Advice", "Illegal Activity",
    "Immoral/Unethical", "Economic Harm",
}


def _safety_text(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip()


def _safety_prompt_hash(record: dict) -> str:
    return hashlib.sha256(_safety_text(record.get("prompt")).encode("utf-8")).hexdigest()


def _safety_eligible(record: dict, languages: set[str]) -> bool:
    if record.get("image_path") or record.get("task_type") != "safety":
        return False
    if languages and record.get("language") not in languages:
        return False
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        return False
    if record.get("input_label") not in {"safe", "unsafe"}:
        return False
    response = record.get("response")
    if response is not None and not isinstance(response, str):
        return False
    if response and response.strip():
        if record.get("response_label") not in {"safe", "unsafe"}:
            return False
    elif record.get("response_label") not in {None, ""}:
        return False
    if not isinstance(record.get("row_id"), str) or not record["row_id"]:
        return False
    categories = record.get("violated_categories")
    if categories is not None and not isinstance(categories, str):
        return False
    return all(category.strip() in _SAFETY_CATEGORIES for category in (categories or "").split(",") if category.strip())


def nemotron_content_safety(
    tokenizer: Any,
    *,
    source_split: str = "train",
    source_dataset: str = NEMOTRON_CONTENT_SAFETY_DATASET,
    dataset_revision: str = NEMOTRON_CONTENT_SAFETY_DATASET_REVISION,
    languages: Iterable[str] = ("en",),
) -> Transform:
    """Render native text moderation inputs and label-only SFT completions.

    The source assistant response is context to classify. It must never become
    the supervised answer or be templated together with the classification
    labels. Generated reasoning and image-grounded/topic-following rows are
    excluded. An empty language sequence permits every source language.
    """
    allowed_languages = set(languages)

    def transform(record: dict) -> dict | None:
        if not _safety_eligible(record, allowed_languages):
            return None
        messages = [{"role": "user", "content": record["prompt"]}]
        lines = [f"User Safety: {record['input_label']}"]
        if record.get("response") and record["response"].strip():
            messages.append({"role": "assistant", "content": record["response"]})
            lines.append(f"Response Safety: {record['response_label']}")
        if "unsafe" in (record["input_label"], record.get("response_label")):
            categories = list(dict.fromkeys(
                category.strip() for category in (record.get("violated_categories") or "").split(",")
                if category.strip()
            ))
            if categories:
                lines.append("Safety Categories: " + ", ".join(categories))
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **_SAFETY_CHAT_KWARGS,
        )
        output_text = "\n".join(lines) + "<end_of_turn>\n"
        pair = [_safety_text(record["prompt"]), _safety_text(record.get("response"))]
        metadata = {
            "source_dataset": source_dataset, "source_revision": dataset_revision,
            "source_split": source_split, "row_id": record["row_id"],
            "prompt_sha256": _safety_prompt_hash(record),
            "interaction_sha256": hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "input_tokens": len(tokenizer.encode(input_text, add_special_tokens=False)),
            "output_tokens": len(tokenizer.encode(output_text, add_special_tokens=False)),
            "input_label": record["input_label"], "response_label": record.get("response_label"),
        }
        for key in ("language", "dataset_source", "provenance", "prompt_source",
                    "response_source", "input_label_source", "response_label_source"):
            metadata[key] = record.get(key)
        return {"input": input_text, "output": output_text, "messages": messages, "metadata": metadata}

    return transform


def _select_content_safety_splits(
    dataset: Mapping[str, Any], tokenizer: Any, *, source_dataset: str,
    revision: str, sizes: dict[str, int], seed: int, languages: tuple[str, ...],
    seq_length: int, max_scan_rows: int,
) -> dict[str, list[dict]]:
    if set(dataset) != {"train", "validation", "test"}:
        raise ValueError("Safety data must preserve the original train/validation/test splits")
    for split, rows in dataset.items():
        if len(rows) > max_scan_rows:
            raise ValueError(f"{split} exceeds max_scan_rows={max_scan_rows}; refusing a partial scan")
    heldout_prompts = {
        _safety_prompt_hash(record)
        for split in ("validation", "test") for record in dataset[split]
        if not record.get("image_path") and record.get("task_type") == "safety" and record.get("prompt")
    }
    selected_prompts: set[str] = set()
    result = {}
    for split in ("validation", "test", "train"):
        transform = nemotron_content_safety(
            tokenizer, source_split=split, source_dataset=source_dataset,
            dataset_revision=revision, languages=languages,
        )
        candidates = [record for record in dataset[split] if _safety_eligible(record, set(languages))
                      and (split != "train" or _safety_prompt_hash(record) not in heldout_prompts)]
        random.Random(f"{seed}:{split}").shuffle(candidates)
        selected = []
        for record in candidates:
            digest = _safety_prompt_hash(record)
            if digest in selected_prompts:
                continue
            formatted = transform(record)
            # The training reader concatenates independently encoded input and
            # answer, then shifts once for next-token targets. Never truncate.
            tokens = formatted["metadata"]["input_tokens"] + formatted["metadata"]["output_tokens"] - 1
            if tokens > seq_length:
                continue
            selected.append(formatted)
            selected_prompts.add(digest)
            if len(selected) == sizes[split]:
                break
        if len(selected) != sizes[split]:
            raise ValueError(f"{split}: requested {sizes[split]} valid safety rows, found {len(selected)}")
        result[split] = selected
    return result


def _safety_tokenizer_fingerprint(tokenizer: Any) -> dict:
    """Identify the tokenizer that actually encoded the cached records."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        serialized = backend.to_str()
    else:
        # Slow tokenizers also expose their actual token-to-id mapping.
        serialized = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    configuration = getattr(tokenizer, "init_kwargs", {})
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "revision": configuration.get("_commit_hash") or configuration.get("revision"),
        "backend_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


def load_nemotron_content_safety_splits(
    tokenizer: Any,
    *,
    path_or_dataset_id: str = NEMOTRON_CONTENT_SAFETY_DATASET,
    revision: str = NEMOTRON_CONTENT_SAFETY_DATASET_REVISION,
    train_size: int = 4096,
    validation_size: int = 256,
    test_size: int = 256,
    seed: int = 42,
    languages: Iterable[str] = ("en",),
    seq_length: int = 2048,
    max_scan_rows: int = 100000,
    cache_dir: str | Path | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, list[dict]]:
    """Load a pinned, deterministic subset with disjoint original source splits.

    Public source parquet files and prepared records use the shared Hugging Face
    cache. A file lock makes the first distributed worker prepare the subset;
    subsequent workers reuse a checksummed atomic cache. Supplying ``dataset``
    processes an in-memory fixture without downloads or disk writes.
    """
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if min(*sizes.values(), seq_length, max_scan_rows) <= 0:
        raise ValueError("Safety subset sizes, sequence length and scan bound must be positive")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Safety dataset revision must be an immutable commit hash")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or "User Safety:" not in template:
        raise ValueError("The original Nemotron content-safety classifier chat template is required")
    languages = tuple(sorted(set(languages)))
    options = dict(source_dataset=path_or_dataset_id, revision=revision, sizes=sizes,
                   seed=seed, languages=languages, seq_length=seq_length, max_scan_rows=max_scan_rows)
    if dataset is not None:
        return _select_content_safety_splits(dataset, tokenizer, **options)

    from datasets import config as datasets_config, load_dataset
    from filelock import FileLock

    parameters = {
        **options, "languages": list(languages), "format_version": 1,
        "tokenizer": _safety_tokenizer_fingerprint(tokenizer),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
    }
    fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir or datasets_config.HF_DATASETS_CACHE) / "nemotron-content-safety"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fingerprint}.json"
    with FileLock(str(path) + ".lock", timeout=600):
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            records = cached["splits"]
            digest = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if cached["parameters"] != parameters or digest != cached["records_sha256"]:
                raise ValueError("Prepared content-safety cache checksum/settings mismatch")
            return records
        raw = load_dataset(path_or_dataset_id, revision=revision, cache_dir=cache_dir, token=False)
        records = _select_content_safety_splits(raw, tokenizer, **options)
        cached = {
            "parameters": parameters, "splits": records,
            "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "license": "CC-BY-4.0",
            "attribution": "NVIDIA Corporation, Nemotron 3.5 Content Safety Dataset, June 2, 2026",
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(cached, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return records


def build_nemotron_content_safety_dataset(
    tokenizer: Any,
    split: str = "train",
    seq_length: int = 2048,
    *,
    path_or_dataset_id: str = NEMOTRON_CONTENT_SAFETY_DATASET,
    revision: str = NEMOTRON_CONTENT_SAFETY_DATASET_REVISION,
    train_size: int = 4096,
    validation_size: int = 256,
    test_size: int = 256,
    seed: int = 42,
    languages: Iterable[str] = ("en",),
    max_scan_rows: int = 100000,
    cache_dir: str | Path | None = None,
):
    """AutoModel dataset factory preserving native moderation prompts and labels.

    The shared AutoModel reader masks the prefix, supervises the separately
    encoded classification target, and shifts next-token labels exactly once.
    This factory applies no additional template, special tokens, or truncation.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown source safety split: {split}")
    from nemotron.steps._runners.automodel import PreformattedTextDataset

    records = load_nemotron_content_safety_splits(
        tokenizer, path_or_dataset_id=path_or_dataset_id, revision=revision,
        train_size=train_size, validation_size=validation_size, test_size=test_size,
        seed=seed, languages=languages, seq_length=seq_length,
        max_scan_rows=max_scan_rows, cache_dir=cache_dir,
    )
    return PreformattedTextDataset(records=records[split], tokenizer=tokenizer, seq_length=seq_length)
