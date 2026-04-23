"""Schema adapters for translating MCQ-style BYOB benchmarks."""

from __future__ import annotations

import copy
import json
from typing import Any

FAITH_COLUMN_TO_KEY = {
    "faith_fluency": "Fluency",
    "faith_accuracy": "Accuracy",
    "faith_idiomaticity": "Idiomaticity",
    "faith_terminology": "Terminology",
    "faith_handling_of_format": "Handling_of_Format",
    "faith_avg": "average",
}


def flatten_mcq_records(
    records: list[dict[str, Any]],
    *,
    text_field: str = "text",
) -> tuple[list[dict[str, str]], list[tuple[int, str, object]]]:
    """Extract translatable MCQ strings while preserving their positions."""
    staged_rows: list[dict[str, str]] = []
    index: list[tuple[int, str, object]] = []

    for rec_idx, record in enumerate(records):
        question = record.get("question")
        if isinstance(question, str) and question.strip():
            staged_rows.append({text_field: question})
            index.append((rec_idx, "question", None))

        options = record.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                if isinstance(value, str) and value.strip():
                    staged_rows.append({text_field: value})
                    index.append((rec_idx, "options_dict", key))
        elif isinstance(options, list):
            for option_idx, value in enumerate(options):
                if isinstance(value, str) and value.strip():
                    staged_rows.append({text_field: value})
                    index.append((rec_idx, "options_list", option_idx))

    return staged_rows, index


def _init_translation_metadata(record: dict[str, Any], target_lang: str) -> dict[str, Any]:
    """Build per-record translation metadata for BYOB outputs."""
    metadata: dict[str, Any] = {
        "target_lang": target_lang,
        "translation": {},
        "segmented_translation": {},
    }
    if "question" in record:
        metadata["translation"]["question"] = record.get("question")
        metadata["segmented_translation"]["question"] = []

    options = record.get("options")
    if isinstance(options, dict):
        metadata["translation"]["options"] = copy.deepcopy(options)
        metadata["segmented_translation"]["options"] = {key: [] for key in options}
    elif isinstance(options, list):
        metadata["translation"]["options"] = copy.deepcopy(options)
        metadata["segmented_translation"]["options"] = [[] for _ in options]
    return metadata


def _lookup_source_text(record: dict[str, Any], kind: str, key: object) -> str:
    """Return the source string for one staged MCQ field."""
    if kind == "question":
        value = record.get("question", "")
    elif kind == "options_dict":
        value = record.get("options", {}).get(key, "")
    else:
        value = record.get("options", [""])[key]
    return value if isinstance(value, str) else str(value)


def _extract_segment_pairs(
    translated_row: dict[str, Any],
    source_text: str,
    translated_text: str,
) -> list[dict[str, str]]:
    """Extract per-string segment pairs from translation metadata if present."""
    metadata_json = translated_row.get("translation_metadata")
    metadata: dict[str, Any] = {}
    if isinstance(metadata_json, dict):
        metadata = metadata_json
    elif isinstance(metadata_json, str) and metadata_json.strip():
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            metadata = {}

    if metadata:
        segmented = metadata.get("segmented_translation")
        if isinstance(segmented, list):
            return segmented
        if isinstance(segmented, dict):
            content_pairs = segmented.get("content")
            if isinstance(content_pairs, list):
                return content_pairs
            for value in segmented.values():
                if isinstance(value, list):
                    return value

    return [{"src": source_text, "tgt": translated_text}]


def restore_mcq_records(
    original_records: list[dict[str, Any]],
    index: list[tuple[int, str, object]],
    translated_rows: list[dict[str, Any]],
    *,
    target_lang: str,
    translated_field: str = "translated_text",
) -> list[dict[str, Any]]:
    """Merge translated strings back into the original MCQ schema."""
    if len(index) != len(translated_rows):
        raise RuntimeError(
            "Translation output length mismatch. This usually means a filter dropped staged rows before "
            "BYOB reassembly completed."
        )

    out = [copy.deepcopy(record) for record in original_records]
    record_metadata = [_init_translation_metadata(record, target_lang) for record in original_records]
    record_score_values = [{column: [] for column in FAITH_COLUMN_TO_KEY} for _ in original_records]
    record_time_totals = [0.0 for _ in original_records]
    record_error_lists = [[] for _ in original_records]

    for (rec_idx, kind, key), translated_row in zip(index, translated_rows):
        translated_text = str(translated_row.get(translated_field, ""))
        segment_pairs = _extract_segment_pairs(
            translated_row=translated_row,
            source_text=_lookup_source_text(original_records[rec_idx], kind, key),
            translated_text=translated_text,
        )

        if kind == "question":
            out[rec_idx]["question"] = translated_text
            record_metadata[rec_idx]["translation"]["question"] = translated_text
            record_metadata[rec_idx]["segmented_translation"]["question"] = segment_pairs
        elif kind == "options_dict":
            out[rec_idx]["options"][key] = translated_text
            record_metadata[rec_idx]["translation"].setdefault(
                "options", copy.deepcopy(original_records[rec_idx].get("options", {}))
            )[key] = translated_text
            record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options", {option_key: [] for option_key in original_records[rec_idx].get("options", {})}
            )[key] = segment_pairs
        else:
            out[rec_idx]["options"][key] = translated_text
            record_metadata[rec_idx]["translation"].setdefault(
                "options", copy.deepcopy(original_records[rec_idx].get("options", []))
            )[key] = translated_text
            record_metadata[rec_idx]["segmented_translation"].setdefault(
                "options", [[] for _ in original_records[rec_idx].get("options", [])]
            )[key] = segment_pairs

        for column in FAITH_COLUMN_TO_KEY:
            value = translated_row.get(column)
            if value is None or value != value:
                continue
            record_score_values[rec_idx].setdefault(column, []).append(float(value))

        time_value = translated_row.get("translation_time")
        if time_value is not None and time_value == time_value:
            record_time_totals[rec_idx] += float(time_value)

        error_value = str(translated_row.get("translation_errors", "")).strip()
        if error_value:
            record_error_lists[rec_idx].append(error_value)

    for rec_idx, metadata in enumerate(record_metadata):
        faith_scores = {
            score_key: sum(values) / len(values)
            for column, score_key in FAITH_COLUMN_TO_KEY.items()
            for values in [record_score_values[rec_idx].get(column, [])]
            if values
        }
        if faith_scores:
            metadata["faith_scores"] = faith_scores
        out[rec_idx]["translation_metadata"] = metadata

        for column in FAITH_COLUMN_TO_KEY:
            values = record_score_values[rec_idx].get(column, [])
            if values:
                out[rec_idx][column] = sum(values) / len(values)

        if record_time_totals[rec_idx]:
            out[rec_idx]["translation_time"] = record_time_totals[rec_idx]
        if record_error_lists[rec_idx]:
            out[rec_idx]["translation_errors"] = "; ".join(record_error_lists[rec_idx])

    return out


def _options_to_list(options: Any) -> list[str]:
    """Normalize MCQ options to a stable ordered list."""
    if isinstance(options, dict):
        return [str(value) for value in options.values()]
    if isinstance(options, list):
        return [str(value) for value in options]
    return []


def format_mcq_for_metrics(question: str, options: Any) -> str:
    """Format an MCQ into a stable string for round-trip quality metrics."""
    choices = _options_to_list(options)
    choices_flat = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
    return f"Question: {question}\nOptions:\n{choices_flat}"
