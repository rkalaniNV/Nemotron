"""Curator-native corpus translation skill runtime."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _normalize_path(path_value: Any, *, field_name: str) -> str:
    """Resolve required path-like config values to strings."""
    if not path_value:
        raise ValueError(f"translation.{field_name} is required")
    return str(path_value)


def _infer_local_dir_format(input_path: Path) -> str:
    """Infer one local dataset format from the files present in a directory."""
    has_jsonl = any(input_path.glob("*.jsonl"))
    has_parquet = any(input_path.glob("*.parquet"))

    if has_jsonl and has_parquet:
        raise ValueError(
            f"Input directory {input_path} mixes JSONL and Parquet files. "
            "Set translation.input_format explicitly or split the dataset."
        )
    if has_jsonl:
        return "jsonl"
    if has_parquet:
        return "parquet"
    raise FileNotFoundError(f"No .jsonl or .parquet files found in {input_path}")


def _infer_input_format(input_path: str, configured_format: str | None) -> str:
    """Infer the input format for the Curator reader."""
    if configured_format and configured_format != "auto":
        return str(configured_format)

    lower_path = input_path.lower()
    if lower_path.endswith(".jsonl") or ".jsonl" in lower_path:
        return "jsonl"
    if lower_path.endswith(".parquet") or ".parquet" in lower_path:
        return "parquet"

    path_obj = Path(input_path)
    if path_obj.exists() and path_obj.is_dir():
        return _infer_local_dir_format(path_obj)

    matches = glob.glob(input_path)
    matched_suffixes = {Path(match).suffix for match in matches}
    if matched_suffixes == {".jsonl"}:
        return "jsonl"
    if matched_suffixes == {".parquet"}:
        return "parquet"

    raise ValueError(
        "Could not infer translation input format. Use a .jsonl/.parquet path, a homogeneous directory, "
        "or set translation.input_format explicitly."
    )


def _build_reader(input_path: str, step_cfg: dict[str, Any]) -> Any:
    """Build the Curator reader stage for the configured dataset."""
    from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

    input_format = _infer_input_format(input_path, step_cfg.get("input_format"))
    files_per_partition = step_cfg.get("files_per_partition")
    blocksize = step_cfg.get("blocksize")

    if input_format == "jsonl":
        return JsonlReader(
            file_paths=input_path,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
        )
    if input_format == "parquet":
        return ParquetReader(
            file_paths=input_path,
            files_per_partition=files_per_partition,
            blocksize=blocksize,
        )
    raise ValueError(f"Unsupported translation input format: {input_format}")


def _build_writer(output_dir: str, step_cfg: dict[str, Any]) -> Any:
    """Build the Curator writer stage for translated output."""
    from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

    output_format = str(step_cfg.get("output_format", "jsonl"))
    if output_format == "jsonl":
        return JsonlWriter(path=output_dir, mode="overwrite")
    if output_format == "parquet":
        return ParquetWriter(path=output_dir, mode="overwrite")
    raise ValueError(f"Unsupported translation output format: {output_format}")


def _build_curator_client(step_cfg: dict[str, Any], *, enable_faith: bool) -> Any | None:
    """Create the Curator async LLM client when the step needs one."""
    backend = step_cfg.get("backend", "llm")
    if backend != "llm" and not enable_faith:
        return None

    from nemo_curator.models.client.openai_client import AsyncOpenAIClient

    server = step_cfg.get("server", {}) or {}
    api_key = server.get("api_key") or os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError(
            "server.api_key is required when backend='llm' or faith_eval.enabled=True "
            "(set NVIDIA_API_KEY or pass translation.server.api_key)"
        )

    return AsyncOpenAIClient(
        max_concurrent_requests=int(step_cfg.get("max_concurrent_requests", 64)),
        base_url=server.get("url", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
    )


def _build_backend_config(step_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract backend-specific config for the selected translation backend."""
    backend = str(step_cfg.get("backend", "llm"))
    if backend == "google":
        return dict(step_cfg.get("google", {}) or {})
    if backend == "aws":
        return dict(step_cfg.get("aws", {}) or {})
    if backend == "nmt":
        return dict(step_cfg.get("nmt", {}) or {})
    return {}


def _normalize_text_field(value: Any) -> str | list[str]:
    """Preserve list-based field paths instead of stringifying them."""
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def _build_translation_stage(step_cfg: dict[str, Any]) -> Any:
    """Build the Curator TranslationStage for one skill config."""
    from nemo_curator.stages.text.translation import TranslationStage

    faith_cfg = step_cfg.get("faith_eval", {}) or {}
    enable_faith = bool(faith_cfg.get("enabled", False))

    return TranslationStage(
        source_lang=str(step_cfg.get("source_lang", "en")),
        target_lang=str(step_cfg.get("target_lang", "hi")),
        text_field=_normalize_text_field(step_cfg.get("text_field", "text")),
        output_field=str(step_cfg.get("output_field", "translated_text")),
        segmentation_mode=str(step_cfg.get("segmentation_mode", "coarse")),
        min_segment_chars=int(step_cfg.get("min_segment_chars", 0)),
        client=_build_curator_client(step_cfg, enable_faith=enable_faith),
        model_name=str((step_cfg.get("server", {}) or {}).get("model", "")),
        backend_type=str(step_cfg.get("backend", "llm")),
        backend_config=_build_backend_config(step_cfg),
        enable_faith_eval=enable_faith,
        faith_threshold=float(faith_cfg.get("threshold", 2.5)),
        faith_model_name=str(faith_cfg.get("model_name", "")),
        segment_level=bool(faith_cfg.get("segment_level", False)),
        filter_enabled=bool(faith_cfg.get("filter_enabled", True)),
        output_mode=str(step_cfg.get("output_mode", "both")),
        merge_scores=bool(step_cfg.get("merge_scores", True)),
        reconstruct_messages=bool(step_cfg.get("reconstruct_messages", False)),
        messages_field=str(step_cfg.get("messages_field", "messages")),
        messages_content_field=str(step_cfg.get("messages_content_field", "content")),
        skip_translated=bool(step_cfg.get("skip_translated", False)),
        translation_column=str(step_cfg.get("translation_column", "translated_text")),
    )


def _load_translation_cfg(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve the translation config section."""
    if isinstance(cfg, DictConfig):
        step_cfg = OmegaConf.select(cfg, "translation", default=cfg)
        if step_cfg is None:
            raise ValueError("Missing required 'translation' config section")
        if isinstance(step_cfg, DictConfig):
            return OmegaConf.to_container(step_cfg, resolve=True)
        return dict(step_cfg)
    if "translation" in cfg:
        return dict(cfg["translation"])
    return dict(cfg)


def translate_data(cfg: DictConfig | dict[str, Any]) -> Path:
    """Translate a dataset with Curator's reader -> translation -> writer pipeline."""
    from nemo_curator.pipeline import Pipeline

    step_cfg = _load_translation_cfg(cfg)
    input_path = _normalize_path(step_cfg.get("input_path"), field_name="input_path")
    output_dir = Path(_normalize_path(step_cfg.get("output_dir"), field_name="output_dir"))

    pipeline = Pipeline(name="translate_corpus")
    pipeline.add_stage(_build_reader(input_path, step_cfg))
    pipeline.add_stage(_build_translation_stage(step_cfg))
    pipeline.add_stage(_build_writer(str(output_dir), step_cfg))
    pipeline.run()

    log.info("Translation complete. Wrote Curator output shards under %s", output_dir)
    return output_dir
