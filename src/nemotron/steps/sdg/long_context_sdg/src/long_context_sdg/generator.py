"""Data Designer adapter around the independently testable EpisodeRunner."""

from __future__ import annotations

import json
from pathlib import Path

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from .checkpoint import append_record
from .config import PipelineConfig
from .executors.base import ExecutionServices
from .generator_config import LongContextEpisodeConfig
from .retrieval import RetrieverClient
from .runtime import EpisodeRunner
from .schemas import CanonicalRecord, EpisodeSeed
from .tool_registry import ToolRegistry


class LongContextEpisodeGenerator(
    ColumnGeneratorCellByCell[LongContextEpisodeConfig],
    ColumnGeneratorWithModelRegistry[LongContextEpisodeConfig],
):
    def generate(self, data: dict) -> dict:
        cfg = None
        seed = None
        retriever = None
        try:
            cfg = PipelineConfig.model_validate(self.config.pipeline)
            aliases = {m.alias for m in cfg.models}
            models = {alias: self.get_model(alias) for alias in aliases}
            raw = data[self.config.episode_input_column]
            seed = (
                EpisodeSeed.model_validate_json(raw)
                if isinstance(raw, str)
                else EpisodeSeed.model_validate(raw)
            )
            retriever = RetrieverClient(cfg.retriever)
            services = ExecutionServices(
                retriever=retriever,
                models=models,
                simulator_alias="assistant",
            )
            registry = ToolRegistry(cfg.tools, services)
            record = EpisodeRunner(cfg).run(
                models, seed, registry, run_id=self.config.run_id
            )
        except Exception as exc:
            record = CanonicalRecord(
                run_id=self.config.run_id,
                config_fingerprint=cfg.fingerprint()
                if cfg is not None
                else "unavailable",
                query_id=seed.query_id if seed is not None else "unknown",
                status="generation_failed",
                validation={
                    "ok": False,
                    "errors": [f"generator setup failed: {exc}"],
                    "warnings": [],
                },
            )
        finally:
            if retriever is not None:
                retriever.close()
        append_record(Path(self.config.checkpoint_path), record)
        data["canonical_record"] = record.model_dump_json()
        data["trajectory_status"] = record.status
        data["trajectory_validation"] = json.dumps(
            record.validation, ensure_ascii=False
        )
        data["structured_messages"] = json.dumps(record.messages, ensure_ascii=False)
        data[self.config.name] = data["structured_messages"]
        return data
