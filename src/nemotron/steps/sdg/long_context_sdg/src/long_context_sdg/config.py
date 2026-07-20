"""YAML configuration and reproducible run fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import ConfigDict, Field, model_validator

from .config_base import StrictConfigModel
from .service_config import ModelConfig, ProviderConfig, RetrieverConfig


class PathsConfig(StrictConfigModel):
    seeds: str
    enriched_seeds: str
    artifacts: str
    generated: str
    canonical: str
    output_dir: str
    export: str


class RunConfig(StrictConfigModel):
    mode: Literal["preview", "create"] = "preview"
    seed: int = 7
    num_records: int = Field(0, ge=0)
    dataset_name: str = "long_context_sdg"
    resume: Literal["never", "always", "if_possible"] = "if_possible"
    max_parallel_workers: int = Field(8, ge=1)
    buffer_size: int | None = Field(None, ge=1)
    otel_metrics_port: int | None = Field(9465, ge=1, le=65535)

    @model_validator(mode="after")
    def valid_dataset_name(self) -> RunConfig:
        if not self.dataset_name.strip():
            raise ValueError("run.dataset_name must be non-empty")
        return self


class TurnRange(StrictConfigModel):
    min: int = Field(6, ge=6, le=40)
    max: int = Field(40, ge=6, le=40)

    @model_validator(mode="after")
    def ordered(self) -> TurnRange:
        if self.max < self.min:
            raise ValueError("turn_budget.max must be >= turn_budget.min")
        return self


class RetrievalNoveltyConfig(StrictConfigModel):
    """Deterministic lexical and observed-evidence retrieval guards."""

    query_lexical_similarity_threshold: float = Field(0.80, ge=0, le=1)
    evidence_lexical_similarity_threshold: float = Field(0.85, ge=0, le=1)
    min_new_chunk_fraction: float = Field(0.50, ge=0, le=1)
    max_low_gain_chain: int = Field(1, ge=1, le=20)
    low_gain_followup_similarity_threshold: float = Field(0.35, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "query_similarity_threshold" in data:
            data.setdefault(
                "query_lexical_similarity_threshold",
                data.pop("query_similarity_threshold"),
            )
        if "evidence_similarity_threshold" in data:
            data.setdefault(
                "evidence_lexical_similarity_threshold",
                data.pop("evidence_similarity_threshold"),
            )
        if "max_low_gain_calls" in data:
            legacy = int(data.pop("max_low_gain_calls"))
            data.setdefault("max_low_gain_chain", max(1, legacy))
        return data


class EpisodePolicyConfig(StrictConfigModel):
    turn_budget: TurnRange = Field(default_factory=TurnRange)
    honor_seed_turn_budget: bool = True
    max_retrieval_calls: int = Field(6, ge=0, le=120)
    max_retrieval_calls_per_turn: int = Field(1, ge=0, le=12)
    retrieval_novelty: RetrievalNoveltyConfig = Field(default_factory=RetrievalNoveltyConfig)
    max_steps_per_turn: int = Field(6, ge=2, le=12)
    max_tool_calls_per_turn: int = Field(2, ge=1, le=12)
    max_tool_calls_per_conversation: int = Field(16, ge=1, le=200)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_retrieval_policy(cls, value: Any) -> Any:
        """Accept optional-only legacy ranges while rejecting forced floors."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy = data.pop("retrieval_calls", None)
        if data.pop("retrieval_depth_weights", None) is not None:
            raise ValueError(
                "episode.retrieval_depth_weights is no longer supported; task/evidence diversity belongs "
                "in query generation"
            )
        if legacy is not None:
            minimum = int((legacy or {}).get("min", 0))
            if minimum:
                raise ValueError(
                    "episode.retrieval_calls.min is no longer supported; remove the forced floor or set it to 0"
                )
            data.setdefault("max_retrieval_calls", int((legacy or {}).get("max", 6)))
        return data

    @model_validator(mode="after")
    def valid_policy(self) -> EpisodePolicyConfig:
        if self.max_tool_calls_per_turn > self.max_tool_calls_per_conversation:
            raise ValueError("max_tool_calls_per_turn cannot exceed max_tool_calls_per_conversation")
        if self.max_retrieval_calls > self.max_tool_calls_per_conversation:
            raise ValueError("max_retrieval_calls cannot exceed max_tool_calls_per_conversation")
        if self.max_retrieval_calls_per_turn > self.max_tool_calls_per_turn:
            raise ValueError("max_retrieval_calls_per_turn cannot exceed max_tool_calls_per_turn")
        return self


class ContextConfig(StrictConfigModel):
    compression_threshold: int = Field(32000, ge=256)
    model_token_limit: int = Field(65536, ge=512)
    recent_raw_turns: int = Field(4, ge=1, le=20)
    min_turns_between_compression: int = Field(3, ge=1)
    compression_token_budget: int = Field(500, ge=100)
    max_reasoning_tokens: int = Field(400, ge=32)

    @model_validator(mode="after")
    def threshold_fits(self) -> ContextConfig:
        if self.compression_threshold >= self.model_token_limit:
            raise ValueError("compression_threshold must be below model_token_limit")
        return self


class ToolConfig(StrictConfigModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tool_schema: dict[str, Any] = Field(alias="schema")
    executor: str
    executor_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return str((self.tool_schema.get("function") or {}).get("name", ""))


class ValidationConfig(StrictConfigModel):
    require_final_answer_each_turn: bool = True


class JudgeConfig(StrictConfigModel):
    enabled: bool = True
    min_score: int = Field(3, ge=1, le=5)
    dimensions: list[str] = Field(default_factory=list)


class ExportConfig(StrictConfigModel):
    format: Literal["messages", "messages_and_tools", "rich"] = "messages_and_tools"


class PipelineConfig(StrictConfigModel):
    paths: PathsConfig
    run: RunConfig = Field(default_factory=RunConfig)
    instructions: str = ""
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[ModelConfig]
    episode: EpisodePolicyConfig = Field(default_factory=EpisodePolicyConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    retriever: RetrieverConfig
    tools: list[ToolConfig]
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    config_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def required_roles_and_tools(self) -> PipelineConfig:
        aliases = [m.alias for m in self.models]
        if len(aliases) != len(set(aliases)):
            raise ValueError("model aliases must be unique")
        missing = {"assistant", "user", "compressor", "judge"} - set(aliases)
        if missing:
            raise ValueError(f"missing model aliases: {sorted(missing)}")
        names = [t.name for t in self.tools]
        if any(not n for n in names):
            raise ValueError("every tool schema must contain function.name")
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if "retrieve" not in names:
            raise ValueError("the default retrieve tool is required")
        provider_names = [provider.name for provider in self.providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("provider names must be unique")
        return self

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.config_dir / p).resolve()

    def fingerprint(self) -> str:
        payload = self.model_dump(exclude={"paths", "run"}, mode="json")
        # The seed changes generated specifications; other run fields only control
        # Data Designer orchestration and native resume behavior.
        payload["run_seed"] = self.run.seed
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def generation_payload(self) -> dict[str, Any]:
        """Return runtime config without Data Designer orchestration identity."""
        payload = self.model_dump(mode="json", exclude={"config_dir"})
        payload["paths"] = {name: "." for name in type(self.paths).model_fields}
        payload["run"] = {
            "mode": "create",
            "seed": self.run.seed,
            "num_records": 0,
            "dataset_name": "embedded",
            "resume": "never",
        }
        return payload


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = PipelineConfig.model_validate(data)
    cfg.config_dir = config_path.parent
    return cfg
