"""YAML configuration and reproducible run fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfigModel(BaseModel):
    """Reject misspelled or obsolete YAML keys instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(StrictConfigModel):
    seeds: str
    enriched_seeds: str
    checkpoint: str
    canonical: str
    output_dir: str
    export: str


class RunConfig(StrictConfigModel):
    mode: Literal["preview", "create"] = "preview"
    seed: int = 7
    num_records: int = Field(0, ge=0)
    retry_failed: bool = False
    retry_quarantine: bool = False


class ProviderConfig(StrictConfigModel):
    name: str
    endpoint: str
    api_key_env: str


class ModelConfig(StrictConfigModel):
    alias: str
    model: str
    provider: str = "primary"
    skip_health_check: bool = False
    inference_parameters: dict[str, Any] = Field(default_factory=dict)


class TurnRange(StrictConfigModel):
    min: int = Field(6, ge=6, le=40)
    max: int = Field(40, ge=6, le=40)

    @model_validator(mode="after")
    def ordered(self) -> TurnRange:
        if self.max < self.min:
            raise ValueError("turn_budget.max must be >= turn_budget.min")
        return self


class PlanningConfig(StrictConfigModel):
    turn_budget: TurnRange = Field(default_factory=TurnRange)
    honor_seed_turn_budget: bool = True
    retrieval_depth_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.25, 2: 0.5, 3: 0.25}
    )
    max_steps_per_turn: int = Field(6, ge=2, le=12)
    ensure_retrieval_turn: bool = True
    first_turn_intents: dict[str, float] = Field(
        default_factory=lambda: {
            "research": 0.12,
            "rewrite": 0.04,
            "clarify": 0.18,
            "user_context": 0.16,
            "scope": 0.14,
            "orientation": 0.12,
            "direct_answer": 0.10,
            "misconception_check": 0.08,
            "example_first": 0.06,
        }
    )
    intents: dict[str, float] = Field(
        default_factory=lambda: {
            "research": 0.13,
            "rewrite": 0.10,
            "clarify": 0.09,
            "deepen": 0.12,
            "compare": 0.09,
            "synthesize": 0.10,
            "recall": 0.06,
            "user_context": 0.06,
            "apply_scenario": 0.08,
            "challenge_assumption": 0.07,
            "summarize": 0.05,
            "misconception_check": 0.05,
        }
    )

    @model_validator(mode="after")
    def valid_distributions(self) -> PlanningConfig:
        if set(self.retrieval_depth_weights) - {1, 2, 3}:
            raise ValueError("retrieval_depth_weights keys must be in 1..3")
        if (
            not self.retrieval_depth_weights
            or sum(self.retrieval_depth_weights.values()) <= 0
        ):
            raise ValueError(
                "retrieval_depth_weights must contain positive total weight"
            )
        if any(weight < 0 for weight in self.retrieval_depth_weights.values()):
            raise ValueError("retrieval_depth_weights must be nonnegative")
        maximum_depth = max(
            depth
            for depth, weight in self.retrieval_depth_weights.items()
            if weight > 0
        )
        if self.max_steps_per_turn <= maximum_depth:
            raise ValueError(
                "max_steps_per_turn must allow every enabled retrieval depth "
                "plus one final-answer step"
            )
        for name, distribution in (
            ("first_turn_intents", self.first_turn_intents),
            ("intents", self.intents),
        ):
            if not distribution or sum(distribution.values()) <= 0:
                raise ValueError(f"planning.{name} must contain positive total weight")
            if any(weight < 0 for weight in distribution.values()):
                raise ValueError(f"planning.{name} weights must be nonnegative")
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


class RetrieverFields(StrictConfigModel):
    id: str = "id"
    text: str = "text"
    title: str = "title"
    source: str = "source"
    score: str = "score"
    url: str = "url"
    date: str = "date"


class RetrieverConfig(StrictConfigModel):
    endpoint: str
    method: Literal["GET", "POST"] = "POST"
    query_field: str = "query"
    top_k_field: str = "top_k"
    top_k: int = Field(4, ge=1, le=100)
    results_path: str = "chunks"
    fields: RetrieverFields = Field(default_factory=RetrieverFields)
    selection: Literal["ranked", "diverse", "sampled"] = "ranked"
    timeout_seconds: float = Field(45, gt=0)
    retries: int = Field(4, ge=1, le=20)
    backoff_seconds: float = Field(1.0, ge=0)
    headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


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
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
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
        # The seed changes generated plans, while the other run fields only control
        # orchestration and retry policy.
        payload["run_seed"] = self.run.seed
        raw = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(raw.encode()).hexdigest()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = PipelineConfig.model_validate(data)
    cfg.config_dir = config_path.parent
    return cfg
