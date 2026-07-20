"""Strict, standalone configuration for the query-generation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ..config_base import StrictConfigModel
from ..service_config import ModelConfig, ProviderConfig, RetrieverConfig

SURFACE_FORMS = frozenset(
    {
        "well_formed",
        "underspecified",
        "retrieval_rewrite",
        "noisy_language",
        "keyword_fragment",
        "overbroad",
        "adjacent_intent",
    }
)


class QueryGenerationRunConfig(StrictConfigModel):
    mode: Literal["create"] = "create"
    seed: int = 7
    dataset_name: str = "long_context_query_synthesis"
    resume: Literal["never", "always", "if_possible"] = "always"

    @model_validator(mode="after")
    def valid_dataset_name(self) -> QueryGenerationRunConfig:
        if not self.dataset_name.strip():
            raise ValueError("run.dataset_name must be non-empty")
        return self


class PersonaLocaleConfig(StrictConfigModel):
    locale: str
    language: str
    weight: float = Field(gt=0)
    asset_revision: str
    narrative_fields: dict[str, float] = Field(
        default_factory=lambda: {
            "persona": 0.40,
            "professional_persona": 0.30,
            "skills_and_expertise": 0.15,
            "hobbies_and_interests": 0.15,
        }
    )
    attribute_fields: list[str] = Field(
        default_factory=lambda: [
            "occupation",
            "education_level",
            "first_language",
            "state",
        ]
    )
    sex: Literal["Male", "Female"] | None = None
    city: str | list[str] | None = None
    age_range: list[int] = Field(default_factory=lambda: [18, 114], min_length=2, max_length=2)
    select_field_values: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def valid_locale(self) -> PersonaLocaleConfig:
        if not self.locale.strip() or not self.language.strip() or not self.asset_revision.strip():
            raise ValueError("persona locale, language, and asset_revision must be non-empty")
        if (
            not self.narrative_fields
            or sum(self.narrative_fields.values()) <= 0
            or any(weight < 0 for weight in self.narrative_fields.values())
        ):
            raise ValueError("persona narrative field weights must have a positive total")
        if self.age_range[0] >= self.age_range[1]:
            raise ValueError("persona age_range minimum must be below maximum")
        if self.age_range[0] < 18 or self.age_range[1] > 114:
            raise ValueError("persona age_range must remain within Data Designer's 18..114 bounds")
        return self


class QueryEvidenceConfig(StrictConfigModel):
    pool_size: int = Field(32, ge=4, le=100)
    bundle_min: int = Field(1, ge=1, le=8)
    bundle_max: int = Field(4, ge=1, le=8)
    max_per_source: int = Field(2, ge=1, le=8)
    max_pair_similarity: float = Field(0.75, ge=0, le=1)
    min_chunk_chars: int = Field(160, ge=1)
    retrievability_top_k: int = Field(8, ge=1, le=100)
    max_lexical_overlap: float = Field(0.35, ge=0, le=1)
    max_verbatim_tokens: int = Field(12, ge=4, le=50)
    duplicate_similarity: float = Field(0.72, ge=0, le=1)
    max_probe_similarity: float = Field(0.80, ge=0, le=1)

    @model_validator(mode="after")
    def valid_bundle(self) -> QueryEvidenceConfig:
        if self.bundle_max < self.bundle_min:
            raise ValueError("query_generation.evidence.bundle_max must be >= bundle_min")
        if self.bundle_max > self.pool_size:
            raise ValueError("query_generation evidence bundle cannot exceed its pool")
        return self


class QueryArchetypeProfile(StrictConfigModel):
    """Evidence shape used to induce a natural retrieval pattern."""

    evidence_scope: Literal["conversational", "single_facet", "multi_facet"]
    bundle_min: int = Field(1, ge=1, le=8)
    bundle_max: int = Field(1, ge=1, le=8)
    min_sources: int = Field(1, ge=1, le=8)
    min_evidence_needs: int = Field(1, ge=0, le=8)

    @model_validator(mode="after")
    def valid_profile(self) -> QueryArchetypeProfile:
        if self.bundle_max < self.bundle_min:
            raise ValueError("archetype bundle_max must be >= bundle_min")
        if self.min_sources > self.bundle_max:
            raise ValueError("archetype min_sources cannot exceed bundle_max")
        if self.evidence_scope == "multi_facet" and self.min_evidence_needs < 2:
            raise ValueError("multi_facet archetypes require at least two evidence needs")
        if self.evidence_scope == "single_facet" and self.min_evidence_needs != 1:
            raise ValueError("single_facet archetypes require exactly one evidence need")
        if self.evidence_scope == "conversational" and self.min_evidence_needs != 0:
            raise ValueError("conversational archetypes require zero evidence needs")
        return self


class QuerySurfaceFormProfile(StrictConfigModel):
    """Quality gates for the visible first utterance, not runtime tool policy."""

    minimum_anchor_recall_gap: float = Field(0.0, ge=0, le=1)
    require_noncanonical_form: bool = False
    require_topic_overlap: bool = False


def _default_surface_form_profiles() -> dict[str, QuerySurfaceFormProfile]:
    return {
        "well_formed": QuerySurfaceFormProfile(),
        "underspecified": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.20,
            require_noncanonical_form=True,
        ),
        "retrieval_rewrite": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.20,
            require_noncanonical_form=True,
        ),
        "noisy_language": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.10,
            require_noncanonical_form=True,
        ),
        "keyword_fragment": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.10,
            require_noncanonical_form=True,
        ),
        "overbroad": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.10,
            require_noncanonical_form=True,
        ),
        "adjacent_intent": QuerySurfaceFormProfile(
            minimum_anchor_recall_gap=0.20,
            require_noncanonical_form=True,
            require_topic_overlap=True,
        ),
    }


def _default_archetype_profiles() -> dict[str, QueryArchetypeProfile]:
    return {
        "single_lookup": QueryArchetypeProfile(
            evidence_scope="single_facet", bundle_min=1, bundle_max=1, min_sources=1, min_evidence_needs=1
        ),
        "comparison": QueryArchetypeProfile(
            evidence_scope="multi_facet", bundle_min=2, bundle_max=3, min_sources=2, min_evidence_needs=2
        ),
        "timeline": QueryArchetypeProfile(
            evidence_scope="multi_facet", bundle_min=2, bundle_max=4, min_sources=1, min_evidence_needs=2
        ),
        "multi_constraint": QueryArchetypeProfile(
            evidence_scope="multi_facet", bundle_min=3, bundle_max=4, min_sources=2, min_evidence_needs=3
        ),
        "conflict_resolution": QueryArchetypeProfile(
            evidence_scope="multi_facet", bundle_min=2, bundle_max=4, min_sources=2, min_evidence_needs=2
        ),
        "clarification": QueryArchetypeProfile(
            evidence_scope="conversational", bundle_min=1, bundle_max=2, min_sources=1, min_evidence_needs=0
        ),
        "insufficient_evidence": QueryArchetypeProfile(
            evidence_scope="multi_facet", bundle_min=2, bundle_max=3, min_sources=1, min_evidence_needs=2
        ),
    }


class QueryGenerationPaths(StrictConfigModel):
    seeds: str
    evidence_manifest: str = "../output/query_generation/evidence_manifest.json"
    candidates: str = "../output/query_generation/candidates.jsonl"
    artifacts: str = "../output/query_generation/data_designer"
    report: str = "../output/query_generation/report.json"


class QueryGenerationConfig(StrictConfigModel):
    num_queries: int = Field(100, ge=1)
    taxonomy_path: str
    generator_alias: str = "assistant"
    judge_alias: str = "judge"
    max_attempts: int = Field(3, ge=1, le=10)
    min_judge_score: int = Field(4, ge=1, le=5)
    min_query_chars: int = Field(8, ge=1)
    max_query_chars: int = Field(400, ge=1)
    persona_locales: list[PersonaLocaleConfig] = Field(default_factory=list)
    archetype_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "single_lookup": 0.25,
            "comparison": 0.20,
            "timeline": 0.15,
            "multi_constraint": 0.20,
            "conflict_resolution": 0.10,
            "clarification": 0.05,
            "insufficient_evidence": 0.05,
        }
    )
    archetype_profiles: dict[str, QueryArchetypeProfile] = Field(default_factory=_default_archetype_profiles)
    persona_mode_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "general_interest": 0.40,
            "situated_need": 0.40,
            "domain_adjacent": 0.20,
        }
    )
    surface_form_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "well_formed": 0.20,
            "underspecified": 0.20,
            "retrieval_rewrite": 0.20,
            "noisy_language": 0.10,
            "keyword_fragment": 0.10,
            "overbroad": 0.10,
            "adjacent_intent": 0.10,
        }
    )
    surface_form_profiles: dict[str, QuerySurfaceFormProfile] = Field(
        default_factory=_default_surface_form_profiles
    )
    evidence: QueryEvidenceConfig = Field(default_factory=QueryEvidenceConfig)

    @staticmethod
    def _valid_weights(name: str, value: dict[str, float]) -> None:
        invalid = not value or sum(value.values()) <= 0 or any(weight < 0 for weight in value.values())
        if invalid:
            raise ValueError(f"query_generation.{name} must have a positive total")

    @model_validator(mode="after")
    def valid_generation(self) -> QueryGenerationConfig:
        if self.max_query_chars < self.min_query_chars:
            raise ValueError("query_generation.max_query_chars must be >= min_query_chars")
        self._valid_weights("archetype_weights", self.archetype_weights)
        self._valid_weights("persona_mode_weights", self.persona_mode_weights)
        self._valid_weights("surface_form_weights", self.surface_form_weights)
        missing_profiles = sorted(
            name
            for name, weight in self.archetype_weights.items()
            if weight > 0 and name not in self.archetype_profiles
        )
        if missing_profiles:
            raise ValueError(f"query_generation archetypes missing profiles: {missing_profiles}")
        missing_surface_profiles = sorted(
            name
            for name, weight in self.surface_form_weights.items()
            if weight > 0 and name not in self.surface_form_profiles
        )
        if missing_surface_profiles:
            raise ValueError(
                "query_generation surface forms missing profiles: "
                f"{missing_surface_profiles}"
            )
        unknown_surface_forms = sorted(
            (set(self.surface_form_weights) | set(self.surface_form_profiles)) - SURFACE_FORMS
        )
        if unknown_surface_forms:
            raise ValueError(
                "query_generation surface forms are unsupported: "
                f"{unknown_surface_forms}"
            )
        for name, profile in self.archetype_profiles.items():
            if profile.bundle_min < self.evidence.bundle_min or profile.bundle_max > self.evidence.bundle_max:
                raise ValueError(
                    f"archetype profile `{name}` bundle range must fit evidence.bundle_min..bundle_max"
                )
        if not self.taxonomy_path.strip():
            raise ValueError("query_generation.taxonomy_path is required")
        if not self.persona_locales:
            raise ValueError("query_generation.persona_locales is required")
        locales = [item.locale for item in self.persona_locales]
        if len(locales) != len(set(locales)):
            raise ValueError("query_generation persona locales must be unique")
        return self


class QueryGenerationPipelineConfig(StrictConfigModel):
    """All and only the dependencies required to create seed queries."""

    paths: QueryGenerationPaths
    run: QueryGenerationRunConfig = Field(default_factory=QueryGenerationRunConfig)
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[ModelConfig]
    retriever: RetrieverConfig
    query_generation: QueryGenerationConfig
    config_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def valid_dependencies(self) -> QueryGenerationPipelineConfig:
        aliases = [model.alias for model in self.models]
        if len(aliases) != len(set(aliases)):
            raise ValueError("query-generation model aliases must be unique")
        required = {
            self.query_generation.generator_alias,
            self.query_generation.judge_alias,
        }
        missing = required - set(aliases)
        if missing:
            raise ValueError(f"query-generation model aliases unavailable: {sorted(missing)}")
        provider_names = [provider.name for provider in self.providers]
        if len(provider_names) != len(set(provider_names)):
            raise ValueError("query-generation provider names must be unique")
        configured = set(provider_names)
        dangling = {model.provider for model in self.models if configured and model.provider not in configured}
        if dangling:
            raise ValueError(f"query-generation models reference unavailable providers: {sorted(dangling)}")
        return self

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.config_dir / path).resolve()

    def generation_payload(self) -> dict[str, Any]:
        """Return runtime config without Data Designer orchestration identity."""
        payload = self.model_dump(mode="json", exclude={"config_dir"})
        payload["paths"] = {name: "." for name in type(self.paths).model_fields}
        payload["run"] = {
            "mode": "create",
            "seed": self.run.seed,
            "dataset_name": "embedded",
            "resume": "never",
        }
        return payload


def load_query_generation_config(
    path: str | Path,
) -> QueryGenerationPipelineConfig:
    from omegaconf import OmegaConf

    config_path = Path(path).resolve()
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = QueryGenerationPipelineConfig.model_validate(data)
    cfg.config_dir = config_path.parent
    return cfg
