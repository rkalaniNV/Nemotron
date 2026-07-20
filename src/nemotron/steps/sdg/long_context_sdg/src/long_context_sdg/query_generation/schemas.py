"""Schemas for taxonomy-driven, persona-conditioned query generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import RetrievalChunk


class QueryGenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaxonomyNode(QueryGenerationModel):
    id: str
    label: str
    description: str = ""
    weight: float = Field(1.0, gt=0)
    seed_queries: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    required_terms: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    children: list[TaxonomyNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def usable_leaf(self) -> TaxonomyNode:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("taxonomy node id and label must be non-empty")
        if not self.children and not any(query.strip() for query in self.seed_queries):
            raise ValueError(f"taxonomy leaf `{self.id}` needs at least one seed query")
        return self


class QueryTaxonomy(QueryGenerationModel):
    version: str
    topics: list[TaxonomyNode]

    @model_validator(mode="after")
    def unique_nodes(self) -> QueryTaxonomy:
        if not self.version.strip() or not self.topics:
            raise ValueError("taxonomy version and topics are required")
        seen: set[str] = set()

        def visit(node: TaxonomyNode) -> None:
            if node.id in seen:
                raise ValueError(f"duplicate taxonomy id `{node.id}`")
            seen.add(node.id)
            for child in node.children:
                visit(child)

        for topic in self.topics:
            visit(topic)
        return self

    def leaves(self) -> list[TaxonomyNode]:
        leaves: list[TaxonomyNode] = []

        def visit(node: TaxonomyNode) -> None:
            if node.children:
                for child in node.children:
                    visit(child)
            else:
                leaves.append(node)

        for topic in self.topics:
            visit(topic)
        return leaves

    def weighted_leaves(self) -> list[tuple[TaxonomyNode, float]]:
        """Return leaves with the product of every weight on their path."""
        leaves: list[tuple[TaxonomyNode, float]] = []

        def visit(node: TaxonomyNode, inherited: float) -> None:
            combined = inherited * node.weight
            if node.children:
                for child in node.children:
                    visit(child, combined)
            else:
                leaves.append((node, combined))

        for topic in self.topics:
            visit(topic, 1.0)
        return leaves


class PersonaProjection(QueryGenerationModel):
    source_dataset: str
    source_revision: str
    source_split: str
    source_id: str
    language: str
    narrative_field: str
    narrative: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class QueryCandidate(QueryGenerationModel):
    query_id: str
    synthesis_fingerprint: str
    candidate_index: int = Field(ge=0)
    taxonomy_id: str
    taxonomy_label: str
    taxonomy_description: str = ""
    taxonomy_required_terms: list[str] = Field(default_factory=list)
    archetype: str
    evidence_scope: Literal["conversational", "single_facet", "multi_facet"]
    minimum_evidence_needs: int = Field(ge=0)
    answerability: Literal["answerable", "insufficient"]
    persona_mode: str
    surface_form: Literal[
        "well_formed",
        "underspecified",
        "retrieval_rewrite",
        "noisy_language",
        "keyword_fragment",
        "overbroad",
        "adjacent_intent",
    ]
    persona_key: str
    persona_locale: str
    language: str
    evidence: list[RetrievalChunk] = Field(min_length=1)


class EvidenceNeed(QueryGenerationModel):
    need: str = Field(min_length=1)
    retrieval_probe: str = Field(min_length=1)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    supported_by_bundle: bool = True


class QueryDraft(QueryGenerationModel):
    query: str = Field(min_length=1)
    naive_query: str = Field(min_length=1)
    role: str = Field(min_length=1)
    expertise: str = Field(min_length=1)
    style: str = Field(min_length=1)
    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)


class QuerySynthesisJudgment(QueryGenerationModel):
    scores: dict[str, int]
    answerability: Literal["answerable", "insufficient"]
    rating: Literal["success", "failure"]
    explanation: str

    @model_validator(mode="after")
    def bounded_scores(self) -> QuerySynthesisJudgment:
        invalid = {name: score for name, score in self.scores.items() if not 1 <= score <= 5}
        if invalid:
            raise ValueError(f"judge scores must be in 1..5: {invalid}")
        return self


class QuerySynthesisRecord(QueryGenerationModel):
    query_id: str
    synthesis_fingerprint: str
    candidate_index: int
    attempt: int = Field(ge=1)
    status: Literal["accepted", "rejected", "generation_failed"]
    taxonomy_id: str
    archetype: str
    language: str
    persona_mode: str
    surface_form: str
    draft: QueryDraft | None = None
    judgment: QuerySynthesisJudgment | None = None
    validation_errors: list[str] = Field(default_factory=list)
    seed: dict[str, Any] | None = None
