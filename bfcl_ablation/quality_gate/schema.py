"""Versioned contracts for A7 checks, thresholds, and human review files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

GATE_CONTRACT_VERSION = "1.0"
HUMAN_LABEL_CONTRACT_VERSION = "1.0"

AuditStatus = Literal["PASS", "CONDITIONAL", "FAIL", "INCONCLUSIVE"]
AuditDimension = Literal["integrity", "study_validity", "release_readiness"]
ReviewKind = Literal[
    "paraphrase_pair",
    "task_semantics",
    "model_disagreement",
    "mutant_triage",
    "control",
]
Severity = Literal["none", "minor", "major", "critical"]
MutantClassification = Literal["equivalent", "unreachable", "real_gap"]


def _non_empty(value: str, *, field: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


class Reviewer(BaseModel):
    """One declared reviewer; IDs are stable but need not contain a real name."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_id: str
    display_name: str | None = None

    @model_validator(mode="after")
    def validate_reviewer(self) -> Reviewer:
        self.reviewer_id = _non_empty(self.reviewer_id, field="reviewer_id")
        if self.display_name is not None:
            self.display_name = _non_empty(self.display_name, field="display_name")
        return self


class ReviewVerdict(BaseModel):
    """One independent judgement of a blinded review item."""

    model_config = ConfigDict(extra="forbid", strict=True)

    reviewer_id: str
    intent_preserved: StrictBool | None = None
    acceptable_for_benchmark: StrictBool
    required_tools: list[str] | None = None
    turn_policy: str | None = None
    mutant_classification: MutantClassification | None = None
    severity: Severity = "none"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_verdict(self) -> ReviewVerdict:
        self.reviewer_id = _non_empty(self.reviewer_id, field="reviewer_id")
        if self.required_tools is not None:
            cleaned = [_non_empty(tool, field="required_tools entry") for tool in self.required_tools]
            if len(cleaned) != len(set(cleaned)):
                raise ValueError("required_tools must not contain duplicates")
            self.required_tools = cleaned
        if self.turn_policy is not None:
            self.turn_policy = _non_empty(self.turn_policy, field="turn_policy")
        if self.notes is not None and not self.notes.strip():
            raise ValueError("notes must be non-empty when present")
        return self


class Adjudication(BaseModel):
    """Final decision used when independent reviewers disagree."""

    model_config = ConfigDict(extra="forbid", strict=True)

    adjudicator_id: str
    intent_preserved: StrictBool | None = None
    acceptable_for_benchmark: StrictBool
    required_tools: list[str] | None = None
    turn_policy: str | None = None
    mutant_classification: MutantClassification | None = None
    severity: Severity = "none"
    notes: str

    @model_validator(mode="after")
    def validate_adjudication(self) -> Adjudication:
        self.adjudicator_id = _non_empty(self.adjudicator_id, field="adjudicator_id")
        self.notes = _non_empty(self.notes, field="notes")
        if self.required_tools is not None and len(self.required_tools) != len(set(self.required_tools)):
            raise ValueError("required_tools must not contain duplicates")
        return self


class HumanReviewItem(BaseModel):
    """One item shown to reviewers without model or oracle verdicts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    item_id: str
    kind: ReviewKind
    source_arm: Literal["a2", "a3", "a5", "a6", "control"]
    source_ref: str
    template_id: str | None = None
    task_id: str | None = None
    variant_index: StrictInt | None = Field(default=None, ge=0)
    reference_text: str | None = None
    candidate_text: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    labels: list[ReviewVerdict] = Field(default_factory=list)
    adjudication: Adjudication | None = None

    @model_validator(mode="after")
    def validate_item(self) -> HumanReviewItem:
        self.item_id = _non_empty(self.item_id, field="item_id")
        self.source_ref = _non_empty(self.source_ref, field="source_ref")
        if self.kind in {"paraphrase_pair", "control"}:
            if not self.reference_text or not self.reference_text.strip():
                raise ValueError(f"{self.kind} requires reference_text")
            if not self.candidate_text or not self.candidate_text.strip():
                raise ValueError(f"{self.kind} requires candidate_text")
        if self.kind == "task_semantics" and not self.candidate_text:
            raise ValueError("task_semantics requires candidate_text")
        if self.kind == "mutant_triage" and "mutant_index" not in self.context:
            raise ValueError("mutant_triage requires context.mutant_index")
        reviewer_ids = [label.reviewer_id for label in self.labels]
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError(f"item {self.item_id!r} contains duplicate reviewer labels")
        verdicts: list[ReviewVerdict | Adjudication] = list(self.labels)
        if self.adjudication is not None:
            verdicts.append(self.adjudication)
        for label in verdicts:
            if self.kind in {"paraphrase_pair", "control"} and label.intent_preserved is None:
                raise ValueError(f"{self.kind} labels require intent_preserved")
            if self.kind == "mutant_triage" and label.mutant_classification is None:
                raise ValueError("mutant_triage labels require mutant_classification")
            if self.kind == "task_semantics" and (
                label.required_tools is None or label.turn_policy is None
            ):
                raise ValueError("task_semantics labels require required_tools and turn_policy")
        return self


class HumanReviewFile(BaseModel):
    """Complete human-review queue plus any labels collected so far."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = HUMAN_LABEL_CONTRACT_VERSION
    rubric_version: Literal["1.0"] = "1.0"
    pack_id: str
    language: str
    reviewers: list[Reviewer] = Field(default_factory=list)
    items: list[HumanReviewItem]

    @model_validator(mode="after")
    def validate_review_file(self) -> HumanReviewFile:
        self.pack_id = _non_empty(self.pack_id, field="pack_id")
        self.language = _non_empty(self.language, field="language")
        reviewer_ids = [reviewer.reviewer_id for reviewer in self.reviewers]
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise ValueError("reviewers must have unique reviewer_id values")
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("items must have unique item_id values")
        declared = set(reviewer_ids)
        for item in self.items:
            used = {label.reviewer_id for label in item.labels}
            if item.adjudication is not None:
                used.add(item.adjudication.adjudicator_id)
            unknown = sorted(used - declared)
            if unknown:
                raise ValueError(f"item {item.item_id!r} references undeclared reviewers: {unknown}")
        return self


class HumanThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_reviewers_per_item: StrictInt = Field(ge=1)
    max_semantic_error_rate: float = Field(ge=0.0, le=1.0)
    max_semantic_error_upper_ci95: float = Field(ge=0.0, le=1.0)
    max_control_miss_rate: float = Field(ge=0.0, le=1.0)


class A2Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_shift_recall: float = Field(ge=0.0, le=1.0)
    max_canonical_false_alarm: float = Field(ge=0.0, le=1.0)
    max_llm_substitution_rate: float = Field(ge=0.0, le=1.0)


class A3Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_unfalsifiable_share: float = Field(ge=0.0, le=1.0)
    max_tools_never_called: StrictInt = Field(ge=0)


class A4Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_argument_false_acceptance: float = Field(ge=0.0, le=1.0)
    max_call_false_acceptance: float = Field(ge=0.0, le=1.0)
    max_state_false_acceptance: float = Field(ge=0.0, le=1.0)
    min_gold_pass_rate: float = Field(ge=0.0, le=1.0)


class A5Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_absolute_score_delta: float = Field(ge=0.0, le=1.0)
    min_paired_tasks: StrictInt = Field(ge=1)
    min_discordant_pairs: StrictInt = Field(ge=1)
    min_model_families: StrictInt = Field(ge=1)
    min_wordings: StrictInt = Field(ge=2)


class A6Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_all_gate_blind_rate: float = Field(ge=0.0, le=1.0)
    max_critical_real_gaps: StrictInt = Field(ge=0)


class ThresholdPolicy(BaseModel):
    """Versioned policy; scientific facts and policy thresholds remain separate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    contract_version: Literal["1.0"] = GATE_CONTRACT_VERSION
    human: HumanThresholds
    a2: A2Thresholds
    a3: A3Thresholds
    a4: A4Thresholds
    a5: A5Thresholds
    a6: A6Thresholds


class AuditCheck(BaseModel):
    """One self-contained check written to A7/checks.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    check_id: str
    arm: str
    dimension: AuditDimension
    status: AuditStatus
    claim: str
    detail: str
    gating: StrictBool = True
    value: Any = None
    threshold: Any = None
    numerator: int | None = None
    denominator: int | None = None
    ci95: tuple[float, float] | None = None
    source_paths: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_check(self) -> AuditCheck:
        self.check_id = _non_empty(self.check_id, field="check_id")
        self.arm = _non_empty(self.arm, field="arm")
        self.claim = _non_empty(self.claim, field="claim")
        self.detail = _non_empty(self.detail, field="detail")
        if self.denominator is not None and self.denominator < 0:
            raise ValueError("denominator must be non-negative")
        if self.numerator is not None and self.numerator < 0:
            raise ValueError("numerator must be non-negative")
        if self.numerator is not None and self.denominator is not None and self.numerator > self.denominator:
            raise ValueError("numerator cannot exceed denominator")
        return self


def rollup(checks: list[AuditCheck], dimension: AuditDimension) -> AuditStatus:
    """Conservative status rollup over gating checks in one dimension."""
    statuses = {check.status for check in checks if check.dimension == dimension and check.gating}
    if not statuses:
        return "INCONCLUSIVE"
    if "FAIL" in statuses:
        return "FAIL"
    if "INCONCLUSIVE" in statuses:
        return "INCONCLUSIVE"
    if "CONDITIONAL" in statuses:
        return "CONDITIONAL"
    return "PASS"
