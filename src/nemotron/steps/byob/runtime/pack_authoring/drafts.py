# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The four authoring calls, each validated against the evidence before it is believed.

Each valid proposal is checkpointed immediately. A rejected candidate is retained next
to its validation feedback. A human must correct it and explicitly record review before
the workflow continues; a model cannot repair or approve benchmark semantics for them.

The order is a dependency, not a preference. Coverage decides what the benchmark is trying
to exercise, and the other three refer to it, so it is produced first and passed forward as
input rather than being re-derived three times with three chances to disagree.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from nemotron.steps.byob.runtime.authoring_workflow.quota import RunQuota
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
    write_canonical_yaml,
    write_text_atomic,
)
from nemotron.steps.byob.runtime.pack_authoring.bundle import EvidenceView
from nemotron.steps.byob.runtime.pack_authoring.grounding import (
    Grounding,
    GroundingError,
    validate_assertion_specs,
    validate_coverage_plan,
    validate_task_templates,
    validate_validation_cases,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    AuthoringModelError,
    ModelCallRecord,
    StructuredCaller,
    call_structured,
)
from nemotron.steps.byob.runtime.pack_authoring.prompts import (
    ASSERTION_PROMPT_VERSION,
    ASSERTION_TASK,
    AUTHORING_SYSTEM_PROMPT,
    COVERAGE_PROMPT_VERSION,
    COVERAGE_TASK,
    TASK_TEMPLATE_PROMPT_VERSION,
    TASK_TEMPLATE_TASK,
    VALIDATION_CASE_PROMPT_VERSION,
    VALIDATION_CASE_TASK,
    build_evidence_payload,
    prompt_hash,
)
from nemotron.steps.byob.runtime.pack_authoring.schemas import (
    AssertionSpecPlan,
    CoveragePlan,
    TaskTemplatePlan,
    ValidationCasePlan,
)


@dataclass(frozen=True)
class DraftBundle:
    """Every drafted artifact, plus the record of the calls that produced them."""

    coverage: CoveragePlan
    validation_cases: ValidationCasePlan
    task_templates: TaskTemplatePlan
    assertions: AssertionSpecPlan
    calls: tuple[ModelCallRecord, ...]

    def as_documents(self) -> dict[str, Any]:
        return {
            "coverage_plan": self.coverage.model_dump(mode="json", exclude_unset=True),
            "validation_cases": self.validation_cases.model_dump(mode="json", exclude_unset=True),
            "task_templates": self.task_templates.model_dump(mode="json", exclude_unset=True),
            "assertion_specs": self.assertions.model_dump(mode="json", exclude_unset=True),
        }


@dataclass(frozen=True)
class DraftReview:
    """An explicit human attestation of corrected files, bound on acceptance."""

    stages: tuple[str, ...]
    reviewed_by: str
    reviewed_at: str

    def __post_init__(self) -> None:
        if not self.stages or not set(self.stages) <= {
            "coverage_plan", "validation_cases", "task_templates", "assertion_specs",
        }:
            raise ValueError("name the corrected draft files being reviewed")
        if not self.reviewed_by.strip():
            raise ValueError("corrected drafts require a human reviewer")
        if datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("draft review timestamp must include a timezone")


@dataclass(frozen=True)
class DraftingContext:
    """What every generator needs, gathered so each call site stays about its own stage."""

    evidence: EvidenceView
    model: AuthoringModel
    cache: ImmutableModelIOCache
    run_dir: Path
    caller: StructuredCaller | None = None
    quota: RunQuota | None = None
    checkpoint_root: Path | None = None
    human_review: DraftReview | None = None
    coverage_digest: str | None = None

    @property
    def grounding(self) -> Grounding:
        return Grounding(evidence=self.evidence)


def _parse(
    stage: str,
    response: dict[str, Any],
    output_format: type[BaseModel],
) -> Any:
    try:
        return output_format.model_validate(response)
    except ValidationError as exc:
        # The structured column already constrains the shape, so this fires when a provider
        # returns something outside its own schema. It is a grounding failure in spirit.
        raise GroundingError(stage, [f"response does not satisfy {output_format.__name__}: {exc}"]) from exc


def _call(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    columns: dict[str, str],
    output_format: type[BaseModel],
) -> tuple[Any, ModelCallRecord]:
    try:
        response, record = call_structured(
            context.model,
            stage_name=stage,
            prompt_version=prompt_version,
            system_prompt=AUTHORING_SYSTEM_PROMPT,
            prompt=task,
            columns=columns,
            output_format=output_format,
            cache=context.cache,
            run_dir=context.run_dir,
            caller=context.caller,
            quota=context.quota,
        )
    except AuthoringModelError as exc:
        if context.checkpoint_root is not None and exc.response is not None:
            write_canonical_yaml(exc.response, context.checkpoint_root / f"{stage}.candidate.yaml")
        raise
    if context.checkpoint_root is not None:
        write_canonical_yaml(response, context.checkpoint_root / f"{stage}.candidate.yaml")
    return _parse(stage, response, output_format), record


def _checkpoint_record(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    output_format: type[BaseModel],
    document: BaseModel,
) -> ModelCallRecord:
    dumped = document.model_dump(mode="json", exclude_unset=True)
    return ModelCallRecord(
        stage=stage,
        prompt_version=prompt_version,
        prompt_hash=prompt_hash(prompt_version, AUTHORING_SYSTEM_PROMPT, task),
        request_hash=sha256_json({"human_checkpoint": dumped}),
        input_hash=sha256_json({"evidence_digest": context.evidence.digest}),
        output_schema_hash=sha256_json(output_format.model_json_schema()),
        model_canonical="human-reviewed",
        served_from_cache=False,
        source="human_checkpoint",
    )


def _run_stage(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    document_name: str,
    output_format: type[BaseModel],
    validate: Callable[[Grounding, Any], Any],
    produce: Callable[[DraftingContext], tuple[Any, ModelCallRecord]],
) -> tuple[Any, ModelCallRecord]:
    """Propose once, or resume an evidence-bound file; errors require human correction."""
    if context.checkpoint_root is None:
        return produce(context)
    path = context.checkpoint_root / f"{document_name}.yaml"
    metadata_path = path.with_suffix(".checkpoint.json")
    binding = {
        "schema_version": "bfcl-draft-checkpoint-v2",
        "evidence_digest": context.evidence.digest,
        "coverage_digest": context.coverage_digest,
        "prompt_hash": prompt_hash(prompt_version, AUTHORING_SYSTEM_PROMPT, task),
        "output_schema_hash": sha256_json(output_format.model_json_schema()),
    }
    recovery = (
        f"A human must correct {path}, then rerun draft with --reviewed-draft {document_name} "
        "--draft-reviewed-by HUMAN --draft-reviewed-at ISO-8601-TIMESTAMP. "
        f"The model proposal, when available, is {context.checkpoint_root / (stage + '.candidate.yaml')}."
    )
    review = context.human_review
    explicitly_reviewed = review is not None and document_name in review.stages
    if path.is_file():
        try:
            document = validate(context.grounding, output_format.model_validate(yaml.safe_load(path.read_text())))
        except (OSError, ValueError, yaml.YAMLError, GroundingError) as exc:
            raise GroundingError(stage, [f"checkpoint is invalid: {exc}", recovery]) from exc
        digest = sha256_json(document.model_dump(mode="json", exclude_unset=True))
        try:
            metadata = json.loads(metadata_path.read_text())
            if any(metadata.get(key) != value for key, value in binding.items()):
                raise ValueError("checkpoint inputs changed")
            if metadata.get("document_digest") != digest or metadata.get("status") not in {
                "model_draft", "human_reviewed",
            }:
                raise ValueError("checkpoint was edited or not accepted")
            record = ModelCallRecord(**metadata["call"])
            if not explicitly_reviewed:
                return document, replace(record, served_from_cache=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            if not explicitly_reviewed:
                raise GroundingError(stage, ["checkpoint metadata is missing, invalid, or stale", recovery]) from None
        record = _checkpoint_record(
            context, stage=stage, prompt_version=prompt_version, task=task,
            output_format=output_format, document=document,
        )
    else:
        rejected_path = context.checkpoint_root / f"{stage}.rejected.txt"
        if explicitly_reviewed or metadata_path.exists() or rejected_path.exists():
            raise GroundingError(stage, ["corrected canonical file is required before resuming", recovery])
        try:
            document, record = produce(context)
        except (AuthoringModelError, GroundingError) as exc:
            write_text_atomic(
                f"stage: {stage}\nerror: {exc}\n{recovery}\n",
                context.checkpoint_root / f"{stage}.rejected.txt",
            )
            write_canonical_json({**binding, "status": "rejected"}, metadata_path)
            # Surface the recovery through the CLI's JSON error, not only a traceback note.
            raise GroundingError(stage, [str(exc), recovery]) from exc
        write_canonical_yaml(document.model_dump(mode="json", exclude_unset=True), path)
    metadata = {
        **binding,
        "status": "human_reviewed" if explicitly_reviewed else "model_draft",
        "document_digest": sha256_json(document.model_dump(mode="json", exclude_unset=True)),
        "call": record.as_dict(),
    }
    if explicitly_reviewed:
        metadata["human_review"] = {
            "reviewed_by": review.reviewed_by,
            "reviewed_at": review.reviewed_at,
        }
    write_canonical_json(metadata, metadata_path)
    return document, record


def draft_coverage_plan(
    context: DraftingContext,
) -> tuple[CoveragePlan, ModelCallRecord]:
    """What the benchmark should exercise, for every published tool."""
    plan, record = _call(
        context,
        stage="mcp_coverage_plan",
        prompt_version=COVERAGE_PROMPT_VERSION,
        task=COVERAGE_TASK,
        columns={"evidence": build_evidence_payload(context.evidence)},
        output_format=CoveragePlan,
    )
    return validate_coverage_plan(context.grounding, plan), record


def draft_validation_cases(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[ValidationCasePlan, ModelCallRecord]:
    """Probes proving each tool behaves as the pack will claim."""
    plan, record = _call(
        context,
        stage="mcp_validation_cases",
        prompt_version=VALIDATION_CASE_PROMPT_VERSION,
        task=VALIDATION_CASE_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=ValidationCasePlan,
    )
    return validate_validation_cases(context.grounding, plan), record


def draft_task_templates(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[TaskTemplatePlan, ModelCallRecord]:
    """The multi-turn tasks, their ordering, and their milestones."""
    plan, record = _call(
        context,
        stage="mcp_task_templates",
        prompt_version=TASK_TEMPLATE_PROMPT_VERSION,
        task=TASK_TEMPLATE_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=TaskTemplatePlan,
    )
    return validate_task_templates(context.grounding, plan), record


def draft_assertion_specs(
    context: DraftingContext,
    coverage: CoveragePlan,
) -> tuple[AssertionSpecPlan, ModelCallRecord]:
    """Declarative predicates over result, state, and trace."""
    plan, record = _call(
        context,
        stage="mcp_assertion_specs",
        prompt_version=ASSERTION_PROMPT_VERSION,
        task=ASSERTION_TASK,
        columns={
            "evidence": build_evidence_payload(context.evidence),
            "coverage": coverage.model_dump_json(indent=2),
        },
        output_format=AssertionSpecPlan,
    )
    return validate_assertion_specs(context.grounding, plan), record


def draft_all(context: DraftingContext) -> DraftBundle:
    """Run the four calls in dependency order, checkpointing every accepted stage."""
    coverage, coverage_record = _run_stage(
        context,
        stage="mcp_coverage_plan",
        prompt_version=COVERAGE_PROMPT_VERSION,
        task=COVERAGE_TASK,
        document_name="coverage_plan",
        output_format=CoveragePlan,
        validate=validate_coverage_plan,
        produce=draft_coverage_plan,
    )
    context = replace(context, coverage_digest=sha256_json(coverage.model_dump(mode="json", exclude_unset=True)))
    cases, cases_record = _run_stage(
        context,
        stage="mcp_validation_cases",
        prompt_version=VALIDATION_CASE_PROMPT_VERSION,
        task=VALIDATION_CASE_TASK,
        document_name="validation_cases",
        output_format=ValidationCasePlan,
        validate=validate_validation_cases,
        produce=lambda active: draft_validation_cases(active, coverage),
    )
    templates, templates_record = _run_stage(
        context,
        stage="mcp_task_templates",
        prompt_version=TASK_TEMPLATE_PROMPT_VERSION,
        task=TASK_TEMPLATE_TASK,
        document_name="task_templates",
        output_format=TaskTemplatePlan,
        validate=validate_task_templates,
        produce=lambda active: draft_task_templates(active, coverage),
    )
    assertions, assertions_record = _run_stage(
        context,
        stage="mcp_assertion_specs",
        prompt_version=ASSERTION_PROMPT_VERSION,
        task=ASSERTION_TASK,
        document_name="assertion_specs",
        output_format=AssertionSpecPlan,
        validate=validate_assertion_specs,
        produce=lambda active: draft_assertion_specs(active, coverage),
    )
    return DraftBundle(
        coverage=coverage,
        validation_cases=cases,
        task_templates=templates,
        assertions=assertions,
        calls=(coverage_record, cases_record, templates_record, assertions_record),
    )
