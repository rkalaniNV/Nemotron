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

Each accepted stage is checkpointed immediately.  A rejected candidate is retained next
to its validation feedback and may receive one explicit repair attempt.  The retry is part
of BFCL's visible authoring policy rather than Data Designer's hidden restart loop.

The order is a dependency, not a preference. Coverage decides what the benchmark is trying
to exercise, and the other three refer to it, so it is produced first and passed forward as
input rather than being re-derived three times with three chances to disagree.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
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
            "coverage_plan": self.coverage.model_dump(mode="json"),
            "validation_cases": self.validation_cases.model_dump(mode="json"),
            "task_templates": self.task_templates.model_dump(mode="json"),
            "assertion_specs": self.assertions.model_dump(mode="json"),
        }


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
    repair_attempts: int = 0
    repair_feedback: str | None = None
    attempt_index: int = 0

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
    if context.repair_feedback is not None:
        task = (
            f"{task}\n\nThis is a single repair attempt. Correct every issue in "
            "{{ repair_feedback }} and return the complete corrected document."
        )
        columns = {**columns, "repair_feedback": context.repair_feedback}
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
    parsed = _parse(stage, response, output_format)
    if context.checkpoint_root is not None:
        _write_yaml(
            parsed.model_dump(mode="json"),
            context.checkpoint_root / f"{stage}.attempt-{context.attempt_index}.candidate.yaml",
        )
    return parsed, record


def _write_yaml(document: object, path: Path) -> Path:
    payload = yaml.safe_dump(
        document,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )
    return write_text_atomic(str(payload), path)


def _stage_error_text(stage: str, attempt: int, exc: Exception) -> str:
    return f"stage: {stage}\nattempt: {attempt}\nerror: {exc}\n"


def _checkpoint_record(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    output_format: type[BaseModel],
    document: BaseModel,
) -> ModelCallRecord:
    dumped = document.model_dump(mode="json")
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


def _load_accepted_checkpoint(
    context: DraftingContext,
    *,
    stage: str,
    prompt_version: str,
    task: str,
    document_name: str,
    output_format: type[BaseModel],
    validate: Callable[[Grounding, Any], Any],
) -> tuple[Any, ModelCallRecord] | None:
    if context.checkpoint_root is None:
        return None
    path = context.checkpoint_root / f"{document_name}.yaml"
    if not path.is_file():
        return None
    try:
        document = output_format.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        document = validate(context.grounding, document)
    except (OSError, ValueError, ValidationError, GroundingError) as exc:
        raise GroundingError(
            stage,
            [f"reviewed checkpoint {path} is invalid: {exc}"],
        ) from exc
    digest = sha256_json(document.model_dump(mode="json"))
    metadata_path = context.checkpoint_root / f"{document_name}.checkpoint.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("evidence_digest") != context.evidence.digest:
            return None
        if metadata.get("document_digest") != digest:
            raise ValueError("checkpoint was edited")
        record = replace(
            ModelCallRecord(**metadata["call"]),
            served_from_cache=True,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        record = _checkpoint_record(
            context,
            stage=stage,
            prompt_version=prompt_version,
            task=task,
            output_format=output_format,
            document=document,
        )
    return document, record


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
    """Run, validate, checkpoint, and at most once repair one drafting stage."""
    accepted = _load_accepted_checkpoint(
        context,
        stage=stage,
        prompt_version=prompt_version,
        task=task,
        document_name=document_name,
        output_format=output_format,
        validate=validate,
    )
    if accepted is not None:
        return accepted
    last_error: Exception | None = None
    for attempt in range(context.repair_attempts + 1):
        attempt_context = replace(
            context,
            repair_attempts=0,
            repair_feedback=None if last_error is None else str(last_error),
            attempt_index=attempt,
        )
        try:
            document, record = produce(attempt_context)
        except (AuthoringModelError, GroundingError) as exc:
            last_error = exc
            if context.checkpoint_root is not None:
                write_text_atomic(
                    _stage_error_text(stage, attempt, exc),
                    context.checkpoint_root / f"{stage}.attempt-{attempt}.rejected.txt",
                )
            if attempt < context.repair_attempts:
                continue
            if context.checkpoint_root is not None:
                path = context.checkpoint_root / f"{stage}.attempt-{attempt}.candidate.yaml"
                accepted_path = context.checkpoint_root / f"{document_name}.yaml"
                exc.add_note(
                    f"Edit {path} and save the reviewed result as {accepted_path}, then rerun to resume."
                )
            raise
        if context.checkpoint_root is not None:
            dumped = document.model_dump(mode="json")
            _write_yaml(
                dumped,
                context.checkpoint_root / f"{document_name}.yaml",
            )
            write_canonical_json(
                {
                    "schema_version": "bfcl-draft-checkpoint-v1",
                    "evidence_digest": context.evidence.digest,
                    "document_digest": sha256_json(dumped),
                    "call": record.as_dict(),
                },
                context.checkpoint_root / f"{document_name}.checkpoint.json",
            )
        return document, record
    raise AssertionError("unreachable drafting stage loop")


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
