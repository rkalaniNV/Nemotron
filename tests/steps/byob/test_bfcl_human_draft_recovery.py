from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import ImmutableModelIOCache
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.pack_authoring.drafts import DraftReview
from nemotron.steps.byob.runtime.pack_authoring.grounding import GroundingError, validate_validation_cases
from nemotron.steps.byob.runtime.pack_authoring.model_client import call_structured
from nemotron.steps.byob.runtime.pack_authoring.provenance import ProvenanceError
from nemotron.steps.byob.runtime.pack_authoring.runner import run_drafting
from nemotron.steps.byob.runtime.pack_authoring.schemas import ValidationCasePlan
from tests.steps.byob.test_bfcl_pack_drafting import (
    MODEL,
    _bundle_document,
    _error_case,
    _FakeCaller,
    _grounding,
    _write,
)


@pytest.mark.parametrize("metadata", [None, "{", "[]"])
def test_missing_or_corrupt_metadata_never_adopts_other_evidence(tmp_path: Path, metadata: str | None) -> None:
    bundle, approval = _write(tmp_path / "first", _bundle_document())
    output = tmp_path / "out"
    run_drafting(bundle, approval, output, MODEL, caller=_FakeCaller(), allow_legacy_v1_model_exposure=True)
    sidecar = output / "drafts" / "coverage_plan.checkpoint.json"
    if metadata is None:
        sidecar.unlink()
    else:
        sidecar.write_text(metadata)
    changed = _bundle_document()
    changed["oracle"]["oracle_version"] = "2.0.0"
    changed.pop("bundle_digest")
    changed["bundle_digest"] = sha256_json(changed)
    bundle, approval = _write(tmp_path / "second", changed)
    caller = _FakeCaller()
    with pytest.raises(GroundingError, match="metadata is missing, invalid, or stale"):
        run_drafting(bundle, approval, output, MODEL, caller=caller, allow_legacy_v1_model_exposure=True)
    assert not caller.stages


def test_schema_failure_retains_candidate_and_field_diagnostics(tmp_path: Path) -> None:
    bundle, approval = _write(tmp_path / "in", _bundle_document())
    caller = _FakeCaller()
    caller.responses["mcp_coverage_plan"] = {"invalid_field": 123}
    output = tmp_path / "out"
    with pytest.raises(GroundingError, match="human must correct"):
        run_drafting(bundle, approval, output, MODEL, caller=caller, allow_legacy_v1_model_exposure=True)
    assert caller.stages == ["mcp_coverage_plan"]
    assert yaml.safe_load((output / "drafts/mcp_coverage_plan.candidate.yaml").read_text()) == {"invalid_field": 123}
    error = (output / "drafts/mcp_coverage_plan.rejected.txt").read_text()
    assert "invalid_field" in error
    assert "tools: Field required" in error
    # Human correction must retain the assisting model, including when its first
    # proposal never became an accepted model checkpoint.
    corrected = output / "drafts/coverage_plan.yaml"
    corrected.write_text(yaml.safe_dump(_FakeCaller().responses["mcp_coverage_plan"]))
    review = DraftReview(("coverage_plan",), "reviewer", "2026-09-09T12:00:00Z")
    with pytest.raises(ProvenanceError, match="drafting model changed"):
        run_drafting(
            bundle, approval, output, replace(MODEL, canonical_id="other/model@1"), caller=caller,
            allow_legacy_v1_model_exposure=True, human_review=review,
        )
    assert caller.stages == ["mcp_coverage_plan"]
    result = run_drafting(
        bundle, approval, output, MODEL, caller=_FakeCaller(),
        allow_legacy_v1_model_exposure=True, human_review=review,
    )
    assert result.provenance.document["model"] == MODEL.as_provenance()
    assert result.drafts.calls[0].source == "human_checkpoint"


def test_model_and_cache_preserve_omission_and_explicit_null(tmp_path: Path) -> None:
    for supplied in (False, True):
        argument = {"name": "unit", "source": "invalid_literal"}
        if supplied:
            argument["literal"] = None
        raw = _error_case(argument).model_dump(mode="json", exclude_unset=True)
        cache = ImmutableModelIOCache(tmp_path / f"cache-{supplied}.jsonl")
        for iteration in range(2):
            result, call = call_structured(
                MODEL, stage_name="stage", prompt_version="1", system_prompt="system", prompt="prompt",
                columns={}, output_format=ValidationCasePlan, cache=cache, run_dir=tmp_path,
                caller=lambda *args, **kwargs: {"stage": raw},
            )
            assert call.served_from_cache is (iteration == 1)
            plan = ValidationCasePlan.model_validate(result)
            if supplied:
                validate_validation_cases(_grounding(), plan)
            else:
                with pytest.raises(GroundingError, match="requires the value"):
                    validate_validation_cases(_grounding(), plan)


def test_editing_coverage_requires_review_of_dependent_drafts(tmp_path: Path) -> None:
    bundle, approval = _write(tmp_path / "in", _bundle_document())
    output = tmp_path / "out"
    run_drafting(bundle, approval, output, MODEL, caller=_FakeCaller(), allow_legacy_v1_model_exposure=True)
    coverage = output / "drafts/coverage_plan.yaml"
    document = yaml.safe_load(coverage.read_text())
    document["cross_tool_notes"] = ["Human-reviewed change in coverage intent."]
    coverage.write_text(yaml.safe_dump(document))
    caller = _FakeCaller()
    with pytest.raises(GroundingError, match="metadata is missing, invalid, or stale"):
        run_drafting(
            bundle, approval, output, MODEL, caller=caller, allow_legacy_v1_model_exposure=True,
            human_review=DraftReview(("coverage_plan",), "reviewer", "2026-09-09T12:00:00Z"),
        )
    assert not caller.stages
    metadata = json.loads((output / "drafts/coverage_plan.checkpoint.json").read_text())
    assert metadata["status"] == "human_reviewed"
