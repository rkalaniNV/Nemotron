"""Bounded, fail-closed orchestration for executable BFCL evaluation."""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_matching import (
    CanonicalCallMatchGate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CANDIDATE_IO_CACHE_FILE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.config import (
    load_eval_config,
    write_resolved_eval_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination import (
    assert_plan_unchanged,
    evaluate_contamination,
    write_contamination_failure,
    write_contamination_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_artifacts import (
    EVAL_REPORT_FILE,
    EvalArtifactSet,
    write_executable_eval_artifacts,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_aggregation import (
    ExecutableCandidateScore,
    aggregate_executable_scores,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver import (
    run_executable_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
    build_executable_task_spec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scorer import (
    score_executable_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_contract import (
    ExecutableTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.oracle_session import (
    OracleSession,
    open_oracle_session,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    BfclEvalConfig,
    EvalCandidate,
    EvalLimits,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedEvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    verify_eval_source,
    write_source_failure_diagnostic,
    write_source_verification_report,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_cache import (
    ToolTraceCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_contract import (
    TOOL_TRACE_CACHE_FILE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    project_published_benchmark,
)


class EvalRunnerError(Exception):
    """A complete executable run cannot be started under this contract."""

    code = "eval_runner_invalid"

    def __init__(self, problem: str, *, recovery: str) -> None:
        self.problem = problem
        self.recovery = recovery
        super().__init__(f"{problem}. Fix: {recovery}")

    def as_report(self) -> dict[str, str]:
        return {
            "code": self.code,
            "subject": "eval.runner",
            "problem": self.problem,
            "expected": "an executable-mode config and complete authorized task set",
            "recovery": self.recovery,
        }


class UnsupportedRunnerModeError(EvalRunnerError):
    """This runner was asked to execute a trace-only evaluation."""

    code = "eval_runner_mode_unsupported"


class CandidateClient(Protocol):
    async def complete(self, request: Any, *, deadline: float | None = None) -> Any: ...

    async def aclose(self) -> None: ...


ClientFactory = Callable[[EvalCandidate, EvalLimits, CandidateIOCache], CandidateClient]
OracleFactory = Callable[
    [VerifiedEvalSource, ExecutableTaskSpec, EvalLimits],
    OracleSession,
]


@dataclass(frozen=True)
class BfclEvalRunResult:
    eval_run_id: str
    config: BfclEvalConfig
    source: VerifiedEvalSource
    plan: EligibleEvalPlan
    task_scores: tuple[ExecutableTaskScore, ...]
    candidate_scores: tuple[ExecutableCandidateScore, ...]
    artifacts: EvalArtifactSet
    resolved_config_path: Path


def _default_client_factory(
    candidate: EvalCandidate,
    limits: EvalLimits,
    cache: CandidateIOCache,
) -> NativeFunctionCallingClient:
    return NativeFunctionCallingClient(candidate, limits, cache)


def _default_oracle_factory(
    source: VerifiedEvalSource,
    task: ExecutableTaskSpec,
    limits: EvalLimits,
) -> OracleSession:
    return open_oracle_session(source=source, task=task, limits=limits)


def _resolve_eval_run_id(
    config: BfclEvalConfig,
    requested: str | None,
) -> str:
    if requested is not None and (not requested.strip() or requested != requested.strip()):
        raise EvalRunnerError(
            "eval_run_id is empty or carries surrounding whitespace",
            recovery="provide a stable non-empty run identifier",
        )
    report_path = config.outputs.output_dir / EVAL_REPORT_FILE
    if report_path.exists():
        try:
            document = json.loads(report_path.read_text(encoding="utf-8"))
            existing = document["eval_run_id"]
            config_hash = document["eval_config_hash"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvalRunnerError(
                "the existing eval report cannot supply a reusable run identity",
                recovery="preserve the output as evidence and use a new output directory",
            ) from exc
        if (
            not isinstance(existing, str)
            or not existing.strip()
            or config_hash != config.eval_config_hash
        ):
            raise EvalRunnerError(
                "the existing eval report belongs to another or invalid run",
                recovery="use the config and output directory that produced it, or start a new run",
            )
        if requested is not None and requested != existing:
            raise EvalRunnerError(
                "eval_run_id conflicts with the immutable report in this output directory",
                recovery=f"reuse {existing!r} or select a new output directory",
            )
        return existing
    if requested is not None:
        return requested
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"bfcl-eval-{timestamp}-{uuid.uuid4().hex}"


async def _cancel_on_failure(
    calls: tuple[Coroutine[Any, Any, ExecutableTaskScore], ...],
) -> tuple[ExecutableTaskScore, ...]:
    tasks = tuple(asyncio.create_task(call) for call in calls)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_candidate_tasks(
    *,
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
    projection: CanonicalExportProjection,
    candidate: EvalCandidate,
    client: CandidateClient,
    tool_trace_cache: ToolTraceCache | None,
    oracle_factory: OracleFactory,
) -> tuple[ExecutableTaskScore, ...]:
    semaphore = asyncio.Semaphore(config.limits.max_parallel_tasks)
    failure = asyncio.Event()
    gate = CanonicalCallMatchGate(config.scoring)

    async def run_one(task_id: str) -> ExecutableTaskScore:
        async with semaphore:
            try:
                if failure.is_set():
                    raise asyncio.CancelledError
                task = build_executable_task_spec(
                    projection,
                    task_id,
                    candidate_alias=candidate.alias,
                    source=source,
                    plan=plan,
                )
                oracle = oracle_factory(source, task, config.limits)
                try:
                    episode = await run_executable_episode(
                        candidate=candidate,
                        limits=config.limits,
                        client=client,  # type: ignore[arg-type]
                        task=task,
                        source=source,
                        plan=plan,
                        oracle=oracle,
                        gate=gate,
                        tool_trace_cache=tool_trace_cache,
                    )
                except BaseException as primary:
                    try:
                        await oracle.close()
                    except Exception as cleanup:
                        primary.add_note(
                            f"oracle cleanup also failed as {type(cleanup).__name__}"
                        )
                    raise
                else:
                    await oracle.close()
                return score_executable_episode(
                    episode=episode,
                    task=task,
                    scoring=config.scoring,
                    plan=plan,
                )
            except BaseException:
                failure.set()
                raise

    return await _cancel_on_failure(
        tuple(
            run_one(task_id)
            for task_id in plan.evaluation_task_ids(candidate.alias)
        )
    )


async def run_bfcl_eval(
    config: BfclEvalConfig,
    *,
    eval_run_id: str | None = None,
    probe_oracle: bool = True,
    client_factory: ClientFactory = _default_client_factory,
    oracle_factory: OracleFactory = _default_oracle_factory,
) -> BfclEvalRunResult:
    """Verify, authorize, execute, aggregate, and publish one bounded eval run."""
    if not config.settings.executable:
        raise UnsupportedRunnerModeError(
            "the executable batch runner received a trace-only config",
            recovery="set eval.mode to [trace, executable] or use the trace runner",
        )
    output_dir = config.outputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _resolve_eval_run_id(config, eval_run_id)
    resolved_config_path = output_dir / "resolved_eval_config.json"
    write_resolved_eval_config(config, resolved_config_path)

    try:
        source = verify_eval_source(config, probe_oracle=probe_oracle)
    except Exception as exc:
        write_source_failure_diagnostic(config, exc)
        raise
    write_source_verification_report(config, source)

    try:
        plan = evaluate_contamination(config, source)
    except Exception as exc:
        write_contamination_failure(config, exc)
        raise
    write_contamination_report(config, plan)

    projection = project_published_benchmark(
        source.evaluation_benchmark.path,
        expected_content_hash=source.evaluation_benchmark.content_hash,
        expected_task_ids=source.task_ids,
    )
    assert_plan_unchanged(config, source, plan)

    temporary_cache: tempfile.TemporaryDirectory[str] | None = None
    if config.outputs.cache_candidate_responses:
        candidate_cache_path = output_dir / CANDIDATE_IO_CACHE_FILE
    else:
        temporary_cache = tempfile.TemporaryDirectory(prefix="bfcl-candidate-cache-")
        candidate_cache_path = Path(temporary_cache.name) / CANDIDATE_IO_CACHE_FILE
    candidate_cache = CandidateIOCache(candidate_cache_path)
    tool_cache_path = output_dir / TOOL_TRACE_CACHE_FILE
    tool_trace_cache = (
        ToolTraceCache(tool_cache_path)
        if config.outputs.cache_tool_results
        else None
    )

    try:
        task_scores: list[ExecutableTaskScore] = []
        candidate_scores: list[ExecutableCandidateScore] = []
        for alias in plan.candidate_aliases:
            candidate = config.candidate(alias)
            client = client_factory(candidate, config.limits, candidate_cache)
            try:
                scores = await _run_candidate_tasks(
                    config=config,
                    source=source,
                    plan=plan,
                    projection=projection,
                    candidate=candidate,
                    client=client,
                    tool_trace_cache=tool_trace_cache,
                    oracle_factory=oracle_factory,
                )
            except BaseException as primary:
                try:
                    await client.aclose()
                except Exception as cleanup:
                    primary.add_note(
                        f"candidate client cleanup also failed as {type(cleanup).__name__}"
                    )
                raise
            else:
                await client.aclose()
            task_scores.extend(scores)
            candidate_scores.append(
                aggregate_executable_scores(
                    scores=scores,
                    plan=plan,
                    candidate_alias=alias,
                )
            )

        artifacts = write_executable_eval_artifacts(
            eval_run_id=run_id,
            config=config,
            plan=plan,
            candidate_scores=tuple(candidate_scores),
            task_scores=tuple(task_scores),
            candidate_io_cache_path=(
                candidate_cache_path
                if config.outputs.cache_candidate_responses
                else None
            ),
            tool_trace_cache_path=(
                tool_cache_path if config.outputs.cache_tool_results else None
            ),
        )
        return BfclEvalRunResult(
            eval_run_id=run_id,
            config=config,
            source=source,
            plan=plan,
            task_scores=tuple(task_scores),
            candidate_scores=tuple(candidate_scores),
            artifacts=artifacts,
            resolved_config_path=resolved_config_path,
        )
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()


def run_bfcl_eval_sync(
    config_path: str | Path,
    *,
    eval_run_id: str | None = None,
    probe_oracle: bool = True,
) -> BfclEvalRunResult:
    """Load a resolved config and execute it from a synchronous entrypoint."""
    config = load_eval_config(config_path)
    return asyncio.run(
        run_bfcl_eval(
            config,
            eval_run_id=eval_run_id,
            probe_oracle=probe_oracle,
        )
    )


__all__ = [
    "BfclEvalRunResult",
    "EvalRunnerError",
    "UnsupportedRunnerModeError",
    "run_bfcl_eval",
    "run_bfcl_eval_sync",
]
