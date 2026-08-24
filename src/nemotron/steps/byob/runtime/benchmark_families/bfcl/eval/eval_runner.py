"""Bounded, fail-closed orchestration for BFCL evaluation, in either mode.

Both runners here verify the source, gate contamination, drive one candidate at a
time over its authorized tasks, aggregate, and publish one immutable artifact set.
They differ only in what a turn costs: executable driving interleaves candidate
turns with live oracle calls, while trace driving releases the tool results the
benchmark already recorded.

The two are separate entry points rather than one mode switch because an artifact
set is one measurement. Both write the same file names into the output directory,
and the metrics inside them are not the same numbers, so a run is one or the
other and a config that asks for both is served by the executable runner, which
scores every trace gate as well.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypeVar

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
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_driver import (
    run_candidate_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_projection import (
    build_conversation_script,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_artifacts import (
    EVAL_REPORT_FILE,
    EvalArtifactSet,
    write_executable_eval_artifacts,
    write_trace_eval_artifacts,
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
    assert_source_unchanged,
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
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_aggregation import (
    TraceCandidateScore,
    aggregate_trace_scores,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scorer import (
    score_trace_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    TraceTaskScore,
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


@dataclass(frozen=True)
class BfclTraceEvalRunResult:
    eval_run_id: str
    config: BfclEvalConfig
    source: VerifiedEvalSource
    plan: EligibleEvalPlan
    task_scores: tuple[TraceTaskScore, ...]
    candidate_scores: tuple[TraceCandidateScore, ...]
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"bfcl-eval-{timestamp}-{uuid.uuid4().hex}"


_ScoreT = TypeVar("_ScoreT")


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)


async def _cancel_on_failure(
    calls: Sequence[Coroutine[Any, Any, _ScoreT]],
) -> tuple[_ScoreT, ...]:
    tasks = tuple(asyncio.create_task(call) for call in calls)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class AuthorizedEvalRun:
    """Everything both modes must establish before a candidate is contacted."""

    eval_config_hash: str
    eval_run_id: str
    resolved_config_path: Path
    source: VerifiedEvalSource
    plan: EligibleEvalPlan
    projection: CanonicalExportProjection


def authorize_bfcl_eval(
    config: BfclEvalConfig,
    *,
    eval_run_id: str | None,
    probe_oracle: bool,
) -> AuthorizedEvalRun:
    """Pin the run identity, verify the source, and gate contamination.

    Every failed gate leaves its own diagnostic behind before the exception
    propagates: a run that stopped here has to be explainable from the output
    directory alone, without the operator's terminal scrollback.
    """
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
    return AuthorizedEvalRun(
        eval_config_hash=config.eval_config_hash,
        eval_run_id=run_id,
        resolved_config_path=resolved_config_path,
        source=source,
        plan=plan,
        projection=projection,
    )


def _authorization(
    config: BfclEvalConfig,
    *,
    eval_run_id: str | None,
    probe_oracle: bool,
    authorized: AuthorizedEvalRun | None,
) -> AuthorizedEvalRun:
    if authorized is None:
        return authorize_bfcl_eval(
            config,
            eval_run_id=eval_run_id,
            probe_oracle=probe_oracle,
        )
    if authorized.eval_config_hash != config.eval_config_hash:
        raise EvalRunnerError(
            "precomputed authorization belongs to another eval config",
            recovery="authorize the exact config immediately before execution",
        )
    if eval_run_id is not None and authorized.eval_run_id != eval_run_id:
        raise EvalRunnerError(
            "precomputed authorization belongs to another eval run id",
            recovery="reuse the authorization's run id or authorize a new run",
        )
    assert_source_unchanged(authorized.source)
    assert_plan_unchanged(config, authorized.source, authorized.plan)
    return authorized


def _candidate_cache(
    config: BfclEvalConfig,
) -> tuple[CandidateIOCache, Path, tempfile.TemporaryDirectory[str] | None]:
    """Open the candidate cache, in the output tree only when it is published.

    A run that does not publish the cache still needs one: the client de-duplicates
    identical turns through it, and a temporary directory keeps that behaviour
    without leaving unpublished evidence beside the artifacts.
    """
    if config.outputs.cache_candidate_responses:
        path = config.outputs.output_dir / CANDIDATE_IO_CACHE_FILE
        return CandidateIOCache(path), path, None
    temporary = tempfile.TemporaryDirectory(prefix="bfcl-candidate-cache-")
    path = Path(temporary.name) / CANDIDATE_IO_CACHE_FILE
    return CandidateIOCache(path), path, temporary


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
                        _add_exception_note(
                            primary,
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
    authorized: AuthorizedEvalRun | None = None,
) -> BfclEvalRunResult:
    """Verify, authorize, execute, aggregate, and publish one bounded eval run."""
    if not config.settings.executable:
        raise UnsupportedRunnerModeError(
            "the executable batch runner received a trace-only config",
            recovery="set eval.mode to [trace, executable] or use run_bfcl_trace_eval",
        )
    authorized = _authorization(
        config,
        eval_run_id=eval_run_id,
        probe_oracle=probe_oracle,
        authorized=authorized,
    )
    run_id = authorized.eval_run_id
    source = authorized.source
    plan = authorized.plan
    projection = authorized.projection
    output_dir = config.outputs.output_dir

    candidate_cache, candidate_cache_path, temporary_cache = _candidate_cache(config)
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
                    _add_exception_note(
                        primary,
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
            resolved_config_path=authorized.resolved_config_path,
        )
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()


async def _run_candidate_trace_tasks(
    *,
    config: BfclEvalConfig,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
    projection: CanonicalExportProjection,
    candidate: EvalCandidate,
    client: CandidateClient,
) -> tuple[TraceTaskScore, ...]:
    semaphore = asyncio.Semaphore(config.limits.max_parallel_tasks)
    failure = asyncio.Event()
    gate = CanonicalCallMatchGate(config.scoring)

    async def run_one(task_id: str) -> TraceTaskScore:
        async with semaphore:
            try:
                if failure.is_set():
                    raise asyncio.CancelledError
                script = build_conversation_script(projection, task_id, source=source)
                episode = await run_candidate_episode(
                    candidate=candidate,
                    limits=config.limits,
                    client=client,  # type: ignore[arg-type]
                    script=script,
                    plan=plan,
                    gate=gate,
                )
                return score_trace_episode(
                    episode=episode,
                    script=script,
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


async def run_bfcl_trace_eval(
    config: BfclEvalConfig,
    *,
    eval_run_id: str | None = None,
    probe_oracle: bool = True,
    client_factory: ClientFactory = _default_client_factory,
    authorized: AuthorizedEvalRun | None = None,
) -> BfclTraceEvalRunResult:
    """Verify, authorize, drive, score, aggregate, and publish one trace-only run.

    No oracle is contacted and no tool is executed, so the run needs no oracle
    session and persists no tool-trace cache: the results a turn releases are the
    ones the benchmark recorded, and source verification already proved those by
    content hash.
    """
    if config.settings.executable:
        raise UnsupportedRunnerModeError(
            "the trace batch runner received an executable config",
            recovery=(
                "run run_bfcl_eval, which scores every trace gate as well, "
                "or set eval.mode to [trace]"
            ),
        )
    authorized = _authorization(
        config,
        eval_run_id=eval_run_id,
        probe_oracle=probe_oracle,
        authorized=authorized,
    )
    plan = authorized.plan
    candidate_cache, candidate_cache_path, temporary_cache = _candidate_cache(config)

    try:
        task_scores: list[TraceTaskScore] = []
        candidate_scores: list[TraceCandidateScore] = []
        for alias in plan.candidate_aliases:
            candidate = config.candidate(alias)
            client = client_factory(candidate, config.limits, candidate_cache)
            try:
                scores = await _run_candidate_trace_tasks(
                    config=config,
                    source=authorized.source,
                    plan=plan,
                    projection=authorized.projection,
                    candidate=candidate,
                    client=client,
                )
            except BaseException as primary:
                try:
                    await client.aclose()
                except Exception as cleanup:
                    _add_exception_note(
                        primary,
                        f"candidate client cleanup also failed as {type(cleanup).__name__}"
                    )
                raise
            else:
                await client.aclose()
            task_scores.extend(scores)
            candidate_scores.append(
                aggregate_trace_scores(
                    scores=scores,
                    plan=plan,
                    candidate_alias=alias,
                )
            )

        artifacts = write_trace_eval_artifacts(
            eval_run_id=authorized.eval_run_id,
            config=config,
            plan=plan,
            candidate_scores=tuple(candidate_scores),
            task_scores=tuple(task_scores),
            candidate_io_cache_path=(
                candidate_cache_path
                if config.outputs.cache_candidate_responses
                else None
            ),
        )
        return BfclTraceEvalRunResult(
            eval_run_id=authorized.eval_run_id,
            config=config,
            source=authorized.source,
            plan=plan,
            task_scores=tuple(task_scores),
            candidate_scores=tuple(candidate_scores),
            artifacts=artifacts,
            resolved_config_path=authorized.resolved_config_path,
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


def run_bfcl_trace_eval_sync(
    config_path: str | Path,
    *,
    eval_run_id: str | None = None,
    probe_oracle: bool = True,
) -> BfclTraceEvalRunResult:
    """Load a resolved trace-only config and drive it from a synchronous entrypoint."""
    config = load_eval_config(config_path)
    return asyncio.run(
        run_bfcl_trace_eval(
            config,
            eval_run_id=eval_run_id,
            probe_oracle=probe_oracle,
        )
    )


def run_declared_eval_sync(
    config_path: str | Path,
    *,
    eval_run_id: str | None = None,
    probe_oracle: bool = True,
) -> BfclEvalRunResult | BfclTraceEvalRunResult:
    """Run whichever evaluation ``eval.mode`` declares, and publish its artifacts.

    The mode is a property of the pinned config, so an operator should not have to
    restate it by picking a function. The mode-specific entry points remain, for a
    caller that wants the config refused rather than served when it declares the
    other mode.
    """
    config = load_eval_config(config_path)
    if config.settings.executable:
        return asyncio.run(
            run_bfcl_eval(
                config,
                eval_run_id=eval_run_id,
                probe_oracle=probe_oracle,
            )
        )
    return asyncio.run(
        run_bfcl_trace_eval(
            config,
            eval_run_id=eval_run_id,
            probe_oracle=probe_oracle,
        )
    )


__all__ = [
    "AuthorizedEvalRun",
    "BfclEvalRunResult",
    "BfclTraceEvalRunResult",
    "EvalRunnerError",
    "UnsupportedRunnerModeError",
    "authorize_bfcl_eval",
    "run_bfcl_eval",
    "run_bfcl_eval_sync",
    "run_bfcl_trace_eval",
    "run_bfcl_trace_eval_sync",
    "run_declared_eval_sync",
]
