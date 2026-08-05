"""Trust + timeout helpers for oracle-pack execution.

Pipeline owns reset/timeouts; packs only expose entrypoints.
Gold claims require process workers. Each episode (reset + tools + state)
must run inside a single worker process so fixture state is coherent.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import multiprocessing as mp
import os
import queue as queue_module
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
_FIRST_STEP = object()


def _advance_episode(
    iterator: Any,
    output: Any = _FIRST_STEP,
) -> dict[str, Any]:
    """Advance a static step iterator or feed output back to an interactive generator."""
    if output is _FIRST_STEP:
        return next(iterator)
    sender = getattr(iterator, "send", None)
    return sender(output) if sender is not None else next(iterator)


@contextmanager
def _restored_import_state() -> Iterator[None]:
    """Keep debug-thread pack imports from leaking into later host imports."""
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    original_bytecode_flag = sys.dont_write_bytecode
    try:
        yield
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode_flag
        for name in set(sys.modules) - original_modules:
            sys.modules.pop(name, None)


class PackTrustError(PermissionError):
    """Raised when an oracle pack path is outside the configured allowlist."""


def assert_pack_allowed(path: Path | str, allowed_roots: Sequence[Path | str]) -> Path:
    """Return the resolved pack path if it sits under an allowlisted root."""
    resolved = Path(path).resolve()
    roots = [Path(root).resolve() for root in allowed_roots]
    if not roots:
        raise PackTrustError("oracle_runtime.allowed_roots is empty; refusing pack load")

    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    roots_display = ", ".join(str(root) for root in roots)
    raise PackTrustError(f"Oracle pack path {resolved} is outside allowlisted roots: {roots_display}")


def _load_module(path: Path, module_name: str, import_root: Path | str | None = None) -> ModuleType:
    """Import a pack module from its file, resolving the pack's own helper modules.

    A pack may split its backend or assertions across several files, and the fingerprint
    already covers the whole pack tree. Importing by file location alone would leave
    ``import helper`` resolving against whatever ``sys.path`` the calling process happened
    to have, so the same pack would load in one invocation and fail in another. The
    module's directory and the pack root are what resolve those imports instead.
    """
    # Importing pack code must not write into the pack: a __pycache__ directory
    # appearing next to backend.py changes the tree the fingerprint covers.
    sys.dont_write_bytecode = True
    # Inserted last-to-first so the module's own directory outranks the pack root.
    roots = [] if import_root is None else [Path(import_root)]
    roots.append(path.parent)
    for root in roots:
        entry = str(root.resolve())
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sanitize_pack_environment() -> None:
    """Remove inherited host secrets before executing allowlisted pack code."""
    keep = {
        key: value
        for key, value in os.environ.items()
        if key in {"LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP"}
    }
    os.environ.clear()
    os.environ.update(keep)


def _timeout_target(
    queue: mp.Queue,
    fn: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        queue.put(("ok", fn(*args, **kwargs)))
    except BaseException:  # noqa: BLE001 — re-raise in parent with remote traceback
        queue.put(("err", traceback.format_exc()))


def run_with_timeout(
    fn: Callable[..., T],
    timeout_s: float,
    /,
    *args: Any,
    worker: str = "process",
    **kwargs: Any,
) -> T:
    """Run ``fn`` under a deadline.

    ``worker="process"`` is required for gold claims. ``worker="thread"`` is
    debug-only: Python cannot terminate a running thread, so a timed-out callable
    may continue in the background and thread mode cannot claim a hard timeout.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")

    if worker == "thread":
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(
                f"debug thread exceeded timeout_s={timeout_s}; the callable may still be running"
            ) from exc
        finally:
            pool.shutdown(wait=future.done(), cancel_futures=True)

    if worker != "process":
        raise ValueError(f"unsupported worker {worker!r}; expected 'process' or 'thread'")

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_timeout_target, args=(queue, fn, args, kwargs))
    proc.start()

    # Read before joining. A result larger than the pipe buffer leaves the child
    # blocked in its feeder thread until the parent drains the queue, so joining
    # first would deadlock until the deadline and report a timeout that never was.
    payload: Any = None
    received = False
    exited = False
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                payload = queue.get(timeout=min(0.05, timeout_s))
                received = True
                break
            except queue_module.Empty:
                if not proc.is_alive():
                    # The child is gone; give its feeder thread a last chance to
                    # flush before calling this a crash rather than a timeout.
                    try:
                        payload = queue.get(timeout=0.5)
                        received = True
                    except queue_module.Empty:
                        exited = True
                    break
                if time.monotonic() >= deadline:
                    break
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                proc.kill()
        proc.join(1.0)
        queue.close()

    if exited:
        raise RuntimeError("worker exited without returning a result")
    if not received:
        raise TimeoutError(f"call exceeded timeout_s={timeout_s}")
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError("worker exited without returning a result")
    if payload[0] == "ok":
        return payload[1]
    raise RuntimeError(f"worker failed:\n{payload[1]}")


def _run_backend_episode_sync_inner(
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    steps: Iterable[dict[str, Any]],
    *,
    sanitize_environment: bool,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
    oracle_holder: list[Any] | None = None,
) -> list[Any]:
    """Run an episode synchronously (debug-thread path only).

    ``sanitize_environment`` is only safe when this runs in a worker process of its
    own. Thread mode shares the parent's environment: clearing it here would strip
    the host session, and a timed-out thread cannot be stopped to restore it.
    """
    from datetime import datetime

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.run_context import RunContext

    if sanitize_environment:
        _sanitize_pack_environment()
    if endpoint_config is not None:
        from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
            EndpointOracleClient,
        )

        module = EndpointOracleClient(
            endpoint_config,
            headers=endpoint_headers or {},
            timeout_s=tool_timeout_s,
        )
    else:
        if backend_path is None:
            raise ValueError("backend_path is required for a local oracle")
        module = _load_module(
            Path(backend_path),
            f"_bfcl_episode_{Path(backend_path).stem}_{task_id}",
            import_root,
        )
    if oracle_holder is not None:
        oracle_holder.append(module)
    assertions_module = None
    if assertions_path is not None:
        assertions_module = _load_module(
            Path(assertions_path),
            f"_bfcl_episode_assertions_{Path(assertions_path).stem}_{task_id}",
            import_root,
        )
    ctx = RunContext(
        clock=datetime.fromisoformat(clock_iso),
        seed=seed,
        timeout_s=tool_timeout_s,
        task_id=task_id,
        turn_index=0,
    )
    outputs: list[Any] = []
    trace: list[dict[str, Any]] = []
    iterator = iter(steps)
    try:
        step = _advance_episode(iterator)
    except StopIteration:
        return outputs
    while True:
        op = step["op"]
        if op == "reset":
            module.reset(ctx=ctx, fixtures=fixtures)
            trace.clear()
            outputs.append(None)
        elif op == "get_state":
            outputs.append(module.get_state())
        elif op == "call_tool":
            arguments = step.get("arguments") or {}
            ctx = replace(ctx, turn_index=int(step.get("turn_index", 0)))
            result = module.call_tool(step["name"], arguments, ctx=ctx)
            trace.append(
                {
                    "tool": step["name"],
                    "arguments": arguments,
                    "result": result,
                    "turn_index": ctx.turn_index,
                }
            )
            outputs.append(result)
        elif op == "list_tools":
            outputs.append(module.list_tools())
        elif op == "metadata":
            metadata = getattr(module, "metadata", None)
            outputs.append(metadata() if callable(metadata) else None)
        elif op == "inspect_backend":
            outputs.append(
                {
                    name: callable(getattr(module, name, None))
                    for name in ("list_tools", "reset", "call_tool", "get_state")
                }
            )
        elif op == "run_assertion":
            if assertions_module is None:
                raise RuntimeError("run_assertion requires assertions_path on the worker")
            try:
                _resolve_assertion(assertions_module, step["name"])(
                    state=module.get_state(),
                    trace=step.get("trace") if step.get("trace") is not None else trace,
                    task=step.get("task") or {},
                    ctx=ctx,
                )
                outputs.append({"name": step["name"], "passed": True, "detail": None})
            except AssertionError as exc:
                outputs.append({"name": step["name"], "passed": False, "detail": str(exc)})
            except Exception as exc:  # noqa: BLE001
                # Type and message only: an object repr can carry an address and
                # would make two identical replays look nondeterministic.
                outputs.append(
                    {
                        "name": step["name"],
                        "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
        else:
            raise ValueError(f"unknown episode op {op!r}")
        try:
            step = _advance_episode(iterator, outputs[-1])
        except StopIteration:
            break
    return outputs


def _run_backend_episode_sync(
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    steps: Iterable[dict[str, Any]],
    *,
    sanitize_environment: bool,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
) -> list[Any]:
    """Run the debug-thread episode and restore process-global import state."""
    oracle_holder: list[Any] = []
    with _restored_import_state():
        try:
            return _run_backend_episode_sync_inner(
                backend_path,
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout_s,
                steps,
                sanitize_environment=sanitize_environment,
                assertions_path=assertions_path,
                import_root=import_root,
                endpoint_config=endpoint_config,
                endpoint_headers=endpoint_headers,
                oracle_holder=oracle_holder,
            )
        finally:
            close = getattr(oracle_holder[0], "close", None) if oracle_holder else None
            if callable(close):
                close()


def _resolve_assertion(module: ModuleType, name: str) -> Callable[..., Any]:
    exported = getattr(module, "ASSERTIONS", None)
    if isinstance(exported, dict) and name in exported:
        candidate = exported[name]
    else:
        candidate = getattr(module, name, None)
    if not callable(candidate):
        raise LookupError(f"assertion {name!r} is not defined")
    return candidate


def _backend_worker(
    connection: Any,
    backend_path: str | None,
    fixtures: dict[str, Any] | None,
    clock_iso: str,
    seed: int,
    task_id: str,
    tool_timeout_s: float,
    assertions_path: str | None = None,
    import_root: str | None = None,
    endpoint_config: Any = None,
    endpoint_headers: dict[str, str] | None = None,
) -> None:
    """Persistent backend worker; parent enforces import and per-operation deadlines."""
    from datetime import datetime

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.run_context import RunContext

    try:
        _sanitize_pack_environment()
        if endpoint_config is not None:
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                EndpointOracleClient,
            )

            module = EndpointOracleClient(
                endpoint_config,
                headers=endpoint_headers or {},
                timeout_s=tool_timeout_s,
            )
        else:
            if backend_path is None:
                raise ValueError("backend_path is required for a local oracle")
            module = _load_module(
                Path(backend_path),
                f"_bfcl_backend_{Path(backend_path).stem}_{abs(hash((backend_path, task_id)))}",
                import_root,
            )
        assertions_module = None
        if assertions_path is not None:
            assertions_module = _load_module(
                Path(assertions_path),
                f"_bfcl_worker_assertions_{abs(hash((assertions_path, task_id)))}",
                import_root,
            )
        ctx = RunContext(
            clock=datetime.fromisoformat(clock_iso),
            seed=seed,
            timeout_s=tool_timeout_s,
            task_id=task_id,
            turn_index=0,
        )
        connection.send(("ready", None, None))
        trace: list[dict[str, Any]] = []
        while True:
            step = connection.recv()
            op = step["op"]
            if op == "close":
                close = getattr(module, "close", None)
                if callable(close):
                    close()
                connection.send(("ok", None, None))
                return
            if op == "reset":
                value = module.reset(ctx=ctx, fixtures=fixtures)
                trace.clear()
            elif op == "get_state":
                value = module.get_state()
            elif op == "call_tool":
                arguments = step.get("arguments") or {}
                ctx = replace(ctx, turn_index=int(step.get("turn_index", 0)))
                value = module.call_tool(step["name"], arguments, ctx=ctx)
                trace.append(
                    {
                        "tool": step["name"],
                        "arguments": arguments,
                        "result": value,
                        "turn_index": ctx.turn_index,
                    }
                )
            elif op == "list_tools":
                value = module.list_tools()
            elif op == "metadata":
                metadata = getattr(module, "metadata", None)
                value = metadata() if callable(metadata) else None
            elif op == "inspect_backend":
                value = {
                    name: callable(getattr(module, name, None))
                    for name in ("list_tools", "reset", "call_tool", "get_state")
                }
            elif op == "run_assertion":
                if assertions_module is None:
                    raise RuntimeError("run_assertion requires assertions_path on the worker")
                # An assertion verdict is data: a failing pack assertion must not kill the worker.
                try:
                    _resolve_assertion(assertions_module, step["name"])(
                        state=module.get_state(),
                        trace=step.get("trace") if step.get("trace") is not None else trace,
                        task=step.get("task") or {},
                        ctx=ctx,
                    )
                    value = {"name": step["name"], "passed": True, "detail": None}
                except AssertionError as exc:
                    value = {"name": step["name"], "passed": False, "detail": str(exc)}
                except Exception as exc:  # noqa: BLE001 — surfaced as a failed assertion, not a crash
                    value = {
                        "name": step["name"],
                        "passed": False,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
            else:
                raise ValueError(f"unknown episode op {op!r}")
            connection.send(("ok", value, getattr(module, "session_id", None)))
    except EOFError:
        return
    except BaseException:  # noqa: BLE001
        try:
            connection.send(("err", traceback.format_exc(), getattr(locals().get("module"), "session_id", None)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        close = getattr(locals().get("module"), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        connection.close()


def _inspect_assertions_module_inner(
    path: str,
    *,
    sanitize_environment: bool,
    import_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Import assertions inside a worker and validate the callable contract."""
    if sanitize_environment:
        _sanitize_pack_environment()
    module = _load_module(Path(path), f"_bfcl_assertions_{abs(hash(path))}", import_root)
    exported = getattr(module, "ASSERTIONS", None)
    if isinstance(exported, dict):
        assertions = {str(name): fn for name, fn in exported.items() if callable(fn)}
    else:
        assertions = {
            name: getattr(module, name)
            for name in dir(module)
            if name.startswith("assert_") and callable(getattr(module, name))
        }

    required = {"state", "trace", "task", "ctx"}
    report: dict[str, dict[str, Any]] = {}
    for name, fn in assertions.items():
        signature = inspect.signature(fn)
        params = signature.parameters
        has_var_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
        missing = sorted(
            arg
            for arg in required
            if arg not in params or params[arg].kind is inspect.Parameter.POSITIONAL_ONLY
        )
        required_extras = sorted(
            param.name
            for param in params.values()
            if param.name not in required
            and param.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and param.default is inspect.Parameter.empty
        )
        valid = (has_var_kwargs or not missing) and not required_extras
        report[name] = {
            "valid": valid,
            "reason": None
            if valid
            else f"missing keyword args={missing}; unsupported required args={required_extras}",
        }
    return report


def _inspect_assertions_module(
    path: str,
    *,
    sanitize_environment: bool,
    import_root: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect assertions without retaining pack modules in thread-debug mode."""
    if sanitize_environment:
        return _inspect_assertions_module_inner(
            path,
            sanitize_environment=True,
            import_root=import_root,
        )
    with _restored_import_state():
        return _inspect_assertions_module_inner(
            path,
            sanitize_environment=False,
            import_root=import_root,
        )


def _stop_process(proc: mp.Process) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(1.0)
    if proc.is_alive():
        proc.kill()
        proc.join(1.0)


@dataclass
class ProcessWorker:
    """Process-worker facade used by oracle validation / replay."""

    default_timeout_s: float = 5.0
    worker: str = "process"

    def __post_init__(self) -> None:
        if self.worker == "thread":
            logger.warning(
                "BFCL oracle_runtime.worker=thread imports pack code into this process with the "
                "host environment visible to it. Only episode_timeout_s bounds the caller's wait; "
                "per-operation timeouts are hard deadlines only in process mode, and a timed-out "
                "thread may keep running. Use thread mode to debug and process mode to certify."
            )

    def call(self, fn: Callable[..., T], /, *args: Any, timeout_s: float | None = None, **kwargs: Any) -> T:
        return run_with_timeout(
            fn,
            timeout_s if timeout_s is not None else self.default_timeout_s,
            *args,
            worker=self.worker,
            **kwargs,
        )

    def run_episode(
        self,
        *,
        backend_path: Path | str | None = None,
        endpoint_config: Any = None,
        fixtures: dict[str, Any] | None,
        clock_iso: str,
        seed: int,
        task_id: str,
        steps: Iterable[dict[str, Any]],
        assertions_path: Path | str | None = None,
        import_root: Path | str | None = None,
        import_timeout_s: float | None = None,
        reset_timeout_s: float | None = None,
        tool_timeout_s: float | None = None,
        assertion_timeout_s: float | None = None,
        episode_timeout_s: float | None = None,
    ) -> list[Any]:
        """Run one backend episode.

        ``steps`` may be a regular iterable or an interactive generator. Interactive
        generators receive each operation result through ``send`` before yielding the
        next step, which lets a later tool call bind arguments from an earlier result
        without restarting the backend. Process workers enforce each operation timeout;
        debug-thread workers can only bound the caller's wait for the whole episode.
        """
        import_timeout = self.default_timeout_s if import_timeout_s is None else import_timeout_s
        reset_timeout = self.default_timeout_s if reset_timeout_s is None else reset_timeout_s
        tool_timeout = self.default_timeout_s if tool_timeout_s is None else tool_timeout_s
        assertion_timeout = self.default_timeout_s if assertion_timeout_s is None else assertion_timeout_s
        episode_timeout = self.default_timeout_s if episode_timeout_s is None else episode_timeout_s
        if (backend_path is None) == (endpoint_config is None):
            raise ValueError("run_episode requires exactly one of backend_path or endpoint_config")
        endpoint_headers: dict[str, str] | None = None
        if endpoint_config is not None:
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                resolve_endpoint_headers,
            )

            endpoint_headers = resolve_endpoint_headers(endpoint_config)

        if self.worker == "thread":
            return run_with_timeout(
                _run_backend_episode_sync,
                episode_timeout,
                None if backend_path is None else str(backend_path),
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout,
                steps,
                sanitize_environment=False,
                assertions_path=None if assertions_path is None else str(assertions_path),
                import_root=None if import_root is None else str(import_root),
                endpoint_config=endpoint_config,
                endpoint_headers=endpoint_headers,
                worker="thread",
            )
        if self.worker != "process":
            raise ValueError(f"unsupported worker {self.worker!r}")

        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        proc = context.Process(
            target=_backend_worker,
            args=(
                child,
                None if backend_path is None else str(backend_path),
                fixtures,
                clock_iso,
                seed,
                task_id,
                tool_timeout,
                None if assertions_path is None else str(assertions_path),
                None if import_root is None else str(import_root),
                endpoint_config,
                endpoint_headers,
            ),
        )
        started = time.monotonic()
        proc.start()
        child.close()
        remote_session_id: str | None = None

        def cleanup_endpoint_session() -> None:
            nonlocal remote_session_id
            if endpoint_config is None or remote_session_id is None:
                return
            from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
                EndpointOracleClient,
            )

            client = EndpointOracleClient(
                endpoint_config,
                headers=endpoint_headers or {},
                timeout_s=tool_timeout,
            )
            client.session_id = remote_session_id
            client.close()
            remote_session_id = None

        def exchange(
            timeout_s: float,
            label: str,
            step: dict[str, Any] | None = None,
            *,
            respect_episode_deadline: bool = True,
        ) -> Any:
            """Send one step and receive its reply without blocking past the deadline.

            ``Connection.poll`` only proves that the first bytes of a message are
            readable. ``Connection.recv`` can then block while a large payload is
            still being written, so the complete exchange has to run behind the same
            deadline as the backend operation.
            """
            nonlocal remote_session_id
            remaining = episode_timeout - (time.monotonic() - started)
            timeout = min(timeout_s, remaining) if respect_episode_deadline else timeout_s
            if timeout <= 0:
                _stop_process(proc)
                error = TimeoutError(f"{label} exceeded timeout_s={timeout_s}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error

            replies: queue_module.Queue[tuple[str, Any]] = queue_module.Queue(maxsize=1)

            def communicate() -> None:
                try:
                    if step is not None:
                        parent.send(step)
                    replies.put(("reply", parent.recv()))
                except BaseException as exc:  # noqa: BLE001 — transferred to caller
                    replies.put(("exception", exc))

            receiver = threading.Thread(target=communicate, daemon=True)
            receiver.start()
            receiver.join(timeout)
            if receiver.is_alive():
                _stop_process(proc)
                parent.close()
                receiver.join(1.0)
                error = TimeoutError(f"{label} exceeded timeout_s={timeout_s}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error
            kind, value = replies.get_nowait()
            if kind == "exception":
                raise RuntimeError(f"worker connection failed during {label}: {value}") from value
            status, payload, session_id = value
            remote_session_id = session_id
            if status == "err":
                _stop_process(proc)
                error = RuntimeError(f"worker failed during {label}:\n{payload}")
                try:
                    cleanup_endpoint_session()
                except Exception as cleanup_exc:  # noqa: BLE001
                    error.add_note(
                        "endpoint session cleanup also failed: "
                        f"{type(cleanup_exc).__name__}"
                    )
                raise error
            return payload

        try:
            exchange(import_timeout, "backend import")
            outputs: list[Any] = []
            operation_timeouts = {"reset": reset_timeout, "run_assertion": assertion_timeout}
            iterator = iter(steps)
            try:
                step = _advance_episode(iterator)
                while True:
                    op = step["op"]
                    output = exchange(operation_timeouts.get(op, tool_timeout), op, step)
                    outputs.append(output)
                    step = _advance_episode(iterator, output)
            except StopIteration:
                pass
            exchange(
                tool_timeout,
                "session close",
                {"op": "close"},
                respect_episode_deadline=False,
            )
            proc.join(min(tool_timeout, max(0.0, episode_timeout - (time.monotonic() - started))))
            _stop_process(proc)
            return outputs
        finally:
            _stop_process(proc)
            parent.close()
            # Covers parent-side generator failures and broken IPC after a reset:
            # neither path passes through exchange's timeout/error cleanup.
            cleanup_endpoint_session()

    def inspect_assertions(
        self,
        assertions_path: Path | str,
        *,
        import_root: Path | str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Discover assertions and validate signatures without importing in the parent."""
        return self.call(
            _inspect_assertions_module,
            str(assertions_path),
            sanitize_environment=self.worker == "process",
            import_root=None if import_root is None else str(import_root),
            timeout_s=timeout_s,
        )
