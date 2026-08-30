# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The one shape every curate step shares: ``run(cfg) -> report``.

Without a uniform seam, anything that drives several steps has to know each one's
private entry point, and each step can only be tested by faking ``sys.argv``.
These tests are a cross-step invariant, so they live apart from any one step's
suite — adding a sixth step with a different shape fails here.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types

import pytest

#: Every curate step, and the module that owns its entry point.
STEP_MODULES = {
    "curate/ingest": "nemotron.steps.curate.scripts.run_ingest",
    "curate/nemo_curator": "nemotron.steps.curate.nemo_curator.step",
    "curate/audit": "nemotron.steps.curate.scripts.run_audit",
    "curate/profile": "nemotron.steps.curate.scripts.run_profile",
    "curate/subset": "nemotron.steps.curate.scripts.run_subset",
    "curate/decontamination": "nemotron.steps.curate.scripts.run_decontamination",
    "curate/flow": "nemotron.steps.curate.scripts.run_flow",
}


@pytest.fixture(scope="module", autouse=True)
def _curator_stub():
    """``nemo_curator/step.py`` imports Curator at module scope; the others do not."""
    names = (
        "nemo_curator",
        "nemo_curator.core",
        "nemo_curator.core.client",
        "nemo_curator.pipeline",
        "nemo_curator.stages",
        "nemo_curator.stages.text",
        "nemo_curator.stages.text.io",
        "nemo_curator.stages.text.io.reader",
        "nemo_curator.stages.text.io.writer",
        "huggingface_hub",
    )
    created = [n for n in names if n not in sys.modules]
    for name in created:
        sys.modules[name] = types.ModuleType(name)
    for module, attr in (
        ("nemo_curator.core.client", "RayClient"),
        ("nemo_curator.pipeline", "Pipeline"),
        ("nemo_curator.stages.text.io.reader", "JsonlReader"),
        ("nemo_curator.stages.text.io.writer", "JsonlWriter"),
    ):
        if not hasattr(sys.modules[module], attr):
            setattr(sys.modules[module], attr, object)
    if not hasattr(sys.modules["huggingface_hub"], "snapshot_download"):
        sys.modules["huggingface_hub"].snapshot_download = lambda **kwargs: None
    yield
    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture(params=sorted(STEP_MODULES), ids=sorted(STEP_MODULES))
def step_module(request):
    return request.param, importlib.import_module(STEP_MODULES[request.param])


def test_every_step_exposes_run(step_module) -> None:
    name, module = step_module

    assert callable(getattr(module, "run", None)), f"{name} has no run()"


def test_run_is_callable_with_only_a_config(step_module) -> None:
    """A caller with a config dict must not need to know a step's extra arguments."""
    name, module = step_module
    parameters = list(inspect.signature(module.run).parameters.values())

    assert parameters, f"{name}.run() takes no arguments"
    assert parameters[0].name == "cfg", f"{name}.run()'s first parameter is not cfg"
    for extra in parameters[1:]:
        assert extra.default is not inspect.Parameter.empty, (
            f"{name}.run() requires {extra.name!r} beyond cfg, so a caller holding only a "
            "config cannot invoke it"
        )


def test_every_step_still_has_a_main(step_module) -> None:
    """The CLI entry point stays; run() is added beside it, not instead of it."""
    name, module = step_module

    assert callable(getattr(module, "main", None)), f"{name} has no main()"


def test_run_does_not_exit_the_process(step_module) -> None:
    """A step that calls SystemExit takes the decision away from its caller.

    Deciding what a failing audit means for the steps after it belongs to
    whatever is running them, so the exit code is mapped in main().
    """
    import ast
    import textwrap

    name, module = step_module
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.run)))

    # Parsed, not grepped: the docstrings say "SystemExit" precisely because they
    # promise not to raise it, and a substring check would fail on the promise.
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(raised, ast.Name) and raised.id == "SystemExit":
                offenders.append(node.lineno)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "exit":
                offenders.append(node.lineno)
            if isinstance(fn, ast.Name) and fn.id == "exit":
                offenders.append(node.lineno)

    assert not offenders, f"{name}.run() exits the process at line(s) {offenders}"


def test_the_step_entry_point_delegates_rather_than_reimplementing(step_module) -> None:
    """step.py must not grow a second copy of the logic in run()."""
    name, module = step_module
    if name == "curate/nemo_curator":
        pytest.skip("its runspec entry point is the module itself, not a delegating shim")

    main_source = inspect.getsource(module.main)
    assert "run(" in main_source, f"{name}.main() does not call run()"


def test_the_module_list_covers_every_step_directory() -> None:
    """A sixth step added without a run() must fail here rather than silently."""
    from pathlib import Path

    root = Path(inspect.getfile(importlib.import_module("nemotron.steps.curate"))).parent
    found = {
        f"curate/{p.name}"
        for p in root.iterdir()
        if p.is_dir() and (p / "step.toml").is_file()
    }

    assert found == set(STEP_MODULES), (
        f"step directories and the seam list disagree: {found ^ set(STEP_MODULES)}"
    )


# -- one corpus reference means one corpus -------------------------------------
#
# There were five path resolvers: integrity.expand_inputs, a private jsonl-only
# copy in run_audit, and bare glob.glob in run_subset and run_decontamination.
# The same reference therefore resolved to different corpora depending on which
# step read it — and a directory, which the flow and curate/ingest both accept,
# resolved to nothing at all in subset and decontamination.


#: Extra arguments a resolver needs beyond the corpus reference.
_RESOLVER_ARGS = {"run_decontamination.resolve_inputs": ("train_glob",)}


def _resolvers():
    """Every corpus-reference resolver in the category, found rather than listed.

    This test used to enumerate them by hand and listed four of the five: it
    claimed "every step resolves the same" while never calling run_profile's,
    which had quietly kept the old directory behaviour. A hand-written list in a
    completeness test is a list that goes stale silently, so the modules are
    scanned instead and a new resolver joins automatically.
    """
    import importlib

    from nemotron.steps.curate.runtime import integrity

    found = {"integrity.expand_inputs": (integrity.expand_inputs, ())}
    for module_name in ("run_profile", "run_audit", "run_subset", "run_decontamination", "run_ingest"):
        module = importlib.import_module(f"nemotron.steps.curate.scripts.{module_name}")
        for attr in ("expand", "resolve_inputs"):
            fn = getattr(module, attr, None)
            if fn is None:
                continue
            key = f"{module_name}.{attr}"
            found[key] = (fn, _RESOLVER_ARGS.get(key, ()))

    return {name: (lambda ref, f=fn, a=args: f(ref, *a)) for name, (fn, args) in found.items()}


def test_the_resolver_scan_finds_every_step() -> None:
    """The scan is the test's own premise; if it finds too few it proves nothing."""
    names = set(_resolvers())

    assert "run_profile.expand" in names, "the one that was missed when this was a hand-written list"
    assert len(names) >= 4, f"only found {sorted(names)}"


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("part_0.jsonl", "part_1.jsonl"):
        (root / name).write_text('{"id":"x","text":"y"}\n', encoding="utf-8")
    return root


def test_every_step_resolves_a_directory_to_the_same_corpus(tmp_path) -> None:
    root = _corpus(tmp_path)

    resolved = {name: sorted(fn(str(root))) for name, fn in _resolvers().items()}

    assert len({tuple(v) for v in resolved.values()}) == 1, resolved
    assert len(next(iter(resolved.values()))) == 2


def test_every_step_resolves_a_glob_to_the_same_corpus(tmp_path) -> None:
    root = _corpus(tmp_path)

    resolved = {name: sorted(fn(f"{root}/**/*.jsonl")) for name, fn in _resolvers().items()}

    assert len({tuple(v) for v in resolved.values()}) == 1, resolved


# -- one user mistake, one kind of refusal -------------------------------------
#
# Every runner exposes run(cfg) and a main() that turns a user-fixable problem
# into exit 2 with a message, and leaves a real bug as a traceback. That contract
# held for three of the five: pointing any step at a corpus that does not exist
# raised IngestError, ConfigError, ConfigError — but FileNotFoundError from
# run_profile and run_audit, which no main() catches, so the same mistake gave a
# raw traceback and exit 1 depending on which step you ran.

_MISSING_INPUT = {
    "run_ingest": lambda t: {"input": f"{t}/nope/*.jsonl", "output_dir": f"{t}/o"},
    "run_profile": lambda t: {
        "input_glob": f"{t}/nope/*.jsonl", "output_dir": f"{t}/o", "language": "vi",
    },
    "run_audit": lambda t: {"target_glob": f"{t}/nope/*.jsonl", "output_dir": f"{t}/o"},
    "run_subset": lambda t: {
        "input_glob": f"{t}/nope/*.jsonl", "output_dir": f"{t}/o", "id_field": "id",
        "token_budgets": [100], "tokenizer": None,
    },
    "run_decontamination": lambda t: {
        "train_glob": f"{t}/nope/*.jsonl", "holdout_glob": f"{t}/nope2/*.jsonl",
        "output_dir": f"{t}/o", "id_field": "id",
    },
}


@pytest.mark.parametrize("module_name", sorted(_MISSING_INPUT))
def test_a_corpus_that_is_not_there_is_refused_the_same_way(module_name, tmp_path) -> None:
    """The message must name the glob, so the user can see their own typo."""
    import importlib

    module = importlib.import_module(f"nemotron.steps.curate.scripts.{module_name}")
    cfg = _MISSING_INPUT[module_name](str(tmp_path))

    with pytest.raises(ValueError) as caught:
        module.run(cfg)

    assert "nope" in str(caught.value), (
        f"{module_name} refused without naming the reference that failed"
    )
