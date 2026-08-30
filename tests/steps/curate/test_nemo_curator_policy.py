# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""F2: approved-policy filtering and the three run modes.

The mode table is a claim about rows and columns, so it is tested by running
data through stages rather than by asserting which class was constructed. The
stubs below transcribe Curator's own ``score_filter.py`` — in particular that
``ScoreFilter`` writes ``score_field`` *before* applying ``keep_document``,
which is why ``both`` is one stage and not ``Score`` followed by ``Filter``.
``test_stub_signatures_match_curator`` pins the transcription wherever Curator
is actually installed.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
import yaml

from nemotron.steps.curate.runtime import policy as policy_module
from nemotron.steps.curate.runtime import registry as signal_registry

# Three clean documents and two that are almost entirely emoji. Scored by
# ``unicode_alpha_numeric`` at the 0.25 default, the split is 3 kept / 2 dropped.
DOCS = [
    {"id": "a", "text": "Curation needs measurements, not defaults inherited from a sample config."},
    {"id": "b", "text": "Tieng Viet la ngon ngu chinh thuc cua Viet Nam."},
    {"id": "c", "text": "A third ordinary sentence, long enough to score like prose."},
    {"id": "d", "text": "🙂🙂🙂🙂🙂🙂🙂🙂"},
    {"id": "e", "text": "★★★ ☆☆☆ ★★★ ☆☆☆"},
]
CLEAN = 3

SIGNAL = "unicode_alpha_numeric"
SCORE_COLUMN = f"__{SIGNAL}"
BASE_COLUMNS = {"id", "text"}


# -- executable stubs ---------------------------------------------------------


@dataclass
class _Score:
    score_fn: Any
    score_field: str
    text_field: str = "text"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.score_field] = df[self.text_field].apply(self.score_fn.score_document)
        return df


@dataclass
class _Filter:
    filter_fn: Any
    filter_field: str
    invert: bool = False

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.filter_field].apply(self.filter_fn)
        return df[~mask if self.invert else mask]


@dataclass
class _ScoreFilter:
    filter_obj: Any
    text_field: str = "text"
    score_field: str | None = None
    invert: bool = False

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        scores = df[self.text_field].apply(self.filter_obj.score_document)
        if self.score_field is not None:
            df[self.score_field] = scores
        mask = scores.apply(self.filter_obj.keep_document)
        return df[~mask if self.invert else mask]


@pytest.fixture
def step(monkeypatch):
    """``step`` with Curator's filter modules replaced by executable stubs."""
    filters = types.ModuleType("nemo_curator.stages.text.filters")
    filters.Score, filters.Filter, filters.ScoreFilter = _Score, _Filter, _ScoreFilter

    mods = {
        "nemo_curator": {},
        "nemo_curator.core": {},
        "nemo_curator.core.client": {"RayClient": object},
        "nemo_curator.pipeline": {"Pipeline": object},
        "nemo_curator.stages": {},
        "nemo_curator.stages.text": {},
        "nemo_curator.stages.text.io": {},
        "nemo_curator.stages.text.io.reader": {"JsonlReader": object},
        "nemo_curator.stages.text.io.writer": {"JsonlWriter": object},
        # Present but without the filter classes, exercising the release shim in
        # text_filter_stages()/score_stage() the same way Curator 1.3 does.
        "nemo_curator.stages.text.modules": {"AddId": object},
        "huggingface_hub": {"snapshot_download": lambda **kwargs: None},
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "nemo_curator.stages.text.filters", filters)

    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    monkeypatch.delitem(sys.modules, "nemotron.steps.curate.nemo_curator.step", raising=False)
    return module


def run(stages) -> pd.DataFrame:
    df = pd.DataFrame(DOCS)
    for stage in stages:
        df = stage.process(df)
    return df


def approved_policy(**overrides) -> dict:
    document = {
        "schema_version": policy_module.SCHEMA_VERSION,
        "approved": True,
        "corpus": {"fingerprint": "sha256:deadbeef"},
        "signals_impl_version": signal_registry.IMPL_VERSION,
        "profile_digest": "sha256:cafe",
        "approval": {
            "method": "manual",
            "approver": "hndo@nvidia.com",
            "date": "2026-08-25",
            "evidence": "retention curve reviewed against a 200-doc sample",
        },
        "thresholds": [{"signal": SIGNAL, "max": 0.25}],
    }
    document.update(overrides)
    return document


def write_policy(tmp_path, document, name="approved_policy.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# -- the mode table -----------------------------------------------------------


def test_filter_mode_drops_rows_and_adds_no_column(step) -> None:
    """The historical shape: gates reject rows and the score is discarded."""
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "filter"))

    assert len(out) == CLEAN
    assert set(out.columns) == BASE_COLUMNS


def test_annotate_mode_keeps_every_row_and_adds_the_column(step) -> None:
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "annotate"))

    assert len(out) == len(DOCS), "annotate must not drop anything"
    assert set(out.columns) == BASE_COLUMNS | {SCORE_COLUMN}


def test_both_mode_drops_rows_and_survivors_keep_the_column(step) -> None:
    out = run(step.policy_stages(approved_policy()["thresholds"], "text", "both"))

    assert len(out) == CLEAN
    assert set(out.columns) == BASE_COLUMNS | {SCORE_COLUMN}
    assert out[SCORE_COLUMN].notna().all()


def test_both_is_one_stage_not_score_then_filter(step) -> None:
    """ScoreFilter writes the score before gating, so a second pass is waste."""
    thresholds = approved_policy()["thresholds"]

    assert len(step.policy_stages(thresholds, "text", "both")) == len(thresholds)


def test_annotate_and_filter_agree_on_which_rows_survive(step) -> None:
    """The modes must differ in what they emit, never in what they judge."""
    annotated = run(step.policy_stages(approved_policy()["thresholds"], "text", "annotate"))
    filtered = run(step.policy_stages(approved_policy()["thresholds"], "text", "filter"))

    kept = annotated[annotated[SCORE_COLUMN] <= 0.25]

    assert sorted(kept["id"]) == sorted(filtered["id"])


def test_the_text_field_is_honoured(step) -> None:
    stages = step.policy_stages(approved_policy()["thresholds"], "body", "annotate")

    assert stages[0].text_field == "body"


def test_an_unknown_mode_is_refused(step, tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"mode": "sieve", "input_glob": "x", "text_field": "text", "output_dir": "o"}))
    monkeypatch.setattr(sys, "argv", ["step.py", "--config", str(cfg)])

    with pytest.raises(ValueError, match="mode must be one of"):
        step.main()


# -- approval gating ----------------------------------------------------------


def test_a_candidate_policy_is_refused(step, tmp_path) -> None:
    """curate/profile emits approved: false on purpose; this is the other half."""
    path = write_policy(tmp_path, approved_policy(approved=False), "candidate_policies.yaml")

    with pytest.raises(policy_module.PolicyNotApprovedError) as excinfo:
        step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})

    message = str(excinfo.value)
    assert str(path) in message, "the error must name which policy was refused"
    assert "approved must be true" in message, "the error must name the unmet field"


def test_a_policy_with_no_approval_block_still_runs(step, tmp_path) -> None:
    """The block is optional; the machine-checkable fields around it are not."""
    document = approved_policy()
    del document["approval"]
    path = write_policy(tmp_path, document)

    thresholds, _, _ = step.resolve_policy(
        {"heuristic_filters": {"approved_policy": str(path)}}
    )

    assert thresholds, "thresholds are what make a policy executable"


def test_the_override_proceeds_and_warns_naming_the_policy(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy(approved=False))

    thresholds, _pack, warnings = step.resolve_policy(
        {"heuristic_filters": {"approved_policy": str(path), "allow_unvalidated_policy": True}}
    )

    assert thresholds, "the override must actually proceed"
    assert warnings, "an override that logs nothing is indistinguishable from approval"
    assert all(str(path) in w for w in warnings)
    assert any("allow_unvalidated_policy" in w for w in warnings)


def test_an_approved_policy_warns_about_nothing(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy())

    thresholds, _pack, warnings = step.resolve_policy({"heuristic_filters": {"approved_policy": str(path)}})

    assert warnings == []
    assert thresholds == [{"signal": SIGNAL, "max": 0.25}]


def test_a_policy_from_another_langpack_is_refused(step, tmp_path) -> None:
    """Thresholds are calibrated against a pack's word lists and character set."""
    path = write_policy(tmp_path, approved_policy(langpack={"tag": "vi", "content_hash": "sha256:aaa"}))

    with pytest.raises(ValueError, match="sha256:aaa"):
        step.resolve_policy(
            {
                "heuristic_filters": {
                    "approved_policy": str(path),
                    "langpack_content_hash": "sha256:bbb",
                }
            }
        )


def test_a_matching_langpack_hash_is_accepted(step, tmp_path) -> None:
    path = write_policy(tmp_path, approved_policy(langpack={"tag": "vi", "content_hash": "sha256:aaa"}))

    thresholds, _pack, _warnings = step.resolve_policy(
        {
            "heuristic_filters": {
                "approved_policy": str(path),
                "langpack_content_hash": "sha256:aaa",
            }
        }
    )

    assert thresholds


def test_an_unknown_signal_is_refused_with_the_allowed_names(step) -> None:
    with pytest.raises(ValueError) as excinfo:
        step.policy_stages([{"signal": "os.system", "max": 1}], "text", "filter")

    message = str(excinfo.value)
    assert "os.system" in message
    assert SIGNAL in message, "the error must list what is allowed"


# -- the neutrality guarantee -------------------------------------------------


def test_no_policy_configured_adds_no_stages(step) -> None:
    """A config that predates F2 must build exactly the pipeline it built before."""
    for cfg in ({}, {"heuristic_filters": None}, {"heuristic_filters": {}}):
        assert step.resolve_policy(cfg) == ([], {}, [])

    for mode in step.MODES:
        assert step.policy_stages([], "text", mode) == []


def test_no_policy_resolves_no_filter_classes(step, monkeypatch) -> None:
    """The empty path must not reach an import.

    Curator moved Score/Filter/ScoreFilter between releases, so resolving them
    runs a fallback chain. A run with no policy configured should not depend on
    that chain resolving at all.
    """

    def boom():
        raise AssertionError("the empty path resolved a filter class")

    monkeypatch.setattr(step, "text_filter_stages", boom)
    monkeypatch.setattr(step, "score_stage", boom)

    for mode in step.MODES:
        assert step.policy_stages([], "text", mode) == []


# -- the transcription these tests rest on ------------------------------------


def test_stub_signatures_match_curator() -> None:
    """Wherever Curator is installed, prove the stubs model the real classes."""
    real = pytest.importorskip("nemo_curator.stages.text.filters")

    for stub, name in ((_Score, "Score"), (_Filter, "Filter"), (_ScoreFilter, "ScoreFilter")):
        actual = inspect.signature(getattr(real, name)).parameters
        for param, spec in inspect.signature(stub).parameters.items():
            assert param in actual, f"{name} has no parameter {param}"
            if spec.default is not inspect.Parameter.empty:
                assert actual[param].default == spec.default, f"{name}.{param} default drifted"

    source = inspect.getsource(real.ScoreFilter.compute_filter_mask)
    assert "df[score_field] = scores" in source, (
        "ScoreFilter no longer annotates before filtering; 'both' mode is no longer one stage"
    )


# -- language-pack signals ----------------------------------------------------
#
# Half the registry is parameterised by a language pack, and curate/profile emits
# candidates for exactly those. A policy that could be produced but not executed
# would be a dead end between the two steps, so these cover the handover.

PACK_SIGNAL = "stopword_ratio"


def pack_policy(**overrides):
    from nemotron.steps.curate.runtime import langpack

    document = approved_policy(
        langpack={
            "language_tag": "vi",
            "content_hash": langpack.load("vi").content_hash,
        },
        thresholds=[{"signal": PACK_SIGNAL, "min": 0.05}],
    )
    document.update(overrides)
    return document


def test_a_policy_naming_a_pack_signal_can_actually_be_executed(step) -> None:
    """The handover curate/profile -> curate/nemo_curator, end to end."""
    document = pack_policy()

    stages = step.policy_stages(
        document["thresholds"], "text", "annotate", document["langpack"]
    )

    assert len(stages) == 1
    assert stages[0].score_fn.score_document("của và những là một trong") > 0


def test_every_pack_backed_signal_in_the_registry_can_be_built(step) -> None:
    """A signal profile can propose but the filter cannot construct is unusable."""
    from nemotron.steps.curate.runtime import langpack
    from nemotron.steps.curate.runtime import registry as signal_registry

    pack = langpack.load("vi")
    spec = {"language_tag": "vi", "content_hash": pack.content_hash}
    supported = [n for n in sorted(signal_registry.PACK_SIGNALS) if n in signal_registry.SIGNALS]
    buildable = [
        n
        for n in supported
        if set(signal_registry.SIGNALS[n].requires) <= set(pack.capabilities)
    ]

    assert buildable, "the vi pack must back at least one registry signal"
    for name in buildable:
        # The bound key must match the signal's direction. Feeding every signal
        # "max" is what hid the gate-inversion defect: a max bound on a
        # min-direction signal used to build a filter that gated the wrong way.
        signal = signal_registry.SIGNALS[name]
        entry = {"signal": name}
        for key in step.BOUND_KEYS[signal.direction]:
            entry[key] = 0.5 if key == "max" else 0.1
        assert step.policy_stages([entry], "text", "filter", spec), name


def test_a_pack_signal_without_a_declared_language_is_refused(step) -> None:
    """There is no default language: a wrong one produces plausible wrong numbers."""
    with pytest.raises(ValueError, match="no langpack.language_tag"):
        step.policy_stages([{"signal": PACK_SIGNAL, "min": 0.05}], "text", "filter", {})


def test_a_pack_that_hashes_differently_than_declared_is_refused(step) -> None:
    """Checked against the pack actually loaded, not another string in the config."""
    spec = {"language_tag": "vi", "content_hash": "sha256:notthepackonthismachine"}

    with pytest.raises(ValueError, match="hashes to"):
        step.policy_stages([{"signal": PACK_SIGNAL, "min": 0.05}], "text", "filter", spec)


def test_a_pack_free_policy_never_loads_a_pack(step, monkeypatch) -> None:
    """Otherwise a machine with no packs installed could not run a policy of
    Curator-only signals, which have nothing to do with language packs."""

    def boom(*args, **kwargs):
        raise AssertionError("a pack-free policy loaded a language pack")

    monkeypatch.setattr(step, "load_policy_pack", boom)

    assert step.policy_stages(approved_policy()["thresholds"], "text", "filter")


def test_resolve_policy_hands_the_pack_spec_to_the_caller(step, tmp_path) -> None:
    path = write_policy(tmp_path, pack_policy())

    thresholds, spec, _ = step.resolve_policy(
        {"heuristic_filters": {"approved_policy": str(path)}}
    )

    assert thresholds[0]["signal"] == PACK_SIGNAL
    assert spec["language_tag"] == "vi"


# -- bound direction ----------------------------------------------------------
#
# A policy's min/max zipped positionally onto a signal's threshold_params builds
# a working pipeline that gates the wrong way round. No test of the pipeline's
# shape can catch that, so it is pinned on behaviour here.


def test_a_max_bound_on_a_min_direction_signal_is_refused(step) -> None:
    """The silent gate inversion: 'drop above 0.9' becoming 'drop below 0.9'."""
    from nemotron.steps.curate.runtime import langpack

    spec = {"language_tag": "vi", "content_hash": langpack.load("vi").content_hash}

    with pytest.raises(ValueError, match="invert the gate"):
        step.policy_stages([{"signal": "stopword_ratio", "max": 0.9}], "text", "filter", spec)


def test_a_min_bound_on_a_max_direction_signal_is_refused(step) -> None:
    with pytest.raises(ValueError, match="invert the gate"):
        step.policy_stages([{"signal": SIGNAL, "min": 0.9}], "text", "filter")


def test_the_correct_bound_still_builds_the_gate_it_says(step) -> None:
    """The other half: the accepted spelling must mean what it reads as."""
    stages = step.policy_stages([{"signal": SIGNAL, "max": 0.25}], "text", "filter")
    document_filter = stages[0].filter_obj

    assert document_filter.keep_document(0.10) is True
    assert document_filter.keep_document(0.90) is False


def test_an_interval_signal_needs_both_bounds(step) -> None:
    """One bound would fix the other at an unstated value."""
    with pytest.raises(ValueError, match="gates from both sides"):
        step.policy_stages([{"signal": "word_count", "min": 50}], "text", "filter")


def test_a_signal_with_no_bound_and_no_default_is_refused(step) -> None:
    with pytest.raises(ValueError, match="no shipped default"):
        step.policy_stages([{"signal": "token_count"}], "text", "filter", {"tokenizer": "m"})


def test_every_registry_signal_has_a_known_direction() -> None:
    """BOUND_KEYS must cover the registry, or a signal fails only when used."""
    import importlib

    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    from nemotron.steps.curate.runtime import registry as signal_registry

    directions = {s.direction for s in signal_registry.SIGNALS.values()}
    assert directions <= set(module.BOUND_KEYS)


# -- requirements -------------------------------------------------------------


def test_a_signal_needing_a_tokenizer_says_so_instead_of_failing_on_a_kwarg(step) -> None:
    """token_count is proposed by profile whenever models.tokenizer is set."""
    with pytest.raises(ValueError, match="needs a tokenizer"):
        step.policy_stages(
            [{"signal": "token_count", "min": 10, "max": 4096}], "text", "filter"
        )


def test_every_requirement_the_registry_declares_is_one_this_step_can_supply() -> None:
    """The check that would have caught token_count before a user did.

    A hand-maintained set of pack-backed names has to be remembered; this derives
    the obligation from the registry itself, so adding a signal with a new
    requirement fails here rather than at someone's pipeline construction.
    """
    import importlib

    module = importlib.import_module("nemotron.steps.curate.nemo_curator.step")
    from nemotron.steps.curate.runtime import registry as signal_registry

    declared = {r for s in signal_registry.SIGNALS.values() for r in s.requires}

    assert declared <= module.KNOWN_REQUIREMENTS, (
        f"registry declares requirement(s) the filter step cannot satisfy: "
        f"{sorted(declared - module.KNOWN_REQUIREMENTS)}"
    )


def test_the_tokenizer_reaches_the_filter(step, monkeypatch) -> None:
    """Asserted on the kwargs, since constructing it needs Curator installed."""
    import dataclasses

    from nemotron.steps.curate.runtime import registry as signal_registry

    seen: dict = {}
    original = signal_registry.SIGNALS["token_count"]
    monkeypatch.setitem(
        signal_registry.SIGNALS,
        "token_count",
        dataclasses.replace(original, factory=lambda **kwargs: seen.update(kwargs) or object()),
    )

    step.policy_stages(
        [{"signal": "token_count", "min": 10, "max": 4096}],
        "text",
        "filter",
        {"tokenizer": "some/model"},
    )

    assert seen["hf_model_name"] == "some/model"
    assert seen["min_tokens"] == 10 and seen["max_tokens"] == 4096


# -- content hash enforcement -------------------------------------------------


def test_a_policy_omitting_the_pack_hash_is_refused(step) -> None:
    """An unenforced guarantee reads exactly like an enforced one.

    The README promises a pack mismatch stops the run; a policy that simply
    declares no hash used to sail past both checks.
    """
    with pytest.raises(ValueError, match="declares no langpack.content_hash"):
        step.policy_stages(
            [{"signal": "stopword_ratio", "min": 0.1}], "text", "filter", {"language_tag": "vi"}
        )


def test_resolve_policy_carries_the_config_tokenizer_and_pack_dir(step, tmp_path) -> None:
    path = write_policy(tmp_path, pack_policy())

    _, spec, _ = step.resolve_policy(
        {
            "heuristic_filters": {
                "approved_policy": str(path),
                "tokenizer": "some/model",
                "langpack_dir": "/somewhere/packs",
            }
        }
    )

    assert spec["tokenizer"] == "some/model"
    assert spec["langpack_dir"] == "/somewhere/packs"
