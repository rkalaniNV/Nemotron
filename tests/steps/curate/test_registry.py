# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The signal allowlist and how signals are selected."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from nemotron.steps.curate.runtime import registry as r


def test_the_registry_imports_without_nemo_curator() -> None:
    """Metadata must be inspectable on a plain CI host; factories import lazily."""
    assert r.SIGNALS
    assert all(callable(s.factory) for s in r.SIGNALS.values())


def test_every_signal_names_itself_consistently() -> None:
    assert all(key == signal.name for key, signal in r.SIGNALS.items())


def test_every_signal_records_how_to_pass_its_thresholds() -> None:
    for signal in r.SIGNALS.values():
        expected = 2 if signal.direction == "interval" else 1
        assert len(signal.threshold_params) == expected, signal.name


#: Signals whose Curator defaults keep everything, so reporting "the shipped
#: default retains 100%" would be true and useless.
NO_MEANINGFUL_DEFAULT = {"token_count"}


def _wraps_a_curator_filter(signal: r.Signal) -> bool:
    """Local signals have no Curator default because Curator does not have them."""
    return signal.name not in r.PACK_SIGNALS


def test_every_curator_signal_records_its_shipped_default() -> None:
    """The report's headline is what the shipped default does to your corpus."""
    for signal in r.SIGNALS.values():
        if not _wraps_a_curator_filter(signal):
            continue
        if signal.name in NO_MEANINGFUL_DEFAULT:
            assert signal.curator_default == (), f"{signal.name} should record no default"
            assert "inf" in signal.notes, "the exemption must be explained where it is defined"
            continue
        assert signal.curator_default, f"{signal.name} has no recorded default"


def test_pack_signals_declare_no_curator_default() -> None:
    """Claiming a Curator default for a signal Curator lacks would invent a baseline."""
    for name in r.PACK_SIGNALS:
        assert r.SIGNALS[name].curator_default == (), name


def test_every_pack_signal_names_the_capability_it_needs() -> None:
    for name in r.PACK_SIGNALS:
        assert r.SIGNALS[name].requires, f"{name} declares no capability"


def test_build_accepts_runtime_supplied_arguments() -> None:
    """A tokenizer object cannot live in a module-level table; it comes from config."""
    captured = {}
    signal = r.Signal(
        name="probe",
        factory=lambda **kw: captured.update(kw),
        direction="max",
        units="ratio",
        grid=r.Grid(0, 1, 4),
        threshold_params=("max_ratio",),
    )
    signal.build(0.3, hf_model_name="some/model")

    assert captured == {"max_ratio": 0.3, "hf_model_name": "some/model"}


def test_interval_signals_carry_a_two_dimensional_grid() -> None:
    """A one-dimensional sweep of a two-sided gate is a test failure, per the plan."""
    for signal in r.SIGNALS.values():
        if signal.direction == "interval":
            assert isinstance(signal.grid, r.IntervalGrid), signal.name
        else:
            assert isinstance(signal.grid, r.Grid), signal.name


def test_an_interval_signal_with_a_flat_grid_is_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="IntervalGrid"):
        r.Signal(
            name="bad",
            factory=lambda **kw: None,
            direction="interval",
            units="words",
            grid=r.Grid(0, 10, 4),
            threshold_params=("min_words", "max_words"),
        )


def test_a_one_sided_signal_with_an_interval_grid_is_rejected() -> None:
    with pytest.raises(TypeError, match="only interval signals"):
        r.Signal(
            name="bad",
            factory=lambda **kw: None,
            direction="max",
            units="ratio",
            grid=r.IntervalGrid(r.Grid(0, 1, 4), r.Grid(1, 2, 4)),
            threshold_params=("max_ratio",),
        )


def test_a_signal_whose_parameter_count_disagrees_is_rejected() -> None:
    with pytest.raises(TypeError, match="exactly one threshold"):
        r.Signal(
            name="bad",
            factory=lambda **kw: None,
            direction="max",
            units="ratio",
            grid=r.Grid(0, 1, 4),
            threshold_params=("a", "b"),
        )


# -- grids --------------------------------------------------------------------


def test_a_grid_spans_its_endpoints() -> None:
    values = r.Grid(0.0, 1.0, 5).values()

    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert len(values) == 5


def test_an_integral_grid_yields_unique_whole_numbers() -> None:
    values = r.Grid(0, 3, 10, integral=True).values()

    assert values == [0, 1, 2, 3]
    assert all(isinstance(v, int) for v in values)


def test_a_grid_needs_at_least_two_points() -> None:
    with pytest.raises(ValueError):
        r.Grid(0, 1, 1).values()


# -- building -----------------------------------------------------------------


def test_build_maps_thresholds_onto_the_filters_own_parameter_names() -> None:
    captured = {}

    signal = r.Signal(
        name="probe",
        factory=lambda **kw: captured.update(kw),
        direction="max",
        units="ratio",
        grid=r.Grid(0, 1, 4),
        threshold_params=("max_ratio",),
        extra_kwargs={"lang": "en"},
    )
    signal.build(0.3)

    assert captured == {"max_ratio": 0.3, "lang": "en"}


def test_build_rejects_the_wrong_number_of_thresholds() -> None:
    with pytest.raises(ValueError, match="takes 1 threshold"):
        r.SIGNALS["non_alpha_numeric"].build(0.1, 0.9)


# -- resolution ---------------------------------------------------------------


def test_naming_nothing_selects_everything_the_run_supports() -> None:
    chosen, warnings = r.resolve(None, capabilities=set())
    # With no pack loaded, every pack signal is skipped along with token_count.

    names = {s.name for s in chosen}
    assert "non_alpha_numeric" in names
    assert "token_count" not in names, "token_count needs a tokenizer"
    assert any("token_count" in w for w in warnings)


def test_a_capability_present_makes_its_signal_available() -> None:
    chosen, _ = r.resolve(None, capabilities={"tokenizer"})

    assert "token_count" in {s.name for s in chosen}


def test_naming_an_unsupported_signal_fails_rather_than_skipping() -> None:
    """Silently dropping a named signal would answer a question nobody asked."""
    with pytest.raises(r.SignalRequirementsUnmet, match="token_count requires"):
        r.resolve(["token_count"], capabilities=set())


def test_naming_an_unknown_signal_fails() -> None:
    with pytest.raises(r.UnknownSignal):
        r.resolve(["definitely_not_a_signal"], capabilities=set())


def test_config_cannot_name_an_import_path() -> None:
    """The allowlist is closed: a config is a document people paste between machines."""
    with pytest.raises(r.UnknownSignal):
        r.resolve(["nemo_curator.stages.text.filters.heuristic.string.WordCountFilter"], set())


def test_resolution_order_is_stable() -> None:
    first, _ = r.resolve(None, capabilities=set())
    second, _ = r.resolve(None, capabilities=set())

    assert [s.name for s in first] == [s.name for s in second]


# -- exclusions ---------------------------------------------------------------


def test_excluded_filters_record_why() -> None:
    """A filter left out silently looks like an oversight to the next reader."""
    assert r.EXCLUDED
    assert all(reason.strip() for reason in r.EXCLUDED.values())


def test_no_excluded_filter_is_also_registered() -> None:
    registered_classes = {
        getattr(s.factory, "__name__", "").removeprefix("build_") for s in r.SIGNALS.values()
    }

    assert not (set(r.EXCLUDED) & registered_classes)


def test_the_direction_ambiguous_repetition_filters_are_excluded() -> None:
    """Their parameter says max_*, their keep_document says >=. Not reportable until resolved."""
    assert "RepeatedLinesFilter" in r.EXCLUDED
    assert "keep_document" in r.EXCLUDED["RepeatedLinesFilter"]


def test_the_hash_order_dependent_ngram_filter_is_excluded() -> None:
    """A score that moves between processes cannot inform a reproducible threshold."""
    assert "RepeatingTopNGramsFilter" in r.EXCLUDED
    assert "PYTHONHASHSEED" in r.EXCLUDED["RepeatingTopNGramsFilter"]

    with pytest.raises(r.UnknownSignal):
        r.resolve(["repeating_top_ngrams"], set())


# A frequency tie between two bigrams that contribute different character counts:
# ("aa", "bbbb") and ("bbbb", "cccccc") both occur twice, and joining them gives
# 7 and 11 characters, so which one max() happens to reach first is visible in the
# score. 14/29 or 22/29 on the same document.
_TIE_PROBE = """
from nemo_curator.stages.text.filters.heuristic.repetition.repetition import (
    RepeatingTopNGramsFilter,
)

print(RepeatingTopNGramsFilter(n=2).score_document("aa bbbb cccccc aa bbbb cccccc"))
"""


def test_the_ngram_exclusion_is_still_justified_upstream() -> None:
    """Guards the reason, not just the entry.

    If Curator makes the tie-break deterministic this fails, which is the signal
    to re-register the filter rather than leave a stale exclusion in place.
    """
    pytest.importorskip("nemo_curator.stages.text.filters.heuristic.repetition.repetition")

    scores = set()
    for seed in range(8):
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _TIE_PROBE],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
        )
        scores.add(result.stdout.strip())

    assert len(scores) > 1, (
        f"one document scored {scores} under eight hash seeds. If upstream now breaks ties "
        "deterministically, remove RepeatingTopNGramsFilter from EXCLUDED and register it."
    )
