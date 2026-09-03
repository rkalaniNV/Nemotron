# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The signals this step knows how to profile.

A closed allowlist in code. Never an import path resolved from config: a
profile config is a document people paste between machines, and a
``_target_``-style import in one would be arbitrary code execution.

Curator's filters do not share one shape. Some are upper bounds, some lower,
and two are intervals over a single score, so "sweep the threshold" means
different things for different signals. That variation is what this module
records, and it is why claiming to "reuse all ~35 filters" without an adapter
layer would be an overclaim.

Every factory imports ``nemo_curator`` lazily. The module has to stay importable
on a plain CI host so its metadata can be tested without the framework present.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

IMPL_VERSION = "curate-runtime-0.2.0"

Direction = Literal["min", "max", "interval", "categorical"]


class SignalRequirementsUnmetError(ValueError):
    """A named signal needs capabilities the loaded language pack lacks."""


class UnknownSignalError(KeyError):
    """A config named a signal that is not in the allowlist."""


@dataclass(frozen=True)
class Grid:
    """Threshold values to sweep for a one-sided signal."""

    lo: float
    hi: float
    points: int = 64
    integral: bool = False

    def values(self) -> list[float]:
        if self.points < 2:
            raise ValueError("a grid needs at least two points")
        step = (self.hi - self.lo) / (self.points - 1)
        raw = [self.lo + step * i for i in range(self.points)]
        if not self.integral:
            return raw
        return sorted({int(round(v)) for v in raw})


@dataclass(frozen=True)
class IntervalGrid:
    """Threshold pairs for a signal gated from both sides.

    A single grid cannot express ``word_count(min, max)``: the retention of a
    lower bound depends on where the upper bound sits. The sweep is a matrix,
    and the report renders it as a surface plus per-axis marginals.
    """

    lo_grid: Grid
    hi_grid: Grid


@dataclass(frozen=True)
class Signal:
    """One profileable quality signal and how to sweep it."""

    name: str
    factory: Callable[..., Any]
    direction: Direction
    units: str
    grid: Grid | IntervalGrid
    threshold_params: tuple[str, ...]
    curator_default: tuple[float, ...] = ()
    requires: tuple[str, ...] = ()
    impl_version: str = IMPL_VERSION
    notes: str = ""
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction == "interval":
            if not isinstance(self.grid, IntervalGrid):
                raise TypeError(
                    f"{self.name}: an interval signal needs an IntervalGrid. A one-dimensional "
                    "sweep cannot express a two-sided gate, and reporting one as if it could "
                    "would understate what the upper bound removes."
                )
            if len(self.threshold_params) != 2:
                raise TypeError(f"{self.name}: an interval signal needs two threshold parameters")
        else:
            if isinstance(self.grid, IntervalGrid):
                raise TypeError(f"{self.name}: only interval signals take an IntervalGrid")
            if len(self.threshold_params) != 1:
                raise TypeError(f"{self.name}: a one-sided signal needs exactly one threshold parameter")

    def build(self, *thresholds: float, **overrides: Any) -> Any:
        """Construct the underlying Curator filter at the given threshold(s).

        ``overrides`` carries anything only the caller can supply — a tokenizer
        object, say — which cannot live in a module-level table.
        """
        if len(thresholds) != len(self.threshold_params):
            raise ValueError(f"{self.name} takes {len(self.threshold_params)} threshold(s), got {len(thresholds)}")
        kwargs = dict(zip(self.threshold_params, thresholds))
        kwargs.update(self.extra_kwargs)
        kwargs.update(overrides)
        return self.factory(**kwargs)


# -- lazy factories -----------------------------------------------------------
#
# Import inside the call so this module loads without nemo_curator installed.
# Curator's filters/__init__ exports only DocumentFilter/Filter/Score/ScoreFilter,
# so the concrete classes come from their own modules.


def _string(name: str) -> Callable[..., Any]:
    def factory(**kwargs: Any) -> Any:
        from nemo_curator.stages.text.filters.heuristic import string as mod

        return getattr(mod, name)(**kwargs)

    factory.__name__ = f"build_{name}"
    return factory


def _repetition(name: str) -> Callable[..., Any]:
    def factory(**kwargs: Any) -> Any:
        from nemo_curator.stages.text.filters.heuristic.repetition import repetition as mod

        return getattr(mod, name)(**kwargs)

    factory.__name__ = f"build_{name}"
    return factory


def _token_count(**kwargs: Any) -> Any:
    from nemo_curator.stages.text.filters.token.token_count import TokenCountFilter

    return TokenCountFilter(**kwargs)


def _local(class_name: str) -> Callable[..., Any]:
    """A signal Curator does not provide, parameterised by a language pack."""

    def factory(**kwargs: Any) -> Any:
        from nemotron.steps.curate.runtime import signals as local

        return getattr(local, class_name)(**kwargs)

    factory.__name__ = f"build_{class_name}"
    return factory


# -- the allowlist ------------------------------------------------------------
#
# PR A registers only signals Curator already ships, so every `requires` is
# empty. The capability machinery is exercised from the first release anyway,
# because adding language-pack signals later must not change how resolution
# behaves.

SIGNALS: dict[str, Signal] = {
    "non_alpha_numeric": Signal(
        name="non_alpha_numeric",
        curator_default=(0.25,),
        factory=_string("NonAlphaNumericFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_non_alpha_numeric_to_text_ratio",),
        notes=(
            "Curator's implementation counts only [a-zA-Z0-9\\n?!,.] as alphanumeric, so any "
            "script outside Latin-ASCII scores high. Profile it before trusting the 0.25 default."
        ),
    ),
    "symbol_to_word": Signal(
        name="symbol_to_word",
        curator_default=(0.1,),
        factory=_string("SymbolsToWordsFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_symbol_to_word_ratio",),
    ),
    "numbers_ratio": Signal(
        name="numbers_ratio",
        curator_default=(0.15,),
        factory=_string("NumbersFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_number_to_text_ratio",),
        notes=(
            "Counts [0-9] only. Devanagari and other non-ASCII digits are Unicode category Nd "
            "and satisfy str.isdigit(), but do not match, so this reads 0 on such text."
        ),
    ),
    "urls_ratio": Signal(
        name="urls_ratio",
        curator_default=(0.2,),
        factory=_string("UrlsFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_url_to_text_ratio",),
    ),
    "bullet_ratio": Signal(
        name="bullet_ratio",
        curator_default=(0.9,),
        factory=_string("BulletsFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_bullet_lines_ratio",),
    ),
    "white_space": Signal(
        name="white_space",
        curator_default=(0.25,),
        factory=_string("WhiteSpaceFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_white_space_ratio",),
    ),
    "parentheses_ratio": Signal(
        name="parentheses_ratio",
        curator_default=(0.1,),
        factory=_string("ParenthesesFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_parentheses_ratio",),
    ),
    "max_word_length": Signal(
        name="max_word_length",
        curator_default=(1000,),
        factory=_string("LongWordFilter"),
        direction="max",
        units="characters",
        grid=Grid(10, 5000, 64, integral=True),
        threshold_params=("max_word_length",),
    ),
    "punctuation": Signal(
        name="punctuation",
        curator_default=(0.85,),
        factory=_string("PunctuationFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_num_sentences_without_endmark_ratio",),
        notes=(
            "Scores the fraction of sentences NOT ending in an end mark, so the gate is an "
            "upper bound. End marks are '.', '!', '?' — a script using another terminator, "
            "such as the Devanagari danda, scores 1.0 on correct text."
        ),
    ),
    "ellipsis": Signal(
        name="ellipsis",
        curator_default=(0.3,),
        factory=_string("EllipsisFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_num_lines_ending_with_ellipsis_ratio",),
    ),
    "words_with_alphabets": Signal(
        name="words_with_alphabets",
        curator_default=(0.8,),
        factory=_string("WordsWithoutAlphabetsFilter"),
        direction="min",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("min_words_with_alphabets",),
    ),
    "repeating_duplicate_ngrams": Signal(
        name="repeating_duplicate_ngrams",
        curator_default=(0.2,),
        factory=_repetition("RepeatingDuplicateNGramsFilter"),
        direction="max",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),
        threshold_params=("max_repeating_duplicate_ngram_ratio",),
        extra_kwargs={"n": 2},
    ),
    "word_count": Signal(
        name="word_count",
        curator_default=(50, 100000),
        factory=_string("WordCountFilter"),
        direction="interval",
        units="words",
        grid=IntervalGrid(
            lo_grid=Grid(0, 200, 32, integral=True),
            hi_grid=Grid(1000, 20000, 32, integral=True),
        ),
        threshold_params=("min_words", "max_words"),
    ),
    "mean_word_length": Signal(
        name="mean_word_length",
        curator_default=(3, 10),
        factory=_string("MeanWordLengthFilter"),
        direction="interval",
        units="characters",
        grid=IntervalGrid(
            lo_grid=Grid(1, 8, 8, integral=True),
            hi_grid=Grid(8, 30, 23, integral=True),
        ),
        threshold_params=("min_mean_word_length", "max_mean_word_length"),
    ),
    "token_count": Signal(
        name="token_count",
        factory=_token_count,
        direction="interval",
        units="tokens",
        grid=IntervalGrid(
            lo_grid=Grid(0, 512, 33, integral=True),
            hi_grid=Grid(1024, 32768, 32, integral=True),
        ),
        threshold_params=("min_tokens", "max_tokens"),
        requires=("tokenizer",),
        notes=(
            "Needs a tokenizer: set models.tokenizer in config. No curator_default is "
            "recorded because Curator's shipped bounds are (0, inf), which keep everything "
            "and would report a meaningless 100% retention."
        ),
    ),
}


# -- signals Curator does not provide ----------------------------------------
#
# Every one is parameterised by a language pack: the character sets, word lists
# and patterns arrive as data, and `requires` names the pack capability each one
# depends on. A pack that does not declare a capability makes its signals absent
# from the report rather than computed on a false premise.

SIGNALS.update(
    {
        "unicode_alpha_numeric": Signal(
            name="unicode_alpha_numeric",
            factory=_local("UnicodeAwareNonAlphaNumericFilter"),
            direction="max",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("max_non_alpha_numeric_to_text_ratio",),
            curator_default=(0.25,),
            notes=(
                "Unicode-correct replacement for non_alpha_numeric: accepts categories L, N and "
                "all of M. Curator's version is ASCII-only, which inverts the signal for "
                "Vietnamese and rejects Devanagari outright."
            ),
        ),
        "script_ratio": Signal(
            name="script_ratio",
            factory=_local("ScriptRatio"),
            direction="min",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("min_script_ratio",),
            requires=("script_ratio",),
            notes="Continuous, unlike HistogramFilter, which returns 0 or 1 and cannot be swept.",
        ),
        "latin_ratio": Signal(
            name="latin_ratio",
            factory=_local("LatinRatio"),
            direction="max",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("max_latin_ratio",),
            requires=("script_ratio",),
        ),
        "foreign_script_ratio": Signal(
            name="foreign_script_ratio",
            factory=_local("ForeignScriptRatio"),
            direction="max",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("max_foreign_ratio",),
            requires=("script_ratio",),
        ),
        "diacritic_ratio": Signal(
            name="diacritic_ratio",
            factory=_local("DiacriticRatio"),
            direction="min",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("min_diacritic_ratio",),
            requires=("diacritic_ratio",),
            notes=(
                "Only meaningful where marks are removable. Devanagari matras are obligatory "
                "vowels, so the Hindi pack does not declare this capability."
            ),
        ),
        "stopword_ratio": Signal(
            name="stopword_ratio",
            factory=_local("StopwordRatio"),
            direction="min",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("min_stopword_ratio",),
            requires=("stopword_ratio",),
        ),
        "stopword_ratio_folded": Signal(
            name="stopword_ratio_folded",
            factory=_local("FoldedStopwordRatio"),
            direction="min",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("min_stopword_ratio_folded",),
            requires=("stopword_ratio_folded",),
            notes=(
                "Recovers text written without marks, which a mark-sensitive gate deletes "
                "outright. Noisier by construction: folding merges distinct words."
            ),
        ),
        "boilerplate_hits": Signal(
            name="boilerplate_hits",
            factory=_local("BoilerplateHits"),
            direction="max",
            units="patterns",
            grid=Grid(0, 20, 21, integral=True),
            threshold_params=("max_boilerplate_hits",),
            requires=("boilerplate_hits",),
        ),
        "sentence_end_ratio": Signal(
            name="sentence_end_ratio",
            factory=_local("SentenceEndRatio"),
            direction="min",
            units="ratio",
            grid=Grid(0.0, 1.0, 64),
            threshold_params=("min_sentence_end_ratio",),
            requires=("sentence_end_ratio",),
            notes=(
                "Terminators come from the pack. Curator's PunctuationFilter looks for '.', '!' "
                "and '?', so a correct Hindi paragraph, which ends with the danda, scores zero."
            ),
        ),
    }
)

#: Signals that need the loaded language pack handed to their constructor.
PACK_SIGNALS = frozenset(
    {
        "script_ratio",
        "latin_ratio",
        "foreign_script_ratio",
        "diacritic_ratio",
        "stopword_ratio",
        "stopword_ratio_folded",
        "boilerplate_hits",
        "sentence_end_ratio",
    }
)


#: Curator filters deliberately left out, and why. Kept in code so the decision
#: is visible to the next person rather than looking like an oversight.
EXCLUDED: dict[str, str] = {
    "CommonEnglishWordsFilter": (
        "hardcodes eight English function words and get_word_splitter('en'); it reports nothing on other languages"
    ),
    "BoilerPlateStringFilter": ("policy_substrings are English cookie/privacy phrases; it matches nothing elsewhere"),
    "RepeatedLinesFilter": (
        "keep_document is `score >= cutoff` while the parameter is named "
        "max_repeated_line_fraction. Until that is resolved upstream the direction cannot be "
        "described honestly in a report meant to inform threshold choices"
    ),
    "RepeatedParagraphsFilter": "same direction ambiguity as RepeatedLinesFilter",
    "RepeatedLinesByCharFilter": "same direction ambiguity as RepeatedLinesFilter",
    "RepeatedParagraphsByCharFilter": "same direction ambiguity as RepeatedLinesFilter",
    "RepeatingTopNGramsFilter": (
        "score_document breaks frequency ties by `max()` over a set of n-gram tuples, so the "
        "winning n-gram — and the score — depends on PYTHONHASHSEED. Measured on Vietnamese "
        "Wikipedia, 9-12% of documents score differently between two processes on identical "
        "input. A threshold chosen from one profile does not reproduce in the next, which is "
        "the one thing this step exists to provide"
    ),
    "HistogramFilter": (
        "returns a binary 0/1 rather than the underlying ratio, so a threshold sweep over it "
        "is meaningless; it also downloads a histogram at construction time"
    ),
    "FastTextQualityFilter": (
        "keep_document draws from a Pareto distribution, so retention is not a function of the threshold"
    ),
    "PornographicUrlsFilter": "binary, not a swept threshold",
    "SubstringFilter": "binary, and its parameter is a substring rather than a threshold",
}


def resolve(
    requested: Sequence[str] | None,
    capabilities: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[Signal], list[str]]:
    """Choose which signals to profile.

    Auto-selection and explicit naming are treated differently on purpose. When
    the caller lists nothing, a signal the pack cannot support is skipped with a
    warning — the run should still produce a report. When the caller names a
    signal, silently dropping it would answer a question they did not ask, so
    that case fails.

    Returns the signals to run and any warnings to carry into the report.
    """
    capabilities = frozenset(capabilities)
    warnings: list[str] = []

    if requested:
        unknown = [name for name in requested if name not in SIGNALS]
        if unknown:
            raise UnknownSignalError(
                f"unknown signal(s) {unknown}. Registered: {sorted(SIGNALS)}. "
                "Signals are a closed allowlist; config cannot name an import path."
            )
        chosen = []
        for name in requested:
            signal = SIGNALS[name]
            missing = tuple(c for c in signal.requires if c not in capabilities)
            if missing:
                raise SignalRequirementsUnmetError(
                    f"{name} requires {list(missing)}, which is not available. "
                    "Supply it or remove the signal from the list."
                )
            chosen.append(signal)
        return chosen, warnings

    chosen = []
    for name in sorted(SIGNALS):
        signal = SIGNALS[name]
        missing = tuple(c for c in signal.requires if c not in capabilities)
        if missing:
            warnings.append(f"skipped {name}: requires {list(missing)}")
            continue
        chosen.append(signal)
    return chosen, warnings
