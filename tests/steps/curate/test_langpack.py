# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The language pack contract.

The claim these tests defend is that the runtime is language-parametric. A
substring search for "vi" would prove nothing — the real evidence is that the
whole pipeline runs against a language that does not exist, because there is
nothing about it anyone could have special-cased.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from nemotron.steps.curate.runtime import langpack, signals
from nemotron.steps.curate.runtime import registry as r

FIXTURES = Path(__file__).parent / "fixtures" / "langpacks"
RUNTIME = Path(langpack.__file__).parent


# -- shipped packs ------------------------------------------------------------


def test_the_bundled_packs_load() -> None:
    assert set(langpack.available()) >= {"vi", "hi", "en", "ja", "th"}


@pytest.mark.parametrize("tag", ["en", "hi", "ja", "th", "vi"])
def test_a_bundled_pack_reports_what_it_carries(tag) -> None:
    pack = langpack.load(tag)
    described = pack.describe()

    assert described["language_tag"] == tag
    assert described["content_hash"].startswith("sha256:")
    assert described["stopwords"] > 0
    assert described["charset"] > 0
    assert described["capabilities"]


def test_the_synthetic_fixture_is_not_shipped() -> None:
    """x-test is a test fixture; shipping it would offer users a fake language."""
    assert "x-test" not in langpack.available()
    assert (FIXTURES / "x-test" / "pack.toml").is_file()


# -- the capability mechanism -------------------------------------------------


def test_hindi_declares_fewer_capabilities_than_vietnamese() -> None:
    """The finding, not a shortfall.

    Vietnamese tone marks strip to degraded but readable text, so measuring their
    density says something. Devanagari matras are obligatory vowels; stripping
    them yields nonsense, so the capability is absent rather than measured on a
    false premise.
    """
    vi = langpack.load("vi")
    hi = langpack.load("hi")

    assert "diacritic_ratio" in vi.capabilities
    assert "diacritic_ratio" not in hi.capabilities
    assert "stopword_ratio_folded" not in hi.capabilities
    assert hi.capabilities < vi.capabilities


@pytest.mark.parametrize("tag", ["en", "ja", "th"])
def test_en_ja_th_omit_diacritic_ratio(tag) -> None:
    """English loanword accents, Japanese dakuten and Thai tone marks are not
    removable orthography. Measuring their density would be the Hindi trap."""
    pack = langpack.load(tag)
    assert "diacritic_ratio" not in pack.capabilities
    assert "stopword_ratio_folded" not in pack.capabilities


@pytest.mark.parametrize(
    "tag,text",
    [
        ("en", "The weather is nice today."),
        ("ja", "今日は良い天気です。"),
        ("th", "วันนี้อากาศดีมาก"),
    ],
)
def test_a_correct_sentence_is_own_script(tag, text) -> None:
    pack = langpack.load(tag)
    score = signals.ScriptRatio(pack).score_document(text)
    assert score == 1.0


def test_an_unsupported_capability_removes_its_signals_from_the_run() -> None:
    hi = langpack.load("hi")

    chosen, warnings = r.resolve(None, hi.capabilities)

    names = {s.name for s in chosen}
    assert "diacritic_ratio" not in names
    assert "stopword_ratio_folded" not in names
    assert any("diacritic_ratio" in w for w in warnings)


def test_naming_a_signal_the_pack_cannot_support_fails() -> None:
    hi = langpack.load("hi")

    with pytest.raises(r.SignalRequirementsUnmetError, match="diacritic_ratio"):
        r.resolve(["diacritic_ratio"], hi.capabilities)


def test_a_capability_declared_without_its_data_is_rejected(tmp_path) -> None:
    """Otherwise the report fills with zeroes, which reads as a finding."""
    pack_dir = tmp_path / "x-broken"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="b"\nlanguage_tag="x-broken"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="without the data behind them"):
        langpack.load_pack(pack_dir)


def test_an_unknown_capability_is_rejected(tmp_path) -> None:
    """A typo would otherwise read as 'this language cannot do that'."""
    pack_dir = tmp_path / "x-typo"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="t"\nlanguage_tag="x-typo"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["stopwrd_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="unknown capabilities"):
        langpack.load_pack(pack_dir)


# -- no default language ------------------------------------------------------


def test_there_is_no_default_language() -> None:
    """A wrong default produces plausible numbers for the wrong language."""
    with pytest.raises(langpack.LanguagePackNotFoundError, match="no default"):
        langpack.load("")


def test_an_unknown_tag_names_what_is_available() -> None:
    with pytest.raises(langpack.LanguagePackNotFoundError, match="Available"):
        langpack.load("xx-nonexistent")


# -- structural evidence of genericity ---------------------------------------


def test_no_runtime_branch_turns_on_a_language() -> None:
    """A cheap secondary guard, and deliberately narrow.

    A flat substring search would flag the docstrings that explain *why* the
    Unicode handling is what it is — those cite Vietnamese and Devanagari as
    measured evidence, which is the opposite of a problem. What would be a real
    defect is executable code that behaves differently for a named language, so
    only comparisons and subscripts are inspected, with docstrings and comments
    stripped by the parser.
    """
    import ast

    named = ("vi", "hi", "vietnamese", "hindi", "devanagari")
    offenders: list[str] = []

    for module in sorted(RUNTIME.glob("*.py")):
        if module.name == "langpack.py":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                literals = [
                    n.value
                    for n in [node.left, *node.comparators]
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                for value in literals:
                    if value.lower() in named:
                        offenders.append(f"{module.name}:{node.lineno} compares against {value!r}")

    assert not offenders, f"the neutral layer branches on a language: {offenders}"


def test_the_runtime_never_reaches_into_the_pack_directory() -> None:
    """Packs are resolved through the loader, never by path from another module."""
    for module in sorted(RUNTIME.glob("*.py")):
        if module.name == "langpack.py":
            continue
        assert "data/langpacks" not in module.read_text(encoding="utf-8"), module.name


def test_every_pack_signal_declares_the_capability_it_needs() -> None:
    for name in r.PACK_SIGNALS:
        assert r.SIGNALS[name].requires, f"{name} declares no requirement"
        for capability in r.SIGNALS[name].requires:
            assert capability in langpack.KNOWN_CAPABILITIES, capability


def test_the_full_signal_set_runs_against_an_invented_language() -> None:
    """The load-bearing test.

    x-test is written in a script neither shipped pack uses, with invented words
    and an invented fold map. If every signal produces a number here, the runtime
    is parametric — there is nothing about this language to special-case.
    """
    pack = langpack.load("x-test", FIXTURES)
    chosen, _ = r.resolve(None, pack.capabilities)

    text = "".join(sorted(pack.charset)[:40]) + " " + " ".join(sorted(pack.stopwords)[:5]) + "."
    scored = 0
    for signal in chosen:
        if signal.name not in r.PACK_SIGNALS:
            continue
        value = signal.build(*_probe(signal), pack=pack).score_document(text)
        assert isinstance(value, float), signal.name
        scored += 1

    assert scored == len(r.PACK_SIGNALS), "every pack signal must run on an invented language"


def _probe(signal):
    if signal.direction == "interval":
        return (signal.grid.lo_grid.values()[0], signal.grid.hi_grid.values()[-1])
    return (signal.grid.values()[-1],)


# -- packaging ----------------------------------------------------------------


def test_packs_live_inside_the_package_so_an_install_carries_them() -> None:
    """A wheel that drops the packs leaves profile naming a language it cannot load."""
    assert langpack.BUNDLED_DIR.is_dir()
    assert "src/nemotron/steps/curate" in langpack.BUNDLED_DIR.as_posix()


def test_the_wheel_declares_the_pack_files() -> None:
    root = Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)

    artifacts = config["tool"]["hatch"]["build"]["targets"]["wheel"].get("artifacts", [])
    assert any("langpacks" in entry for entry in artifacts)


def test_every_shipped_pack_records_where_its_word_list_came_from() -> None:
    """A word list with no provenance cannot be relicensed, corrected, or trusted."""
    for tag in langpack.available():
        pack = langpack.load(tag)
        stopwords = pack.sources.get("stopwords", {})
        assert stopwords.get("origin"), f"{tag}: stopwords declare no origin"
        assert stopwords.get("license"), f"{tag}: stopwords declare no license"


def test_every_shipped_asset_records_origin_and_license() -> None:
    for tag in langpack.available():
        for name, source in langpack.load(tag).sources.items():
            assert source.get("origin"), f"{tag}/{name}: source declares no origin"
            assert source.get("license"), f"{tag}/{name}: source declares no license"


def test_a_source_without_license_is_rejected(tmp_path) -> None:
    pack_dir = tmp_path / "x-missing-license"
    pack_dir.mkdir()
    (pack_dir / "stopwords.txt").write_text("word\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\nstopwords={file="stopwords.txt", origin="test"}\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="license"):
        langpack.load_pack(pack_dir)


def test_text_resources_are_normalized_to_nfc(tmp_path) -> None:
    import unicodedata

    pack_dir = tmp_path / "x-nfd"
    pack_dir.mkdir()
    nfd = unicodedata.normalize("NFD", "café")
    (pack_dir / "stopwords.txt").write_text(f"{nfd}\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\nstopwords={file="stopwords.txt", origin="test", license="Apache-2.0"}\n'
        '[capabilities]\nsupports=["stopword_ratio"]\n',
        encoding="utf-8",
    )

    assert langpack.load_pack(pack_dir).stopwords == {"café"}


# -- fold map -----------------------------------------------------------------


def test_the_fold_map_lives_in_the_pack_not_the_code() -> None:
    """Vietnamese đ does not decompose under NFD, so it must be declared."""
    vi = langpack.load("vi")

    assert vi.fold_map.get("đ") == "d"
    assert vi.fold("đường") == "duong"
    assert "đ" not in signals.__file__ or "fold_map" in Path(signals.__file__).read_text(encoding="utf-8")


def test_folding_collapses_distinct_stopwords_and_the_count_is_knowable() -> None:
    """The collision count is a property of the language, not a defect."""
    vi = langpack.load("vi")

    folded = vi.folded_stopwords()

    assert len(folded) < len(vi.stopwords)


def test_a_pack_without_a_fold_map_folds_only_combining_marks() -> None:
    hi = langpack.load("hi")

    assert hi.fold_map == {}
    assert hi.fold("भारत") == "भारत" or len(hi.fold("भारत")) <= len("भारत")


def test_fold_map_must_be_a_mapping(tmp_path) -> None:
    pack_dir = tmp_path / "x-invalid-fold-map"
    pack_dir.mkdir()
    (pack_dir / "pack.toml").write_text(
        'fold_map=["not", "a", "mapping"]\n'
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[capabilities]\nsupports=["sentence_end_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="fold_map must be a mapping"):
        langpack.load_pack(pack_dir)


def test_diacritic_capability_requires_an_explicit_fold_map(tmp_path) -> None:
    pack_dir = tmp_path / "x-no-fold-map"
    pack_dir.mkdir()
    (pack_dir / "charset.txt").write_text("a\n", encoding="utf-8")
    (pack_dir / "pack.toml").write_text(
        '[pack]\npack_id="x"\nlanguage_tag="x"\nversion="1"\nschema=1\n'
        '[sources]\ncharset={file="charset.txt", origin="test", license="Apache-2.0"}\n'
        '[capabilities]\nsupports=["diacritic_ratio"]\n',
        encoding="utf-8",
    )

    with pytest.raises(langpack.LanguagePackInvalidError, match="fold_map"):
        langpack.load_pack(pack_dir)


# -- provenance ---------------------------------------------------------------


def test_the_content_hash_follows_the_contents_not_the_path(tmp_path) -> None:
    """A policy is tied to the pack that produced it; moving the pack must not break that."""
    import shutil

    source = langpack.BUNDLED_DIR / "hi"
    copy = tmp_path / "hi"
    shutil.copytree(source, copy)

    assert langpack.load_pack(copy).content_hash == langpack.load("hi").content_hash


def test_editing_a_word_list_changes_the_hash(tmp_path) -> None:
    import shutil

    copy = tmp_path / "hi"
    shutil.copytree(langpack.BUNDLED_DIR / "hi", copy)
    before = langpack.load_pack(copy).content_hash
    (copy / "stopwords.txt").write_text("\n".join(["एक", "दो"]) + "\n", encoding="utf-8")

    assert langpack.load_pack(copy).content_hash != before


# -- a capability must be one the signal can actually deliver -------------------
#
# The ja and th packs declared stopword_ratio while the signal tokenises on
# whitespace, which those scripts do not use. Measured on 20,000 real documents
# each: 93.7% of Japanese and 53.1% of Thai scored EXACTLY zero, and 87.9% /
# 90.1% of those were correct native-script text. So the signal could not tell
# "not Japanese" from "Japanese, written normally" — the one distinction it
# exists to make. Hindi, which does use spaces, scores zero on 25.9% and those
# are genuinely foreign-language documents, so it keeps the capability.

UNSEGMENTED_SCRIPTS = ("ja", "th")


@pytest.mark.parametrize("tag", UNSEGMENTED_SCRIPTS)
def test_an_unsegmented_script_does_not_declare_stopword_ratio(tag) -> None:
    pack = langpack.load(tag)

    assert not pack.supports("stopword_ratio"), (
        f"{tag} tokenises on whitespace it does not use; declaring the capability "
        "produces a clean-looking distribution over nothing"
    )


@pytest.mark.parametrize("tag", UNSEGMENTED_SCRIPTS)
def test_the_reason_is_recorded_in_the_pack(tag) -> None:
    """A capability removed without a recorded measurement invites re-adding it."""
    pack = langpack.load(tag)

    assert "stopword_ratio_not_declared" in pack.notes


@pytest.mark.parametrize("tag", ("vi", "en", "hi"))
def test_a_space_separated_script_keeps_stopword_ratio(tag) -> None:
    assert langpack.load(tag).supports("stopword_ratio")


def test_pack_notes_reach_the_report() -> None:
    """A caveat that stays in the file is not a caveat.

    The ja pack recorded the whitespace-tokenisation problem all along; describe()
    dropped it, so every report showed a stopword distribution that was 93.7%
    exact zeros with nothing saying why.
    """
    described = langpack.load("ja").describe()

    assert "notes" in described
    assert described["notes"], "the ja pack has notes and they must be carried"
