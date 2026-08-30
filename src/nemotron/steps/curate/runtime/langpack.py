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

"""Language packs: everything a signal needs to know about one language.

The runtime holds no language-specific code. A pack is data — word lists,
character sets, regex patterns, a fold map — plus a declaration of which
capabilities it can support. Adding a language means adding a directory.

The capability declaration is the load-bearing part, and it is not bureaucracy.
Vietnamese diacritics carry tone and can be stripped to yield degraded but
readable text, so a diacritic ratio measures something. Devanagari matras are
obligatory vowels; stripping them yields nonsense, so the Hindi pack simply does
not declare that capability and every figure derived from it is absent from the
Hindi report rather than silently computed on a meaningless basis.

The fold map lives here rather than in code for the same reason. Vietnamese
``đ`` does not decompose under NFD, so it has to be declared somewhere — and
declaring it in the runtime would put one language inside the language-neutral
layer.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

SCHEMA_VERSION = 1

BUNDLED = "bundled"

#: Directory holding the packs that ship with this step.
BUNDLED_DIR = Path(__file__).resolve().parents[1] / "data" / "langpacks"

#: Every capability a pack may declare. A pack naming something outside this set
#: is rejected: a typo would otherwise read as "this language cannot do that".
KNOWN_CAPABILITIES = frozenset(
    {
        "script_ratio",
        "diacritic_ratio",
        "stopword_ratio",
        "stopword_ratio_folded",
        "boilerplate_hits",
        "sentence_end_ratio",
    }
)

DEFAULT_SENTENCE_TERMINATORS = (".", "!", "?")


class LanguagePackNotFoundError(FileNotFoundError):
    """No pack for the requested tag."""


class LanguagePackInvalidError(ValueError):
    """A pack exists but does not meet the contract."""


@dataclass(frozen=True)
class LanguagePack:
    """One language's data, loaded and validated."""

    pack_id: str
    language_tag: str
    version: str
    capabilities: frozenset[str]
    stopwords: frozenset[str]
    charset: frozenset[str]
    boilerplate: tuple[re.Pattern[str], ...]
    fold_map: dict[str, str]
    sentence_terminators: tuple[str, ...]
    content_hash: str
    origin: Path
    sources: dict[str, Any] = field(default_factory=dict)
    #: The pack's own ``[notes]`` table. Carried into every report rather than
    #: left in the file: a caveat a reader never sees is not a caveat. The ja
    #: pack recorded that whitespace tokenisation undercounts Japanese function
    #: words, and that note stayed on disk while the report showed a clean-looking
    #: stopword distribution that was 93.7% exact zeros.
    notes: dict[str, str] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def describe(self) -> dict[str, Any]:
        """What the report records about which pack produced its numbers."""
        return {
            "pack_id": self.pack_id,
            "language_tag": self.language_tag,
            "version": self.version,
            "content_hash": self.content_hash,
            "capabilities": sorted(self.capabilities),
            "stopwords": len(self.stopwords),
            "charset": len(self.charset),
            "boilerplate_patterns": len(self.boilerplate),
            "fold_map_entries": len(self.fold_map),
            "sentence_terminators": list(self.sentence_terminators),
            "notes": dict(self.notes),
        }

    def fold(self, text: str) -> str:
        """Remove the marks this language treats as removable.

        Applies the pack's explicit fold map first — for characters that do not
        decompose, such as Vietnamese ``đ`` — then drops combining marks.
        """
        for source, target in self.fold_map.items():
            text = text.replace(source, target)
        return "".join(
            ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch)
        )

    def folded_stopwords(self) -> frozenset[str]:
        """The stopword list with marks removed, for un-marked text.

        Folding collapses distinct words together, so this set is smaller than
        the original. Callers that report on it should say so: the collision
        count is a property of the language, not a defect in the list.
        """
        return frozenset(self.fold(word) for word in self.stopwords)


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise LanguagePackInvalidError(f"pack file is missing: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _content_hash(directory: Path, files: Iterable[Path]) -> str:
    """Hash the pack's contents so a policy can be tied to the pack that made it.

    Order-independent, and over file contents rather than paths, so moving the
    pack does not change it and editing a word list does.
    """
    lines = []
    for path in sorted(files):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{path.relative_to(directory)}\t{digest}")
    joined = "\n".join(sorted(lines))
    return f"sha256:{hashlib.sha256(joined.encode('utf-8')).hexdigest()}"


def load_pack(directory: str | Path) -> LanguagePack:
    """Load and validate one pack directory."""
    root = Path(directory)
    manifest_path = root / "pack.toml"
    if not manifest_path.is_file():
        raise LanguagePackNotFoundError(f"no pack.toml under {root}")

    with manifest_path.open("rb") as handle:
        raw = tomllib.load(handle)

    pack = raw.get("pack") or {}
    if pack.get("schema") != SCHEMA_VERSION:
        raise LanguagePackInvalidError(
            f"{manifest_path}: pack.schema must be {SCHEMA_VERSION}, got {pack.get('schema')!r}"
        )
    for key in ("pack_id", "language_tag", "version"):
        if not pack.get(key):
            raise LanguagePackInvalidError(f"{manifest_path}: pack.{key} is required")

    declared = raw.get("capabilities", {}).get("supports")
    if not isinstance(declared, list) or not declared:
        raise LanguagePackInvalidError(f"{manifest_path}: capabilities.supports must be a non-empty list")
    unknown = sorted(set(declared) - KNOWN_CAPABILITIES)
    if unknown:
        raise LanguagePackInvalidError(
            f"{manifest_path}: unknown capabilities {unknown}. Known: {sorted(KNOWN_CAPABILITIES)}. "
            "A typo here would read as 'this language cannot do that'."
        )
    capabilities = frozenset(declared)

    sources = raw.get("sources") or {}
    files = [manifest_path]

    def _load(name: str) -> list[str]:
        entry = sources.get(name)
        if not entry:
            return []
        path = root / entry["file"]
        files.append(path)
        return _read_lines(path)

    stopwords = frozenset(_load("stopwords"))
    charset = frozenset("".join(_load("charset")))
    boilerplate_raw = _load("boilerplate")

    patterns = []
    for pattern in boilerplate_raw:
        try:
            patterns.append(re.compile(pattern))
        except re.error as exc:
            raise LanguagePackInvalidError(f"{root}: boilerplate pattern {pattern!r} does not compile: {exc}") from exc

    fold_map = {str(k): str(v) for k, v in (raw.get("fold_map") or {}).items()}
    terminators = tuple(
        (raw.get("orthography") or {}).get("sentence_terminators") or DEFAULT_SENTENCE_TERMINATORS
    )

    loaded = LanguagePack(
        pack_id=pack["pack_id"],
        language_tag=pack["language_tag"],
        version=str(pack["version"]),
        capabilities=capabilities,
        stopwords=stopwords,
        charset=charset,
        boilerplate=tuple(patterns),
        fold_map=fold_map,
        sentence_terminators=terminators,
        content_hash=_content_hash(root, files),
        origin=root,
        sources=sources,
        notes={str(k): str(v) for k, v in (raw.get("notes") or {}).items()},
    )

    _assert_capabilities_are_backed(loaded, manifest_path)
    return loaded


#: What each capability needs the pack to actually carry. Declaring a capability
#: without the data behind it would produce a report full of zeroes rather than
#: an error, which is the failure mode this check exists to prevent.
CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "script_ratio": ("charset",),
    "diacritic_ratio": ("charset",),
    "stopword_ratio": ("stopwords",),
    "stopword_ratio_folded": ("stopwords", "fold_map"),
    "boilerplate_hits": ("boilerplate",),
    "sentence_end_ratio": ("sentence_terminators",),
}


def _assert_capabilities_are_backed(pack: LanguagePack, manifest_path: Path) -> None:
    missing: list[str] = []
    for capability in sorted(pack.capabilities):
        for requirement in CAPABILITY_REQUIREMENTS.get(capability, ()):
            if not getattr(pack, requirement):
                missing.append(f"{capability} needs {requirement}")
    if missing:
        raise LanguagePackInvalidError(
            f"{manifest_path}: capabilities declared without the data behind them: {missing}"
        )


def resolve_dir(langpack_dir: str | Path | None) -> Path:
    """Where to look for packs. ``'bundled'`` means the ones that ship here."""
    if langpack_dir in (None, "", BUNDLED):
        return BUNDLED_DIR
    return Path(langpack_dir)


def load(language_tag: str, langpack_dir: str | Path | None = BUNDLED) -> LanguagePack:
    """Load the pack for a BCP-47 tag.

    There is deliberately no default language. A wrong default silently produces
    wrong numbers for a corpus, which is worse than an error.
    """
    if not language_tag:
        raise LanguagePackNotFoundError(
            "language is required and has no default: a wrong default silently produces "
            "wrong numbers. Set it to the BCP-47 tag of the corpus."
        )

    root = resolve_dir(langpack_dir)
    candidate = root / language_tag
    if candidate.is_dir():
        return load_pack(candidate)

    available = sorted(p.name for p in root.iterdir() if (p / "pack.toml").is_file()) if root.is_dir() else []
    raise LanguagePackNotFoundError(
        f"no pack for {language_tag!r} under {root}. Available: {available or 'none'}. "
        "See data/langpacks/SPEC.md to author one."
    )


def available(langpack_dir: str | Path | None = BUNDLED) -> list[str]:
    """Tags with a pack in the given directory."""
    root = resolve_dir(langpack_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "pack.toml").is_file())
