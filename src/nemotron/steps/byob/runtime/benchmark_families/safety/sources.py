# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy source inventory shared by every safety stage.

A source is named by its path relative to input_dir in POSIX form, for example
``sub/card.md``. Hidden files and directories are not sources. Symbolic links are
refused so the source set cannot reach outside input_dir.
"""

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_FINGERPRINT_VERSION = 1
_CHUNK_BYTES = 1024 * 1024


class SourceError(ValueError):
    """input_dir cannot be turned into a source inventory."""


class SourceChangedError(SourceError):
    """A policy file changed between the scan and the read. The config is fine; prepare again."""


def scan_sources(input_dir: Path) -> dict[str, str]:
    """Return every policy source as ``{relative POSIX path: sha256 of its bytes}``."""
    sources: dict[str, str] = {}
    links: list[str] = []
    try:
        for root, dirs, files in os.walk(input_dir, followlinks=False):
            base = Path(root)
            for name in list(dirs):
                if name.startswith("."):
                    dirs.remove(name)
                elif (base / name).is_symlink():
                    links.append((base / name).relative_to(input_dir).as_posix())
                    dirs.remove(name)
            dirs.sort()
            for name in sorted(files):
                if name.startswith("."):
                    continue
                path = base / name
                relative = path.relative_to(input_dir).as_posix()
                if path.is_symlink():
                    links.append(relative)
                    continue
                sources[relative] = _sha256(path)
    except OSError as exc:
        raise SourceError(f"cannot read input_dir: {exc}") from exc
    if links:
        raise SourceError(
            "input_dir contains symbolic links, which could pull in files from outside it: "
            + ", ".join(sorted(links))
            + ". Copy the files into input_dir instead."
        )
    if not sources:
        raise SourceError(f"input_dir contains no policy files: {input_dir}")
    return sources


def source_fingerprint(sources: dict[str, str]) -> str:
    """One digest for the whole source set, sensitive to names and contents."""
    digest = hashlib.sha256(f"safety-sources-v{SOURCE_FINGERPRINT_VERSION}\n".encode())
    for relative in sorted(sources):
        digest.update(f"{relative}\0{sources[relative]}\n".encode())
    return digest.hexdigest()


def canonical_source_path(value: Any) -> str | None:
    """Return the path when it is already canonical, otherwise None.

    Canonical means relative, POSIX separators, and no empty, ``.`` or ``..`` segments.
    """
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        return None
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return PurePosixPath(value).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
