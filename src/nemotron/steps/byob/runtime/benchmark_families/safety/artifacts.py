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

"""Atomic writers for files inside a safety run directory."""

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` in one step, so a crash leaves the old file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: Path, data: Any) -> None:
    write_atomic(path, dump_json(data))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_atomic(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    write_atomic(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
