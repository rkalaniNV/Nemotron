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

"""Safety adapter around the shared MCQ Curator semantic-dedup workflow.

The coverage cell still owns meaning. This pass compares scored user-turn text the
same way MCQ compares formatted questions: embed, cluster, and mark near-duplicates.
Python decides which marked rows to drop so contrast pairs stay intact.
"""

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def scored_turn_text(case: Mapping[str, Any]) -> str:
    """The triggering user message: the scored user turn, else the last user turn."""
    turns = case.get("turns") if isinstance(case.get("turns"), list) else []
    users: list[str] = []
    scored = ""
    for turn in turns:
        if not isinstance(turn, Mapping) or turn.get("role") != "user":
            continue
        text = turn.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        stripped = text.strip()
        users.append(stripped)
        if turn.get("scored") is True:
            scored = stripped
    return scored or (users[-1] if users else "")


def semantic_partitions(cases: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Partition by label and language so similarity never merges unlike safety decisions."""
    partitions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        text = scored_turn_text(case)
        if not text:
            continue
        action = case.get("expected_action")
        action_key = str(action) if action is not None else ""
        language = case.get("language")
        if isinstance(language, str) and language.strip():
            language_tags = language.split(",")
        else:
            languages = case.get("languages")
            language_tags = (
                [item for item in languages if isinstance(item, str)]
                if isinstance(languages, list)
                else []
            )
        language_key = ",".join(
            sorted(
                {
                    unicodedata.normalize("NFC", tag.strip()).casefold()
                    for tag in language_tags
                    if tag.strip()
                }
            )
        )
        key = (
            unicodedata.normalize("NFC", action_key).casefold(),
            language_key,
        )
        partitions.setdefault(key, []).append(dict(case))
    return partitions


def run_mcq_semantic_dedup(
    cases: Sequence[Mapping[str, Any]],
    *,
    work_dir: Path | None,
    settings: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return duplicate case ids using one MCQ Curator workflow per safety partition."""
    if work_dir is None:
        raise ValueError("semantic deduplication requires a work directory")
    partitions = [
        (partition, rows)
        for partition, rows in sorted(semantic_partitions(cases).items())
        if len(rows) >= 2
    ]
    if not partitions:
        return ()

    import pandas as pd

    from nemotron.steps.byob.runtime.deduplication import TextSemanticDeduplication

    class TextSemanticDeduplicationSafety(TextSemanticDeduplication):
        """MCQ semantic workflow with safety scored-turn text and case ids."""

        def prepare_input_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
            texts = [scored_turn_text(row.to_dict()) for _index, row in dataset.iterrows()]
            return pd.DataFrame({"id": dataset["case_id"].astype(str), "text": texts})

        def _mark_duplicates(self, dataset: pd.DataFrame) -> pd.DataFrame:
            import glob
            import os

            marked = dataset.copy()
            paths = glob.glob(os.path.join(self.output_path, "duplicates/*.parquet"))
            if not paths:
                marked["is_duplicate"] = False
                return marked
            duplicates = pd.concat([pd.read_parquet(path) for path in paths])
            marked["is_duplicate"] = marked["case_id"].astype(str).isin(duplicates["id"].astype(str))
            return marked

    duplicate_ids: set[str] = set()
    for partition, rows in partitions:
        n_rows = len(rows)
        configured = settings.get("n_clusters", 1)
        n_clusters = configured if isinstance(configured, int) and not isinstance(configured, bool) else 1
        n_clusters = max(1, min(n_clusters, n_rows - 1))
        partition_id = hashlib.sha256(
            json.dumps(partition, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        config = SimpleNamespace(
            output_dir=str(work_dir / "stage_cache"),
            expt_name=f"semantic_deduplication/{partition_id}",
            semantic_deduplication_config={
                "model_identifier": settings["model_identifier"],
                "n_clusters": n_clusters,
                "eps": settings["eps"],
                "remove_duplicates": False,
            },
        )
        marked = TextSemanticDeduplicationSafety(config).run(pd.DataFrame(rows))
        duplicate_ids.update(
            str(case_id) for case_id in marked.loc[marked["is_duplicate"], "case_id"].tolist()
        )
    return tuple(sorted(duplicate_ids))
