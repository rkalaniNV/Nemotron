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

"""BYOB MCQ assessment: dataset summaries and optional category plots."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_MMLU_PRO_PARQUET_COLUMNS = ("question", "options", "answer", "category")


def _letter_from_answer_index(idx: int) -> str:
    return chr(ord("A") + int(idx))


def build_mcq_assessment_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate statistics for a final BYOB / MMLU-Pro-style MCQ parquet."""
    n = len(df)
    cat_counts: dict[str, int] = {}
    if n and "category" in df.columns:
        cat_counts = {str(k): int(v) for k, v in df["category"].value_counts().items()}

    answer_letters: Counter[str] = Counter()
    if n and "answer" in df.columns:
        for a in df["answer"].dropna():
            answer_letters[str(a).strip().upper()[:1]] += 1
    elif n and "answer_index" in df.columns:
        for idx in df["answer_index"].dropna():
            try:
                answer_letters[_letter_from_answer_index(int(idx))] += 1
            except (TypeError, ValueError):
                continue

    n_options: list[int] = []
    if n and "options" in df.columns:
        for opts in df["options"]:
            if isinstance(opts, (list, tuple)):
                n_options.append(len(opts))
            else:
                n_options.append(0)
    options_len = {
        "min": min(n_options) if n_options else None,
        "max": max(n_options) if n_options else None,
    }

    return {
        "num_questions": n,
        "num_categories": len(cat_counts),
        "counts_by_category": cat_counts,
        "answer_letter_counts": dict(answer_letters),
        "options_per_question": options_len,
        "columns": list(df.columns),
    }


def write_assessment_artifacts(
    benchmark_parquet: str | Path,
    output_dir: str | Path,
    *,
    write_json: bool = True,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Write ``assessment_summary.json`` and optional ``category_counts.png``.

    Args:
        benchmark_parquet: Path to ``benchmark.parquet`` (MMLU-Pro-style columns).
        output_dir: Directory for ``assessment_summary.json`` and plots.
        write_json: When False, skip ``assessment_summary.json``.
        write_plots: When True, write ``category_counts.png`` if matplotlib is available.

    Returns:
        The assessment summary dictionary (same content as the JSON file when ``write_json``).
    """
    path = Path(benchmark_parquet)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_MMLU_PRO_PARQUET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Benchmark parquet missing columns {missing}; "
            f"expected at least {REQUIRED_MMLU_PRO_PARQUET_COLUMNS}"
        )

    summary = build_mcq_assessment_summary(df)
    summary["benchmark_parquet"] = str(path.resolve())
    summary["assessment_output_dir"] = str(out.resolve())

    if write_json:
        json_path = out / "assessment_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info("Wrote %s", json_path)

    if write_plots and summary.get("counts_by_category"):
        _try_write_category_bar_chart(summary["counts_by_category"], out / "category_counts.png")

    return summary


def _try_write_category_bar_chart(counts_by_category: dict[str, int], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed; skip category_counts.png")
        return

    categories = list(counts_by_category.keys())
    values = [counts_by_category[c] for c in categories]
    if not categories:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.35), 5))
    ax.bar(range(len(categories)), values, color="#1f77b4", alpha=0.85)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Question count")
    ax.set_title("BYOB benchmark — questions per category")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)
