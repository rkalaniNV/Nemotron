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

"""Optional utility: MMLU-Pro parquet → nemo-skills-style JSONL.

Not used by the BYOB CLI or ``assessment_utils``; keep only if you hand off benchmarks
to nemo-skills or another tool that expects this JSONL layout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MMLUProProcessor:
    """Processor for MMLU-Pro dataset — converts parquet format to nemo-skills JSONL format."""

    @staticmethod
    def process(input_file: str, output_file: str) -> None:
        """Convert MMLU-Pro parquet file to nemo-skills JSONL format.

        Args:
            input_file: Path to input parquet file (must be .parquet)
            output_file: Path to output JSONL file
        """
        if not input_file.endswith(".parquet"):
            raise ValueError(f"MMLUProProcessor only accepts .parquet files, got: {input_file}")

        df = pd.read_parquet(input_file)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                options = row["options"]
                options_dict = {chr(ord("A") + i): str(option) for i, option in enumerate(options)}
                options_text = "\n".join(f"{letter}) {option}" for letter, option in options_dict.items())

                category = str(row["category"]).replace(" ", "_")

                entry = {
                    "expected_answer": row["answer"],
                    "examples_type": f"mmlu_pro_few_shot_{category}",
                    "subset_for_metrics": category,
                    "problem": f"{row['question']}\n\n{options_text}",
                    "options": options_text,
                    **options_dict,
                }

                f.write(json.dumps(entry) + "\n")

        logger.info("Successfully processed %s entries", len(df))
