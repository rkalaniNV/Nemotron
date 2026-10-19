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

"""Publication statuses for the safety family.

The shared BYOB dispatcher already turns an exception with ``cli_exit_code``
into the process status. These codes belong to safety publication, not to a
successful ``stage=prepare``.
"""

from enum import Enum
from pathlib import Path


# str + Enum rather than StrEnum: the project supports Python 3.10.
class RunStatus(str, Enum):
    """Business outcome of a safety publication run."""

    GOLD_COMPLETED = "gold_completed"
    SILVER_COMPLETED = "silver_completed"
    PREVIEW_COMPLETED = "preview_completed"
    REVIEW_REQUIRED = "review_required"
    PRIVACY_BLOCKED = "privacy_blocked"
    INVALID_CONFIG = "invalid_config"
    PIPELINE_INCOMPLETE = "pipeline_incomplete"
    STAGE_UNAVAILABLE = "stage_unavailable"
    PREPARE_REQUIRED = "prepare_required"
    INVALID_RUN_ARTIFACT = "invalid_run_artifact"
    RUN_LOCKED = "run_locked"
    NO_USABLE_SOURCES = "no_usable_sources"
    NO_USABLE_RULES = "no_usable_rules"
    MODEL_UNAVAILABLE = "model_unavailable"
    QUALITY_GATES_FAILED = "minimum_dataset_quality_not_met"


EXIT_CODES = {
    RunStatus.GOLD_COMPLETED: 0,
    RunStatus.SILVER_COMPLETED: 2,
    RunStatus.PREVIEW_COMPLETED: 2,
    RunStatus.REVIEW_REQUIRED: 3,
    RunStatus.PRIVACY_BLOCKED: 4,
    RunStatus.INVALID_CONFIG: 1,
    RunStatus.PIPELINE_INCOMPLETE: 1,
    RunStatus.STAGE_UNAVAILABLE: 1,
    RunStatus.PREPARE_REQUIRED: 1,
    RunStatus.INVALID_RUN_ARTIFACT: 1,
    RunStatus.RUN_LOCKED: 1,
    RunStatus.NO_USABLE_SOURCES: 1,
    RunStatus.NO_USABLE_RULES: 1,
    RunStatus.MODEL_UNAVAILABLE: 1,
    RunStatus.QUALITY_GATES_FAILED: 1,
}


class SafetyStatus(Exception):  # noqa: N818 - also carries Gold and review_required, which are not errors
    """A safety outcome the shared BYOB CLI can print and exit with."""

    def __init__(self, status: RunStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = status.value
        self.cli_exit_code = EXIT_CODES[status]


def explain(status: RunStatus, run_dir: Path) -> str:
    """Explain a publication outcome without requiring the reader to know exit numbers."""
    review_page = run_dir / "review.html"
    review_packet = run_dir / "publication_review_packet.yaml"
    quarantine = run_dir / "quarantine" / "files.jsonl"
    if status is RunStatus.REVIEW_REQUIRED:
        return (
            "Review required. This is not an error. "
            "The publication review packet is ready. Open the review page and record each decision. "
            "Then run: python -m nemotron.steps.byob.scripts.approve_safety_review --config <config>. "
            "It signs the packet and publishes. "
            f"Review page: {review_page}. Review packet: {review_packet}."
        )
    if status is RunStatus.PRIVACY_BLOCKED:
        return (
            "Privacy check stopped the run. "
            "No safe policy document remains, so no external model was called. "
            f"Quarantine report: {quarantine}."
        )
    if status is RunStatus.GOLD_COMPLETED:
        return f"Gold benchmark published. Output directory: {run_dir}."
    if status is RunStatus.SILVER_COMPLETED:
        return (
            "Silver benchmark published. Gold remains blocked until the recorded "
            f"coverage gaps are filled. Output directory: {run_dir}."
        )
    if status is RunStatus.PREVIEW_COMPLETED:
        return f"Preview benchmark published. This result is not a comparable score. Output directory: {run_dir}."
    if status is RunStatus.PIPELINE_INCOMPLETE:
        return (
            "Publication did not finish, so no benchmark was published. "
            f"Output directory: {run_dir}."
        )
    if status is RunStatus.QUALITY_GATES_FAILED:
        return (
            "Minimum dataset quality gates were not met, and quality_gates.on_failure is fail, "
            "so no benchmark was published. The review packet still lists the unmet gates. "
            f"Review page: {review_page}."
        )
    if status is RunStatus.NO_USABLE_RULES:
        return (
            "No policy rule can be used: every extracted or pre-authored rule was quarantined or rejected, "
            "so no case was generated. Each rule's reasons are in the flags file. "
            f"Flags: {run_dir / 'stage_cache' / 'policy_flags.jsonl'}."
        )
    if status is RunStatus.MODEL_UNAVAILABLE:
        return (
            "The rule-extraction model could not be reached or returned an unusable reply. "
            "Windows already extracted are cached, so rerunning the same command continues from there. "
            f"Output directory: {run_dir}."
        )
    if status is RunStatus.STAGE_UNAVAILABLE:
        return (
            "This safety stage is not connected yet. "
            "This result is not a Silver or Preview publication. "
            f"Output directory: {run_dir}."
        )
    if status is RunStatus.PREPARE_REQUIRED:
        return (
            "The config or the policy files differ from the ones this run directory was prepared with, "
            "or the run was never prepared. Run stage=prepare or stage=all first. "
            f"Output directory: {run_dir}."
        )
    if status is RunStatus.INVALID_RUN_ARTIFACT:
        return (
            "A file inside the run directory is damaged, stale, or incomplete. The config itself is valid. "
            f"Output directory: {run_dir}."
        )
    if status is RunStatus.RUN_LOCKED:
        return f"Another safety process is writing this run directory. Output directory: {run_dir}."
    if status is RunStatus.NO_USABLE_SOURCES:
        return (
            "No policy file could be used: every file was rejected as unsupported, unreadable, or invalid. "
            "No external model was called. Each file's reason is in the quarantine report. "
            f"Quarantine report: {quarantine}."
        )
    return f"Invalid safety configuration. {status.value}."


def abort(status: RunStatus, run_dir: Path, detail: str = "") -> SafetyStatus:
    """Build the exception a stage raises when it must stop the shared CLI."""
    message = explain(status, run_dir)
    if detail:
        message = f"{message} {detail}"
    return SafetyStatus(status, message)
