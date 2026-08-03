# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Experiment inspection commands.

Mirrors ``nemo experiment logs`` / ``status`` but runs inside our process, so
the nemo-run and Lepton client patches in :mod:`nemo_runspec.run` are applied
first. The upstream ``nemo`` CLI never imports ``nemo_runspec``, which is why
tailing a Lepton job through it dies on chunk-split UTF-8 characters.
"""

from __future__ import annotations

import typer

experiment_app = typer.Typer(
    name="experiment",
    help="Inspect nemo-run experiments (logs, status)",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _load_experiment(experiment_id: str):
    """Apply client patches, then reconstruct the experiment by id or title."""
    from nemo_runspec.run import patch_lepton_log_stream_incremental_decode

    patch_lepton_log_stream_incremental_decode()

    from nemo_run.run.experiment import Experiment

    try:
        return Experiment.from_id(experiment_id)
    except AssertionError:
        # Bare titles are convenient when you just want the newest run.
        try:
            return Experiment.from_title(experiment_id)
        except AssertionError as exc:
            raise typer.BadParameter(str(exc), param_hint="EXPERIMENT_ID") from exc


def _resolve_job_id(exp, job: str) -> str:
    """Accept either a job name or a positional index, as ``nemo`` does."""
    job_ids = [j.id for j in exp.jobs]
    if job in job_ids:
        return job
    if job.isdigit() and int(job) < len(job_ids):
        return job_ids[int(job)]
    raise typer.BadParameter(f"job {job!r} not found; available: {job_ids}", param_hint="JOB")


@experiment_app.command("logs")
def logs(
    experiment_id: str = typer.Argument(..., help="Experiment id, or title for the latest run"),
    job: str = typer.Argument("0", help="Job name or index within the experiment"),
    regex: str | None = typer.Option(None, "--regex", help="Only print matching lines"),
) -> None:
    """Tail a job's logs without dying on multi-byte characters."""
    exp = _load_experiment(experiment_id)
    exp.logs(_resolve_job_id(exp, job), regex=regex)


@experiment_app.command("status")
def status(
    experiment_id: str = typer.Argument(..., help="Experiment id, or title for the latest run"),
) -> None:
    """Show the status of every job in an experiment."""
    _load_experiment(experiment_id).status()
