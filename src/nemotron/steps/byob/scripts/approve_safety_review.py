#!/usr/bin/env python3
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

"""Sign the current safety review packet, and publish once every required role has signed."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from nemotron.steps.byob.runtime.benchmark_families.safety.pipeline import SIGNER_ROLES, sign_safety_packet
from nemotron.steps.byob.runtime.benchmark_families.safety.status import EXIT_CODES, RunStatus, SafetyStatus
from nemotron.steps.byob.scripts.runtime import run_byob


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="The same safety config used for the run")
    parser.add_argument("--approved-by", help="Reviewer name. Defaults to the login name.")
    parser.add_argument(
        "--role", choices=SIGNER_ROLES, help="Needed only when several roles are still missing a signature."
    )
    parser.add_argument("--note")
    parser.add_argument("--sign-only", action="store_true", help="Record the signature without publishing.")
    return parser


def _login_name() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return ""


def main() -> None:
    args = _parser().parse_args()
    try:
        result = sign_safety_packet(
            args.config, approved_by=args.approved_by or _login_name(), role=args.role, note=args.note
        )
        print(f"Signed by {result['approved_by']} as {result['role']}.")
        if result["replaced_older_signoff"]:
            print("Signatures on an older version of the packet were dropped.")
        if result["missing_roles"]:
            print(f"Still waiting for: {', '.join(result['missing_roles'])}. That approver runs this command too.")
            raise SystemExit(EXIT_CODES[RunStatus.REVIEW_REQUIRED])
        if args.sign_only:
            print("Every required role has signed. Run stage=generate to publish.")
            return
        output = run_byob(config=args.config, stage="generate", family="safety")
    except SafetyStatus as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        raise SystemExit(exc.cli_exit_code) from exc
    if output is not None:
        print(output)


if __name__ == "__main__":
    main()
