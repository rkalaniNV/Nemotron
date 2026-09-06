<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Author a Pack with Model Assistance

Use this guide to produce a reviewed Oracle Pack from a conventional source package, with a model drafting the parts a model is allowed to propose. The guided command is `bfcl_author.py`. It is stateful: each subcommand binds its output into a session, and when a gate refuses progress it prints the next safe command.

A model in this flow may propose a tool coverage plan, validation cases, task-template plans, and declarative assertion specifications. It may not change the backend, the endpoint behavior, the tool schemas, or the fixtures, and it may not certify its own output, invent fixture bindings, approve anything, or bypass executable Gold validation. Everything it proposes passes through the same replay and Gold gate as a hand-written pack, which is why {doc}`author-a-pack` and this guide converge on one publication contract.

## Before You Start

- Install the BYOB dependencies with `uv sync --extra byob`, and prepare a source package in one of the two supported layouts below.
- Prepare a domain brief, a reviewed statement of what the source is for, which is sanitized and bound into the evidence.
- Prepare a probe plan, which you need for certification tier A1 or A2 and therefore for a Gold release.
- Have a certification key pair and its allowlisted key identifier available.

### Source layouts

A `local_python` source requires `backend.py` as its only import-closure root, a reviewed `tools.json`, and a canonical `dependency-lock.json`, and may add `fixtures.json`. An `http_package` source requires a strict, secret-free `endpoint_config.yaml` and its companion `tools.json`. `src/nemotron/steps/byob/references/bfcl-conventional-source-packages.md` is the normative description of both, including the dependency-lock format and the static-inspection rules.

### Enable the adapter

Live source intake is disabled for every adapter by default.

```bash
export BFCL_ENABLE_LOCAL_PYTHON=1
```

The equivalent variable for the HTTP adapter is `BFCL_ENABLE_HTTP_PACKAGE`. Without the variable, or an equivalent reviewed `adapter_rollout` policy block, intake fails with `adapter_rollout_disabled`. Accepted values are `1`, `true`, `yes`, `0`, `false`, and `no`, case-insensitive. Offline artifact verification, review, approval, and freeze do not require the flag; only operations that inspect or execute a source do.

:::{important}
Publication currently supports `local_python`. An `http_package` source reaches intake, drafting, review, and freeze, but publication is deliberately refused until an independently verified publication adapter exists. Plan for that limitation before you onboard an HTTP source, because the refusal arrives at the last step.
:::

## Step 1: Intake and Certification

`author` runs source intake and produces transport-neutral evidence. Two decisions are required at this first command rather than deferred, because evidence that has already been collected cannot be retroactively declared clean.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author \
  --ci author \
  --workspace /srv/bfcl/authoring/warehouse \
  --source /srv/sources/warehouse-package \
  --brief /srv/sources/domain-brief.txt \
  --adapter local_python \
  --pack-id warehouse_assets \
  --pack-version 0.1.0 \
  --required-tier A2 \
  --held-out-not-applicable-reason "The asset catalog is public reference data." \
  --held-out-reviewed-by reviewer@example.test \
  --certification-private-key /srv/bfcl/keys/certification-private.pem \
  --certification-key-id warehouse-authoring \
  --probe-plan /srv/sources/probe-plan.json
```

The first decision is held-out status: supply either `--held-out-policy` or `--held-out-not-applicable-reason`, always together with `--held-out-reviewed-by`. The second is the probe plan, because A1 and A2 are earned from observed probe outcomes and nothing else can supply them. `--adapter auto` is the default and recognizes any supported layout, though naming the adapter is clearer in automation; `--ci` guarantees the command never prompts; and flags after the guided ones, including the certification key flags above, are delegated to the underlying intake command.

The probe plan is one transport-neutral document. It names a case per published tool, at least one structured error if the source has error codes, and a case the tool cannot finish inside the deadline; without that last case, timeout cleanup stays unobserved and certification cannot reach A2. Certification then awards one of three tiers. A0 proves source identity and catalog integrity. A1 adds bounded read-only observation. A2 adds deterministic reset, episode isolation, confirmation safety, mutation declaration, timeout cleanup, and result coverage. Stop if the report does not record the tier you need: drafting may inspect lower tiers, but a Gold freeze requires A2, and no approval can raise a certification tier.

## Step 2: Answer the Open Questions

Intake may raise digest-bound open questions about the source. Apply the reviewed answers as a new evidence revision:

```bash
python -m nemotron.steps.byob.scripts.bfcl_author answer \
  --workspace /srv/bfcl/authoring/warehouse \
  --evidence <EVIDENCE_BUNDLE_JSON> \
  --questions <OPEN_QUESTIONS_JSON> \
  --answers <REVIEWED_ANSWERS_JSON>
```

## Step 3: Authorize Exposure, Then Approve the Evidence

The source owner first authorizes the exact redacted evidence subject for exposure to a model. This is the pre-model boundary, and it is the reason the evidence bundle is sanitized before anything is sent anywhere. Supply either `--authorized-by` or `--organizational-policy-digest`, not both; `--output` writes the record to a path you choose.

```bash
python -m nemotron.steps.byob.scripts.bfcl_author authorize \
  --workspace /srv/bfcl/authoring/warehouse \
  --subject <MODEL_EXPOSURE_SUBJECT_JSON> \
  --authorized-by owner@example.test

python -m nemotron.steps.byob.scripts.bfcl_author approve \
  --workspace /srv/bfcl/authoring/warehouse \
  --boundary evidence \
  --approved-by reviewer@example.test
```

:::{important}
Evidence approval and release approval are different decisions, and the first cannot be replaced by the second. Evidence approval says a reviewer inspected this exact source and normalized evidence and considers it fit to draft from. Release approval, later, says a reviewer inspected the finished pack and its fresh validation and considers it fit to publish. Approving the release does not retroactively authorize the model exposure that already happened, so the command sequence requires both in order.
:::

Both approvals are digest-bound, so if the source, brief, redaction, observations, certification, or resolved authoring configuration changes afterwards, the approval goes stale and must be redone against the new digest.

## Step 4: Draft

```bash
python -m nemotron.steps.byob.scripts.bfcl_author draft \
  --workspace /srv/bfcl/authoring/warehouse
```

Drafting issues bounded, cached, structured model requests and stops at proposals: it writes them beside a pack rather than into one. Unknown tools, unsupported assertions, ungrounded arguments, malformed output, and cache conflicts all fail closed.

## Step 5: Assemble the Candidate Pack

```bash
python -m nemotron.steps.byob.scripts.bfcl_author assemble \
  --workspace /srv/bfcl/authoring/warehouse
```

The assembler derives everything mechanical from evidence that is already trusted: pack identity and the manifest from the verified bundle, `tools.json` from the certified catalog, and `assertions.py` from drafts that compiled without a blocker. For a `local_python` source, `backend.py` and `fixtures.json` are copied byte for byte from the fingerprinted tree; for a session-based source there is nothing to copy, so the pack names the certified endpoint and takes its fixtures from the reviewed probe plan those sessions were opened with. What remains is what a model that has only read a catalog must not state: slot bindings to fixture columns, turn policies, per-language user turns, and the validation cases that decide the tier. Those arrive in one reviewed `bfcl-candidate-pack-supplement-v1` YAML file, and every tool and assertion it names is checked back against the evidence and the compiled assertions, so a supplement cannot introduce a tool the source never published or an assertion nobody compiled. For a pack assembled outside a guided session, `assemble_candidate_pack.py` takes the same inputs explicitly through `--evidence`, `--source`, `--drafts`, `--supplement`, `--output`, and `--probe-plan`.

## Step 6: Build the Review Packet

```bash
python -m nemotron.steps.byob.scripts.bfcl_author review \
  --workspace /srv/bfcl/authoring/warehouse \
  --adapter-kind local_python
```

Review assembles independently verified certification, fresh validation, the answered questions, and the complete candidate pack into one deterministic packet. `--adapter-kind` defaults to `mcp_mode_a`, so name your own adapter explicitly.

## Step 7: Approve the Release, Then Freeze

```bash
python -m nemotron.steps.byob.scripts.bfcl_author approve \
  --workspace /srv/bfcl/authoring/warehouse \
  --boundary release \
  --approved-by reviewer@example.test

python -m nemotron.steps.byob.scripts.bfcl_author freeze \
  --workspace /srv/bfcl/authoring/warehouse
```

Release approval binds the exact review packet, so a rebuilt packet needs its new digest approved. `--acknowledge-warning` and `--acknowledge-finding` may each be repeated to record an acknowledged item. Freeze then seals the pack and all reviewed sidecars as immutable bytes; never edit frozen bytes.

## Step 8: Publish

```bash
python -m nemotron.steps.byob.scripts.bfcl_author publish \
  --workspace /srv/bfcl/authoring/warehouse \
  --adapter-kind local_python
```

Publication reruns fresh Gold validation against the frozen pack and then runs the ordinary generation pipeline as `stage=all`; it does not trust the validation evidence recorded during review. Choose the release budget and the mixes with {doc}`publish-a-release`.

## Verify Success

The certification report records the tier you required, `A2` for a Gold release, and the exposure authorization and both approvals reference current digests. Assembly wrote `candidate_pack_provenance.json` recording the digest of every input and every file it produced. Fresh validation of the candidate pack reports `gold_eligible: true`, and publication wrote `run_manifest.json` beside `benchmark.parquet` and `benchmark_raw.parquet`.

## Common Failures

| Reported code | What to do |
| --- | --- |
| `adapter_rollout_disabled` | Enable the reviewed adapter policy or its environment variable. Do not bypass the gate downstream. |
| `adapter_under_certified` | Collect the missing A2 observations. Approval cannot raise certification, and the probe plan needs its timeout case. |
| `source_identity_mismatch` | The source tree changed after certification. Assemble the exact revision intake certified, or rerun intake. |
| `supplement_assertion_unknown` | The supplement names an assertion drafting never compiled. Draft and compile that specification first. |
| `review_approval_stale` | Rebuild the review packet and approve its new digest. |
| `session_binding_drift` | Restore the bound artifact, or resume with `bfcl_author resume --workspace <DIR> --next <STEP>`. |

## Next Steps

- Onboard a running MCP server instead of a package: {doc}`mcp-server`, or take the frozen pack to publication scale with {doc}`publish-a-release`.
- Read where the authorization boundaries sit and why: {doc}`../explanation/authoring-flows`.
