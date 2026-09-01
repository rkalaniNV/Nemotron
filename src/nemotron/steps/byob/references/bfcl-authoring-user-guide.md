# BFCL assisted-authoring user guide

This guide covers Flow 2: producing a reviewed Oracle Pack from one source declaration
and one domain brief. The manual Oracle Pack path remains unchanged and is documented in
[bfcl-oracle-pack.md](bfcl-oracle-pack.md). Current adapter and publication status is
test-linked in [bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md).

## Install and inspect the CLI

Install the BYOB dependencies. MCP transport users also install the isolated `bfcl-mcp`
extra. The guided command is intentionally stateful and prints the next safe command when
a gate refuses progress.

<!-- doc-smoke: bfcl-author-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author --help
```

## Inputs and adapter selection

`author` accepts `--source`, `--brief`, and a workspace. `--adapter auto` recognizes a
local Python source package, reviewed HTTP package, or MCP Mode A intake. Pack ID and
version come from policy or explicit confirmation; CI never guesses or prompts.

<!-- doc-smoke: bfcl-author-author-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author author --help
```

Source layouts are defined in
[bfcl-conventional-source-packages.md](bfcl-conventional-source-packages.md) and
[bfcl-mcp-user-guide.md](bfcl-mcp-user-guide.md). Live adapters are fail-closed behind
the policy described in [bfcl-authoring-rollout.md](bfcl-authoring-rollout.md)
([`test_bfcl_authoring_rollout_policy.py`](../../../../../tests/steps/byob/test_bfcl_authoring_rollout_policy.py)).

## The two authorization boundaries

The normal command sequence is:

1. `author` creates transport-neutral evidence.
2. `answer` applies any digest-bound open questions.
3. `authorize` grants model exposure for the exact evidence subject.
4. `approve --boundary evidence` separately approves that evidence for drafting.
5. `draft` runs bounded, cached structured model calls.
6. `review` assembles independently verified certification, fresh validation, answered
   questions, and the complete candidate pack.
7. `approve --boundary release` approves the exact review packet.
8. `freeze` seals the pack and all reviewed sidecars.
9. `publish` reruns fresh Gold validation and `stage=all`.

Pre-model authorization cannot be replaced by final release approval. Session ordering
and stale-binding refusal are exercised by
[`test_bfcl_authoring_cli.py`](../../../../../tests/steps/byob/test_bfcl_authoring_cli.py)
and
[`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py).

Use each subcommand’s help as the authoritative argument synopsis:

<!-- doc-smoke: bfcl-author-review-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author review --help
```

<!-- doc-smoke: bfcl-author-publish-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author publish --help
```

## Certification and publication

A0 proves identity and catalog integrity, A1 adds bounded read-only observation, and A2
adds deterministic reset, isolation, confirmation safety, mutation declaration, timeout
cleanup, and result coverage. Drafting may inspect lower tiers, but Gold freeze requires
A2 ([`test_bfcl_authoring_release.py`](../../../../../tests/steps/byob/test_bfcl_authoring_release.py)).

Local Python and MCP Mode A have publication adapters. HTTP packages share intake,
review, and freeze but publication is intentionally refused until an independently
verified publication adapter exists
([`test_bfcl_authoring_e2e.py`](../../../../../tests/steps/byob/test_bfcl_authoring_e2e.py)).

The release contract is
[bfcl-authoring-release-v2.md](bfcl-authoring-release-v2.md). Legacy MCP v1 releases
remain verifiable but are not byte-identical to v2.

## Operations

- Structured events: [bfcl-authoring-events.md](bfcl-authoring-events.md), verified by
  [`test_bfcl_authoring_events.py`](../../../../../tests/steps/byob/test_bfcl_authoring_events.py).
- Credential references and rotation:
  [bfcl-authoring-credentials.md](bfcl-authoring-credentials.md), verified by
  [`test_bfcl_authoring_credentials.py`](../../../../../tests/steps/byob/test_bfcl_authoring_credentials.py).
- Model-cache retention:
  [bfcl-authoring-cache-retention.md](bfcl-authoring-cache-retention.md), verified by
  [`test_bfcl_authoring_retention.py`](../../../../../tests/steps/byob/test_bfcl_authoring_retention.py).
- Release revocation:
  [bfcl-authoring-revocation.md](bfcl-authoring-revocation.md), verified by
  [`test_bfcl_release_revocation.py`](../../../../../tests/steps/byob/test_bfcl_release_revocation.py).

<!-- doc-smoke: guide-purge-help -->
```shell
python -m nemotron.steps.byob.scripts.bfcl_author purge-cache --help
```

<!-- doc-smoke: guide-revoke-help -->
```shell
python -m nemotron.steps.byob.scripts.revoke_authoring_release --help
```

## Troubleshooting

- `adapter_rollout_disabled`: explicitly enable the reviewed adapter policy; do not
  bypass it in a downstream command.
- `credential_context_stale`: rerun intake and certification after principal,
  permission, or credential-reference drift.
- `session_binding_drift`: restore the bound artifact or resume from the last verified
  session.
- `adapter_under_certified`: collect the missing A2 observations; approval cannot raise
  certification.
- `review_approval_stale`: rebuild review and approve its new digest.
- `release_revoked`: consult the authenticated registry and replacement fingerprint;
  never edit frozen bytes.

## Contract index

- Adapter declaration and evidence:
  [bfcl-transport-neutral-intake.md](bfcl-transport-neutral-intake.md)
- Certification profiles:
  [bfcl-source-adapter-certification-profiles.md](bfcl-source-adapter-certification-profiles.md)
- Adding an adapter:
  [bfcl-adapter-authoring-certification-guide.md](bfcl-adapter-authoring-certification-guide.md)
- MCP transport:
  [bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md)
- Held-out and model exposure:
  [bfcl-unified-authoring-plan.md](bfcl-unified-authoring-plan.md)
- Release:
  [bfcl-authoring-release-v2.md](bfcl-authoring-release-v2.md)
- Multi-domain rollout evidence:
  [bfcl-authoring-broader-evaluation.md](bfcl-authoring-broader-evaluation.md)
- Workflow acceptance criteria and their owning tests:
  [bfcl-workflow-acceptance-matrix.md](bfcl-workflow-acceptance-matrix.md)
