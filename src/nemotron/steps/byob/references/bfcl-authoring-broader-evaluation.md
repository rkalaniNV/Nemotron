# BFCL multi-domain rollout evidence

UA-1206 uses `bfcl-onboarding-ablation-rollout-v2` to publish three domain slots
without filling missing observations. The contract binds raw observation files,
collection states, immutable run trees, exclusions, a signed independent-reviewer
attestation, an evaluator pin record, per-domain reports, and the final
descriptive or causal decision.

## Locked scope

The selected domains are:

1. `tiny_library` — existing nine-run pilot;
2. `inventory` — selected additional domain;
3. `banking_vn` — selected additional domain.

The selected target route is `azure/openai/gpt-5.6-sol` through the NVIDIA inference
API. This route is **not yet a pinned evaluator identity**: an immutable revision or
weights digest has not been supplied. `NGC_API_KEY` is only a credential reference and
must never enter evidence bytes.

## Reviewer identity is signed, not typed

A completed domain record used to carry `reviewer_identity` as a string the
operator supplied while publishing their own runs. That string proved nothing, so
`bfcl-onboarding-domain-review-v1` replaces it with an Ed25519 attestation held by
the reviewer:

- the reviewer signs a `ReviewedBundle` — the protocol, ablation input, ablation
  report, evaluator pin, exclusions, and the digests of all nine observations and
  run trees;
- publication recomputes that bundle from the raw files and refuses any
  attestation covering different bytes, so a bundle shown for review cannot be
  swapped for another before release;
- the reviewer names the operator they are independent of, which makes
  "reviewer differs from operator" a signed claim rather than an operator's;
- which reviewer key is trusted is supplied by whoever publishes the rollout, not
  read from the operator's bundle manifest.

An untrusted key, an invalid signature, a review timestamped before the last run,
an incomplete checklist, or a reviewer equal to the operator all fail closed
([`test_bfcl_mcp_ablation_rollout.py`](../../../../../tests/steps/byob/test_bfcl_mcp_ablation_rollout.py)).

## The evaluator pin reuses the evaluation config contract

`bfcl-onboarding-evaluator-pin-v1` does not restate what "immutable" means. A
pinned evaluator is validated by constructing `CandidateModelIdentity` from the
evaluation config contract, which requires an immutable revision or a weights
digest and refuses moving pointers such as `main`, `latest`, or a branch ref. The
pin record adds only the non-secret serving route, the *name* of the credential
environment variable, and a digest of the provider evidence the pin was read
from; a value shaped like a credential is refused.

The absence of a pin is a record, not a gap: `unpinned` states either
`target_evaluation_not_run` or `immutable_pin_unavailable`, and the rollout
publishes `target_model_pin_missing` instead of implying a score it cannot
support. A domain whose observations carry scores cannot claim the target model
was never run, and a pin that does not match the ablation input's
`evaluator_model` is refused.

## Current evidence classification

The rollout remains **descriptive and unimplemented as broader evaluation evidence**.
The published decision is `mcp610-rollout/rollout.json`:

- `tiny_library` has nine digest-verifiable onboarding runs. The whole bundle
  re-verifies from `mcp610-tiny-library.bundle.json`, but no independent reviewer
  has signed an attestation, so the slot is published as missing with
  `reviewer_missing` rather than as a completed record;
- `inventory` has no locked nine-run observation set;
- `banking_vn` has no locked nine-run observation set;
- the pilot's evaluator pin is honestly recorded as `target_evaluation_not_run`,
  so no target-model scores exist;
- the hosted model’s immutable revision or weights digest is still pending.

These are missing evidence, not zero-valued observations. The rollout validator refuses
`causal` while any domain slot is missing, any evaluation score set is incomplete,
pinned model identities differ across completed domains, or reviewer and operator are
the same.

## Collection and publication rules

Each domain follows the same predeclared nine-run sequence: three repetitions of
`manual`, `llm_backend`, and `llm_mcp`. A completed domain record requires:

- all nine raw observations and complete collection states;
- all nine immutable run-artifact directories matching `run_digest`;
- an explicit exclusion record for every run, including zero exclusions;
- a reason for every positive excluded duration;
- a signed reviewer attestation from a trusted key, distinct from the operator;
- an evaluator pin record bound to the input's `evaluator_model`;
- a reproducible ablation report;
- all nine target-model scores for a causal claim.

`observations_are_live` is always true and `synthetic_substitution_allowed` is always
false in completed evidence. Tampered observation metrics, run trees, report bytes,
duplicate JSON keys, or stale digests fail closed.

## Operator and reviewer commands

Record how the target model was identified. The operator does this; only the
environment variable name is stored:

<!-- doc-smoke: evaluator-pin-help -->
```shell
python -m nemotron.steps.byob.scripts.record_bfcl_evaluator_pin --help
```

An independent reviewer verifies a bundle and signs the digests they verified.
`verify` prints the derived bundle without signing, `sign` re-verifies and signs,
and `check` re-checks a signature against the bundle:

<!-- doc-smoke: domain-review-help -->
```shell
python -m nemotron.steps.byob.scripts.review_bfcl_domain_evidence --help
```

Inspect the rollout assembler:

<!-- doc-smoke: ablation-rollout-help -->
```shell
python -m nemotron.steps.byob.scripts.assemble_bfcl_ablation_rollout --help
```

`--domain-evidence` publishes a missing-domain record; `--domain-bundle` rebuilds a
completed domain from its raw files and the review attestation its manifest names,
and `--trusted-reviewer-key` supplies the keys the publishing authority accepts.
Together they must describe exactly three domains. A standalone self-hashed
“complete” summary is not trusted; complete evidence is always rebuilt from all raw
observations, collection states, run directories, exclusions, and reports. The
present protocol is explicitly descriptive, so `--evidence-kind causal` always
fails with `causal_design_unimplemented`. A future causal protocol requires separate
preregistration and analysis contracts. The emitted `rollout_digest` binds all domain
records, decision authority, rationale, and blockers.

## What closing UA-1206 still requires

The two contracts above are implemented and tested. Closing the task needs
evidence that no contract can produce:

1. an independent reviewer generates a key pair, verifies
   `mcp610-tiny-library.bundle.json`, and signs an attestation, which turns the
   pilot slot into a completed domain record;
2. real `inventory` and `banking_vn` nine-run sets are collected under the same
   protocol and independently reviewed;
3. an immutable revision or weights digest for the target route is supplied and
   all nine runs per domain are scored against it.

## Claim boundary

The current single-domain quality and effort measurements are descriptive. They do not
support a causal claim or a default-UX decision. This document and the support matrix
must retain the **unimplemented** label for the multi-domain causal claim until real
`inventory` and `banking_vn` runs, independent review, and uniformly pinned
target-model evaluations are published.
