---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Schema reference for candidate and approved filter policies used by curate/profile and curate/nemo_curator."
topics: ["Curation", "Reference", "Policy"]
tags: ["Reference", "Curation", "Configuration"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Filter Policy File

A *filter policy* is a YAML document that names the quality signals to gate on and the threshold for each.
`curate/profile` writes a *candidate policy* with `approved: false`.
A person promotes a candidate into an *approved policy* by choosing thresholds and recording who approved them and on what evidence.
`curate/nemo_curator` executes only approved policies unless the `allow_unvalidated_policy` override is set.

The schema is implemented in `src/nemotron/steps/curate/nemo_curator/runtime/policy.py`.

## Candidate Policy

`curate/profile` writes `candidate_policies.yaml` with this top-level shape:

```yaml
schema_version: 1
approved: false
note: "..."
corpus:
  glob: ./output/vi/ingested/*.jsonl
  document_count: 200000
  fingerprint: <sha256>
langpack:
  id: vi-example
  language_tag: vi
  version: 1.0.0
  content_hash: <sha256>
signals_impl_version: <string>
profile_digest: sha256:<hex>
candidates:
  - signal: unicode_alpha_numeric
    kind: threshold
    direction: max
    units: ratio
    bands: [...]
    note: "..."
  - signal: mean_word_length
    kind: interval
    surface_axes: {min: [...], max: [...]}
    note: "two-sided gate; choose a (min, max) pair from the surface"
```

| Field | Meaning |
| --- | --- |
| `approved` | Always `false` in a candidate. The writer refuses to emit a candidate marked approved. |
| `corpus.fingerprint` | Order-independent SHA-256 over the document text (and identifier, when present) of the profiled corpus. The flow compares it against the corpus present at application time. |
| `langpack` | Identifier, BCP-47 tag, version, and content hash of the language pack the signals were measured with. |
| `signals_impl_version` | Version of the signal implementations. |
| `profile_digest` | Content hash of `profile_report.json`, excluding its producer block, that links the policy to the report it came from. |
| `candidates` | One entry per profiled signal. A `threshold` entry lists retention-stable bands for a one-sided gate; an `interval` entry lists the axes of the retention surface for a two-sided gate. |

The candidate entries describe what each threshold would remove.
They do not establish that what it removes is low quality; that judgement is the approval.

## Approved Policy

An approved policy carries the candidate's provenance forward and adds the chosen thresholds and an approval record:

```yaml
schema_version: 1
approved: true
corpus:
  glob: ./output/vi/ingested/*.jsonl
  document_count: 200000
  fingerprint: <sha256>
langpack:
  id: vi-example
  language_tag: vi
  version: 1.0.0
  content_hash: <sha256>
signals_impl_version: <string>
profile_digest: sha256:<hex>
approval:
  approver: you@example.com
  date: 2026-01-01
  method: manual
  evidence: >-
    What was examined and why these values were chosen.
thresholds:
  - {signal: unicode_alpha_numeric, max: 0.3333}
  - {signal: numbers_ratio, max: 0.2540}
  - {signal: script_ratio, min: 0.4286}
```

### Required Fields

An approved policy must carry `schema_version`, `approved`, `corpus`, `signals_impl_version`, `profile_digest`, and `thresholds`.
Validation additionally requires:

| Field | Rule |
| --- | --- |
| `schema_version` | Must be `1`. |
| `approved` | Must be `true` for the policy to be executable. |
| `signals_impl_version` | Non-empty string. |
| `profile_digest` | A SHA-256 digest. |
| `corpus.fingerprint` | A SHA-256 digest. |
| `approval.approver` | Non-empty when `approved` is `true`. |
| `approval.evidence` | Non-empty when `approved` is `true`. |
| `approval.method` | `manual` or `ablation`. |
| `approval.date` | Optional provenance. |
| `thresholds` | Non-empty list. Each entry names a `signal` from the registry and carries `min`, `max`, or both. A signal whose direction is `max` must carry `max`; a `min` signal must carry `min`; an `interval` signal must carry both, with `min` not greater than `max`. |

A policy names signals by their registry name only.
It cannot name an import path.

### Promotion Through the Flow

The flow driver promotes a candidate when the flow configuration carries an `approve` block; refer to {doc}`flow-config`.
The promoted document is written to `<output_root>/policy/approved_policy.yaml` and wired into `curate/nemo_curator` automatically.
Promotion refuses a candidate that is already approved, an empty threshold list, a missing `approver` or `evidence`, and a threshold for a signal the profile did not measure.

### Manual Promotion

A policy may also be written by hand, for example when thresholds come from an earlier project.
Set `heuristic_filters.approved_policy` to the file path.
Every rule in the table above still applies, including the SHA-256 `corpus.fingerprint` and `profile_digest`; a hand-written policy that cannot supply them is not approved and requires the override described next.

## Unvalidated Override

```yaml
heuristic_filters:
  approved_policy: ./output/vi/profile/candidate_policies.yaml
  allow_unvalidated_policy: true
```

With `allow_unvalidated_policy: true`, `curate/nemo_curator` executes a policy that does not meet the approval contract, logs a warning that names the unmet rules on every run, and records `policy.status: override_unvalidated` in `run_manifest.json`.
The flow report then carries `policy_status: override_unvalidated`.
The override records that a policy was applied without approval; it does not make the policy approved.

## Policy Status in the Manifest

`run_manifest.json` records one of four states under `policy.status`:

| Status | Meaning |
| --- | --- |
| `approved` | An approved policy was applied. |
| `override_unvalidated` | A policy was applied under `allow_unvalidated_policy`. |
| `unapproved` | A policy was configured but applied no thresholds. |
| `unresolved` | The policy file could not be loaded or resolved. |

## Related Pages

- {doc}`../explanation/two-run-curation`
- {doc}`../how-to/apply-a-known-policy`
- {doc}`../how-to/evaluate-a-policy`
- [`src/nemotron/steps/curate/nemo_curator/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/curate/nemo_curator/README.md)
