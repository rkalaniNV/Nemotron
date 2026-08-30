# Curate Flow

One config, five steps, one command. You provide this file and your data.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## What It Removes

Running the category by hand is five configs plus a hand-written approved
policy, and the paths between them have to agree. Two of those agreements fail
*silently*:

- `curate/nemo_curator` ships `emit_manifest: null`; `curate/audit` ships
  `declared_manifest: null`. Two nulls in two files. An audit against a producer
  that emitted no manifest reports counts as **informational and claims
  nothing** — which reads exactly like a clean result.
- `curate/subset` can stratify on a score column that `curate/nemo_curator`
  only writes in `mode: annotate` or `both`.

The flow derives both from one place, so they cannot disagree, and refuses the
second at preflight instead of at the point of use.

## Where To Start

Four configs ship with the step, run by name:

| `-c <name>` | For |
|---|---|
| `tiny` | smoke test on packaged fixtures, CPU, no downloads |
| `default` | the shape, commented, to copy |
| `vi_c4` | worked example: Vietnamese web corpus |
| `hi_sangraha` | worked example: Hindi with mixed web/OCR/ASR provenance |

```bash
uv run nemotron steps run curate/flow -c tiny            # verify it runs
uv run nemotron steps run curate/flow -c vi_c4 \
    corpus.input='./raw_jsonl/*.jsonl' output_root=./output/vi
```

Copy `vi_c4.yaml` next to your data and edit it — that is your config. Every
number in it was measured on real documents, and the two examples differ on
purpose: the Hindi one shows a pack that declares fewer capabilities, and a
provenance column that changes which gate is safe.

## Run It Twice, On Purpose

```bash
# 1. measure, filter without a policy, check
uv run nemotron steps run curate/flow -c ./vi_c4.yaml

# read output/curate/profile/profile_report.json, decide, write the approve block

# 2. apply thresholds a person signed for
uv run nemotron steps run curate/flow -c ./vi_c4.yaml
```

**The gate is not collapsible into one run.** `candidate_policies.yaml` does not
exist until the profile has read your corpus, so on run 1 there is nothing to
approve — and the filter runs with no threshold gating rather than with a
default someone would inherit without reading.

Writing the `approve:` block is the deliberate act. It goes through
`runtime/policy.py::promote`, the only function in `steps/curate` permitted to
mark a policy approved, which checks that the bound direction matches the
signal, that the signal was actually profiled on this corpus, and that a corpus
fingerprint is present.

On top of that the flow **recomputes the corpus fingerprint and compares it to
the one the approval was granted against**. That is what stops a config being
copied to a different corpus with its thresholds *and* its approval signature
attached — the failure mode a shared config file otherwise invites.

## Preview Before Running

```bash
uv run nemotron steps run curate/flow -c ./vi_c4.yaml --plan
```

Writes `flow_plan.json` with every derived per-step config and prints what would
run. Nothing executes. A flow that fails forty minutes in, after the filter has
rewritten the corpus, is worse than five separate commands — so everything
refusable is refused before the first step starts.

## `enabled: false`

A disabled step is skipped. A later step that needs its output either **reuses**
a previous run's artifact under the same `output_root`, or the flow **refuses**
and names the step that would have produced it.

Refusing matters more than it sounds. Two cases would otherwise look like
results:

| Disabled | Downstream symptom without the check |
|---|---|
| `filter` | audit reports `attribution.available: false` — "nobody recorded why records left", not "no records left" |
| `filter` in `mode: filter` | subset stratifies on a `__signal` column that was never written |

## Order, And Why

```
profile   reads the UNFILTERED corpus
   │      (profiling the output measures gates that already ran)
   ▼
filter    corpus -> filtered_jsonl + run_manifest + ledger
   │
   ├─► audit             completeness + attribution, against the derived manifest
   ├─► subset            nested token-budget tiers
   └─► decontamination   holdout overlap  ← the only step needing a GPU
```

## GPU

`curate/flow` declares `gpus_per_node = 0`, because four of the five steps need
none. Enabling `decontamination` with the similarity pass needs
`run.env.gpus_per_node: 1` in your config, or `skip_similarity: true` to run the
exact source-identity pass on CPU alone.

## Output Layout

Everything lands under `output_root`:

```
output/curate/
├── flow_plan.json        derived per-step configs, written before anything runs
├── flow_report.json      what ran, what was skipped, every warning, audit verdict
├── profile/              profile_report.json, candidate_policies.yaml, sample_manifest.json
├── policy/               approved_policy.yaml   (only once you write approve:)
├── filtered_jsonl/       the corpus, run_manifest.json, curation_ledger.json
├── audit/                audit_report.json
├── subset/               plan.json, budget_<N>_<unit>/, subset_report.json
└── decontaminated/       train_decontaminated.jsonl, decontamination_report.json
```

## Escape Hatches

Any key you write inside a `steps.<name>` block overrides the derivation. A flow
config is not a cage — an escape hatch that needs a second file is not one.

The five steps remain independently runnable; this step does not replace them.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_flow.py`
- Each step's own contract: `../{profile,nemo_curator,audit,subset,decontamination}/README.md`

## Guardrails

- Read `flow_report.json`'s `warnings` before quoting any number from a run.
- `audit_passed: false` fails the flow's exit code. A flow that "succeeded"
  while its audit failed would be the same silent success the audit exists to
  catch.
- Do not set `approve.verify_corpus: false` to make a copied config run. That
  check is the one thing standing between a shared config and an approval
  signature applied to data nobody approved it for.
