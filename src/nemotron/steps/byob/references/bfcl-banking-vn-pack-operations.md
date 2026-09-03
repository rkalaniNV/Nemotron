# banking_vn oracle pack: file map, commands, and release record

This document sits outside
[`data/banking_vn_oracle_pack/`](../data/banking_vn_oracle_pack/README.md) on
purpose. An eval verifies that the pack it executes is byte-for-byte the pack
that certified the gold traces it is scoring, and that check hashes *every* file
in the pack directory — documentation included, because a fingerprint blind to a
file cannot tell a comment from a policy the backend reads. A published
benchmark therefore freezes the whole directory: editing even the pack's
`README.md` makes every eval of `banking-vn-gold-v1-1392` fail with
`eval_source_oracle_pack_drift` until the bytes are restored.

Notes about the pack that are not themselves pack inputs belong here, where they
can be revised freely. The pack's own `README.md` is frozen at the revision
generation recorded.

## File map

- `manifest.yaml`: identity, Vietnamese prompts, frozen clock, primary keys,
  absent IDs, and authoritative paths.
- `tools.json`: nine strict schemas, including mutation and confirmation flags.
- `fixtures.json`: deterministic account, card, transaction, transfer, VietQR,
  and dispute state plus template slot inventory.
- `backend.py`: isolated reset/state, deterministic IDs/results, business errors,
  confirmation, and mutation semantics.
- `task_templates.yaml`: 42 templates spanning every supported turn policy.
- `assertions.py`: result, state, dependency, confirmation, error, and no-tool checks.
- `validation_cases.yaml`: direct success, invalid, not-found, safe confirmation,
  and committed-mutation probes for the tools.
- `README.md`: slot inventory, publication scale, and the declared mixes.

## Validate, smoke, and generate

From the repository root:

```bash
SMOKE="$(pwd)/src/nemotron/steps/byob/bfcl/config/banking_vn.yaml"
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config "$SMOKE" \
  --output-dir /tmp/bfcl-banking-vn-validation
python -m nemotron.steps.byob.scripts.run \
  --config "$SMOKE" \
  --stage prepare
python -m nemotron.steps.byob.scripts.run \
  --config "$SMOKE" \
  --stage generate
```

Validation exits zero only after schema, direct probes, deterministic replay,
assertions, reset, and confirmation safety pass. Inspect
`oracle_validation_report.json`, stage reports, both Parquet files, and
`run_manifest.json` beneath `/tmp/bfcl/banking_vn_validation/`.

The deterministic Gold profile targets 232 tasks in each of six categories:

```bash
python -m nemotron.steps.byob.scripts.run \
  --config "$(pwd)/src/nemotron/steps/byob/bfcl/config/banking_vn.gold.yaml" \
  --stage all
```

The paraphrased profile is separate because it requires model credentials and
records model-exposure provenance:

```bash
export NGC_API_KEY=REPLACE_ME
python -m nemotron.steps.byob.scripts.run \
  --config "$(pwd)/src/nemotron/steps/byob/bfcl/config/banking_vn.gold.paraphrase.yaml" \
  --stage all
```

Any of these that regenerate into the pack directory, or that are run with an
edited pack, produce a benchmark certified against different bytes. That is
correct behaviour, not a nuisance: the resulting benchmark is a different
artifact and cannot be scored against the old one.

## Reference release

The scale claims in the pack's `README.md` are not projections. This pack
produced a gold release on 2026-09-01 under `banking_vn.gold.paraphrase.yaml`,
and a local snapshot of it is kept at
`BFCL/releases/banking-vn-gold-v1-1392/` alongside this checkout. Read that
snapshot rather than regenerating when you need the numbers a statement there
depends on.

The release identifies itself by content, not by path, so the snapshot is
verifiable wherever it is mounted:

| Identity | Value |
| --- | --- |
| `run_id` | `bfcl_banking_vn_gold_paraphrase_v1_1392-20260901T233211455629Z-277950452534-cb1472102ab04cadb74be58666e1160b` |
| pack `content_hash` | `sha256:f1d6ab3ae97df6c1090cd46031484aa1c4e5c91e87d3f5ccde346e3e7d645718` |
| `benchmark.parquet` | `sha256:d40ba8d3ec5fd7778a42a0f4359feacfec14e6be08cf4a26de14c3ef922e58f6` |
| `benchmark_raw.parquet` | `sha256:e988c246dccbafbf5a2c3638f2204a8de1c10da97f85160bffb3ddbc01ae1d94` |
| `generation_config_hash` | `sha256:9dec917235992be2b2d888016ca27ecf9daab7a1f141abf530b7522ad577be48` |

The pack `content_hash` is the one that matters for reading the snapshot as
evidence about the pack files: the manifest recorded it from the eight pack
files that generation actually loaded, so it is what ties those rows to that
directory rather than to a copy that had drifted from it. Those eight files
still hash to that value, and they do so only as long as the directory is left
alone — which is why this document is not in it.

Every count below comes from `run_manifest.json` in that snapshot, which is
retained in full — `stage_counts` for the funnel and
`semantic_deduplication.report.actual_counts` for the realized mixes.

Stage 4 supplied 2,824 canonical candidates, one guarded Vietnamese variant was
requested for each eligible binding (2,568 requested, 2,542 accepted, 26
rejected), and every one of the 5,366 expanded rows passed replay, schema, and
surface-quality validation. Stage 11 then selected exactly 1,392 rows and
reported no unmet target, which is the concrete form of the fail-closed claim
the pack's `README.md` makes. The realized `turn_policy` counts are
`single_turn` 265, `irrelevant` 232, `dependent_call` 209, `clarify_only` 144,
`missing_slot` 139, `correction` 111, `confirmation` 98, `multi_tool` 97, and
`negative_path` 97 — so slightly over four fifths of the release exercises
something other than a plain lookup. Difficulty landed at 348 easy, 418 medium,
and 626 hard, turns at 974 single and 418 multi, and all six categories at 232.

The release has also been scored end to end rather than only for trace
plausibility, which is the point of shipping an executable pack: `gpt-oss-120b`
reached 53.4% task success (744 of 1,392) in combined `trace` and `executable`
mode on 2026-09-01. Treat that figure as a recorded result, not as something the
local snapshot proves — the eval artifacts are deliberately not kept here, so
reproducing or auditing it means re-running the eval against
`benchmark.parquet`.
