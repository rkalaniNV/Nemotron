# banking_vn oracle pack

Bundled reference pack for BFCL Vietnamese banking and payments.
Offline only — no network. Process-worker isolation is required for gold claims.

Pack id: `banking_vn`
Tools (9): `get_account_balance`, `get_card_limit`, `get_transaction_status`,
`list_recent_transactions`, `get_transfer_fee`, `create_transfer`,
`get_vietqr_payment_status`, `get_dispute_status`, `create_dispute`

Currency: VND. Rails: `napas` / `internal`. QR: VietQR only.

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
- `README.md`: scale, release evidence, file map, and runnable commands.

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

Absent ids (documented, never inserted). Each one is slot inventory for a
`negative_path` template, so the count per collection bounds how many distinct
not-found cases that collection can publish:
- `ACC-ABSENT-1` through `ACC-ABSENT-8`
- `CARD-ABSENT-1` through `CARD-ABSENT-4`
- `TXN-ABSENT-1` through `TXN-ABSENT-8`
- `VQ-ABSENT-1` through `VQ-ABSENT-8`
- `DSP-ABSENT-1` through `DSP-ABSENT-8`
- `TRF-ABSENT-1`

The 42 templates cover every policy edge the pipeline supports — `single_turn`,
`missing_slot`, `confirmation`, `correction` (a read-only transaction id or a
VietQR reference replaced mid-conversation, and a transfer amount replaced and
re-confirmed before the transfer is issued, because only a mutation needs the
confirmation re-obtained), `multi_tool` (parallel call groups of two and three),
`dependent_call` (arguments read from recent-transaction, VietQR, or dispute
results, including three-call chains), `negative_path` (unknown ids in four
collections, an insufficient-funds rejection, and a dispute refused on a
non-disputable transaction), `clarify_only` (one per read category), and
`irrelevant`.

No template narrows `tools_present`, so every row offers the full nine-tool
catalog. Tool selection is therefore a nine-way choice on every row rather than
the one-of-two choice a `required + 1 distractor` exposure would give, and the
`clarify_only` and `irrelevant` rows have to decline while every plausible tool
is in reach.

The current [Gold publication configuration](../../bfcl/config/banking_vn.gold.yaml)
binds 232 deterministic cases in each category (1,392 total). Scale comes from
real slot inventory: fixture-backed accounts, cards, transactions, VietQR
payments and disputes; mutation-safe `dispute_eligible` transactions crossed
with valid dispute reasons; documented absent ids for the not-found paths; and
three unsupported-service matrices of 29 × 8, 12 × 8, and 12 × 8. It does not
copy rows or count paraphrases as new task semantics.

That configuration also enables challenge selection. Stage 4 supplies 2,824
pack-specific candidates, from which Stage 11 selects exactly 1,392 rows at the
declared 25% easy, 30% medium, 45% hard difficulty mix, a 30%
rendered-multiturn mix, and a declared `policy_mix` that reserves 15% for
dependent calls, 10% for clarify-only refusals, 8% for corrections, 7% each for
parallel calls, confirmations, and documented failures, and holds plain
single-call lookups to 19%. Two further caps make the count mean what it says:
`max_execution_case_reuse: 1` admits each executable case once, so all 1,392
published rows call different tools or different arguments against different
state, and `max_rows_per_intent: 120` keeps the cheap out-of-scope inventory
from crowding out narrower intents. These values describe this pack and are not
BFCL framework defaults.

For model-authored wording diversity, use
[`banking_vn.gold.paraphrase.yaml`](../../bfcl/config/banking_vn.gold.paraphrase.yaml).
It keeps the same executable publication target and requests one guarded
Vietnamese variant only for templates that explicitly opt in. As shipped it
routes through the Data Designer provider `nvidia_inference_api` and reads
`NGC_API_KEY`; both are deployment choices you can replace. The profile fails
closed unless the selected set reaches 1,392 rows with at least 15% exact
masked-surface uniqueness and no more than eight uses of one exact masked
surface. Model variants retain their canonical executable-case
identity and immutable request/response cache lineage.

Because a variant shares its canonical row's executable-case identity,
`max_execution_case_reuse: 1` publishes one of the two, not both. The paraphrase
profile therefore buys wording diversity — the model wrote about three quarters
of the published rows — rather than extra rows, and the executable content of the
release stays identical to the template-only profile.

Those gates are reachable here only because of the style axes. The 42 templates
supply 42 canonical masked surfaces, while the 35 paraphrase-eligible templates
crossed with the 20 style axes reach 700 more, over a candidate pool of 2,824
canonical rows plus one variant per eligible binding. Without the axes the same
model collapses onto one wording per template, the pool holds 42 surfaces, and
Stage 11 caps publication at 336 rows and aborts.

The eight-use cap has to clear the target in every category, not only in total,
because the publication set is balanced at 232 rows per category, so each
category needs at least 29 masked surfaces of its own. The thinnest category is
`out_of_scope` with three templates, and all three are paraphrase-eligible, so it
reaches 63. The declared category, difficulty, turn, and policy mixes are all met
exactly at 1,392 with no unmet target.

Frozen clock: `2026-03-02T09:00:00+07:00` (`clock_step: null`).

## Reference release

The claims above are not projections. This pack produced a gold release on
2026-09-01 under `banking_vn.gold.paraphrase.yaml`, and a local snapshot of it
is kept at `BFCL/releases/banking-vn-gold-v1-1392/` alongside this checkout.
Read that snapshot rather than regenerating when you need the numbers a
statement here depends on.

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
evidence about *these* files: the manifest recorded it from the eight pack files
that generation actually loaded, so it is what ties those rows to this directory
rather than to a copy that had drifted from it. The eight files here still hash
to that value.

Every count below comes from `run_manifest.json` in that snapshot, which is
retained in full — `stage_counts` for the funnel and
`semantic_deduplication.report.actual_counts` for the realized mixes.

Stage 4 supplied 2,824 canonical candidates, one guarded Vietnamese variant was
requested for each eligible binding (2,568 requested, 2,542 accepted, 26
rejected), and every one of the 5,366 expanded rows passed replay, schema, and
surface-quality validation. Stage 11 then selected exactly 1,392 rows and
reported no unmet target, which is the concrete form of the fail-closed claim
made above. The realized `turn_policy` counts are `single_turn` 265,
`irrelevant` 232, `dependent_call` 209, `clarify_only` 144, `missing_slot` 139,
`correction` 111, `confirmation` 98, `multi_tool` 97, and `negative_path` 97 —
so slightly over four fifths of the release exercises something other than a
plain lookup. Difficulty landed at 348 easy, 418 medium, and 626 hard, turns at
974 single and 418 multi, and all six categories at 232.

The release has also been scored end to end rather than only for trace
plausibility, which is the point of shipping an executable pack: `gpt-oss-120b`
reached 53.4% task success (744 of 1,392) in combined `trace` and `executable`
mode on 2026-09-01. Treat that figure as a recorded result, not as something the
local snapshot proves — the eval artifacts are deliberately not kept here, so
reproducing or auditing it means re-running the eval against
`benchmark.parquet`.
