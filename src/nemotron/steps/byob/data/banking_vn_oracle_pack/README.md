# banking_vn oracle pack

Bundled reference pack for BFCL Vietnamese banking and payments.
Offline only — no network. Process-worker isolation is required for gold claims.

Pack id: `banking_vn`
Tools (9): `get_account_balance`, `get_card_limit`, `get_transaction_status`,
`list_recent_transactions`, `get_transfer_fee`, `create_transfer`,
`get_vietqr_payment_status`, `get_dispute_status`, `create_dispute`

Currency: VND. Rails: `napas` / `internal`. QR: VietQR only.

Absent ids (documented, never inserted):
- `ACC-ABSENT-1`, `CARD-ABSENT-1`
- `TXN-ABSENT-1` through `TXN-ABSENT-8`
- `TRF-ABSENT-1`, `VQ-ABSENT-1`, `DSP-ABSENT-1`

Templates cover every policy edge the pipeline supports — `single_turn`,
`missing_slot`, `confirmation`, `correction` (transfer amount replaced and
re-confirmed before the transfer is issued), `multi_tool` (parallel call
groups), `dependent_call` (arguments read from recent-transaction, VietQR, or
dispute results), `negative_path` (unknown ids and an insufficient-funds
rejection), `clarify_only`, and `irrelevant` — and every template exposes at
least one distractor tool.

The current [Gold publication configuration](../../bfcl/config/banking_vn.gold.yaml)
binds 232 deterministic cases in each category (1,392 total). Scale comes from
real slot inventory: fixture-backed accounts, cards, transactions, VietQR
payments and disputes; mutation-safe `dispute_eligible` transactions crossed
with valid dispute reasons; and a 29-service × 8-topic unsupported-service
matrix. It does not copy rows or count paraphrases as new task semantics.

That configuration also enables challenge selection. The 28 canonical
templates include dependent transaction lookups from recent activity, VietQR
payments, and disputes; a fee-then-transfer confirmation flow; parallel
balance/activity reconciliation; withheld-slot recovery across four domains;
and a transaction-id correction flow. Stage 4 supplies 1,896 pack-specific
candidates, from which Stage 11 selects exactly 1,392 rows at the declared 45%
easy, 20% medium, 35% hard, and 23.6% rendered-multiturn mix. These values
describe this pack and are not BFCL framework defaults.

For model-authored wording diversity, use
[`banking_vn.gold.paraphrase.yaml`](../../bfcl/config/banking_vn.gold.paraphrase.yaml).
It keeps the same executable publication target and requests one guarded
Vietnamese variant only for templates that explicitly opt in. As shipped it
routes through the Data Designer provider `nvidia_inference_api` and reads
`NGC_API_KEY`; both are deployment choices you can replace. The profile fails
closed unless the selected set reaches 1,392 rows, at
least 15% exact masked-surface uniqueness, and no more than eight uses of one
exact masked surface. Model variants retain their canonical executable-case
identity and immutable request/response cache lineage.

Those gates are reachable here only because of the style axes. The 28 templates
supply 28 canonical masked surfaces, while the 18 paraphrase-eligible templates
crossed with the 20 style axes reach about 340 more, for roughly 368 distinct
masked surfaces over a 3,364-row candidate pool. That admits about 1,592 rows
under the eight-use cap, against the 1,392 target. Without the axes the same
model collapses onto one wording per template, the pool holds 28 surfaces, and
Stage 11 caps publication at 186 rows and aborts.

Frozen clock: `2026-03-02T09:00:00+07:00` (`clock_step: null`).
