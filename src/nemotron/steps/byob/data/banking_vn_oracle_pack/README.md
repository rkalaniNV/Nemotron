# banking_vn oracle pack

Bundled reference pack for BFCL Vietnamese banking and payments.
Offline only — no network. Process-worker isolation is required for gold claims.

Pack id: `banking_vn`
Tools (9): `get_account_balance`, `get_card_limit`, `get_transaction_status`,
`list_recent_transactions`, `get_transfer_fee`, `create_transfer`,
`get_vietqr_payment_status`, `get_dispute_status`, `create_dispute`

Currency: VND. Rails: `napas` / `internal`. QR: VietQR only.

Absent ids (documented, never inserted):
- `ACC-ABSENT-1`, `CARD-ABSENT-1`, `TXN-ABSENT-1`, `TXN-ABSENT-2`
- `TRF-ABSENT-1`, `VQ-ABSENT-1`, `DSP-ABSENT-1`

Templates cover every policy edge the pipeline supports — `single_turn`,
`missing_slot`, `confirmation`, `correction` (transfer amount replaced and
re-confirmed before the transfer is issued), `multi_tool` (one parallel
`call_group`), `dependent_call` (latest transaction id read out of
`list_recent_transactions`), `negative_path` (unknown id and an insufficient-funds
rejection), `clarify_only`, and `irrelevant` — and every template exposes at least
one distractor tool.

Frozen clock: `2026-03-02T09:00:00+07:00` (`clock_step: null`).
