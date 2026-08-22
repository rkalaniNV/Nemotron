# BFCL ablation — arm `a3`

Pack: `/localhome/local-hndo/Nemotron/bfcl_ablation/_generated/packs/a3_full`

## 1. Authoring friction

| file | lines |
| --- | ---: |
| backend.py | 465 |
| task_templates.yaml | 371 |
| validation_cases.yaml | 71 |
| assertions.py | 182 |
| tools.json | 162 |
| fixtures.json | 68 |
| manifest.yaml | 28 |
| run_config | 43 |
| **TOTAL** | **1390** |

20 templates across 6 categories and 6 turn policies (18.6 template lines each).

## 2. Distribution — joint (category x policy)

Cell = task count. `.` = feasible but unwritten. `--` = structurally empty.

| category | clarify_only | confirmation | correction | dependent_call | irrelevant | missing_slot | multi_tool | negative_path | single_turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balance_inquiry | 1 | -- | . | . | 1 | . | 4 | . | . |
| dispute | 1 | . | . | . | 1 | 1 | . | 1 | . |
| out_of_scope | . | -- | -- | -- | 2 | -- | -- | -- | -- |
| qr_payment | 1 | -- | . | . | 2 | . | . | 1 | . |
| transaction_status | 3 | -- | . | . | 1 | . | . | 1 | 1 |
| transfer | 5 | . | . | . | 1 | . | . | . | . |

17/54 cells populated; 27 unwritten; 10 structurally empty.

> Structural emptiness is judged against the arm's declared category tool universes and, for dependent_call, against producer/consumer edges probed from the backend.

| turn_policy | tasks | share |
| --- | ---: | ---: |
| clarify_only | 11 | 39.3% |
| irrelevant | 8 | 28.6% |
| missing_slot | 1 | 3.6% |
| multi_tool | 4 | 14.3% |
| negative_path | 3 | 10.7% |
| single_turn | 1 | 3.6% |

## 3. Coverage

| collection | key | rows | bound | coverage | never bound |
| --- | --- | ---: | ---: | ---: | --- |
| accounts | account_id | 8 | 5 | 62% | ACC-006, ACC-007, ACC-008 |
| cards | card_id | 4 | 4 | 100% | - |
| disputes | dispute_id | 4 | 1 | 25% | DSP-0002, DSP-0003, DSP-0004 |
| fee_schedule | None | 6 | 0 | n/a | - |
| transactions | transaction_id | 16 | 1 | 6% | TXN-1002, TXN-1003, TXN-1004, TXN-1005, TXN-1006, TXN-1007 (+9) |
| transfer_scenarios | None | 2 | 0 | n/a | - |
| transfers | transfer_id | 4 | 0 | 0% | TRF-0001, TRF-0002, TRF-0003, TRF-0004 |
| vietqr_payments | payment_ref | 6 | 0 | 0% | VQ-1001, VQ-1002, VQ-1003, VQ-1004, VQ-1005, VQ-1006 |

Fixture entities bound: 11/50.
Tools never called in any expected trace: `create_dispute`, `create_transfer`, `get_transfer_fee`, `get_vietqr_payment_status`, `list_recent_transactions`.

## 4. Surface diversity

28 tasks -> 28 distinct raw first turns -> **20 distinct slot-masked** (1.0 per template).

Slot-masking substitutes each bound value with its slot name, so two tasks that 
differ only by account id collapse to one sentence.

| category | tasks | distinct masked | ratio |
| --- | ---: | ---: | ---: |
| balance_inquiry | 6 | 3 | 0.50 |
| dispute | 4 | 4 | 1.00 |
| out_of_scope | 2 | 2 | 1.00 |
| qr_payment | 4 | 4 | 1.00 |
| transaction_status | 6 | 5 | 0.83 |
| transfer | 6 | 2 | 0.33 |

Most-repeated masked utterances:

- x5 — `Mình muốn chuyển tiền từ tài khoản {account_id}.`
- x4 — `Mình muốn kiểm tra số dư tài khoản {account_id} và hạn mức thẻ {card_id}.`
- x2 — `Mình muốn biết trạng thái giao dịch cho tài khoản {account_id}.`

**Lexical-shortcut probe:** not runnable. A generalization-gap probe needs several phrasings per intent. This arm produced 20 distinct masked surfaces across 20 templates; with one surface per template there is no held-out phrasing to test on. The probe becomes available at A2.

## 5. Publish funnel

| stage | rows | survival | lost here |
| --- | ---: | ---: | ---: |
| expand | 28 | 100.0% | 0 |
| state_machine | 28 | 100.0% | 0 |
| render_accepted | 28 | 100.0% | 0 |
| expected_trace_derived | 28 | 100.0% | 0 |
| schema_valid | 28 | 100.0% | 0 |
| replay_valid | 28 | 100.0% | 0 |
| benchmark_raw | 28 | 100.0% | 0 |
| published | 28 | 100.0% | 0 |


Publish rate 100.0% (28/28). Gold rows 28/28 (100.0%). Run gold_eligible: `True`, mode `template_only`.

## 6. Proposal accept/drop

56 proposals requested over 44 cells; 56 returned; **20 accepted (35.7%)**.

The authored line count in section 1 is the size of a pack the model wrote, not human friction: no person authored a task in this arm. It is comparable with A0's and A1's line counts only as a measure of how much pack a given number of tasks costs.

| drop bucket | proposals |
| --- | ---: |
| assertion_failed | 5 |
| expected_trace_invalid | 4 |
| generation_failed | 8 |
| plan_invalid | 3 |
| schema_invalid | 13 |
| slot_source_invalid | 3 |

| gate | dropped |
| --- | ---: |
| compile | 3 |
| oracle_validation | 20 |
| schema | 13 |

## 7. Coverage against the spec

44 feasible cells, 10 declared structurally empty. 17 feasible cells covered (38.6%), 15 met their target.

Feasible cells the model could not fill:

- `balance_inquiry` x `single_turn` (target 1)
- `balance_inquiry` x `missing_slot` (target 1)
- `balance_inquiry` x `correction` (target 1)
- `balance_inquiry` x `dependent_call` (target 2)
- `balance_inquiry` x `negative_path` (target 1)
- `dispute` x `single_turn` (target 1)
- `dispute` x `confirmation` (target 2)
- `dispute` x `correction` (target 2)
- `dispute` x `multi_tool` (target 1)
- `dispute` x `dependent_call` (target 1)
- `out_of_scope` x `clarify_only` (target 1)
- `qr_payment` x `single_turn` (target 1)
- `qr_payment` x `missing_slot` (target 1)
- `qr_payment` x `correction` (target 1)
- `qr_payment` x `multi_tool` (target 1)
- `qr_payment` x `dependent_call` (target 2)
- `transaction_status` x `missing_slot` (target 1)
- `transaction_status` x `correction` (target 1)
- `transaction_status` x `multi_tool` (target 1)
- `transaction_status` x `dependent_call` (target 2)
- `transfer` x `single_turn` (target 2)
- `transfer` x `missing_slot` (target 1)
- `transfer` x `confirmation` (target 1)
- `transfer` x `correction` (target 1)
- `transfer` x `multi_tool` (target 2)
- `transfer` x `dependent_call` (target 1)
- `transfer` x `negative_path` (target 1)

Cells declared structurally empty, with the claim that makes them so:

- `balance_inquiry` x `confirmation` — no tool in this category requires confirmation
- `out_of_scope` x `single_turn` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `missing_slot` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `confirmation` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `correction` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `multi_tool` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `dependent_call` — category exposes no tool, and every other policy must call one
- `out_of_scope` x `negative_path` — category exposes no tool, and every other policy must call one
- `qr_payment` x `confirmation` — no tool in this category requires confirmation
- `transaction_status` x `confirmation` — no tool in this category requires confirmation

## 8. Selection bias

### Tool choice, against uniform within each category

| tool | observed | expected | obs/exp |
| --- | ---: | ---: | ---: |
| create_dispute | 0 | 0.67 | 0.0 |
| create_transfer | 0 | 0.0 | - |
| get_account_balance | 1 | 1.0 | 1.0 |
| get_card_limit | 1 | 1.0 | 1.0 |
| get_dispute_status | 1 | 0.67 | 1.5 |
| get_transaction_status | 4 | 2.17 | 1.85 |
| get_transfer_fee | 0 | 0.0 | - |
| get_vietqr_payment_status | 0 | 0.5 | 0.0 |
| list_recent_transactions | 0 | 1.0 | 0.0 |

Pearson statistic against the conditional-uniform null: **3.885**. Tools never required: `create_dispute`, `create_transfer`, `get_transfer_fee`, `get_vietqr_payment_status`, `list_recent_transactions`.

### Accept rate by policy — bias that survives a controlled sampler

| policy | proposed | accepted | rate |
| --- | ---: | ---: | ---: |
| clarify_only | 7 | 6 | 0.8571 |
| confirmation | 3 | 0 | 0.0 |
| correction | 6 | 0 | 0.0 |
| dependent_call | 8 | 0 | 0.0 |
| irrelevant | 8 | 8 | 1.0 |
| missing_slot | 6 | 1 | 0.1667 |
| multi_tool | 7 | 1 | 0.1429 |
| negative_path | 5 | 3 | 0.6 |
| single_turn | 6 | 1 | 0.1667 |

Spread between the easiest and hardest policy: **1.0**.

### Entity choice

11/42 fixture rows bound (26.2%).

| collection | rows | bound | TVD from uniform |
| --- | ---: | ---: | ---: |
| accounts | 8 | 5 | 0.4773 |
| cards | 4 | 4 | 0.0 |
| disputes | 4 | 1 | 0.75 |
| transactions | 16 | 1 | 0.9375 |
| transfers | 4 | 0 | 1.0 |
| vietqr_payments | 6 | 0 | 1.0 |

### What the oracle could not check

14 of 20 accepted templates (70.0%) are clarify_only or irrelevant. Their expected trace is empty by construction and their only available assertion, `assert_no_tool_called`, passes exactly when the trace is empty — so replay, determinism and assertions all succeed regardless of what the request says. Their accept rate is not evidence that the model got them right.

Of those, these declare a tool whose every required parameter the customer already stated, so the request was answerable and the gold behaviour is wrong:

- `txn_status_missing_account_id` (clarify_only) — answerable by `list_recent_transactions`
- `txn_status_missing_txn_id` (clarify_only) — answerable by `get_transaction_status`


### Against A0's human-authored mix

- `required_tools`: total variation distance A3 vs A0 = **0.5042**; only A3 uses nothing; only A0 uses `create_dispute`, `create_transfer`, `get_transfer_fee`, `get_vietqr_payment_status`, `list_recent_transactions`
- `success_assertions`: total variation distance A3 vs A0 = **0.6431**; only A3 uses nothing; only A0 uses `assert_dispute_opened`, `assert_only_corrected_amount_transferred`, `assert_recent_transactions_listed`, `assert_status_checked_from_listed_transaction`, `assert_transfer_committed`, `assert_transfer_fee_reported`, `assert_transfer_rejected_for_funds`, `assert_vietqr_status_reported`

## 9. Does the pack still reach gold

Validation rounds: 3. Final tier `gold`, gold_eligible=True.

Published 28/28 expanded tasks (publish rate 100.0%), gold rows 28 (100.0%).

No template lost an instance after validation.
