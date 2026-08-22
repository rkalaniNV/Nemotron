# BFCL ablation — arm `a1`

Pack: `/localhome/local-hndo/Nemotron/bfcl_ablation/_generated/packs/a1_full`

## 1. Authoring friction

| file | lines |
| --- | ---: |
| backend.py | 465 |
| task_templates.yaml | 393 |
| validation_cases.yaml | 71 |
| assertions.py | 182 |
| tools.json | 162 |
| fixtures.json | 68 |
| manifest.yaml | 28 |
| run_config | 43 |
| **TOTAL** | **1412** |

17 templates across 6 categories and 9 turn policies (23.1 template lines each).

## 2. Distribution — joint (category x policy)

Cell = task count. `.` = feasible but unwritten. `--` = structurally empty.

| category | clarify_only | confirmation | correction | dependent_call | irrelevant | missing_slot | multi_tool | negative_path | single_turn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balance_inquiry | 1 | -- | . | . | . | 1 | 1 | . | 3 |
| dispute | . | 3 | . | . | . | . | . | . | 3 |
| out_of_scope | . | -- | . | . | 3 | . | . | . | . |
| qr_payment | . | -- | . | . | . | . | . | . | 6 |
| transaction_status | . | -- | . | 1 | . | . | . | 1 | 4 |
| transfer | . | 2 | 1 | . | . | . | . | 1 | 2 |

15/54 cells populated; 35 unwritten; 4 structurally empty.

> Structural emptiness is inferred from the union of tools_present across the templates a category already has. A category whose templates all expose one tool is reported as unable to host multi_tool even if the domain could.

| turn_policy | tasks | share |
| --- | ---: | ---: |
| clarify_only | 1 | 3.0% |
| confirmation | 5 | 15.2% |
| correction | 1 | 3.0% |
| dependent_call | 1 | 3.0% |
| irrelevant | 3 | 9.1% |
| missing_slot | 1 | 3.0% |
| multi_tool | 1 | 3.0% |
| negative_path | 2 | 6.1% |
| single_turn | 18 | 54.5% |

## 3. Coverage

| collection | key | rows | bound | coverage | never bound |
| --- | --- | ---: | ---: | ---: | --- |
| accounts | account_id | 8 | 3 | 38% | ACC-003, ACC-004, ACC-005, ACC-007, ACC-008 |
| cards | card_id | 4 | 1 | 25% | CARD-002, CARD-003, CARD-004 |
| disputes | dispute_id | 4 | 3 | 75% | DSP-0004 |
| fee_schedule | None | 6 | 0 | n/a | - |
| transactions | transaction_id | 16 | 4 | 25% | TXN-1003, TXN-1004, TXN-1005, TXN-1006, TXN-1007, TXN-1008 (+6) |
| transfer_scenarios | None | 2 | 0 | n/a | - |
| transfers | transfer_id | 4 | 0 | 0% | TRF-0001, TRF-0002, TRF-0003, TRF-0004 |
| vietqr_payments | payment_ref | 6 | 6 | 100% | - |

Fixture entities bound: 17/50.
Every declared tool appears in at least one expected trace.

## 4. Surface diversity

33 tasks -> 33 distinct raw first turns -> **17 distinct slot-masked** (1.0 per template).

Slot-masking substitutes each bound value with its slot name, so two tasks that 
differ only by account id collapse to one sentence.

| category | tasks | distinct masked | ratio |
| --- | ---: | ---: | ---: |
| balance_inquiry | 6 | 5 | 0.83 |
| dispute | 6 | 2 | 0.33 |
| out_of_scope | 3 | 1 | 0.33 |
| qr_payment | 6 | 1 | 0.17 |
| transaction_status | 6 | 4 | 0.67 |
| transfer | 6 | 4 | 0.67 |

Most-repeated masked utterances:

- x6 — `Mã VietQR {payment_ref} đã trừ tiền chưa?`
- x3 — `Hồ sơ tra soát {dispute_id} đang ở trạng thái nào?`
- x3 — `Mở tra soát cho giao dịch {transaction_id}, lý do {reason}.`
- x3 — `Mình muốn đăng ký {service} thì làm thủ tục thế nào?`
- x2 — `Cho mình xem số dư tài khoản {account_id}.`

**Lexical-shortcut probe:** not runnable. A generalization-gap probe needs several phrasings per intent. This arm produced 17 distinct masked surfaces across 17 templates; with one surface per template there is no held-out phrasing to test on. The probe becomes available at A2.

## 5. Publish funnel

| stage | rows | survival | lost here |
| --- | ---: | ---: | ---: |
| expand | 33 | 100.0% | 0 |
| state_machine | 33 | 100.0% | 0 |
| render_accepted | 33 | 100.0% | 0 |
| expected_trace_derived | 33 | 100.0% | 0 |
| schema_valid | 33 | 100.0% | 0 |
| replay_valid | 33 | 100.0% | 0 |
| benchmark_raw | 33 | 100.0% | 0 |
| published | 33 | 100.0% | 0 |


Publish rate 100.0% (33/33). Gold rows 33/33 (100.0%). Run gold_eligible: `True`, mode `template_only`.

## 6. Simplification

| file | A0 | A1 | saved |
| --- | ---: | ---: | ---: |
| backend.py | 465 | 465 | 0 |
| task_templates.yaml | 473 | 393 | 80 |
| validation_cases.yaml | 199 | 71 | 128 |
| assertions.py | 182 | 182 | 0 |
| tools.json | 162 | 162 | 0 |
| fixtures.json | 68 | 68 | 0 |
| manifest.yaml | 50 | 28 | 22 |
| run_config | 43 | 43 | 0 |
| **TOTAL** | 1642 | 1412 | 230 |

**230 lines removed (14.0%) with no model involved.**

### Field-level derivability

| verdict | fields |
| --- | ---: |
| derived_exact | 39 |
| not_derivable | 1 |
| surface_changed | 6 |

Fields that stayed authored because derivation did not reproduce them:

- `bn_balance_and_card_parallel` / `call_order` — authored="any" derived="strict"

### Milestone-compiler rules the round trip does not vouch for

The compiler chooses among the milestone lists `_check_policy_shape` would accept. Every choice below compiles, but `banking_vn` contains no template that exercises it, so nothing has checked the choice against a human's intent.

- `confirmation` — confirmation over a multi-call batch: every confirming template in banking_vn requires exactly one tool, so 'confirm once for the batch' versus 'confirm per call' has never been decided against a human's choice
- `dependent_call` — any milestone placed between the producing and consuming call
- `missing_slot` — the multi-slot branch: banking_vn withholds exactly one slot anywhere

Validation cases: 23 authored -> 5 authored + 22 generated.

# Equivalence: `a0` vs `a1`

**EQUIVALENT**

| check | result |
| --- | --- |
| `set(task_id)` equal | YES (33 vs 33) |
| `expected_tool_calls` equal | YES (33 tasks compared) |
| `assistant_milestones` / plan equal | YES (33 tasks compared) |
| validation-case coverage held | YES (22 -> 22 (tool, outcome) pairs) |
| gold eligibility preserved | YES (True -> True) |
| published rows | 33 -> 33 |
| opening user turn identical | YES (0 changed) |
| all user turns identical | NO (8 tasks reworded) |

## Reworded turns (expected at A1)

> Simulator replies are drawn from pack-wide canonical templates at A1, so a changed first-turn surface would be a real regression while a changed later turn is the intended trade.

- `banking_vn__bn_balance_withheld_account__0f34f3476873fef5`
  - baseline:  ['Cho mình xem số dư tài khoản của mình với.', 'Tài khoản ACC-001 nhé.']
  - candidate: ['Cho mình xem số dư tài khoản của mình với.', 'ACC-001 nhé.']
- `banking_vn__bn_create_dispute_single__069e782f69b74266`
  - baseline:  ['Mở tra soát cho giao dịch TXN-1011, lý do goods_not_received.', 'Tôi xác nhận mở tra soát.']
  - candidate: ['Mở tra soát cho giao dịch TXN-1011, lý do goods_not_received.', 'Tôi xác nhận thực hiện.']
- `banking_vn__bn_create_dispute_single__9a13aff31ee10eef`
  - baseline:  ['Mở tra soát cho giao dịch TXN-1001, lý do goods_not_received.', 'Tôi xác nhận mở tra soát.']
  - candidate: ['Mở tra soát cho giao dịch TXN-1001, lý do goods_not_received.', 'Tôi xác nhận thực hiện.']
- `banking_vn__bn_create_dispute_single__fac3dccb0bfe459a`
  - baseline:  ['Mở tra soát cho giao dịch TXN-1015, lý do goods_not_received.', 'Tôi xác nhận mở tra soát.']
  - candidate: ['Mở tra soát cho giao dịch TXN-1015, lý do goods_not_received.', 'Tôi xác nhận thực hiện.']
- `banking_vn__bn_create_transfer_single__97e3d31fdcc1c1d1`
  - baseline:  ['Chuyển ngay 200000đ từ ACC-001 sang 9876543210 ngân hàng 970436 qua kênh napas.', 'Tôi xác nhận thực hiện giao dịch.']
  - candidate: ['Chuyển ngay 200000đ từ ACC-001 sang 9876543210 ngân hàng 970436 qua kênh napas.', 'Tôi xác nhận thực hiện.']

## 7. Run-config minimization (separate degree of freedom)

Run config 43 -> 17 lines; total authoring 1642 -> 1386 (256 saved, 15.6%).

Settings dropped: `surface_generation`, `surface_quality_validation`, `semantic_deduplication_config`, `config_status`, `lineage.profile_influenced_surface`, `lineage.judge_advisory`, `lineage.roles`, `oracle_runtime.tool_timeout_s`, `oracle_runtime.assertion_timeout_s`, `oracle_runtime.import_timeout_s`, `oracle_runtime.reset_timeout_s`, `oracle_runtime.episode_timeout_s`

Verdict against `a0`: **EQUIVALENT**
