# BFCL ablation — arm `a4`

Pack: `src/nemotron/steps/byob/data/banking_vn_oracle_pack`

Metric contract `1.0`. 33 tasks gated, 265 mutations, 899 oracle episodes.

## 1. False acceptance by operator class

An assertion that passes on a corrupted episode is a **false acceptance**: it is
not a check. Lower is better. Advisory operators are excluded — those are
corruptions an assertion is *right* to accept — and reported separately in §3.

| assertions | lines | call level | argument level | state level | unmutated pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `human` | 182 | 0.380 (41/108) | 0.610 (86/141) | 0.000 (0/6) | 34/34 |
| `llm_blind` | 1426 | 0.300 (30/100) | 0.248 (32/129) | 0.000 (0/4) | 32/34 |
| `llm_feedback` | 1668 | 0.000 (0/96) | 0.024 (3/123) | 0.000 (0/4) | 31/34 |
| `null_control` | — | 1.000 (108/108) | 1.000 (141/141) | 1.000 (6/6) | 34/34 |

`null_control` is an assertion suite that does nothing. It must score 1.000
everywhere; anything less would mean the harness detects corruptions by accident
and every other row is unreadable.

## 2. Per operator

| operator | class | semantics | trials | detected | false accept | rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `drop_call` | call level | strict | 33 | 31 | 2 | 0.0606 |
| `duplicate_call_mutating` | call level | strict | 7 | 1 | 6 | 0.8571 |
| `duplicate_call_readonly` | call level | advisory | 26 | 0 | 26 | 1.0000 |
| `inject_extra_call` | call level | strict | 34 | 4 | 30 | 0.8824 |
| `perturb_numeric_large` | argument level | strict | 32 | 4 | 28 | 0.8750 |
| `perturb_numeric_plus_one` | argument level | strict | 32 | 4 | 28 | 0.8750 |
| `reorder_calls` | call level | strict | 1 | 0 | 1 | 1.0000 |
| `state_reverted` | state level | strict | 6 | 6 | 0 | 0.0000 |
| `swap_identity_argument` | argument level | strict | 33 | 28 | 5 | 0.1515 |
| `swap_identity_result` | argument level | strict | 44 | 19 | 25 | 0.5682 |
| `swap_tool` | call level | strict | 33 | 31 | 2 | 0.0606 |

Rates above are the **human** suite — the pack as authored.

## 3. Over-strictness (advisory operators)

Detection on an advisory operator is not a win. `duplicate_call_readonly` repeats
an idempotent read: the episode is wasteful, not wrong, so an assertion that
rejects it is refusing correct behaviour. This is the counterweight to §1 — an arm
can always buy a lower false-acceptance rate by rejecting more.

| assertions | advisory trials | rejected | rejection rate |
| --- | ---: | ---: | ---: |
| `human` | 26 | 0 | 0.0000 |
| `llm_blind` | 26 | 8 | 0.3077 |
| `llm_feedback` | 26 | 18 | 0.6923 |
| `null_control` | 26 | 0 | 0.0000 |

## 4. Operator inventory

What the gate could ask, before anything was scored.

| operator | class | delivery | mutations | tasks |
| --- | --- | --- | ---: | ---: |
| `drop_call` | call level | reexecute | 31 | 29 |
| `duplicate_call_mutating` | call level | reexecute | 7 | 7 |
| `duplicate_call_readonly` | call level | reexecute | 24 | 22 |
| `inject_extra_call` | call level | reexecute | 33 | 33 |
| `perturb_numeric_large` | argument level | trace | 30 | 22 |
| `perturb_numeric_plus_one` | argument level | trace | 30 | 22 |
| `reorder_calls` | call level | reexecute | 1 | 1 |
| `state_reverted` | state level | state_reset | 6 | 6 |
| `swap_identity_argument` | argument level | reexecute | 31 | 29 |
| `swap_identity_result` | argument level | trace | 41 | 26 |
| `swap_tool` | call level | reexecute | 31 | 29 |



## 5. Reading this

- **Rows have different denominators.** An (assertion, task) pair whose *unmutated*
  episode fails is dropped from scoring, so the arms are not scored on identical
  trial sets. The dropped trials are systematically the ones the LLM suites got wrong.
- **`llm_feedback` is scored in-sample.** It is shown which mutations `llm_blind`
  survived and asked to rewrite so those fail, then scored on that same mutation set.
  Its rate is a training-set number, not a held-out one.
- **Read §1 and §3 together.** A suite that scores 0.000 in §1 while rejecting most
  of §3 has traded false acceptance for false rejection, not eliminated error.

