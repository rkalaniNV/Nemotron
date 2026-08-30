# BFCL ablation — arm `a6` (is the oracle itself falsifiable?)

Pack: `src/nemotron/steps/byob/data/banking_vn_oracle_pack`. Metric contract `1.0`.
151 single-edit mutants of `backend.py` (465 lines), judged against 23 validation cases, 33 replayed tasks and a full pipeline run. 3755 oracle episodes.

## 1. The headline

**43.7% of observable backend corruptions pass every check the pack ships** (45 of 103).

Read the raw survival count carefully — on its own it is the wrong number. A mutant that
survives everything is usually one the benchmark never *executes*, which is a coverage
finding, not a checking one. The number that measures checking is the share of mutants that
demonstrably changed something and were still accepted.

| outcome | mutants | share |
| --- | ---: | ---: |
| unobservable — nothing the pack runs reaches it | 48 | 31.8% |
| observable, caught by a check the pack ships | 58 | 38.4% |
| **observable, caught by nothing the pack ships** | **45** | **29.8%** |

That is the oracle-side view of the hole A4 measured from the assertion side as 0.610
argument-level false acceptance. Two independent methods, one gap.

## 2. What killed each mutant

| layer | mutants | share | ships with the pack? |
| --- | ---: | ---: | --- |
| `L0_import` | 0 | 0.0% | yes |
| `L1_validation_cases` | 47 | 31.1% | yes |
| `L2_expected_traces` | 45 | 29.8% | **no — reference added by this arm** |
| `L3_assertions` | 11 | 7.3% | yes |
| `L4_oracle_validation` | 0 | 0.0% | yes |
| `L5_pipeline` | 0 | 0.0% | yes |
| `survived` | 48 | 31.8% | — |

**`L4_oracle_validation` and `L5_pipeline` killed nothing.** Every mutant that reached them
passed. The oracle-validation checks and a full generation run to 33 published rows at tier
`gold` added **zero** detection over the cheap layers — A0's finding that the gates never
fire, reproduced against a deliberately corrupted oracle rather than against clean input.

`L2_expected_traces` is **not** a check the pack ships. It is a differential comparison
against the unmutated backend, added here because nothing in the pack states what a tool
should *return*. Every mutant in that row changed an observable value or the final state and
was accepted by all four shipped layers.

## 3. By mutation family

| family | mutants | survived | survival rate |
| --- | ---: | ---: | ---: |
| `contract` | 41 | 3 | 0.073 |
| `guard` | 52 | 17 | 0.327 |
| `state` | 13 | 5 | 0.385 |
| `value` | 45 | 23 | 0.511 |

## 4. By operator

| operator | mutants | survived | survival rate |
| --- | ---: | ---: | ---: |
| `delete_guard` | 26 | 16 | 0.615 |
| `delete_state_write` | 13 | 5 | 0.385 |
| `drop_result_key` | 41 | 3 | 0.073 |
| `flip_comparison` | 23 | 12 | 0.522 |
| `invert_guard` | 26 | 1 | 0.038 |
| `negate_bool_literal` | 3 | 2 | 0.667 |
| `perturb_int_literal` | 13 | 8 | 0.615 |
| `swap_arithmetic` | 6 | 1 | 0.167 |

## 5. Unpinned lines

33 of the 68 lines carrying a mutant have at least
one single-edit corruption that nothing detects:

```
15, 73, 106, 108, 114, 121, 145, 147, 150, 152, 159, 163, 165, 168, 244, 245, 248, 252, 265, 274, 293, 299, 308, 313, 315, 340, 353, 413, 415, 426, 435, 445, 457
```

## 6. Observable but unchecked (45)

changed a returned value or the final state, and no check the pack ships noticed:

| line | operator | edit |
| ---: | --- | --- |
| 88 | `perturb_int_literal` | 1 -> 2 |
| 90 | `flip_comparison` | Eq -> NotEq |
| 326 | `swap_arithmetic` | Sub -> Add |
| 51 | `perturb_int_literal` | 0 -> 1 |
| 51 | `perturb_int_literal` | 0 -> 1 |
| 51 | `perturb_int_literal` | 0 -> 1 |
| 326 | `swap_arithmetic` | Sub -> Add |
| 346 | `swap_arithmetic` | Add -> Sub |

## 7. Reading this

- **A surviving mutant is not necessarily a bug.** It is a line the *benchmark* does not
  depend on. Some are genuinely unreachable given the fixtures; those are still a finding,
  because they are lines a pack author paid to write and maintain that no task exercises.
- **Equivalent mutants inflate survival.** An edit with no observable effect on any input
  cannot be killed by anything and is not evidence of a weak pack. Survivors reaching L5
  need hand-triage before the headline number is quoted, the way A4's strict/advisory
  split did — that reclassification moved a headline from 0.137 to 0.380.
- **This arm uses no model.** Everything here is deterministic and reproduces exactly.

