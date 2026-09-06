# BFCL ablation — A7 Independent Quality Gate

**Publication decision: `NOT_READY`.**

A7 audits frozen A0–A6 evidence. It does not rerun the production pipeline, call an LLM, or turn missing human evidence into a pass.

## Decision split

- `integrity`: **PASS**
- `study_validity`: **INCONCLUSIVE**
- `release_readiness`: **FAIL**

`study_validity` asks whether each experimental conclusion is supported. `release_readiness` asks whether the generated benchmark variants are safe to publish.

## Headline warnings

- A0/A3 publish and gold rates are throughput, not content-quality evidence.
- A2 FROZEN checks task IDs and expected calls; it is not semantic equivalence.
- A4 must be read per strict operator class; aggregate false acceptance hides the weakness.
- A5 assertion agreement is 100%, but assertion accuracy is 32/33 on each wording.
- A6 raw blind_rate is not an all-layer result; current evidence bounds it at 3.7%-45.8%.

## Human-review coverage

Complete items: **0/80**; declared reviewers: 0; unadjudicated disagreements: 0.

- Label issue: no human label file supplied

- Paraphrase prevalence sample: 0/51 reviewed; errors=0.
- Intent-shift controls: 0/17 reviewed; misses=0.

## Checks by arm

### Global

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `G-ARTIFACTS` | integrity | **PASS** | - | 12/12 required artifacts are readable |
| `G-EVIDENCE-STUDY_VALIDITY` | study_validity | **PASS** | - | All required artifacts are present |
| `G-EVIDENCE-RELEASE_READINESS` | release_readiness | **PASS** | - | All required artifacts are present |
| `G-METRIC-VERSION` | integrity | **PASS** | {"a0": "1.0", "a1": "1.0", "a2": "1.0", "a3": "1.0", "a4": "1.0", "a5": "1.0", "a6": "1.0"} | Observed versions: {'a0': '1.0', 'a1': '1.0', 'a2': '1.0', 'a3': '1.0', 'a4': '1.0', 'a5': '1.0', 'a6': '1.0'} |
| `G-ARM-TAGS` | integrity | **PASS** | - | All arm tags agree |
| `G-VERSION-ENFORCEMENT` | study_validity | **CONDITIONAL** | - | A7 enforces it, but A0-A6 runners only record the version |

### A0

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A0-ACCOUNTING` | integrity | **PASS** | {"expanded": 33, "policy_total": 33} | Policy total 33; expanded 33 |
| `A0-TOOL-COVERAGE` | release_readiness | **PASS** | 0 | All declared tools are called |
| `A0-SURFACE-BASELINE` | study_validity | **PASS** | - | 17 masked surfaces across 17 templates |
| `A0-PUBLISH-IS-THROUGHPUT` | study_validity | **CONDITIONAL** | 1.0000 | Publish rate is 100.0%; it supplies no content-quality evidence |
| `A0-FIXTURE-COVERAGE` | release_readiness | **CONDITIONAL** | - | 17/50 fixture entities are bound; the denominator includes backend-only rows |
| `A0-BUDGET-SWEEP` | study_validity | **PASS** | {"surfaces": {"6": 17, "12": 17, "24": 17}, "tasks": {"6": 33, "12": 55, "24": 91}} | Tasks by budget: {6: 33, 12: 55, 24: 91}; distinct surfaces: {6: 17, 12: 17, 24: 17} |

### A1

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A1-EQUIVALENCE` | study_validity | **PASS** | {"conversation_plans": true, "expected_tool_calls": true, "opening_turn": true, "task_ids": true, "validation_coverage": true} | Gates {'task_ids': True, 'expected_tool_calls': True, 'conversation_plans': True, 'validation_coverage': True, 'opening_turn': True}; metrics='EQUIVALENT'; proof='EQUIVALENT' |
| `A1-LOC` | integrity | **PASS** | {"baseline": 1642, "candidate": 1412, "saved": 230} | 1642 - 1412 = 230 |
| `A1-DIALOGUE-SCOPE` | release_readiness | **CONDITIONAL** | {"dialogues_changed": 8, "opening_changed": 0} | 0 opening turns and 8 complete conversations changed |

### A2

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A2-FROZEN-SCOPE` | integrity | **PASS** | 12 | 12 rungs checked; FROZEN is not treated as semantic equivalence |
| `A2-FROZEN-NOT-SEMANTIC` | study_validity | **CONDITIONAL** | - | Semantic preservation requires independent human labels |
| `A2-POOL-COVERAGE` | integrity | **PASS** | 0 | All templates have pools |
| `A2-DIVERSITY` | study_validity | **PASS** | 77 | Maximum 77 surfaces at budget=24 N=20 |
| `A2-LLM-CHECKER` | study_validity | **CONDITIONAL** | {"canonical_false_alarm": 0.0588, "shift_recall": 0.9412, "substitution_rate": 0.0217} | shift recall=94.1%, canonical false alarm=5.9%, substitution flags=2.2% |
| `A2-GENERATION-DENOMINATOR` | study_validity | **CONDITIONAL** | - | 5/328 examined candidates=1.52%; this is not all candidates returned because quota overflow was not inspected |
| `A2-HUMAN-SEMANTICS-STUDY_VALIDITY` | study_validity | **INCONCLUSIVE** | - | Human prevalence labels complete for 0/51 required pairs |
| `A2-HUMAN-SEMANTICS-RELEASE_READINESS` | release_readiness | **INCONCLUSIVE** | - | Human prevalence labels complete for 0/51 required pairs |
| `A2-HUMAN-CONTROLS` | study_validity | **INCONCLUSIVE** | - | Misses 0/0; required controls 17 |

- `A2-FROZEN-SCOPE` caveat: task IDs and expected calls do not depend on opening-turn wording
- `A2-DIVERSITY` caveat: the maximum combines paraphrase fan-out with a larger task budget
- `A2-LLM-CHECKER` caveat: generator and checker use the same model family

### A3

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A3-PROPOSAL-ACCOUNTING` | integrity | **PASS** | {"accepted": 20, "dropped": 36, "requested": 56} | 20 accepted + 36 dropped = 56 requested |
| `A3-SKEW-FINDING` | study_validity | **PASS** | {"tools_never_called": ["create_dispute", "create_transfer", "get_transfer_fee", "get_vietqr_payment_status", "list_recent_transactions"], "unfalsifiable_share": 0.7} | Unfalsifiable share=70.0%; tools never called=5 |
| `A3-CANDIDATE-COVERAGE` | release_readiness | **FAIL** | {"tools_never_called": 5, "unfalsifiable_share": 0.7} | Unfalsifiable share=70.0%; never-called tools=['create_dispute', 'create_transfer', 'get_transfer_fee', 'get_vietqr_payment_status', 'list_recent_transactions'] |
| `A3-PROVENANCE` | study_validity | **CONDITIONAL** | - | Recorded pack '/localhome/local-hndo/Nemotron/bfcl_ablation/_generated/packs/a3_full' exists=False |
| `A3-HUMAN-SEMANTICS` | study_validity | **INCONCLUSIVE** | - | The current artifact does not preserve a complete blinded A3 review queue |

- `A3-SKEW-FINDING` caveat: this identifies stack-level survivorship bias, not an LLM-only causal effect
- `A3-PROVENANCE` caveat: other a3f/a3g/a3h runs exist without a recorded selection rationale

### A4

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A4-TRIAL-RECONCILIATION` | integrity | **PASS** | {"argument_level": {"false_accept": 86, "trials": 141}, "call_level": {"false_accept": 41, "trials": 108}, "state_level": {"false_accept": 0, "trials": 6}} | Recomputed totals: {'argument_level': {'false_accept': 86, 'trials': 141}, 'call_level': {'false_accept': 41, 'trials': 108}, 'state_level': {'false_accept': 0, 'trials': 6}} |
| `A4-ARGUMENT_LEVEL-FAR` | release_readiness | **FAIL** | 0.6099 | False acceptance=61.0%; policy maximum=5.0% |
| `A4-CALL_LEVEL-FAR` | release_readiness | **FAIL** | 0.3796 | False acceptance=38.0%; policy maximum=5.0% |
| `A4-STATE_LEVEL-FAR` | release_readiness | **PASS** | 0.0000 | False acceptance=0.0%; policy maximum=0.0% |
| `A4-HUMAN-WEAKNESS-FINDING` | study_validity | **PASS** | {"argument_level": 0.6099, "call_level": 0.3796, "state_level": 0.0} | Strict FAR by class: {'argument_level': 0.6099, 'call_level': 0.3796, 'state_level': 0.0} |
| `A4-GOLD-FLOOR` | release_readiness | **PASS** | 1.0000 | Gold pass rate=100.0% |
| `A4-FEEDBACK-GENERALIZATION` | study_validity | **INCONCLUSIVE** | - | Feedback assertions were authored and scored on the same mutation plans |

### A5

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A5-PAIR-RECONCILIATION` | integrity | **PASS** | {"assertion": {"a0_correct": 32, "a0_only": 0, "a2_correct": 32, "a2_only": 0, "discordant": 0, "n": 33}, "ast_match": {"a0_correct": 32, "a0_only": 2, "a2_correct": 30, "a2_onl... | Recomputed: {'ast_match': {'n': 33, 'a0_correct': 32, 'a2_correct': 30, 'a0_only': 2, 'a2_only': 0, 'discordant': 2}, 'assertion': {'n': 33, 'a0_correct': 32, 'a2_correct': 32, ... |
| `A5-BEHAVIORAL-FLIPS` | study_validity | **PASS** | - | AST discordant=2/33; assertion discordant=0/33 |
| `A5-EFFECT-ESTIMATE` | study_validity | **INCONCLUSIVE** | {"discordant": 2, "mcnemar_p": 0.5, "n": 33} | n=33 (minimum 100); discordant=2 (minimum 25); McNemar p=0.5 |
| `A5-RELEASE-STABILITY` | release_readiness | **FAIL** | -0.0606 | Observed AST score delta=-6.1%; allowed absolute delta=2.0% |
| `A5-EXTERNAL-VALIDITY` | study_validity | **INCONCLUSIVE** | - | The stored experiment uses one model family and one selected A2 wording |
| `A5-HUMAN-DISAGREEMENT` | study_validity | **INCONCLUSIVE** | - | Reviewed 0/2; rejected as unacceptable=0 |

- `A5-BEHAVIORAL-FLIPS` caveat: both observed flips belong to one template
- `A5-RELEASE-STABILITY` caveat: statistical significance is not required to treat observed release regressions as blockers

### A6

| check | dimension | status | value | detail |
| --- | --- | --- | --- | --- |
| `A6-TRIAGE-PARTITION` | integrity | **PASS** | {"L1_validation_cases": 47, "L2_expected_traces": 45, "L3_assertions": 11, "survived": 48} | metrics survived=48; trial survivors=48; triage partition=48; verdict rows=48 |
| `A6-BLIND-BOUNDS` | study_validity | **INCONCLUSIVE** | {"lower_count": 4, "lower_rate": 0.0374, "observable": 107, "upper_count": 49, "upper_rate": 0.4579} | Current evidence bounds blind mutants at 4/107–49/107 (3.7%–45.8%); 45 L2 rows lack L4/L5 outcomes |
| `A6-RELEASE-BLIND-RATE` | release_readiness | **INCONCLUSIVE** | {"lower_count": 4, "lower_rate": 0.0374, "observable": 107, "upper_count": 49, "upper_rate": 0.4579} | Allowed ≤5.0%; observed interval 3.7%–45.8% |
| `A6-REAL-GAPS` | study_validity | **PASS** | {"critical_real_gaps": 0, "real_gaps": 4} | Triaged real gaps=4; high/critical real gaps=0 |
| `A6-CRITICAL-GAPS` | release_readiness | **PASS** | 0 | High/critical real gaps=0 |
| `A6-HUMAN-TRIAGE` | study_validity | **INCONCLUSIVE** | - | Reviewed 0/4; classification agreements=0 |

- `A6-BLIND-BOUNDS` caveat: raw metrics.blind_rate=45/103 is not an all-layer measurement

## What blocks a clean release

- **INCONCLUSIVE** `A2-HUMAN-SEMANTICS-STUDY_VALIDITY` — Human prevalence labels complete for 0/51 required pairs
- **INCONCLUSIVE** `A2-HUMAN-SEMANTICS-RELEASE_READINESS` — Human prevalence labels complete for 0/51 required pairs
- **INCONCLUSIVE** `A2-HUMAN-CONTROLS` — Misses 0/0; required controls 17
- **FAIL** `A3-CANDIDATE-COVERAGE` — Unfalsifiable share=70.0%; never-called tools=['create_dispute', 'create_transfer', 'get_transfer_fee', 'get_vietqr_payment_status', 'list_recent_transactions']
- **INCONCLUSIVE** `A3-HUMAN-SEMANTICS` — The current artifact does not preserve a complete blinded A3 review queue
- **FAIL** `A4-ARGUMENT_LEVEL-FAR` — False acceptance=61.0%; policy maximum=5.0%
- **FAIL** `A4-CALL_LEVEL-FAR` — False acceptance=38.0%; policy maximum=5.0%
- **INCONCLUSIVE** `A4-FEEDBACK-GENERALIZATION` — Feedback assertions were authored and scored on the same mutation plans
- **INCONCLUSIVE** `A5-EFFECT-ESTIMATE` — n=33 (minimum 100); discordant=2 (minimum 25); McNemar p=0.5
- **FAIL** `A5-RELEASE-STABILITY` — Observed AST score delta=-6.1%; allowed absolute delta=2.0%
- **INCONCLUSIVE** `A5-EXTERNAL-VALIDITY` — The stored experiment uses one model family and one selected A2 wording
- **INCONCLUSIVE** `A5-HUMAN-DISAGREEMENT` — Reviewed 0/2; rejected as unacceptable=0
- **INCONCLUSIVE** `A6-BLIND-BOUNDS` — Current evidence bounds blind mutants at 4/107–49/107 (3.7%–45.8%); 45 L2 rows lack L4/L5 outcomes
- **INCONCLUSIVE** `A6-RELEASE-BLIND-RATE` — Allowed ≤5.0%; observed interval 3.7%–45.8%
- **INCONCLUSIVE** `A6-HUMAN-TRIAGE` — Reviewed 0/4; classification agreements=0

## Artifact provenance

| artifact | present | metrics version | sha256 |
| --- | --- | --- | --- |
| `a0_metrics` | True | 1.0 | `e03997e5c50f5989` |
| `budget_sweep` | True | - | `f5816043e262476a` |
| `a1_metrics` | True | 1.0 | `dadc946878e938fc` |
| `a1_equivalence` | True | - | `df2fa6963c93af2f` |
| `a2_metrics` | True | 1.0 | `41a62cf49b9f6f1e` |
| `a3_metrics` | True | 1.0 | `6493dced19cde690` |
| `a4_metrics` | True | 1.0 | `54e8b6cc81d2421c` |
| `a4_trials` | True | - | `5c15c10e9f47ad38` |
| `a5_metrics` | True | 1.0 | `0a1e5dbfb9dae266` |
| `a5_trials` | True | - | `ed053c7d801c8afe` |
| `a6_metrics` | True | 1.0 | `4acb21bf7eb07600` |
| `a6_trials` | True | - | `467f2a1f73a2dbc8` |
| `a6_triage` | True | - | `dfdc9a58e94032e3` |

Threshold policy: `bfcl_ablation/quality_gate/defaults.yaml` (`f1e9f1b1643a4aee`), contract `1.0`.
