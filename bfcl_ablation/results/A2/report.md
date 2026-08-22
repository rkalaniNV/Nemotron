# A2 — LLM surface generation (wording only)

Model: `openai/gpt-oss-120b`. 0 calls made, 426 served from cache.

## 1. Paraphrase generation

Requested 19 paraphrases per template on top of the authored sentence. Pool sizes (incl. index 0): [20].

| rejection reason | variants |
| --- | ---: |
| duplicate | 5 |

Rejection rate: 1.5% of everything the model returned.

## 2. Surface diversity vs. frozen ground truth

| budget | N | tasks | distinct masked surfaces | ceiling | % of ceiling | surfaces/template | task_id equal | expected_tool_calls equal | verdict | published tasks on a substituted-intent surface |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 6 | 1 | 33 | 17 | 17 | 100% | 1.0 | YES | YES | FROZEN | 0/33 |
| 6 | 2 | 33 | 23 | 26 | 88% | 1.353 | YES | YES | FROZEN | 0/33 |
| 6 | 3 | 33 | 22 | 30 | 73% | 1.294 | YES | YES | FROZEN | 2/33 |
| 6 | 5 | 33 | 28 | 32 | 88% | 1.647 | YES | YES | FROZEN | 1/33 |
| 6 | 10 | 33 | 29 | 33 | 88% | 1.706 | YES | YES | FROZEN | 3/33 |
| 6 | 20 | 33 | 31 | 33 | 94% | 1.824 | YES | YES | FROZEN | 2/33 |
| 24 | 1 | 91 | 17 | 17 | 100% | 1.0 | YES | YES | FROZEN | 0/91 |
| 24 | 2 | 91 | 33 | 34 | 97% | 1.941 | YES | YES | FROZEN | 0/91 |
| 24 | 3 | 91 | 41 | 48 | 85% | 2.412 | YES | YES | FROZEN | 3/91 |
| 24 | 5 | 91 | 55 | 72 | 76% | 3.235 | YES | YES | FROZEN | 2/91 |
| 24 | 10 | 91 | 67 | 90 | 74% | 3.941 | YES | YES | FROZEN | 4/91 |
| 24 | 20 | 91 | 77 | 91 | 85% | 4.529 | YES | YES | FROZEN | 3/91 |

Ceiling is `sum over templates of min(N, pool size, tasks for that template)`; a rung that hits it has extracted all the variety the budget allows. The shortfall is collision: the variant a task gets is `seed % effective_N`, an unbalanced draw, so two tasks on the same template can land on the same variant while another goes unused. That is why a rung can lose ground to the rung below it — the assignment is not monotone in N.

## 3. Intent check — the number that decides whether any of this is safe

An independent call reads the sentence plus the tool catalogue and nothing else, and names the tools the request needs. Disagreement with the template's `required_tools` flags the sentence.

| population | n | flagged | rate |
| --- | ---: | ---: | ---: |
| canonical (authored, correct by construction) | 17 | 1 | 5.9% |
| paraphrases (assumed good) | 323 | 42 | 13.0% |
| injected intent shifts (should all be caught) | 34 | 32 | 94.1% |

Recall on injected shifts: **94.1%**. False-alarm floor on the authored sentences: 5.9%. Flag rate on generated paraphrases: 13.0%.

On 15/34 shifts the checker named exactly the tool the decoy was steered towards, which is the strong form of catching one.

0 of 34 decoys were stopped by the placeholder/literal/tool-name guards before the checker saw them.

### What the flags on paraphrases actually are

Flag shapes: `{'under_predicted': 35, 'substituted': 7}`. Only a *substituted* prediction says the sentence now asks for a different tool; that is 7/323 paraphrases (2.2%). The rest are the checker disagreeing about how many calls the opening turn implies.

| turn policy | paraphrases | flagged | rate | shapes |
| --- | ---: | ---: | ---: | --- |
| dependent_call | 19 | 18 | 94.7% | {'under_predicted': 18} |
| missing_slot | 19 | 16 | 84.2% | {'under_predicted': 16} |
| confirmation | 38 | 8 | 21.1% | {'substituted': 7, 'under_predicted': 1} |
| clarify_only | 19 | 0 | 0.0% | - |
| correction | 19 | 0 | 0.0% | - |
| irrelevant | 19 | 0 | 0.0% | - |
| multi_tool | 19 | 0 | 0.0% | - |
| negative_path | 38 | 0 | 0.0% | - |
| single_turn | 133 | 0 | 0.0% | - |

### What this costs the delivered benchmark

The checker is a measurement in A2, not a gate: a flagged variant is still published. The last column of section 2 is therefore the contamination the arm actually shipped.

| budget | N | published | flagged surface | substituted surface |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 1 | 33 | 0 | 0 |
| 6 | 2 | 33 | 2 | 0 |
| 6 | 3 | 33 | 4 | 2 |
| 6 | 5 | 33 | 2 | 1 |
| 6 | 10 | 33 | 5 | 3 |
| 6 | 20 | 33 | 4 | 2 |
| 24 | 1 | 91 | 0 | 0 |
| 24 | 2 | 91 | 8 | 0 |
| 24 | 3 | 91 | 9 | 3 |
| 24 | 5 | 91 | 11 | 2 |
| 24 | 10 | 91 | 14 | 4 |
| 24 | 20 | 91 | 15 | 3 |

> Recall is measured against decoys the same model family was asked to shift, so it bounds the checker's power against LLM-written drift and says nothing about drift a human author would introduce. False alarms on the canonical sentences are the floor: those templates are correct by construction, so every flag there is the checker being wrong.
