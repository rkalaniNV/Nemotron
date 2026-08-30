# BFCL ablation — arm `a5` (cross-wording target-model evaluation)

Target model: `openai/gpt-oss-120b`. Metric contract `1.0`.
33 paired tasks, 33 with a different opening turn. 128 oracle episodes.

A0 and A2 carry identical `task_id`s — the hash covers pack, template, fixture refs and
slot bindings, not the surface — so each task is its own control and the test is paired.

## 1. Headline

| verdict | A0 accuracy | A2 accuracy | delta | paired agreement | discordant | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ast_match` | 0.970 | 0.909 | -0.061 | 0.939 | 2 | 0.5000 |
| `assertion` | 0.970 | 0.970 | +0.000 | 1.000 | 0 | 1.0000 |

95% CI on A0 accuracy (0.8468, 0.9946), on A2 (0.7643, 0.9686) (Wilson).

`ast_match` is the headline: the model's calls equal `expected_tool_calls`. `assertion`
is the same episode judged by the pack's own `success_assertions`, which A4 scored at
0.610 false acceptance on argument-level corruptions — the gap between the two rows is
that leniency priced on real model output.

## 2. Contingency (ast_match)

| | A2 correct | A2 wrong |
| --- | ---: | ---: |
| **A0 correct** | 30 | 2 |
| **A0 wrong** | 0 | 1 |

Off-diagonal cells are the wording effect. McNemar conditions on exactly those.

## 3. Per turn policy (ast_match)

| policy | n | A0 | A2 | delta | flipped down | flipped up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clarify_only` | 1 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `confirmation` | 5 | 1.00 | 0.60 | -0.40 | 2 | 0 |
| `correction` | 1 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `dependent_call` | 1 | 0.00 | 0.00 | +0.00 | 0 | 0 |
| `irrelevant` | 3 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `missing_slot` | 1 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `multi_tool` | 1 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `negative_path` | 2 | 1.00 | 1.00 | +0.00 | 0 | 0 |
| `single_turn` | 18 | 1.00 | 1.00 | +0.00 | 0 | 0 |

Most cells hold one task. A per-cell rate at n=1 is an anecdote with a decimal point;
the counts are printed so no rate is read without its denominator.

## 4. Where the two verdicts disagree

- assertions passed while the calls were wrong: **2**
- calls were right while assertions failed: **0**

## 5. What this does not show

- **One target model, and it is the generator's own family.** A2's paraphrases and this
  model both come from `gpt-oss-120b`, so a model scoring well on its own family's
  wording is a self-preference result as much as a robustness one. A second family is
  the first thing to add.
- **One paraphrase per task.** `--a2-run` selects one variant index; the result is the
  effect of *that* wording, not of paraphrasing in general.
- **Later user turns are replayed, not simulated.** A real user would react to what the
  model actually said. Replaying the canned turns keeps a second model out of the
  measurement path, at the cost of realism on the 8 multi-turn tasks.
- **n = 33.** McNemar on a handful of discordant pairs has low power; a p above 0.05
  here is 'not detected at this n', not 'no effect'.

