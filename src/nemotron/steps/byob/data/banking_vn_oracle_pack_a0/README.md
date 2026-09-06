# banking_vn oracle pack — A0 ladder snapshot

This is the `banking_vn` pack as it stood at commit `c012ce3` ("Add executable
BFCL benchmark pipeline"), recovered so the SOV-866 ablation ladder's baseline
can be *reproduced* rather than only transcribed from Jira.

## Why a snapshot is needed

The A0 and A1 closure evidence reports 17 templates and a
`tasks_per_category` budget of 6, yielding 33 published tasks. The live pack has
since grown to 42 templates, and `balance_inquiry` alone now declares 10, so
budget 6 is no longer structurally valid: oracle validation check 7
(`representative_generation_contract`) correctly refuses it, because a template
would lose its only instance. The live pack therefore cannot reproduce the
ladder baseline at all, and its minimum viable budget is 10.

Template counts across the pack's history:

| Commit | Templates | Minimum viable budget |
|---|---:|---:|
| `c012ce3` | 17 | 5 |
| `e8f088c` | 22 | 6 |
| `2e2f410` | 28 | 7 |
| `834d3ca` | 28 | 7 |
| `3825e9f` (live) | 42 | 10 |

## What it reproduces

Generated template-only at budget 6 through the current pipeline
(`results/ladder-eval/ladder_a0pack_b6_template.yaml`), this pack yields:

- 33 published tasks, matching A0's `published_task_count`;
- 17 distinct templates, matching A0's `distinct_surface_count`;
- 9 of 9 required tools, matching A0's `tool_coverage`;
- 33 of 33 gold-eligible, matching A0's `gold_eligible_rate`.

## What it does not claim

The pipeline is the current one, not the pipeline A0 and A1 ran against. The
agreement above is evidence that this is the right pack, not proof that the two
runs are byte-identical. Any measurement taken here must state the pipeline it
used.

This snapshot is read-only reproduction input. Authoring changes belong in
`banking_vn_oracle_pack`.
