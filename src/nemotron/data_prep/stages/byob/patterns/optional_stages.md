# When to turn optional BYOB stages on (optimizer heuristics)

| Trigger / goal | Suggestion |
|----------------|------------|
| Benchmark feels too “small” in wrong-answer space | `do_distractor_expansion: true` to grow choices (e.g. 4 → 10). |
| Need to assert questions cover the source window | `do_coverage_check: true` (adds work; use when coverage matters for audits). |
| Tight budget or first smoke run | `do_distractor_expansion: false` and `do_coverage_check: false` until core quality looks good. |
| Too many near-duplicate items after generation | Rely on semantic dedup (on by default in the notebook); tune YAML under `semantic_deduplication_config`. |
| Scores from judge/filter look noisy | Lower batch sizes or review `prompt_config` in YAML before re-enabling optional stages. |

Add more `patterns/*.md` files here for domain-specific triggers (e.g. legal vs. medical corpora).
