# Pattern: enable-faith-for-high-value-data

## Orient

Some datasets need post-translation quality evidence, not just translated output.

## Recommend

Enable `faith_eval.enabled=true` and attach scores to the output.

## Verify

Confirm whether low-scoring rows should be dropped or merely annotated.

## Boundaries

Do not enable row dropping unless the downstream workflow can tolerate filtered corpus output.
