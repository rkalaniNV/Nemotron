# Pattern: disable-faith-filter-for-byob

## Orient

BYOB MCQ translation requires a one-to-one mapping between staged source strings and translated output rows.

## Recommend

Force `faith_eval.filter_enabled=false` for `nemotron.steps.byob`.

## Verify

Check that the user still gets score columns in the output even though rows are not dropped inline.

## Boundaries

This rule is specific to BYOB benchmark translation and should not be applied to ordinary corpus translation by default.
