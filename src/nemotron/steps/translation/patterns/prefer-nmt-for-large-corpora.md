# Pattern: prefer-nmt-for-large-corpora

## Orient

Large plain-text corpora are often better served by a local NMT backend when cost and throughput matter more than nuanced instruction following.

## Recommend

Prefer `backend=nmt` for `nemotron.steps.translation` when a local server exists.

## Verify

Check that the data is mostly plain text and that a reachable NMT HTTP server is available.

## Boundaries

Do not use this pattern for structured chat data or when FAITH-scored high-fidelity translation is the priority.
