# eval — what “correct” looks like

This folder is for **golden or smoke cases**: minimal inputs and expected *shapes* (and optionally fixture small parquets) so BYOB changes stay contract-stable.

- Prefer **one row** examples per concern (generation schema, post-filter columns, final benchmark columns).
- Do not commit large blobs; use tiny synthetic text and paths under `Nemotron/` test data if you add formal tests.
- Add cases as JSON/YAML sidecars or pytest fixtures as the test harness grows.

The LLM will not be deterministic; golden tests should check **column presence**, **dtype-ish constraints**, and **invariants** (e.g. exactly one correct option index) rather than exact text.
