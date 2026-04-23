# Quality And Failure Modes

FAITH scoring is optional.

Enable it when the translated corpus is high-value and you need quality evidence. Disable it when throughput is the priority.

When `faith_eval.segment_level=true`, Curator scores exploded segment rows before reassembly and then aggregates document-level `faith_*` columns after reassembly. This is the preferred mode when whole-document FAITH requests would be too large.

Common failure modes:
- missing API keys
- server health check failures
- rate limits on hosted endpoints
- mixed-format input directories
- single-file datasets that need preprocessing before Curator partitioning is enough

`skip_translated=true` is safe for resume-style runs when the output field already exists.
