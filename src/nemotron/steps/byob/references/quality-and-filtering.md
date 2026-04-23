# Quality And Filtering

BYOB reassembly requires a one-to-one mapping between staged source strings and translated rows.

Because of that, FAITH row filtering must stay disabled during translation.

If quality gating is needed, use:
- attached `faith_*` columns for review
- a second generic translation pass with reversed languages
- Curator `TextQualityMetricStage` over the restored benchmark strings
