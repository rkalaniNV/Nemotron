---
id: add-new-benchmark-family
title: Add a new BYOB benchmark family
tags: [byob, extensibility, benchmark]
triggers:
  - Add GSM8K-style benchmark generation.
  - Extend BYOB beyond multiple-choice questions.
steps: [byob]
confidence: high
---

Before coding, read `references/new-family-checklist.md` and answer the task-format, final-schema,
source-example, quality-gate, and validation questions. Then create a new package under
`runtime/benchmark_families/`, keep family logic local, and register it.
