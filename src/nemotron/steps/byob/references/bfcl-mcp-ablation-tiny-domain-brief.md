# tiny_library onboarding brief

Build a BFCL Oracle Pack for a small English-language library circulation domain. This document
is the only domain-specific input shared across the three onboarding flows.

## Domain state

The initial state contains:

- Books:
  - `BK-100`, title `Algorithms`, available, 2 copies
  - `BK-200`, title `Databases`, available, 1 copy
  - `BK-300`, title `Networks`, on loan, 0 copies
- Patrons:
  - `P-1`, name `Ada`
  - `P-2`, name `Grace`
- No initial loans

Use the frozen clock `2026-03-02T09:00:00+07:00` and seed `7`.

## Required capabilities

The benchmark must let a model:

1. retrieve the current status of a book by book ID; and
2. check out an available book to a known patron after explicit user confirmation.

A successful checkout reduces the available copy count, marks the book on loan when no copies
remain, and adds an active loan with a deterministic ID. A request without explicit confirmation
must return a stable pending status and leave all state unchanged.

## Required failures

Return machine-readable stable error codes for:

- malformed argument types;
- unknown book IDs;
- unknown patron IDs;
- unavailable books; and
- unsupported tool names.

Negative calls must not partially mutate state.

## Benchmark coverage

Include examples covering:

- one successful status lookup;
- one missing-book lookup;
- one successful confirmed checkout;
- one unconfirmed checkout;
- one malformed confirmation value;
- a multi-call status request; and
- an irrelevant request that should not call a tool.

Every exposed tool needs successful and negative validation coverage. The oracle must support
deterministic reset, complete state inspection, isolated episodes, deterministic replay, and
bounded execution.

## Deliverable

Produce a complete canonical Oracle Pack and a generated benchmark that passes fresh BFCL
validation. The flow under test determines whether pack fields are authored manually, drafted by
an LLM over a conventional backend, or drafted from MCP discovery evidence. Do not copy files
from another flow or a prior repetition.
