# BFCL MCP support matrix

This page states what the MCP onboarding path can prove today. It is intentionally stricter
than a feature list: an implemented transport does not make an endpoint Gold-eligible.

## Rollout status

The MCP path is **experimental and feature-flagged**. Discovery, deterministic catalog
normalization, the MCP-to-BFCL gateway, evidence-bound authoring, P4–P11 evidence, review, freeze,
and publication handoff exist for Mode A. A discovery-only gateway still attests `L0`; only a
gateway restarted with strict BFCL-produced target evidence and a passing controlled P9 suite can
attest `L2`, and the existing BFCL Gold Gate independently reruns and verifies that evidence.

Live discovery, gateway startup, and intake are disabled by default. Set
`BFCL_ENABLE_MCP_MODE_A=1`. `BFCL_ENABLE_EXPERIMENTAL_MCP` remains an alias for one
deprecation window; if both variables are present they must agree. Omitted, false,
misspelled, unknown-kind, and conflicting settings fail closed. Offline review and
verification of already-produced artifacts do not need a rollout flag.

Do not describe an MCP pack as Gold merely because it has:

- an endpoint conformance document;
- a completed human review;
- an immutable frozen release; or
- a gateway that can execute tools.

Those are necessary evidence, not substitutes for fresh BFCL validation.

## Capability matrix

| Capability | Current status | Gold requirement or boundary |
| --- | --- | --- |
| Strict `mcp_oracle.yaml` | Implemented | Unknown keys and literal credentials are rejected. |
| stdio transport | Implemented, experimental | Host-owned executable path, digest, argv, and cwd policy are mandatory. |
| Streamable HTTP transport | Implemented, experimental | HTTPS is mandatory except explicit loopback debugging; redirects and ambient proxy trust are disabled. |
| MCP protocol | SDK v2 runtime, profile `2026-07-28` | Runtime refuses MCP SDK majors other than 2. |
| Tool discovery | Implemented | All pages must be consumed within configured limits. |
| Catalog normalization | Implemented | Unsupported or ambiguous schemas are excluded with reasons; no guessing. |
| Business/control separation | Implemented | Control tools are never model-facing. |
| Synchronous tool calls | Implemented | One request must produce one bounded result. |
| JSON-object `structuredContent` | Implemented | Required for the current profile; non-object projection is deferred. |
| MCP Tasks / claimed results | Refused | A conversation-level contract is required first. |
| `InputRequiredResult` | Refused | BFCL v1 cannot authorize hidden extra MCP exchanges. |
| Reset/state/end control mapping | Implemented and probed | P4/P5 prove lifecycle and deterministic fresh replay. |
| Process/namespace episode isolation | Implemented and probed | P6 interleaves two live endpoint sessions and checks state separation. |
| Confirmation boundary | Implemented and probed | P8 proves an unconfirmed mutation leaves state unchanged. |
| Resources and prompts | Out of scope | They are not Oracle Pack truth sources in this profile. |
| Dynamic `listChanged` catalogs | Refused for a run | Catalog identity must remain pinned from discovery through publication. |
| LLM authoring | Implemented as evidence-bound drafting | Model output is a draft; compilation, validation, and review remain authoritative. |
| Review and immutable freeze | Implemented | Approval covers exact bytes and cannot raise conformance level. |
| BFCL publication handoff | Implemented for Mode A | Fresh Gold plus independently verified publishable `L2` remains mandatory. |

## Supported operating modes

- **Mode A — cooperative server:** server exposes reviewed describe/reset/state/end controls.
- **Mode B — shimmed server:** a reviewed, fingerprinted adapter supplies the missing controls.
- **Mode C — immutable read-only snapshot:** only snapshot-safe, non-mutating behavior is eligible;
  snapshot and read-only boundaries must be explicit.

Mode is not a quality tier. `L0`, `L1`, and `L2` describe independently attained conformance:

- `L0`: discovery identity is reproducible.
- `L1`: executable behavior has passed the P4 boundary, but is not publishable.
- `L2`: reset, state, isolation, mutation, confirmation, replay, and failure handling have all
  passed the applicable independent probes and the endpoint may proceed to the Gold Gate.

## Deferred extensions

The following are explicit deferrals, not silently supported behavior:

- non-object `structuredContent` projection;
- MCP Tasks and elicitation/input-required exchanges;
- mid-run catalog refresh;
- trust in server annotations without an explicit profile decision;
- signed-release `L2` without a cryptographic verifier; and
- publication from a server that cannot expose deterministic state and episode isolation.

The normative contracts are
[bfcl-mcp-oracle-contract.md](bfcl-mcp-oracle-contract.md),
[bfcl-mcp-architecture-decision.md](bfcl-mcp-architecture-decision.md), and
[bfcl-mcp-threat-model.md](bfcl-mcp-threat-model.md).
