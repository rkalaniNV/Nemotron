# BFCL MCP Oracle Contract

Tasks: `MCP-002` (profile), `MCP-003` (`mcp_oracle.yaml` schema), `MCP-004` (tool
normalization), `MCP-005` (result and error mapping).

Profile id: `bfcl-mcp-oracle-v1`.
Read with `bfcl-oracle-pack.md`, which defines the pack, backend, and endpoint
contracts this document maps onto, and with `bfcl-mcp-architecture-decision.md`,
which explains why the mapping happens in a gateway.

The profile is written for three situations, not one: a server the operator built for
BFCL, a server the operator cannot modify, and a read-only third-party server with
static data. Requirements are therefore stated as a **level** (how much trust the
server has earned) crossed with a **mode** (how the episode control plane is obtained).
A requirement that applies to only one mode says so.

## 1. Conformance Levels

| Level | Claim | Requires | Publication |
| --- | --- | --- | --- |
| `L0` discovery | The catalog can be normalized into `tools.json` | `tools/list` normalizes without exclusions the operator did not accept (§7) | Authoring intake only. No oracle, no execution claim. |
| `L1` executable | Episodes can be reset, called, and read | A working control plane (§4), the result mapping total over every included tool (§8), probes `P1`–`P4` | `lineage.policy: smoke_no_publication` only |
| `L2` certifiable | The oracle is trustworthy enough for BFCL to certify | `L1` plus probes `P5`–`P11` and server-derived identity (§9) | Publishable if BFCL's own gold gate passes |

`L2` is a precondition, not a verdict. A pack on an `L2` server still has to pass
`run_oracle_validation`; the levels only describe what the MCP side contributed.
A level is recorded in the gateway's conformance artifact and in the pack's
provenance, so a pack can never be published on the strength of an unstated
assumption about its server.

## 2. Operating Modes

| Mode | Situation | Reset strategy | State strategy | Fixtures | Reachable level |
| --- | --- | --- | --- | --- | --- |
| `A` cooperative | The operator controls the server and can add the control tools of §4 | `control_tool` | `control_tool` | pushed to the server | `L2` |
| `B` shimmed | The server cannot be changed; the gateway owns a wrapper that launches, seeds, and inspects it | `process_restart` or `namespace` | `read_only_projection` or a shim-provided control tool | pushed by the shim | `L2` when the shim can seed and isolate |
| `C` read-only snapshot | A third-party server with data the operator cannot seed, exposing non-mutating tools only | `no_op_verified` | `read_only_projection` | **snapshotted from** the server at discovery | `L2` for read-only domains |

Mode `C` inverts the usual direction of fixtures. Nothing is pushed; `fixtures.json`
is a snapshot of the server's own data taken during discovery, so template slots bind
to values that exist upstream. Its digest is part of `content_digest`, which means
upstream data drift is detected as an identity change rather than as a run of
mysteriously failing tasks. The cost is stated plainly: mode `C` exposes no mutating
tool, so it cannot cover the `confirmation`, `correction`-of-a-mutation, or
mutation-assertion task categories, and a pack built on it is narrower by
construction rather than by accident.

Mode `B` is the one that makes this contract usable against servers in the wild. The
shim is gateway-side code, not pack code, so it is not fingerprinted as pack content;
its version enters the catalog digest instead (§9).

## 3. Gateway Obligations

The gateway exposes BFCL Oracle HTTP v1 and nothing else. Routes and payloads are
fixed by `EndpointOracleClient`; `protocol_version` is always the literal
`bfcl-oracle-http-v1`, which names the BFCL contract and is unrelated to the MCP
version.

| BFCL route | Gateway obligation |
| --- | --- |
| `GET /v1/metadata` | Return `{protocol_version, oracle_id, oracle_version, content_digest}` with exactly those four keys, all non-empty strings, `content_digest` matching `sha256:<64 hex>`. Derived per §9. |
| `GET /v1/tools` | Return `{"tools": [<name>, ...]}` — the published business tool names in catalog order (§7). Control tools never appear. The client re-reads metadata before every list, so identity must be stable within a run. |
| `POST /v1/sessions` | Body carries `context{clock, seed, timeout_s, task_id}` and `fixtures`. Open one isolated episode, apply the reset strategy, propagate the frozen clock and seed, and return `{session_id, oracle{...identity...}}`. The identity must equal the metadata identity or the client aborts. |
| `POST /v1/sessions/{id}/calls` | Body carries `{name, arguments, turn_index}`. Invoke `tools/call` on the bound episode and return the mapped result object (§8). |
| `GET /v1/sessions/{id}/state` | Return the episode's state as a JSON object, per the mode's state strategy. This is what assertions, `expect.state_unchanged`, and check `D1` read. |
| `DELETE /v1/sessions/{id}` | End the episode and release its resources. |

Robustness rules that are not visible in the route table but decide whether a run can
be trusted:

- **An unknown or expired `session_id` is an error, never a new session.** Silently
  seeding a fresh episode would let a task that lost its session pass against empty
  state, which is the one failure mode that produces a plausible wrong benchmark
  instead of a visible defect. Answer with a named failure (`mcp_session_unknown`)
  and let the run fail.
- **`DELETE` is idempotent.** `reset()` closes the previous session before opening a
  new one, and `close(suppress_errors=True)` can arrive after a failure, so a delete
  for an already-gone session must succeed rather than raise.
- **A call is never retried.** `EndpointOracleClient` does not retry mutating
  requests, and the gateway must not retry `tools/call` either: a transport failure
  after the server applied a mutation is indistinguishable from one before, so a
  retry can double-apply. Report `mcp_call_failed` and let the task drop.
- **Concurrency is supported, not relied upon.** BFCL generation drives one session
  at a time, so correctness must not depend on parallelism; but the gateway must
  serve at least two concurrent episodes, otherwise the isolation probe `P6` cannot
  be written. A session ceiling and an idle TTL bound orphaned episodes, and an
  expiry surfaces as `mcp_session_unknown`.
- **`fixtures` may be `null`.** A pack without `fixtures.json` sends no fixtures;
  that means "no seeding requested", not "empty state".
- **In mode `C` the fixtures BFCL sends must equal the pinned snapshot.** Nothing is
  pushed, but the pack still transmits its `fixtures.json`, and that document is the
  snapshot the identity covers. A mismatch means the pack's slot values describe data
  the server does not have, so it is answered with `mcp_fixtures_not_snapshot` rather
  than ignored — ignoring it would bind templates to values that only exist in the
  pack.
- **`context.clock` arrives as `datetime.isoformat()` output** and may or may not
  carry a timezone offset. Parse it, do not reinterpret it, and never substitute the
  host clock.
- **`turn_index` is advisory.** It records the conversation position for logs. Tool
  results must not depend on it, or the oracle's behavior would depend on how the
  surface was rendered.
- **Nothing is invented.** A missing capability is an error at the route, not a
  default value. BFCL treats every returned object as oracle truth and would certify
  a fabricated one.

## 4. Episode Control Plane

MCP standardizes discovery and invocation only, so the episode operations are carried
as **reserved MCP tools** where the server can provide them. Tools are the one
extension point every MCP SDK and both transports support; a custom JSON-RPC method
is not portable and would not survive Streamable HTTP method routing. Names come from
`mcp_oracle.yaml`, never hardcoded, the same way a pack chooses its own confirmation
vocabulary.

| Operation | Input | Required output | Strategy alternatives |
| --- | --- | --- | --- |
| `describe_oracle` | none | `{oracle_id, oracle_version, content_digest}` | Absent, §9's fallback applies and the level is capped at `L1`. |
| `reset_episode` | `{fixtures, context{clock, seed, timeout_s, task_id}}` | `{episode_id}` | `control_tool`; or `process_restart` / `namespace` (mode `B`); or `no_op_verified` (mode `C`). |
| `get_episode_state` | `{episode_id}` | a JSON object | `control_tool`; or `read_only_projection` — a declared, ordered list of non-mutating calls whose canonicalized results form the state document. |
| `end_episode` | `{episode_id}` | any object | `control_tool`; or process termination / namespace teardown. Must be idempotent either way. |

Whichever strategy is used, these hold:

- **Reset replaces, never merges.** A second episode must not inherit the first's
  mutations. Under `no_op_verified` the strategy is only valid if no exposed tool can
  mutate observed state, and probe `P10` must confirm exactly that.
- **A `read_only_projection` is deterministic and pure.** The declared probe calls
  run in declared order, their results are serialized with the repo's
  `canonical_json`, and they must not mutate. Reading state must be repeatable,
  because check `D1` compares state around a call.
- **Episode binding is verified, not assumed.** Allowed channels, in order of
  preference: `transport` (one process or namespace per episode, so no token is
  needed), `meta` (`_meta.bfcl.episode_id` on the `tools/call` request, the
  specification's designated extension slot), or `argument` (an injected argument
  named in config). A server that accepts `_meta` and ignores it routes every call to
  one shared episode while looking healthy, so probe `P6` must prove binding works
  rather than trusting that it does. With `argument`, the injected name must not
  appear in the model-facing `tools.json`: the evaluated model never sees an episode
  id.
- **Isolation.** Two concurrent episodes must not observe each other's state, and a
  new episode must not inherit a previous one's. BFCL check `I1` separately requires
  `oracle_runtime.worker: process`; the gateway's own isolation mode is declared in
  config and probed by `P6`.
- **Determinism.** Given the same fixtures and the same `context`, two fresh episodes
  must produce identical results and identical state. Wall-clock reads, unseeded
  randomness, unpinned upstream dependencies, and counters that survive a reset all
  violate this, and check `D1` is what detects them.
- **Frozen time and seed.** `context.clock` and `context.seed` must be honored by
  whatever generates timestamps or identifiers.
- **Bounded calls.** A call must return or fail within `limits.tool_timeout_s`. Check
  `T1` requires that a hung tool is killed on the same path pack tools run through,
  so the gateway must cancel in flight — `notifications/cancelled` on stdio, closing
  the response stream on HTTP — and answer the route with a failure rather than
  hanging.
- **Structured errors.** Every failure the pack is expected to certify must arrive as
  a machine-readable code (§8). Check `D2` requires `error.code` on every observed
  structured error, and validation cases compare `expect.error_code`.
- **Mutation honesty.** A tool declared mutating must change state in at least one
  successful probe, and a tool that changes observed state must be declared
  (check `M1`). Declarations come from reviewed config, never from MCP `annotations`,
  which the specification requires clients to treat as untrusted.
- **Confirmation protocol.** A gated tool must return the pending status without
  mutating state when the confirmation argument is false, matching the pack
  manifest's `confirmation.{parameter, status_field, pending_status}` — which default
  to `confirm`, `status`, and `awaiting_confirmation`.

## 5. `mcp_oracle.yaml` Schema

The file lives in the oracle pack tree, so `pack_fingerprint` covers it and its
content is part of what a Gold verdict describes. It holds env var **names** only,
never secret values, mirroring `_reject_model_secrets`. Unknown keys are rejected
rather than ignored, mirroring `_reject_unknown`: a misspelled control name that
silently defaulted would be certified as a working oracle.

```yaml
profile_version: bfcl-mcp-oracle-v1
mode: A                              # A | B | C, per §2
mcp_protocol_versions: ["2026-07-28", "2025-06-18"]   # accepted at initialize

transport:
  kind: stdio                        # stdio | streamable_http
  # kind: stdio
  command: ["python", "-m", "acme_banking_mcp"]
  cwd: /opt/acme-banking-mcp         # must resolve under an allowed root
  env_passthrough: [ACME_MCP_TOKEN, ACME_MCP_TENANT]
  # kind: streamable_http
  # url: https://mcp.internal.example.com/mcp
  # auth: {bearer_token_env: ACME_MCP_TOKEN, headers: {X-Tenant: ACME_MCP_TENANT}}
  # tls: {ca_bundle_path: ./ca/internal-root.pem}

expected:
  server_name: acme-banking-mcp
  server_version: "3.2.0"
  tool_catalog_digest: "sha256:<64 hex>"
  oracle_id: acme-banking
  oracle_version: "3.2.0"
  content_digest: "sha256:<64 hex>"  # omit only when describe_oracle is absent

control:
  reset_strategy: control_tool       # control_tool | process_restart | namespace | no_op_verified
  state_strategy: control_tool       # control_tool | read_only_projection
  describe_oracle: bfcl.describe_oracle
  reset_episode: bfcl.reset_episode
  get_episode_state: bfcl.get_episode_state
  end_episode: bfcl.end_episode
  episode_binding: meta              # transport | meta | argument
  # episode_argument: episode_id     # required only when episode_binding=argument
  # state_projection:                # required only when state_strategy=read_only_projection
  #   - {tool: get_account, arguments: {account_id: acc_1}}
  #   - {tool: list_recent_transactions, arguments: {account_id: acc_1, limit: 10}}

fixtures:
  direction: pushed                  # pushed | snapshot (snapshot is mode C)
  # snapshot_calls:                  # required only when direction=snapshot
  #   - {tool: list_accounts, arguments: {}, collection: accounts}

tools:
  include: [get_account, list_recent_transactions, create_transfer]
  aliases: {"admin.audit.read": admin_audit_read}
  mutates: [create_transfer]
  requires_confirmation: [create_transfer]
  trust_annotations: false

results:
  error_path: error                  # where the server puts it; republished as "error"
  status_field: status               # must equal the pack manifest confirmation vocabulary
  pending_status: awaiting_confirmation

isolation: process_per_episode        # process_per_episode | namespace_per_episode

limits:
  connect_timeout_s: 5
  handshake_timeout_s: 5
  tool_timeout_s: 5
  reset_timeout_s: 10
  episode_timeout_s: 60
  max_response_bytes: 10485760
  max_tools: 64
  max_catalog_pages: 20
  max_concurrent_episodes: 4
  session_idle_ttl_s: 300
```

Validation rules the loader enforces:

- `profile_version` must be a profile this build implements; an unknown value is
  refused instead of being treated as the current one.
- `mode` must be consistent with the strategies it implies: `no_op_verified` requires
  an empty `tools.mutates`, `snapshot` requires `snapshot_calls`, and
  `read_only_projection` requires `state_projection`. A mode that contradicts its
  strategies is refused rather than reconciled.
- Exactly one transport shape: `command` with `stdio`, `url` with
  `streamable_http`. Mixing them is refused.
- `url` must be `https` with no credentials, query, or fragment. `http` is allowed
  only for `localhost` and only under a non-publication lineage policy.
- `command` must be an argument vector, never a shell string, and its executable must
  resolve under an allowed root. `env_passthrough` names variables to forward; every
  other variable is withheld from the child.
- `ca_bundle_path` must resolve under an allowed root and inside the pack tree, so it
  participates in the fingerprint the way the endpoint CA bundle does.
- `expected.tool_catalog_digest` is required. `expected.content_digest` is required
  unless `control.describe_oracle` is omitted, in which case §9's fallback applies
  and the level is capped at `L1`.
- `tools.include` is an explicit ordered allowlist. A discovered tool that is not
  listed is excluded; a listed tool that is not discovered is an error. There is no
  implicit "expose everything".
- `aliases` maps a discovered MCP name to the published BFCL name (§7). Values must
  be unique, must not collide with an unaliased name, and must not name a control
  tool.
- `mutates` and `requires_confirmation` must reference published names, and
  `requires_confirmation` requires the tool to declare the manifest's confirmation
  parameter.
- `trust_annotations` must be `false` at `L2`.
- `episode_argument` is required when and only when `episode_binding: argument`, and
  the named argument must not be a declared model-facing parameter of any published
  tool.
- `results.error_path` names where the server places its error object inside
  `structuredContent`; the gateway republishes it under the key `error`, because
  `_classify_result` classifies on exactly that key and nothing else.
  `results.status_field` and `results.pending_status` must equal the pack manifest's
  `confirmation.status_field` and `confirmation.pending_status`; disagreement is
  refused rather than resolved in favor of one file, since BFCL classifies with the
  manifest's vocabulary while the gateway would be mapping with another.
- Every value in `limits` must be a positive finite number; `NaN` and infinities are
  refused before they reach deadline arithmetic, as elsewhere in BFCL config.
- No key may hold a secret-looking literal. Credentials are named, never inlined.

## 6. MCP Version Support

`mcp_protocol_versions` lists the revisions the operator accepts. The version
negotiated at `initialize` must be one of them, and the negotiated value — not the
list — enters the catalog digest, so a server that silently downgrades changes the
pack's identity instead of changing its behavior unnoticed.

The profile depends on three features that arrived at different revisions, so support
is stated per feature rather than pinned to one specification date:

| Feature | Available from | If the negotiated version lacks it |
| --- | --- | --- |
| `structuredContent` on a tool result | `2025-06-18` | Mode `B` with a shim that projects `content` into a declared structured shape, from reviewed config. The gateway never parses free text on its own. |
| `outputSchema` on a tool definition | `2025-06-18` | Authoring evidence is weaker; assertions must be drafted from probes instead. Not a blocker. |
| `annotations` on a tool definition | `2025-03-26` | No effect: annotations are untrusted and unused for declarations either way. |

`ttlMs` and `cacheScope` on `tools/list` exist only in the newest revisions and are
ignored in every version: a cache directive is not part of a tool contract, and
honoring one would let a stale catalog back a fresh certification.

## 7. Tool Normalization

`tools.json` is an OpenAI-compatible function catalog whose parameter schema is
checked by `validate_tool_definition`, which enforces a deliberately small JSON Schema
subset. Normalization maps an MCP tool definition into that subset, or excludes the
tool with a recorded reason. It never widens the subset and never guesses.

| MCP field | BFCL destination |
| --- | --- |
| `name` | `function.name`, after the name rules below |
| `description` | `function.description`, after the text rules below |
| `inputSchema` | `function.parameters` |
| `outputSchema` | not published; recorded as authoring evidence and as input for assertion drafting |
| `annotations` | not published; recorded as unverified metadata only |
| `title`, `icons`, `_meta`, `x-mcp-header` | dropped |

`x-mutates` and `x-requires-confirmation` are siblings of `function` on the tool
entry, written from `tools.mutates` and `tools.requires_confirmation`. They are
internal: `project_model_facing_tools` strips them before a row is published.

### Names

MCP allows dots and up to 128 characters. Function-calling endpoints commonly accept
only `[A-Za-z0-9_-]` up to 64 characters, so a name discovery accepts can fail at
evaluation time instead. A discovered name is published unchanged when it matches
`^[A-Za-z0-9_-]{1,64}$`; otherwise it needs an explicit `tools.aliases` entry that
does, and the gateway keeps the alias-to-real mapping for its own `tools/call`. An
unmatched, unaliased name is excluded as `name_not_publishable`. Names are
case-sensitive and must be unique after aliasing.

### Parameter schemas

Allowed keywords are exactly the ones `validate_function_schema` accepts: `type`,
`description`, `title`, `default`, `examples`, `deprecated`, `readOnly`, `writeOnly`,
`format`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`,
`pattern`, `minLength`, `maxLength`, `minimum`, `maximum`, `minItems`, `maxItems`.
Allowed types are `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`.

- `inputSchema` must be `type: object`, or the tool is excluded as
  `parameters_not_object`.
- `$ref`, `allOf`, `anyOf`, `oneOf`, and `not` are excluded as
  `unsupported_schema_keyword`. MCP permits them and BFCL refuses them, so the tool is
  reported rather than silently flattened — an incorrectly inlined `$ref` would ship a
  schema the server does not actually enforce.
- Any other unrecognized keyword is excluded as `unsupported_schema_keyword`. Dropping
  it would publish a looser contract than the server implements.
- `additionalProperties` must be a boolean and is preserved as declared. It is never
  synthesized: adding `false` would make BFCL reject arguments the server accepts, and
  adding `true` would claim the opposite.
- `required` must list declared properties only.
- An unsatisfiable constraint is refused, not published: `minimum` above `maximum`, an
  empty `enum`, a duplicated `enum` value, or an `enum` or `const` value the declared
  `type` rejects. Check 3 would fail the pack anyway; failing at intake names the tool.
- Order is deterministic: tools sorted by published name, object keys canonicalized
  through `canonical_json`, `required` sorted. Two discoveries of an unchanged server
  must produce byte-identical `tools.json`.

### Descriptions

A description is model-facing text from an untrusted source, so it is treated as data.
It is length-capped, must be plain text, and a description carrying imperative
instructions to the assistant, tool-selection guidance, or anything resembling a
system prompt is flagged for review before it can enter a published pack. See
`bfcl-mcp-threat-model.md` §2.

### Mutation and confirmation

`annotations.readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`
may be shown to a reviewer as a suggestion and recorded as evidence, but they never
become the declaration: check `M1` compares the declaration against observed state
change, and an untrusted hint could certify a false claim.

### Catalog paging

`tools/list` may return `nextCursor`. All pages are followed up to
`limits.max_catalog_pages`; a truncated catalog is an error, not a smaller catalog.
`listChanged` notifications are not subscribed to during a run: the catalog is pinned
by digest, so a change is drift to report rather than an update to apply.

## 8. Result And Error Mapping

`call_tool` must return an object BFCL can classify into exactly one of `success`,
`structured_error`, or `awaiting_confirmation` — the three classes
`validation_cases.yaml` declares through `expect.result_class`. MCP's result shape is
looser than that, so the mapping is total and explicit.

| MCP `tools/call` result | Mapped BFCL result | Class |
| --- | --- | --- |
| `isError` absent or false, `structuredContent` is a JSON object carrying no error at `results.error_path`, and its `status_field` is absent or not `pending_status` | `structuredContent` verbatim | `success` |
| `isError` absent or false, `structuredContent` is an object whose `status_field` equals `pending_status` | `structuredContent` verbatim | `awaiting_confirmation` |
| `isError: true` **and** the object at `results.error_path` carries `code` as a non-empty string | that object under the key `error` | `structured_error` |
| `isError` absent or false but the object at `results.error_path` carries `code` | that object under the key `error`, and the inconsistency is recorded | `structured_error` |
| `isError: true` without a machine-readable code | route failure `mcp_unstructured_error` | none |
| `structuredContent` absent, or not a JSON object | route failure `mcp_result_not_object` | none |
| `InputRequiredResult` | route failure `mcp_input_required_unsupported` | none |
| A task handle (Tasks extension) | route failure `mcp_async_task_unsupported` | none |
| JSON-RPC protocol error | route failure `mcp_protocol_error` | none |
| Timeout or cancellation | route failure `mcp_call_timeout` | none |
| Transport failure mid-call | route failure `mcp_call_failed`, never a retry | none |

The `content` array is never the source of truth. It is human-facing text a server may
format freely, and MCP only *recommends* mirroring `structuredContent` into it. It is
retained in the diagnostic artifacts and never reaches a published row.

`structuredContent` may legally be any JSON value, including an array or a scalar.
BFCL reads results as mappings — `_classify_result` looks up the `error` key and a
status field, and assertions index fields — so a non-object result is refused rather
than wrapped. Wrapping would invent a key the server never emitted, and every
assertion and validation case would then be written against gateway-invented
structure. A **declared** projection is a different matter and is the deferred
extension in §11: the distinction that matters is whether a human reviewed and pinned
the shape, not whether the gateway could guess it.

The error object passes through as-is beyond requiring `code`, so a server may carry
`entity`, `id`, `field`, and `message` exactly as `bfcl-oracle-pack.md` describes for
a Python backend. A route failure is a defect in the oracle or the transport, not a
business outcome: it surfaces as a failing probe or a dropped task and can never
satisfy a negative validation case.

Worked example of a confirmation-gated mutation:

```json
// tools/call → create_transfer {"to_account": "acc_2", "amount_vnd": 200000, "confirm": false}
{
  "content": [{"type": "text", "text": "Confirm transfer of 200,000 VND to acc_2?"}],
  "structuredContent": {"status": "awaiting_confirmation", "amount_vnd": 200000}
}
// mapped to {"status": "awaiting_confirmation", "amount_vnd": 200000} → awaiting_confirmation
// check 6 then requires the state read to be unchanged
```

```json
// tools/call → create_transfer {"to_account": "acc_9", "amount_vnd": 200000, "confirm": true}
{
  "content": [{"type": "text", "text": "No such account: acc_9"}],
  "structuredContent": {"error": {"code": "account_not_found", "entity": "account", "id": "acc_9"}},
  "isError": true
}
// mapped to {"error": {"code": "account_not_found", ...}} → structured_error, satisfies expect.error_code
```

## 9. Identity And Digests

BFCL compares oracle identity at four moments — prepare, session creation,
generation, and publication — so the gateway must publish an identity that changes
when the oracle's behavior changes and stays stable when nothing does. Every digest
below is `sha256` over `row_schema.canonical_json` of the stated document, so key
order, separators, and non-ASCII handling are decided once, in code the pipeline
already uses.

- **`tool_catalog_digest`** covers
  `{profile_version, mode, negotiated_mcp_version, server_name, server_version, tools: [normalized definitions in published order], control: {resolved names and strategies}, adapter_version, shim_version}`.
  Volatile fields — `nextCursor`, `ttlMs`, `cacheScope`, request ids — are excluded,
  because a cache directive or a pagination token is not part of the contract.
- **`content_digest`** is the digest the server reports through `describe_oracle`. It
  is the server's own statement about its domain content — fixture handling, business
  rules, data version — which a catalog digest cannot see. In mode `C` it must also
  cover the fixture snapshot, so upstream data drift changes identity.
- **Fallback.** Without `describe_oracle`, the gateway reports the catalog digest as
  `content_digest` and records `content_digest_source: catalog_fallback`. That pack is
  capped at `L1`: a server can change its business logic without changing one byte of
  its catalog, so the fallback cannot support a claim that the oracle being replayed
  is the oracle that was certified.
- **Handshake pinning.** `expected.server_name`, `expected.server_version`, and the
  negotiated MCP version are checked before any probe runs, because a probe against
  an unexpected server produces confident results about the wrong thing.

## 10. Conformance Probes

`MCP-4xx` implements these as a preflight the gateway runs before BFCL sees the
endpoint. Each probe names the BFCL check it protects, so a failure explains itself in
the vocabulary of the existing gold gate.

| Probe | Passes when | Protects | Needed for |
| --- | --- | --- | --- |
| `P1 handshake` | negotiated MCP version, server name, and server version match `expected` | identity comparison | `L1` |
| `P2 catalog` | all pages retrieved, digest matches `expected.tool_catalog_digest`, every `tools.include` entry found, published set is exactly `tools.include` | check 3, `TM-13` | `L1` |
| `P3 normalization` | every included tool maps into the supported schema subset | check 3 | `L0` |
| `P4 control` | the mode's reset, state, and teardown strategies work with the declared shapes | reset, state, close | `L1` |
| `P5 reset` | two fresh episodes with identical fixtures and context yield identical state | check `D1` | `L2` |
| `P6 isolation` | two concurrent episodes do not observe each other's mutations, and a bound call reaches its own episode | check `I1`, binding | `L2` |
| `P7 error shape` | at least one deliberately invalid call returns `error.code` | check `D2` | `L2` |
| `P8 confirmation` | a gated tool called with confirmation false returns the pending status and leaves state unchanged | check 6 | `L2` when any tool is gated |
| `P9 timeout` | a call exceeding `limits.tool_timeout_s` is cancelled and reported | check `T1` | `L2` |
| `P10 mutation` | every `tools.mutates` entry changes state in at least one successful probe, and no undeclared tool does | check `M1` | `L2` |
| `P11 rejected shapes` | no included tool returns `InputRequiredResult`, a task handle, or non-object content | §8 | `L2` |

A probe whose precondition failed is recorded as skipped, never as passed, following
the rule `run_oracle_validation` already applies. `P8` is skipped rather than passed
when no tool is gated, and a mode `C` configuration therefore cannot claim
confirmation coverage.

## 11. Deferred Extension Points

Each of these is a decision, not an omission. They are listed so a later reader can
tell the difference.

| Deferred | Why, and what would unblock it |
| --- | --- |
| Declared projection of non-object `structuredContent` | The refusal in §8 is about unreviewed guessing, not about the shape itself. A per-tool projection pinned in config and fingerprinted would be reviewable; it needs its own probe proving the projection is total over observed results. |
| MCP Tasks extension and `InputRequiredResult` | Both turn one BFCL call into a multi-round-trip exchange with server-side lifecycle, while `expected_trace` and `executable_replay` assume one call, one result. Needs a conversation-level contract first. |
| `listChanged`-driven catalog refresh | A catalog that can change mid-run cannot be pinned by digest. Would need a rule for what a mid-run change means for rows already generated. |
| Multi-server tool aggregation | MCP scopes name uniqueness per server, so aggregation needs a disambiguation and provenance scheme per tool, and `content_digest` would have to compose. |
| MCP resources and prompts as oracle truth | Neither is executable state, and a prompt from an untrusted server entering a benchmark is threat `TM-01` with fewer controls. |
| Sampling requests from the server | Would let the oracle call a model during certification, making the certified result model-dependent. |
