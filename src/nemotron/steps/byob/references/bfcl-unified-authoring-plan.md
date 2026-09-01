# BFCL unified authoring plan (Flow 2, Epics 7–12)

Task prefix: `UA`. Status: in progress; Epics 7–11 and UA-1201–UA-1205 are
completed.
Continues Epics 0–6 of the MCP integration
(`bfcl-mcp-architecture-decision.md`, `bfcl-mcp-support-matrix.md`).

Scope: turn the assisted-authoring architecture described in
`BFCL_LLM_Generated_Workflow_Overview.docx` into implemented, tested behaviour. The
DOCX is product and architecture context; this version-controlled plan, the referenced
contracts, and named tests are the implementation acceptance source. A DOCX claim is
not complete until it is represented by a repository contract and executable test.

Non-scope: the manual flow. A hand-authored Oracle Pack keeps its current inputs and
runs `prepare_bfcl → Gold Gate → generate_bfcl → publish` unchanged. Minimal input
(one source plus one domain brief) applies only to the LLM-assisted lane.

## Current state

Epics 7–11 and UA-1201–UA-1205 are implemented. Flow 2 now has a static adapter
registry, A0–A2 certification, evidence schema v2, conventional and MCP intake,
held-out proof and question/answer handling, transactional revisions, a two-boundary
guided CLI, shared review/freeze/publication contracts, structured events, credential
lifecycle, cache retention, revocation, and test-linked operator documentation.

The support source is
[bfcl-authoring-support-matrix.md](bfcl-authoring-support-matrix.md); the operator entry
point is [bfcl-authoring-user-guide.md](bfcl-authoring-user-guide.md). MCP Mode B,
MCP Mode C, dynamically installed third-party adapters, and the UA-1206 multi-domain
rollout evidence are explicitly **unimplemented**.

## Cross-cutting constraints

Every task must hold these:

1. No new trust path. `load_pack`, `prepare_bfcl`, `run_oracle_validation`,
   `derive_pack_tier`, and `generate_bfcl` stay authoritative.
2. Every new artifact is digest-bound and reproducible from recorded inputs.
3. Failures fail closed. Ambiguity, drift, timeout, and unproven business meaning stop
   the run instead of degrading it.
4. Adapter-declared capabilities are untrusted claims. Only a BFCL-owned verifier can
   issue a certification report or raise an attained certification tier.
5. Adapter certification gates only the LLM-assisted release envelope. It must not
   alter the manual-pack contract or require manual packs to carry adapter metadata.
6. Existing MCP and manual tests stay green. Legacy v1 releases remain verifiable;
   new v2 releases are deterministic and semantically equivalent, but are not required
   to be byte-identical to v1.

## Contract decisions fixed by this plan

- **Descriptor versus certification:** `AdapterDescriptor` declares identity and
  capabilities. `AdapterCertificationReport` is a separate BFCL-issued artifact with
  probe evidence, attained tier, issuer, contract version, and digest.
- **Certification scope:** `A0–A2` applies to Flow 2 authoring and release. The existing
  canonical Gold Gate remains source-independent and unchanged for manual packs.
- **Conditional probes:** an A2 profile may mark a capability-specific probe
  `not_applicable` only with a machine-readable reason accepted by the profile. A
  missing probe is never equivalent to not applicable.
- **HTTP source package:** an HTTP authoring source is a reviewed declaration containing
  `endpoint_config.yaml` plus a tool-schema artifact such as `tools.json`. Oracle HTTP
  v1 `/v1/tools` supplies names only and is not silently treated as schema discovery.
- **Two authorization boundaries:** pre-model evidence authorization controls exposure
  of untrusted source text; final semantic approval authorizes freeze/publication. A
  guided session may present both simply, but it must not collapse or bypass either.
- **Schema migration:** v1 source bytes and v2 normalized bytes have distinct digests.
  The migration record binds both digests and the transformer identity. Drafting
  approval binds the normalized bytes that the model reads.
- **Adapter loading:** the built-in registry is a static allowlist. Detection performs
  no network call, subprocess launch, or import of user-controlled Python. Explicit
  plug-ins are future work and require a separate trust policy.

## Execution order

`Epic 7 → Epic 9 → Epic 8 → Epic 10 → Epic 11 → Epic 12`

Epic 9 precedes Epic 8 on purpose: held-out status and the evidence contract determine
the evidence shape, so settling them first avoids writing the two new adapters twice.

Epic 7 internal order:

`UA-701 → UA-702 → UA-707 → UA-704 → UA-703 → UA-706 → UA-705`

Suggested first reviewable slice: `UA-701` and `UA-702`. The contract and vocabulary
must exist before migration or shared-library conversion. These tasks change formats,
so they require compatibility tests rather than a claim of no behaviour change.

## Task completion template

Every implementation task must include in its PR:

1. the contract/schema or code deliverable named below;
2. unit and negative tests, including tamper/unknown-field cases where applicable;
3. migration or compatibility evidence when a persisted artifact changes;
4. documentation of new refusal reason codes and operator action;
5. a mapping from the task ID to one or more named tests.

## Dependency gates and required test files

| Gate | Required tasks | Exit condition |
| --- | --- | --- |
| G7A — schema foundation | UA-701, UA-702, UA-707 | Generic contracts and domain brief exist without changing an existing runtime path |
| G7B — compatibility | UA-704, UA-703, UA-706, UA-705 | Certification exists before migration; MCP runs through v2; v1 remains verifiable |
| G9 — safe model input | UA-901–UA-906 | Held-out, questions, answers, and pre-model authorization are bound before drafting |
| G8 — adapter parity | UA-801–UA-807 | Local, HTTP, and MCP produce equivalent generic evidence and verifier outcomes |
| G10 — durable workflow | UA-1001–UA-1006 | Revisions and resume survive crash, tampering, and concurrency tests |
| G11 — user path | UA-1101–UA-1106 | Guided CLI and shared release work end to end without bypassing either approval |
| G12 — rollout | UA-1201–UA-1206 | Operations controls and broader evaluation support a production-readiness decision |

Minimum named test ownership:

- `test_bfcl_source_adapter_contract.py` — UA-701, UA-702, UA-806.
- `test_bfcl_source_adapter_registry.py` — UA-705.
- `test_bfcl_source_evidence_migration.py` — UA-703.
- `test_bfcl_source_adapter_certification.py` — UA-704.
- `test_bfcl_domain_brief.py` — UA-707.
- Existing `test_bfcl_pack_authoring.py` and `test_bfcl_pack_drafting.py` —
  UA-706 compatibility and recorded-output regression.
- `test_bfcl_local_authoring_adapter.py` — UA-801, UA-802.
- `test_bfcl_http_authoring_adapter.py` — UA-803, UA-804.
- `test_bfcl_source_intake.py` — UA-805.
- `test_bfcl_endpoint_conformance.py` — UA-807.
- `test_bfcl_authoring_held_out.py` — UA-901, UA-902.
- `test_bfcl_authoring_questions.py` — UA-903, UA-904, UA-905.
- `test_bfcl_authoring_authorization.py` — UA-906.
- `test_bfcl_authoring_revisions.py` — UA-1001, UA-1003, UA-1004, UA-1006.
- `test_bfcl_authoring_workspace_lock.py` — UA-1002.
- `test_bfcl_authoring_quotas.py` — UA-1005.
- `test_bfcl_authoring_cli.py` — UA-1101, UA-1102, UA-1105.
- `test_bfcl_authoring_release.py` — UA-1103, UA-1104.
- `test_bfcl_authoring_e2e.py` — UA-1106.
- `test_bfcl_authoring_events.py`, `test_bfcl_authoring_credentials.py`,
  `test_bfcl_authoring_retention.py`, and `test_bfcl_release_revocation.py` —
  UA-1201 through UA-1204 respectively.
- `test_bfcl_authoring_documentation.py` — UA-1205.
- `test_bfcl_mcp_ablation.py` and `test_bfcl_mcp_ablation_rollout.py` — UA-1206 contract
  and rollout gating. The rollout decision stays descriptive until real multi-domain
  observations exist, so these tests own the gate, not the evidence.
- Existing MCP, manual, endpoint, held-out, and publication suites remain regression
  owners for every gate.

## Epic 7 — Adapter contract and certification ladder

Closes: "one versioned source-adapter contract"; "trust follows a certification ladder,
not a transport name".

- **UA-701 (M) Adapter contract module.** Add `runtime/source_adapters/contract.py`
  defining the `OracleSourceAdapter` protocol, capability enum (`describe_tools`,
  `describe_state`, `pin_identity`, `observe`, `reset_state`, `get_state`),
  `AdapterDescriptor`, and dedicated errors. The descriptor contains declarations but
  no attained tier. **Acceptance:** strict serialization rejects unknown authority
  fields; a fake adapter can be exercised without importing MCP; no adapter can create
  a certification report. **Status:** implemented and covered by
  `test_bfcl_source_adapter_contract.py`.
- **UA-702 (M) Transport-neutral evidence schema.** Define `bfcl-source-evidence-v2`
  with a `source_adapter` block, a `certification` block, declared `capabilities`, and
  `unresolved_gaps`. `certification` references a separately digested BFCL-issued
  report; it is not copied from the adapter descriptor. **Acceptance:** canonical
  round-trip is byte-stable; missing, duplicate, unknown authority fields, and digest
  tampering fail closed. **Status:** implemented and covered by
  `test_bfcl_source_adapter_contract.py`.
- **UA-707 (M) Domain-brief contract and prompt boundary.** Add a bounded UTF-8
  domain-brief input with content digest, language, optional policy metadata, prose
  hygiene scan, PII/redaction report, and explicit precedence below executable
  evidence. Add the sanitized brief and its digest to the evidence bundle, model input,
  request hash, and provenance. **Acceptance:** changing one brief byte changes the
  request hash; instructions embedded in the brief are fenced as untrusted data;
  oversized, undecodable, or policy-violating briefs stop before a model call.
  **Status:** implemented in the v2 contract and covered by
  `test_bfcl_domain_brief.py`; activation in the current drafting runner belongs to
  UA-706.
- **UA-703 (M) Version negotiation and migration.** Read `bfcl-mcp-evidence-v1` and
  transform it deterministically to v2. Persist a migration record containing source
  digest, normalized digest, transformer ID/version/digest, and warnings. Evidence
  approval binds normalized v2 bytes; legacy approval cannot silently authorize a
  transformed document. **Acceptance:** v1 remains verifiable, v2 is deterministic,
  unknown/newer versions fail closed, and approval swap tests prove both digests are
  enforced. **Status:** implemented with strict v1/v2 negotiation, an immutable
  migration record binding source, normalized, and transformer digests, explicit
  reviewed context for v2-only fields, and a v2 approval that binds the source,
  normalized evidence, migration record, and warning acknowledgements. Covered by
  `test_bfcl_source_evidence_migration.py`; activation in shared drafting remains
  UA-706.
- **UA-704 (L) Certification ladder.** Implement tiers `A0` (identity and catalog
  integrity), `A1` (bounded read-only observation, structured error shape), and `A2`
  (reset determinism, episode isolation, confirmation safety, mutation declaration,
  bounded timeout and cleanup, observed result coverage). Map MCP `L0/L1/L2` and
  `P1–P11` onto the ladder as the reference realization. Define profile-owned
  conditional applicability and reason codes. The BFCL verifier, not the adapter,
  emits `AdapterCertificationReport`. **Acceptance:** forged/self-issued reports,
  missing probes, invalid `not_applicable`, issuer mismatch, and report drift fail;
  Flow 2 freeze refuses below A2 while manual Gold tests remain unchanged.
  **Status:** certification profile/report, independent rebinding verifier, and the
  deterministic MCP P1–P11 projection are implemented and covered by
  `test_bfcl_source_adapter_certification.py`. MCP's legacy labels are not treated as
  authority: P1–P3 can establish A0, P4 plus the applicable P7 evidence can establish
  A1, and the complete applicable P1–P11 set can establish A2. Activation of the A2
  check at the shared Flow 2 freeze boundary belongs to UA-1103; existing manual Gold
  and MCP release regression tests remain unchanged.
- **UA-705 (S) Adapter registry.** Resolve exactly one adapter from a source
  declaration using a static built-in allowlist and side-effect-free detection.
  Ambiguous matches stop before the first model call. Allow only explicitly namespaced,
  non-authoritative extension fields. **Acceptance:** detection performs no network,
  subprocess, or user-code import; zero and multiple matches have stable reason codes.
  **Status:** implemented with inert source blocks, an immutable metadata-only built-in
  allowlist for local Python, HTTP package, and MCP Mode A, stable fail-closed errors,
  digest-bound declarations, and bounded namespaced JSON extensions that cannot carry
  authority fields or affect selection. Covered by
  `test_bfcl_source_adapter_registry.py`.
- **UA-706 (M) Migrate shared drafting.** Point `pack_authoring/bundle.py` at v2 and
  replace `attained_level` string handling with verified `certification.tier`. Preserve
  a compatibility reader at the boundary, not MCP branches inside prompts or drafting.
  **Acceptance:** recorded-model output is unchanged for semantically equivalent v1/v2
  evidence, and cache keys differ when normalized evidence differs. **Status:**
  implemented. Shared drafting now verifies the BFCL certification report and v2
  approval before model access, supports native v2 and explicitly bound v1 migrations,
  fences the domain brief, keys model requests by normalized evidence digest, and keeps
  direct v1 loading as a non-certified compatibility boundary. Covered by
  `test_bfcl_pack_drafting.py`; related MCP and manual release behavior is unchanged.

Done when: the MCP lane runs end to end through the new contract, and every existing
MCP behaviour test stays green. Persisted v1 fixtures may be supplemented with v2
fixtures, but legacy verification tests must remain.

## Epic 8 — Conventional adapters

Closes: "local Python and BFCL Oracle HTTP v1 as first-class adapters".

- **UA-807 (L) Provider-neutral conformance and origin.** Generalize the endpoint
  conformance verifier and origin/lineage reader so provider-specific evidence is
  selected by a strict profile rather than `provider_kind == "mcp"`. Keep MCP wrappers
  and legacy imports. This task precedes every conventional endpoint implementation.
  **Acceptance:** MCP fixtures retain existing verdicts; local and HTTP reports use the
  same generic verifier; unknown providers, unknown profiles, origin ambiguity,
  profile/report mismatch, and lineage drift fail with stable reason codes. **Status:**
  implemented. Conformance parsing and tier derivation now resolve an immutable
  `(provider_kind, profile_version)` contract; probe coverage, suite identity, timeout
  evidence, snapshot semantics, caps, and publishable level come from that profile.
  Unknown and explicitly mismatched profiles return stable refusal codes. The endpoint
  Gold check accepts the same strict profile registry, while the v1 MCP wire-field names
  remain compatibility fields. Origin loading resolves exactly one immutable
  `(schema_version, provider_kind, origin)` profile by a pack-local lineage path, rejects
  ambiguity and registry drift, and validates endpoint bindings generically. Production
  publication and evaluation use the generic reader; `load_mcp_origin` remains a
  compatibility wrapper. Covered by `test_bfcl_endpoint_conformance.py` and
  `test_bfcl_mcp_release_review.py`.
- **UA-804 (L) Published certification profiles and refusal taxonomy.** Define the
  adapter-neutral probe runner/verifier contract and publish the `A0–A2` profile for
  each adapter kind before implementing its probes. An adapter only returns typed raw
  observations. The BFCL certification service selects the profile, enforces budgets,
  derives applicability, and signs the report; an intake runner only orchestrates and
  persists verified outputs. **Acceptance:** each required probe declares owner,
  input digest, execution budget, safety class, cleanup rule, outcome schema, and
  allowlisted `not_applicable` reasons. Every refusal uses a documented stable code,
  including identity drift, catalog mismatch, missing reviewed schema, unsafe probe,
  timeout, cleanup failure, nondeterminism, state leakage, and under-certification.
  **Status:** implemented. The immutable built-in registry now publishes distinct
  `mcp_mode_a`, `local_python`, and `http_package` profiles. Every generic probe binds
  BFCL ownership and evidence issuer, input/outcome schema versions, safety class,
  call/time budgets, cleanup semantics, closed failure codes, and profile-owned
  applicability reasons. `AdapterProbeObservation` deliberately excludes authority
  fields; BFCL projects it into digest-bound outcomes, derives the tier, applies the
  descriptor ceiling, and signs the report. Unknown/missing reasons and unregistered
  adapter kinds fail closed. The contract and taxonomy are documented in
  `bfcl-source-adapter-certification-profiles.md` and covered by
  `test_bfcl_source_adapter_certification.py`. Actual local and HTTP probe execution
  remains UA-802 and UA-803 respectively.
- **UA-801 (L) `local_python` adapter and A0 identity.** Normalize `tools.json` and
  deterministic fixture metadata, then define an effective-content identity covering
  `backend.py`, the statically resolved in-pack import closure, interpreter/build
  identity, platform ABI, and declared dependency-lock digest. The import-closure
  grammar and dependency-lock policy are contract prerequisites, not implementation
  guesses. Reject dynamic or undeclared imports that escape the reviewed closure.
  **Acceptance:** changes to any identity input invalidate A0 evidence; symlink escape,
  namespace-package ambiguity, missing lock state, dynamic import, and unknown source
  encoding fail with stable codes; unrelated host files do not affect identity.
  **Status:** implemented. A local source is a reviewed directory containing
  `backend.py`, `tools.json`, and canonical `dependency-lock.json`, with optional
  `fixtures.json`. Static AST traversal covers executable in-pack imports and package
  initializers without importing code; namespace packages, ambiguous module/package
  pairs, dynamic import/code operations, undeclared third-party imports, unknown
  encodings, syntax errors, and symlink escape fail closed. The A0 identity binds the
  exact closure, canonical reviewed catalog, deterministic fixture metadata, dependency
  lock, CPython implementation/build, SOABI, platform, and machine identity. Covered by
  `test_bfcl_local_authoring_adapter.py`. Its descriptor is deliberately capped at
  `describe_tools` plus `pin_identity` with identity-only safety; UA-802 must issue a
  digest-distinct descriptor before A1/A2 execution.
- **UA-803 (L) `http_package` adapter.** Treat one source package as endpoint
  declaration plus reviewed companion tool-schema artifact. Verify Oracle HTTP v1
  identity and attestation, exact name/schema equality against `/v1/tools`, TLS and
  redirect policy, credential references, and endpoint probes through UA-807 and the
  UA-804 HTTP profile. Never infer schemas from names or observed values.
  **Acceptance:** missing or stale schema artifact, catalog mismatch, endpoint identity
  drift, cross-origin redirect, unsupported auth, attestation mismatch, and unbounded
  response fail closed before evidence issuance.
  **Status:** implemented. An HTTP source is a reviewed directory containing
  `endpoint_config.yaml` and `tools.json`. The adapter reuses the Oracle HTTP v1
  HTTPS/TLS/auth/redirect/size contract, requires a pinned conformance attestation,
  resolves only credential references, compares the exact live name set with the
  separately reviewed schemas, and binds endpoint identity, catalog, attestation,
  config, and optional CA bundle into A0 evidence. The production conformance registry
  now includes `bfcl-http-oracle-v1`; no schema is inferred from endpoint names or
  observed values. Catalog loading is shared with `local_python`. Covered by
  `test_bfcl_http_authoring_adapter.py`; A1/A2 endpoint probes remain part of this
  adapter's later execution phase and UA-806 integration.
- **UA-802 (L) Local observation probes.** After UA-804 and UA-801, execute read-only
  and reset-isolated probes through the existing `ProcessWorker` under a dedicated
  least-privilege execution policy. Adapter code does not assign tiers. The independent
  verifier derives A1/A2 from raw observations. **Acceptance:** nondeterminism, timeout,
  escaped import, process-child escape, state leakage, unknown commit state, incomplete
  reset, cleanup failure, and undeclared mutation produce stable failures; probe
  fixtures cannot overlap held-out material.
  **Status:** implemented. `bfcl-local-probe-plan-v1` bounds successful, error, and
  timeout representatives and requires success coverage for every reviewed tool.
  `bfcl-local-least-privilege-v1` statically refuses host-file, dynamic-code, network,
  process, and unreviewed dependency surfaces before starting code. A digest-distinct
  reset-isolated descriptor executes catalog/symbol alignment, success/error cases,
  deterministic replay, fresh-process isolation, confirmation non-mutation, hard
  timeout plus recovery, mutation truthfulness, and result-shape coverage through
  `ProcessWorker`. Probe inputs and observed values are held-out scanned; persisted
  evidence contains bounded shapes/digests rather than raw state. Identity is rechecked
  before and after execution, and only the shared certification service derives A1/A2.
  Covered by `test_bfcl_local_authoring_adapter.py`.
- **UA-805 (M) Transport-neutral intake runner.** After UA-804 and both adapters, add
  one shared orchestrator that invokes a resolved adapter, obtains independently signed
  certification, applies the Epic 9 held-out/redaction and exposure gates, and emits
  intake provenance plus `bfcl-source-evidence-v2`. MCP uses the same orchestrator
  through a compatibility wrapper. **Acceptance:** local, HTTP, and MCP evidence pass
  the same generic loader with no adapter branch in prompts or drafting; failed runs
  publish no partial authoritative output.
  **Status:** implemented. `source_adapters/intake.py` resolves inert declarations,
  dispatches only to the fixed local/HTTP collectors, projects BFCL-owned observations
  through the published profile, derives and signs the tier, constructs the shared v2
  evidence schema, proves held-out redaction, and emits the exact model-exposure
  subject. It publishes evidence, observations, trust reports, source brief, exposure
  subject, and digest-bound provenance through a sibling staging directory followed by
  one atomic rename. A refusal leaves no output directory. The MCP v2 runner now keeps
  discovery and legacy pack compatibility at its edge but delegates certification,
  brief sanitization, migration-backed evidence construction, held-out proof, and
  exposure-subject construction to the same `finalize_source_intake` trust spine.
  Local and HTTP parameterized coverage plus the existing MCP suite verify the common
  envelope and compatibility path. See `bfcl-transport-neutral-intake.md`.
- **UA-806 (M) Adapter contract and integration tests.** Prove all adapters emit the
  same schema and obey the same digest, authorization, held-out, timeout, cleanup, and
  refusal rules. **Acceptance:** one parameterized suite runs against local, HTTP, and
  MCP fixtures; adapter-specific exceptions require a versioned profile rule. Tests
  cover A0 intake, A1 observation eligibility, A2 Gold eligibility, stale certification,
  changed credentials/principal, and every stable refusal code.
  **Status:** implemented. A single parameterized matrix executes real local and HTTP
  intake plus the MCP compatibility runner, reloads every result through the same
  `SourceEvidenceDocument`, and independently verifies evidence shape, signed
  certification input binding, profile-bound observations, brief source/redaction,
  held-out proof, exposure subject, human authorization, and stale identity refusal.
  The drafting loader resolves persisted profile IDs only through the fixed published
  registry and verifies an optional observations sidecar, allowing execution-plan-bound
  local A1/A2 reports to reload without an MCP-specific profile branch.
  All three A0 fixtures are rejected at an A2 boundary. Reset-isolated local fixtures
  prove positive A1 and A2 intake, explicit lower-tier gaps, timeout/cleanup gating, and
  atomic refusal when A1 evidence is requested as A2. MCP now persists the same
  `source_observations.json` sidecar and binds it in provenance; its legacy observation
  shape is allowed only under `mcp-mode-a-v1`. HTTP tests bind credential references
  into source identity without persisting secret values and retain live principal,
  catalog, and attestation drift checks. The taxonomy guard covers every stable enum
  member and disallows free-form profile failure reasons. Covered by
  `test_bfcl_source_intake.py` and the focused certification/adapter suites.

Mandatory order is `UA-807 → UA-804 → {UA-801, UA-803} → UA-802 → UA-805 → UA-806`;
UA-801 and UA-803 may proceed in parallel after both prerequisites. Tier timing is
explicit: A0 is required before evidence can be issued; A1 is required before
model-visible observed results can be used; A2 is required by the shared Flow 2 freeze
and fresh-Gold boundary. Catalog-only drafting may proceed from A0 only when the
evidence marks all unobserved behavior unresolved and model exposure is separately
authorized.

Done when: a local-Python source and an HTTP source both reach a reviewed draft using
the same drafting library and the same evidence schema as MCP, and neither can reach
Flow 2 Gold freeze below a freshly verified A2 report.

## Epic 9 — Held-out strictness and the question/answer loop

Closes: "held-out status is explicit before model access"; "unresolved authority becomes
a question, never a guess".

- **UA-901 (M) Held-out contract at intake.** Accept only `required` with a policy path
  or `not_applicable` with a reviewed reason. Remove the `not_declared` acceptance path
  from assisted-authoring review and freeze without changing manual pack loading.
  **Acceptance:** both valid states work; absence blocks before domain brief or source
  prose reaches a model. **Status:** implemented. Assisted-authoring v2 evidence now
  embeds a content-addressed `bfcl-held-out-decision-v1`; MCP v2 intake requires exactly
  one reviewed `required` policy or `not_applicable` reason before reading intake/source
  inputs, and v2 review rejects missing or conflicting policy state. Legacy/manual pack
  loading retains its existing behavior. A legacy review may display `not_declared`
  only while verifying a pre-existing v1 release; that state cannot enter new assisted
  drafting or satisfy the v2 Flow 2 freeze contract. Covered by
  `test_bfcl_authoring_held_out.py` and MCP/source-evidence regressions.
- **UA-902 (M) Redaction proof.** Prove held-out identifiers and content are absent
  from the evidence bundle before any model call. Persist only a policy digest and
  redaction report. **Acceptance:** exact, normalized, encoded, and nested occurrences
  in model-visible payloads are detected by tests. **Status:** implemented. Runtime-only
  terms are derived from held-out fixture/template identifiers and optional reserved
  content, then scanned recursively across the finalized v2 evidence for exact,
  Unicode NFKC/casefold, URL-percent, base64, and nested-JSON forms. Cleartext terms are
  never persisted; the BFCL-signed report contains only the policy/decision/evidence
  bindings, an aggregate term commitment, sanitized finding locations, and its own
  digest. A canonical policy schema rejects unknown selector fields; any non-empty
  reservation requires runtime-only reserved content. Drafting, review, and freeze
  independently reload the reviewed policy/content, recompute the commitment, and
  re-scan the exact evidence before use. Intake writes through a staging directory and
  publishes it only after redaction succeeds.
  Covered by `test_bfcl_authoring_held_out.py` and drafting/MCP regressions.
- **UA-903 (M) Open-question artifact.** Add a bounded, digest-bound
  `open_questions.json` with typed answer slots, source evidence references, allowed
  answer domain, impact, and stable question identity. **Acceptance:** arbitrary free
  text cannot override executable facts or authority fields. **Status:** implemented.
  `pack_authoring/questions.py` provides deterministic question/set identities,
  resolvable `#/unresolved_gaps/<index>` references, fenced prompts and string answers,
  bounded impact declarations, and
  closed boolean/numeric/enum answer domains. Targets are restricted to
  `/semantic/...` through the same validator used by persisted semantic answers, and
  explicitly reject identity, adapter, certification, signature, digest, and tier
  authority segments; open artifacts cannot contain answers.
  Duplicate keys, stale evidence, tampering, excessive size/count, and unbounded string
  answers fail closed. Covered by `test_bfcl_authoring_questions.py`.
- **UA-904 (M) Answer application.** Apply answers as a new content-addressed evidence
  revision linked in provenance. Never mutate an existing bundle in place. **Acceptance:**
  stale, duplicate, extra, or type-invalid answers fail; old approval cannot authorize
  the new evidence revision. **Status:** implemented. Closed-domain
  `bfcl-question-answers-v1` artifacts bind the parent evidence and exact question-set
  digest. Applying a complete answer set creates a new `bfcl-source-evidence-v2` bundle
  containing semantic-only answers plus an immutable parent/question/answer revision
  link, and emits a digest-bound revision record. Existing evidence is unchanged and
  content-addressed revision writes refuse replacement. Stale, missing, extra,
  duplicate, wrong-type, out-of-range, and approval-reuse cases are covered by
  `test_bfcl_authoring_questions.py`.
- **UA-905 (S) Tests.** Undeclared held-out status blocks before the first model call;
  an answered question set resumes only after digest verification. **Acceptance:**
  `test_bfcl_authoring_questions.py` covers complete, incomplete, stale, tampered, and
  replayed answer sets; `test_bfcl_authoring_held_out.py` proves call count remains zero
  on pre-model failure. **Status:** implemented. The drafting pre-model gate now
  deterministically replays parent evidence + open questions + answers and requires the
  resulting digest to equal the supplied revision before loading its new approval.
  Missing/partial resume artifacts, parent/question/answer swaps, tampering, stale sets,
  and non-revision replay attempts fail closed. Migrated evidence sets its revision root
  to the legacy migration source digest and separately verifies the normalized origin.
  Integration coverage additionally proves
  malformed held-out state yields zero model calls and a complete verified revision
  resumes with reviewed semantic answers in the prompt.
- **UA-906 (M) Pre-model authorization contract.** Keep evidence-exposure authorization
  distinct from final semantic approval. Permit either named human authorization or a
  preconfigured organizational policy digest; never infer consent from final approval.
  **Acceptance:** drafting cannot start without one exact authorization, and changing
  evidence, domain brief, redaction report, or policy invalidates it. **Status:**
  implemented. `bfcl-model-exposure-authorization-v1` binds the exact evidence,
  domain-brief content/source/redaction report, held-out decision/policy, and held-out
  redaction proof. Drafting requires either a named-human authorization or an exact
  configured organizational-policy digest before the first model call. Final semantic
  approval uses a different schema and cannot substitute. The authorization is recorded
  separately in draft provenance and carried into review. Direct v1 model exposure is
  denied by default; the library-only compatibility opt-in is explicit and is not
  exposed by the production drafting CLI. Covered by
  `test_bfcl_authoring_authorization.py` and pre-model drafting regressions.

## Epic 10 — Transactional revisions, resume, workspace safety

Closes: "content-addressed immutable revisions"; "fail-closed resume"; "single-writer
transactional workspaces"; "bounded model cost".

- **UA-1001 (L) Revision store.** `revisions/<content-address>/` with a manifest and
  atomic directory rename on explicitly supported local filesystems. Writes never
  overwrite an earlier artifact. **Acceptance:** fsync ordering and crash-injection
  tests prove a revision is either absent or complete; unsupported filesystems fail
  closed unless an equivalent transaction backend is configured.
  **Status:** implemented. `runtime/authoring_workflow/revision_store.py` provides a
  manifest-last, content-addressed store with durable file and directory fsync ordering,
  atomic directory rename, closed-world digest verification, stable refusal codes, and
  an explicit allowlist of local filesystems. Evidence-answer revisions retain their
  existing API through a compatibility wrapper in `pack_authoring/questions.py`.
  Crash injection covers every payload/manifest write and fsync boundary, both sides of
  rename, immutable duplicate writes, unsupported filesystems, and tampered, missing,
  or extra artifacts in `test_bfcl_authoring_revisions.py`.
- **UA-1002 (M) Workspace locking.** Single-writer lock, stale-lock detection, and
  per-run/per-tenant namespacing. Lock metadata includes run ID, tenant ID, host, PID,
  creation time, and renewable lease. **Acceptance:** a lock is never stolen from a
  live owner; stale recovery is explicit and audited.
  **Status:** implemented. `runtime/authoring_workflow/workspace_lock.py` combines an
  OS-held nonblocking advisory lock with digest-bound owner metadata and a renewable
  lease. Tenant/run identifiers resolve to separate lock namespaces; symlink and
  non-regular lock artifacts fail closed. Expired orphan metadata cannot be reused
  implicitly: recovery requires an actor and reason and durably appends a digest-bound
  audit record before publishing the replacement lease. A real spawned-process test
  proves that even an attempted recovery cannot steal from a live owner. Covered by
  `test_bfcl_authoring_workspace_lock.py`.
- **UA-1003 (M) Fail-closed resume.** Verify source, evidence, resolved config, and
  revision digests before resuming. Refuse partial writes and stale approvals.
  **Acceptance:** a resumability matrix names each state and permitted next command;
  corruption and concurrent ownership never fall back to restart-in-place.
  **Status:** implemented. `runtime/authoring_workflow/resume.py` stores each
  digest-bound session state as an immutable RevisionStore entry and requires an active
  matching WorkspaceLease for every new state. Resume reacquires and retains exclusive
  ownership while re-verifying source bytes, internal evidence and source-identity
  digests, canonical resolved configuration, content-addressed evidence revisions, and
  evidence approval binding. The closed resumability matrix exposes exactly one bounded
  set of next commands per phase. Staging remnants, draft files without committed
  provenance, corrupted session/revision artifacts, stale approvals, and concurrent
  ownership return stable errors with recovery instructions; none restart in place.
  Covered by `test_bfcl_authoring_revisions.py`.
- **UA-1004 (M) Refusal records.** Persist the Stage F classification (deterministic
  materialization, model-owned proposal, user-owned source contract, oracle-owned
  behaviour, operational infrastructure) plus the operator authorization required to
  create the next revision. **Acceptance:** refusal records contain sanitized findings
  only and cannot carry target-model output or scores.
  **Status:** implemented. `runtime/authoring_workflow/refusal.py` publishes the five
  Stage F ownership classes, their permitted remediation actions, digest-bound refusal
  records, and separate operator authorizations tied to the exact refusal and parent
  session. Findings use only bounded codes, artifact roles, classifications, and
  evidence digests; the strict schema has no prose, model-output, response, metric, or
  score field and rejects those concepts in codes. Both artifacts are independently
  immutable in RevisionStore and require the active tenant/run WorkspaceLease.
  `AuthoringResumeGate.open_authorized_revision()` keeps the refused phase terminal
  unless a persisted matching authorization is verified. Covered by
  `test_bfcl_authoring_refusals.py`.
- **UA-1005 (S) Run quotas.** Bound calls, tokens, batch size, and wall-clock time per
  run; replay from the immutable cache before incurring new spend. **Acceptance:** quota
  accounting is deterministic and cache hits consume no provider-call quota.
  **Status:** implemented. `runtime/authoring_workflow/quota.py` defines strict,
  digest-bound call, conservative token-unit, batch, and wall-clock limits with stable
  refusal codes. `call_structured()` performs immutable-cache lookup first; a hit records
  only a cache hit, while a miss atomically reserves provider-call and deterministic
  input-plus-output token units before invocation. Failed provider calls retain their
  reservation, preventing retries from bypassing policy. Shared drafting uses a bounded
  four-call default policy and persists the deterministic quota snapshot in draft
  provenance. Covered by `test_bfcl_authoring_quotas.py` and pack-drafting regressions.
- **UA-1006 (M) Tests.** Interrupted run, concurrent run, tampered revision, exhausted
  quota. **Acceptance:** crash injection is exercised at every revision-store write
  boundary, concurrency uses two real processes, and each rejected state has a stable
  reason code and recovery instruction.
  **Status:** implemented. The Epic 10 suite injects crashes before and after every
  payload, manifest, directory-fsync, exclusive-rename, and parent-fsync boundary and
  proves the visible revision is absent or complete. Workspace contention uses a real
  spawned process and verifies a live owner cannot be stolen. The cross-component
  failure matrix covers interrupted staging, concurrent ownership, tampered revisions,
  missing revision authorization, and exhausted quota. Revision, lock, resume, refusal,
  and quota exceptions now expose both a stable `code` and non-empty `recovery`.
  Covered by `test_bfcl_authoring_workflow_failures.py` and the focused Epic 10 suites.

## Epic 11 — Guided CLI and shared release envelope

Closes: "two required user inputs"; "the guided command is a thin dispatcher, not a
bypass"; "one release envelope for every adapter".

- **UA-1101 (L) `scripts/bfcl_author.py`.** `author --source <path|uri> --brief <path>`
  plus `resume`, `answer`, `review`, `approve`, `freeze`, and `publish`. Each subcommand
  delegates to existing runtime operations. UX explicitly presents pre-model
  authorization and final approval as two boundaries while keeping normal input to
  source plus brief. **Acceptance:** CLI help and state-specific errors tell the user
  the next safe command; no interactive prompt is required in CI mode.
  **Status: completed.** The guided command now detects local Python, HTTP-package, and
  MCP sources (including canonical file URIs), delegates intake and drafting to the
  existing runtime paths, applies reviewed answers through immutable revisions, and
  exposes model authorization separately from evidence and final release approval.
  `resume` validates the requested transition through `AuthoringResumeGate`; failures
  return stable recovery guidance. Release commands deliberately reject non-MCP adapters
  until UA-1103/UA-1104 provide the shared release kernel. The conventional intake
  wrapper is `scripts/build_source_intake.py`. CLI, CI-mode, boundary, dispatch, and
  adapter-scope behavior are covered by `test_bfcl_authoring_cli.py`.
- **UA-1102 (M) Resolved configuration.** Emit and hash `resolved_authoring_config.json`
  containing derived defaults; prompt only for values policy cannot derive safely.
  Pack ID candidates are deterministic slugs but require confirmation; pack version is
  policy-provided or explicitly confirmed, never inferred from server prose.
  **Acceptance:** the resolved document identifies the origin of every value
  (`user`, `policy`, `adapter`, `derived`) and is part of every downstream digest.
  **Status: completed.** `authoring_workflow/resolved_config.py` defines the strict,
  immutable `bfcl-resolved-authoring-config-v1` contract, deterministic pack-ID
  candidates, reviewed policy loading, explicit/CI-safe confirmation, origin metadata,
  canonical hashing, and tamper detection. Guided authoring writes and binds the config
  before intake, commits it into the resumable session, and passes its digest through
  conventional/MCP intake provenance and model-exposure subjects. Guided drafting adds
  the verified digest to draft provenance. MCP intake cross-checks its reviewed pack
  declaration against the resolved identity, so server prose cannot select a version.
  Example policy and focused config/CLI/downstream-binding tests are included.
- **UA-1103 (L) Adapter-neutral release kernel.** Extract generic review/freeze/handoff
  code into `runtime/authoring_release/`; retain `runtime/mcp/release/` compatibility
  wrappers and public imports. Adapter-specific records are supplied through typed
  hooks. Version the v2 release format and preserve verification of v1 MCP releases;
  do not promise v1/v2 byte identity. **Acceptance:** API compatibility tests, v1
  verification fixtures, deterministic v2 fixtures, and semantic-equivalence checks.
  **Status: completed.** `runtime/authoring_release/` now owns typed review,
  approval, atomic freeze, full pack-file sealing, version-dispatched loading, and fresh
  publication handoff. `ReleaseAdapter` and `PublicationAdapter` isolate pack semantics,
  review facts, reviewed sidecars, Gold interpretation, and origin-manifest checks.
  New records use `bfcl-authoring-*-v2`; freeze requires A2 and an exact blocker/risk
  approval. MCP v1 modules and signatures remain available, the shared loaders continue
  to verify v1 releases, and `McpReleaseAdapter` supplies v2 hooks. Determinism,
  v1 compatibility, semantic equivalence, tamper refusal, and hook boundaries are
  covered by `test_bfcl_authoring_release.py`.
- **UA-1104 (M) Generalized review.** Extend the review packet and approval checklist to
  any adapter, including independently verified certification, pre-model authorization,
  and answered questions. **Acceptance:** final approval binds the complete candidate
  pack and cannot substitute for pre-model authorization.
  **Status: completed.** `authoring_release/assembly.py` verifies the complete shared
  trust record for local Python, HTTP-package, and MCP sources: signed certification,
  intake/evidence/config/draft bindings, exact model-exposure authorization, answered
  revision replay, candidate-pack identity/fingerprint, and validation. The v2 checklist
  now names independent certification, pre-model authorization, and answered questions;
  missing or stale records block final approval. Neutral review, approval, and freeze
  scripts replace MCP-only guided dispatch, while publication remains MCP-scoped pending
  end-to-end origin work. Parameterized three-adapter and boundary tests cover the
  generalized path.
- **UA-1105 (S) Per-adapter rollout policy.** Add a generic per-kind policy while
  retaining `BFCL_ENABLE_EXPERIMENTAL_MCP` as a documented compatibility alias for one
  deprecation window. **Acceptance:** omitted, false, malformed, unknown-kind, and
  conflicting legacy/new settings fail closed.
  **Status: completed.** `authoring_workflow/rollout.py` defines strict, fail-closed
  decisions for local Python, HTTP-package, and MCP Mode A. Environment controls override
  reviewed per-adapter policy, and the effective decision is digest-bound in
  `bfcl-resolved-authoring-config-v2`; v1 configs remain verifiable but cannot authorize
  new live intake. The legacy MCP flag delegates to the generic policy for one documented
  window, with disagreement against `BFCL_ENABLE_MCP_MODE_A` rejected. Guided and direct
  intake enforce the resolved decision, while offline review/freeze remain ungated.
  Omitted, false, malformed, unknown-kind, conflict, precedence, compatibility, and CLI
  boundary behavior are covered by rollout and existing MCP tests.
- **UA-1106 (L) End-to-end tests.** Local source and MCP source through the guided CLI →
  pre-model authorization → review → freeze → fresh Gold → benchmark and run manifest.
  Include an HTTP source through freeze. **Acceptance:** all three adapters produce the
  same release-envelope schema; local and MCP complete `stage=all`; a below-A2 source,
  stale approval, and changed domain brief are rejected before publication.
  **Status: completed.** `test_bfcl_authoring_e2e.py` verifies one shared v2 envelope
  across local Python, HTTP-package, and MCP Mode A releases; local and MCP guided
  publication both require a fresh Gold report before invoking `stage=all` and emitting
  benchmark, raw benchmark, and run manifest artifacts. It also verifies the A2 freeze
  boundary, stale final-approval binding, and domain-brief evidence invalidation.
  `publish_authoring_release.py` now dispatches v2 local/MCP handoff through typed
  publication adapters. HTTP completes review/freeze but still refuses publication
  until transport-specific origin verification is available.
  **Post-review hardening:** guided commands now advance a fail-closed v2 session and
  bind authorization, review, approval, freeze, and publication artifacts while v1
  sessions remain verifiable. Freeze installs generic authoring origin lineage for every
  adapter and requires the exact reviewed sidecar set. Review reruns BFCL prepare from a
  bound validation config; publication requires a Gold-eligible manifest with no
  ineligibility reasons. Resolved source/brief inputs and current rollout revocations are
  rechecked immediately before live intake. A real local `stage=all` test complements the
  hermetic MCP transport boundary.

## Epic 12 — Operations, observability, rollout evidence

Closes: the production robustness invariants and the platform-operations risk controls.

- **UA-1201 (M) Structured events.** Emit adapter identity, certification tier, refusal
  reason, revision authorization, validation verdict, and freeze digests, with payloads
  selected from an allowlisted schema rather than redacted after arbitrary serialization.
  **Acceptance:** event payload tests prove source prose, fixture values, credentials,
  and model responses are absent.
  **Status (2026-08-29): completed.** `runtime/authoring_workflow/events.py` defines
  strict per-event payloads, digest-bound envelopes, and a locked/fsynced JSONL sink.
  Guided intake, review, freeze, publication validation, CLI refusals, persisted Stage F
  refusals, and revision authorizations emit only selected digests, enums, booleans, and
  stable codes. `test_bfcl_authoring_events.py` owns schema, tamper, persistence, and
  sensitive-corpus absence coverage; `bfcl-authoring-events.md` documents operation and
  failure semantics.
- **UA-1202 (M) Credential lifecycle.** Reference credentials by name only, resolve from
  environment or a secret manager, and support rotation without recording the secret.
  Record a non-secret authorization-context/principal digest; rotation or permission
  changes invalidate observations and require revalidation. **Acceptance:** secrets do
  not enter config, cache, events, provenance, exceptions, or release bytes.
  **Status (2026-08-29): completed.** `runtime/authoring_workflow/credentials.py`
  provides strict environment/secret-manager references, ephemeral redacted values,
  injectable providers, and digest-bound principal/permission authorization contexts.
  HTTP and MCP transports share resolution, authenticated authoring binds the context
  into live identity, evidence, release configuration, and allowlisted events, and
  reflected HTTP/provider failures cannot retain secret text. Token value rotation keeps
  the context stable; reference, principal, or permission drift requires fresh intake.
  `test_bfcl_authoring_credentials.py` owns lifecycle, rotation, secret-manager,
  exception, endpoint, MCP, and event absence coverage.
- **UA-1203 (S) Cache retention.** Enforce retention and access policy for the model I/O
  cache and provide a purge tool. **Acceptance:** dry-run and execute modes enumerate
  the same eligible records; active/referenced revisions are retained; purge produces
  a digest-bound audit record without response content.
  **Status (2026-08-29): completed.**
  `runtime/authoring_workflow/cache_retention.py` verifies immutable sessions and bound
  draft provenance, protects active/uncommitted heads, and builds content-bound purge
  plans for `authoring_io_cache.jsonl` only. Guided `purge-cache` defaults to dry-run;
  execute requires the reviewed plan digest, rewrites under workspace/cache locks with
  fsync and atomic replacement, and appends a private digest-bound audit containing no
  model response. `test_bfcl_authoring_retention.py` owns dry-run/execute parity,
  references, active sessions, stale plans, lock/access/path failures, audit privacy,
  and CLI coverage.
- **UA-1204 (M) Revoke and supersede.** Define a revocation record for a published
  release, signed or stored in an authenticated registry. Publication prevents new use
  of revoked identities; consumer verification rejects or warns according to policy.
  Document that downstream copies cannot be physically recalled. **Acceptance:** stale,
  unsigned, wrong-issuer, and conflicting revocation records fail closed.
  **Status (2026-08-29): completed.** `runtime/authoring_release/revocation.py`
  defines signed records, signed freshness-bounded registry snapshots, monotonic
  generations, strict supersession chains, and reject/warn consumer verdicts.
  Publication handoff rechecks a supplied registry verifier at each commit boundary,
  eval source verification accepts the same policy callback, and
  `revoke_authoring_release.py` issues locked atomic registry updates. Stale,
  unsigned, wrong-issuer/key, rollback, and conflicting-chain cases are owned by
  `test_bfcl_release_revocation.py`; `bfcl-authoring-revocation.md` documents that
  downstream copies cannot be physically recalled.
- **UA-1205 (M) Documentation.** Update the user guide, support matrix, and contract
  references, and add a guide for authoring and certifying a new adapter. **Acceptance:**
  every supported/experimental/target statement links to a named test or is explicitly
  labeled unimplemented; all CLI examples run in documentation smoke tests.
  **Completed 2026-08-29:** `bfcl-authoring-user-guide.md` is the operator and contract
  index; `bfcl-authoring-support-matrix.md` and the MCP matrix bind support claims to
  named tests or `unimplemented`; `bfcl-adapter-authoring-certification-guide.md`
  defines the extension checklist; `test_bfcl_authoring_documentation.py` executes every
  registered authoring shell example and rejects unsupported matrix claims or broken
  test/contract links.
- **UA-1206 (L) Broader evaluation.** Repeat the ablation on two additional domains with
  an independent reviewer, and run pinned target-model evaluation before any causal or
  default-UX claim. **Acceptance:** protocol, exclusions, raw observations, artifact
  digests, reviewer identity, and missing runs are published; no synthetic observations
  are substituted; the rollout decision states whether the evidence is descriptive or
  causal.
  **In progress 2026-08-30:** `bfcl-onboarding-ablation-rollout-v2` and
  `assemble_bfcl_ablation_rollout.py` bind three domain slots, raw
  observation/state/run digests, explicit exclusions, independent identities, evaluator
  pins, missing runs, and a digest-bound descriptive/causal decision.
  Reviewer identity and the model pin are now verifiable rather than declared.
  `bfcl-onboarding-domain-review-v1` (`runtime/mcp/ablation_review.py`) makes an
  independent review an Ed25519 attestation over the exact reviewed bundle —
  protocol, input, report, evaluator pin, exclusions, and all nine observation and
  run-tree digests — which publication recomputes from the raw files; the trusted
  reviewer key is supplied by the publishing authority, never by the operator's
  bundle manifest. `bfcl-onboarding-evaluator-pin-v1`
  (`runtime/mcp/ablation_evaluator_pin.py`) validates a pin through the existing
  `CandidateModelIdentity` contract instead of a local regex, so moving pointers
  are refused, credential-shaped values cannot enter evidence, and an absent pin is
  recorded as `target_evaluation_not_run` or `immutable_pin_unavailable`.
  `bfcl-onboarding-domain-bundle-v1` declares the raw files once so the reviewer and
  the publisher verify the same tree, and `review_bfcl_domain_evidence.py` plus
  `record_bfcl_evaluator_pin.py` are the reviewer and operator entry points.
  The published decision is `mcp610-rollout/rollout.json`: the tiny_library pilot
  bundle re-verifies from `mcp610-tiny-library.bundle.json` but has no signed
  reviewer attestation, inventory and banking_vn have no live runs, and no immutable
  `azure/openai/gpt-5.6-sol` pin exists. The implementation therefore refuses causal
  status and UA-1206 remains incomplete pending independent review of the pilot,
  live runs for the two additional domains, and a real target-model pin.

## Definition of done for the whole plan

1. A user creates a Gold-eligible pack from one source declaration and one domain brief,
   answering only unresolved semantic or safety questions and completing the two
   explicit authorization boundaries.
2. Local Python, BFCL Oracle HTTP v1, and MCP Mode A all satisfy the same adapter
   contract, emit the same evidence schema, and are independently certified on the same
   ladder for Flow 2. Manual packs remain unchanged.
3. Held-out status is explicit before the first model call, with proven redaction.
4. Every correction is a visible, authorized, content-addressed revision, and resume is
   fail-closed.
5. Release freeze, fresh Gold validation, and publication share one envelope across
   adapters, with v1 compatibility and a documented revocation path.
6. The workflow document's acceptance criteria are each backed by a named test.
7. The HTTP adapter requires reviewed schemas and never invents them from tool names.
8. Adapter declarations cannot self-certify, and Flow 2 certification cannot change the
   manual Gold contract.

### Audit status

Conditions 3, 4, 5, 6, 7, and 8 are met. Conditions 1 and 2 are partially met. Nothing
in this section may be marked met on the strength of prose; each line names the owning
test.

- Condition 1. Partially met. No test starts from only a source declaration plus a
  domain brief and reaches a Gold-eligible pack with unmocked validation. The closest
  cases either stop at A0 intake or begin from a pre-built frozen session. Held-out
  selection is also not a first-class argument of `bfcl_author author`, so the minimal
  two-input claim does not hold at the CLI.
- Condition 2. Partially met. All three adapters share the contract and the v2 evidence
  schema at A0, but only `local_python` runs A1 and A2 through intake. HTTP and MCP
  production intake certify A0 only, so "independently certified on the same ladder" is
  unproven for two of the three adapters.
- Condition 4. Met. `test_bfcl_authoring_revisions.py` owns the resume refusals,
  including `artifact_missing` and `session_invalid`, and
  `test_bfcl_authoring_cli.py::test_answer_commits_a_revision_that_must_be_reauthorized_and_reapproved`
  carries one correction through `answer`, the withdrawn right to draft, re-authorization,
  and re-approval.
- Condition 5. Met, with one declared boundary.
  `test_bfcl_authoring_release.py` owns the shared envelope, v1 compatibility, and
  `fresh_validation_stale`;
  `test_bfcl_authoring_generalized_review.py::test_review_refuses_a_validation_report_the_bound_config_did_not_write`
  owns `validation_report_path_mismatch`; and `test_bfcl_release_revocation.py` drives
  `publish_authoring_release` with revocation-registry flags. HTTP publication remains
  refused by design and is labelled unimplemented in the support matrix.
- Condition 6. Met. The workflow document's section 13 acceptance criteria are
  transcribed into [bfcl-workflow-acceptance-matrix.md](bfcl-workflow-acceptance-matrix.md)
  with the source digest recorded, and
  `test_bfcl_authoring_documentation.py::test_workflow_acceptance_criteria_are_backed_by_named_tests`
  fails when a criterion names no test or names a test function that does not exist.
  Task-to-test ownership for this plan is machine-checked separately by
  `test_bfcl_authoring_documentation.py::test_every_plan_task_is_owned_by_an_existing_named_test`.
  The document itself lives outside this repository, so no test can detect that it
  changed; re-transcribe the matrix when its digest moves.
- Condition 8. Met. Self-certification is closed by signature and issuer checks in
  `test_bfcl_source_adapter_certification.py`, and
  `test_bfcl_stages.py::test_manual_oracle_packs_require_no_flow_two_adapter_metadata`
  plus `test_bfcl_stages.py::test_manual_gold_tier_ignores_adapter_and_certification_fields`
  assert that manual packs need no adapter metadata and that Flow 2 records cannot move
  the manual Gold verdict.
