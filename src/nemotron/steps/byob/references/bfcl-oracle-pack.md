# BFCL Oracle Pack Contract

The `byob/bfcl` family generates function-calling benchmarks from an executable
oracle pack. All domain content lives in the pack; `runtime/benchmark_families/bfcl`
stays domain-agnostic.

## Pack Layout

A pack is a directory under an `oracle_runtime.allowed_roots` entry (default
`byob/data`). Paths come from `manifest.yaml.paths`, from `oracle_pack.*` config
overrides, or from these defaults:

Relative paths in the BFCL config resolve from the checked-in `steps/byob` root,
not from the YAML file's directory or the shell working directory. External packs
should use absolute `manifest_path` / `allowed_roots` values. Paths declared inside
`manifest.yaml.paths` remain relative to that pack's own directory.

| File | Required | Purpose |
| --- | --- | --- |
| `manifest.yaml` | yes | `pack_id`, `version`, optional `paths` overrides. |
| `tools.json` | yes | OpenAI-style tool array plus pack-local `x-*` keys. |
| `backend.py` | one oracle required | Local executable oracle. Mutually exclusive with `endpoint_config.yaml`. |
| `endpoint_config.yaml` | one oracle required | BFCL Oracle HTTP v1 endpoint, expected identity, TLS, and secret environment references. Mutually exclusive with `backend.py`. |
| `held_out.yaml` | optional | Versioned fixture primary IDs and template IDs excluded from normal generation; referenced by `manifest.yaml` as `held_out: held_out.yaml`. |
| `fixtures.json` | no | Seed state passed to `reset`. |
| `task_templates.yaml` | yes | Task templates with `slots` and `paraphrase`. |
| `validation_cases.yaml` | yes | Declared probes proving tool behavior. |
| `assertions.py` | yes | Success assertions referenced by templates. |

`backend.py` and `assertions.py` may import helper modules from the pack: the worker
puts the module's own directory and the pack root on `sys.path` before importing, and
the fingerprint already covers every file in the pack tree. Those two directories take
precedence during pack execution, so a helper must not be named after a module the pack
itself imports (`yaml.py`, `json.py`). Anything a pack imports from outside its own tree
is invisible to the fingerprint, so a pack that depends on such a module can be replayed
against a different version of it without the run noticing.

Every model-facing entry is validated as an OpenAI function tool: `type` is
`function`, `function` is a mapping, `description` is text when present, and
`strict` is boolean when present. `tools.json` keys read only by the pack pipeline
are `x-mutates` and `x-requires-confirmation`.
A tool marked `x-requires-confirmation` must expose the confirmation parameter and
must not mutate state when that parameter is `false`. `x-mutates` declares that a
tool changes state; validation compares the claim against what the probes actually
did and rejects a tool that changed state without declaring it.
### Manifest Keys

| Key | Required | Purpose |
| --- | --- | --- |
| `pack_id`, `version` | yes | Row provenance and cache identity. |
| `paths` | no | Override any default file location. |
| `primary_keys.<collection>` | when ambiguous | Field that identifies a fixture row. Convention covers `<singular>_id` and `id`; a collection carrying its own id plus a foreign key must declare which is which. |
| `absent_ids.<collection>` | when used | Ids that deliberately do not exist, for `absent:` slots. |
| `languages`, `default_language` | no | Surface languages the pack offers and the one to use when config names none. |
| `system_prompt` / `system_prompt_path` | no | Inline or file system prompt; the frozen default is used otherwise. |
| `assistant_turn_templates.<type>.<lang>` | per milestone type used | Pack-wide assistant text, overridable per template. |
| `confirmation.parameter` / `.status_field` / `.pending_status` | no | The pack's own confirmation vocabulary; defaults to `confirm`, `status`, `awaiting_confirmation`. |
| `surface_guards.tool_names_exempt` | no | Tool names that are ordinary words of the domain language and must not count as tool-name leakage. |

## Backend Contract

`backend.py` exposes four callables; the pipeline owns reset, clock, seed, and
timeouts, so the backend must not read wall-clock time or the environment:

```python
def list_tools() -> list[str]: ...
def reset(*, ctx, fixtures=None) -> None: ...
def call_tool(name: str, arguments: dict, *, ctx) -> dict: ...
def get_state() -> dict: ...
```

`ctx` is a `RunContext` carrying `clock`, `seed`, `timeout_s`, `task_id`, and
`turn_index`. Every `call_tool` return value must be JSON-serializable. Errors are
returned as data, never raised:

```python
{"error": {"code": "not_found", "entity": "books", "id": "BK-9", "field": "book_id", "message": "..."}}
```

Only `code` is required — it is what scoring compares. `entity`, `id`, `field`, and
`message` are optional detail for a reviewer, so a domain whose failures are not
about one entity is not forced to pad the envelope with nulls.

### HTTPS Endpoint Contract

An endpoint-backed pack exposes the same oracle behavior through BFCL Oracle
HTTP v1. `endpoint_config.yaml` declares:

- `protocol_version: bfcl-oracle-http-v1`
- an HTTPS `base_url`
- expected `oracle_id`, `oracle_version`, and `sha256:` `content_digest`
- optional bearer-token and custom-header environment-variable names
- an optional allowlisted CA bundle plus request- and response-size limits

The fixed routes are `GET /v1/metadata`, `GET /v1/tools`,
`POST /v1/sessions`, `POST /v1/sessions/{id}/calls`,
`GET /v1/sessions/{id}/state`, and `DELETE /v1/sessions/{id}`.
Creating a session is the reset operation: the request carries the frozen
`RunContext` fields and fixtures, and the response returns a unique session id
plus the endpoint identity. Every replay gets a new session.

`GET /v1/metadata` returns the identity object, and `GET /v1/tools` returns
`{"tools": ["tool_name", ...]}`. Session creation uses:

```json
{
  "context": {
    "clock": "2026-03-02T09:00:00+07:00",
    "seed": 7,
    "timeout_s": 5.0,
    "task_id": "pack__template__0001"
  },
  "fixtures": {}
}
```

and returns:

```json
{
  "session_id": "opaque-session-id",
  "oracle": {
    "protocol_version": "bfcl-oracle-http-v1",
    "oracle_id": "example",
    "oracle_version": "1.0.0",
    "content_digest": "sha256:..."
  }
}
```

A call request is
`{"name": "tool_name", "arguments": {}, "turn_index": 0}` and its response is
the tool result itself. The state route returns a JSON object. Session ids are
opaque and URL-escaped by the client. A successful session DELETE may return
either an empty `204 No Content` response or a JSON object.

Remote identity must match the config during prepare, session creation,
generation, and publication. A changed version or digest aborts the run.
Requests are not redirected and mutating requests are never retried. Secret
values are resolved from the environment at runtime and are not included in
fingerprints, reports, logs, or manifests.

`assertions.py` functions must accept exactly the keyword arguments
`(*, state, trace, task, ctx)`. Return `None` on success, raise `AssertionError`
on failure, or return `{"status": "not_applicable", "detail": "..."}` when the
declared predicate does not apply. Any other return is an infrastructure error.
Export assertions through an `ASSERTIONS` dict or name them `assert_*`.
Optional literal `ASSERTION_CAPABILITIES` entries declare boolean `trace` and
`executable` support plus category `state`, `path`, `result`, `final_answer`, or
`unclassified`; omitted entries default to executable-only and unclassified.
Validation and executable projection both read the mapping as a literal
assignment rather than importing it, so a computed mapping, an unknown assertion
name, or a malformed entry fails validation with
`invalid_assertion_capability`. That reason is separate from `invalid_signature`,
which is only about the assertion's keyword arguments.
Assertions run inside the
same process worker as the local backend or endpoint client. For endpoints,
`state` is fetched from the active remote session; `trace` holds the calls the
episode just made: `[{"tool": ..., "arguments": {...}, "result": {...}}]`.

## Template Surface Requirements

Generation renders every assistant and user turn from the pack, never from a model:

- `user_turn_templates.<lang>` supplies the first user turn; `{slot}` placeholders
  are substituted with bound values.
- Placeholders follow one rule everywhere, surface text and call arguments alike:
  `{name}` is replaced in a single pass, so a bound value that itself contains braces
  is inserted as data rather than rescanned, and a name nothing bound fails the run.
  A milestone argument that is exactly `{name}` keeps the slot's own type; a
  placeholder inside a longer argument string is substituted in place, so a call and
  the turn describing it always show the same value. Required tool parameters may be
  filled from same-named slots as a shorthand. Optional parameters are never injected:
  declare them under milestone `args` when the gold call should send them.
- `user_simulator_turns[].content_template.<lang>` supplies later user turns. Each
  entry's `after` must resolve to exactly one milestone — use a milestone `id`
  whenever the same milestone type repeats. An `ask_confirm` needs such an entry;
  a closing `ask_for_slot` does not, because the conversation ends there.
- Every template declares one supported `turn_policy`, and every declared milestone
  `id` is a unique non-empty string. Missing policies and duplicate ids are rejected
  before references can silently bind to the wrong conversation edge.
- A template may declare `edge_signatures` as a unique list of non-empty,
  pack-defined labels for rare behavior not already distinguished by
  `turn_policy`. Expansion carries these labels into Stage 11, where
  deduplication and balancing preserve at least one selected row for every
  complete language/policy/edge bucket.
- A policy is a claim about the conversation's shape, because it is what consumers
  slice the published rows by, so each one is held to that shape:

| `turn_policy` | the shape it must plan |
| --- | --- |
| `single_turn` | exactly one user turn, and at least one tool call |
| `multi_tool` | at least two tool calls |
| `missing_slot` | every slot marked `visible_in_first_turn: false` is named by an `ask_for_slot` before the first call and receives a simulator reply |
| `confirmation` | an `ask_confirm` answered by a `user_simulator_turns` entry before the call batch it authorizes |
| `correction` | a user turn carrying `slot_updates` |
| `dependent_call` | `call_order: strict`, and a call that reads an earlier call's result |
| `negative_path` | at least one `success_assertions` entry, which is what states the refusal the call must produce |
| `clarify_only` | no tool call, ending in `ask_for_slot` |
| `irrelevant` | no tool call, ending in `decline` |
- `assistant_turn_templates.<milestone_type>.<lang>` supplies text for
  `ask_for_slot`, `ask_confirm`, `decline`, and `final_answer`. Declare it on the
  manifest (pack-wide) or override it per template. A milestone with no template
  fails the run rather than inventing text.
- The render language is chosen from the languages `user_turn_templates` offers —
  that is the one block every task renders — narrowed by config `language` or
  `manifest.default_language` / `languages` when the pack offers several. Every
  other block the plan renders must state that same language: `render` lists all
  the gaps at once before any task is rendered, so a pack that declares assistant
  text in one language and simulator turns in another hears about both.
- `{slot_name}` in assistant text resolves to the slot that milestone asks about:
  the milestone's own `slot:` when declared, otherwise the template's single withheld
  slot. A template that withholds two slots and asks about one must name it, because
  a question that lists both is not the question the conversation needs. The slot may
  carry `label.<lang>` so the question reads as prose rather than quoting the slot key.
- A visible slot must appear verbatim in the rendered user turn. `must_preserve`,
  `must_omit`, and `must_not_mention` are re-checked in Python after rendering and
  a violating row is dropped, so a template whose prose paraphrases a slot value
  (`rail: napas` rendered as "qua Napas") is rejected — spell the value out or
  make the slot hidden.
- `must_not_mention` entries are forbidden phrases, matched case-insensitively over
  every user turn. The reserved entry `tool_names` means "no tool name may appear";
  it is also checked whenever `surface_generation.prevent_tool_name_leakage` is on,
  so declaring a phrase adds a rule instead of replacing the leakage guard. A tool
  name is matched as a whole word, and a name that is an ordinary word of the domain
  language can be exempted through `manifest.surface_guards.tool_names_exempt`.
- Guard matching reads a value as a whole token, so a value buried in a longer one was
  never stated: an amount of `4` is not stated by `400000`, and a rail of `us` is not
  stated by `status`. The class excluded follows the value's own edges, so a number may
  carry a unit written against it (`200000` is stated by `200000đ`) while a word must
  not sit inside a longer word.
- Every slot must declare `visible_in_first_turn`. The flag is what puts the slot in
  the preserve set or the omit set, so omitting it would leave it in neither. A
  visible slot's initial value must occur in the **first** user turn; values introduced
  by corrections are preserved across the full conversation.
- `must_omit` covers the **first** turn only. That is what makes a withheld slot
  workable: the value stays out of the opening request and arrives in the
  `user_simulator_turns` reply that answers the assistant's `ask_for_slot`.
- A call may carry `confirm: true` only when its own assistant turn is covered by a
  confirmation: the user answered an earlier `ask_confirm`, no later turn corrected
  a value, and no earlier call batch already consumed that confirmation. One reply
  authorizes the next call batch only; a separate later action needs another
  `ask_confirm`. The check reads positions in the plan, not the `turn_policy` label.
- The system prompt comes from `manifest.system_prompt`, `manifest.system_prompt_path`,
  or the frozen `bfcl/prompts/default_system.txt`. The row stores its
  `system_prompt_id` and `messages[0]` is the resolved text. A declared prompt
  path must remain under `oracle_runtime.allowed_roots` and participates in the
  pack fingerprint. The frozen prompt is English, so a non-English render must declare
  its own prompt rather than publish a mixed-language gold row; under
  `lineage.policy: smoke_no_publication` the mismatch is only a warning, because that
  run publishes nothing.

## Validation Cases

Each entry probes one tool call. Keys:

- `id`: unique probe id.
- `tool`, `arguments`: the call to run.
- `reset_before`: `true` starts a fresh episode; `false` chains onto the previous
  case inside the same worker process, which lets a probe observe state produced
  by the case before it. The first case in the file cannot use `false`.
- `expect.result_class`: `success`, `structured_error`, or `awaiting_confirmation`.
- `expect.error_code`: expected `error.code`, or `null` for success.
- `expect.state_unchanged`: assert `get_state()` is identical before and after.
- Any other `expect.<field>`: compared against `result[<field>]`, which is how
  business outcomes such as `status: rejected_insufficient_funds` are pinned.
- `coverage: negative`: mark a probe as the negative case when the tool signals a
  miss through a normal result instead of `error` or `awaiting_confirmation`
  (for example a collection query returning an empty list).
- `notes`: free text for reviewers.

Every tool in `tools.json` needs at least one success probe and one negative
probe. A negative probe is any case whose result is a `structured_error` or
`awaiting_confirmation`, or one explicitly marked `coverage: negative`. A negative
probe may deliberately violate the schema to test backend error handling. A success
probe must satisfy the model-facing parameter schema; backend-only arguments cannot
prove success coverage.

## Validation Report And Tiers

`stage=prepare` normalizes the pack into `output_dir/expt_name/stage_cache/` and
writes `oracle_validation_report.json` containing `tier`, `gold_eligible`,
`pack_fingerprint`, per-check failures, and pack stats. Checks:

1. `template_tool_names` — templates only reference declared tools.
2. `template_slot_sources` — every slot source resolves and its filter matches rows.
3. `backend_schema_alignment` — `list_tools()` and `tools.json` agree, and every
   parameter schema is one this pipeline enforces. A constraint that no argument can
   satisfy is refused rather than certified: an unsupported keyword, a malformed or
   inconsistent bound (`minimum` above `maximum`), an empty `enum`, a repeated `enum`
   value, `required` name or type-union member, and an `enum` or `const` value the
   declared `type` rejects. Local `$ref` through `$defs` or `definitions` and
   `allOf` conjunctions are supported and enforced recursively; external, missing,
   and cyclic references are refused. Because defaults participate in call
   comparison, every declared `default` must itself satisfy the schema under which
   it would be inserted.
4. `assertions_importable` — every template declares at least one `success_assertions`
   entry, and each referenced assertion exists with a valid signature. A template
   naming none has no statement of success, so replay could only confirm that its
   trace ran, not that it was right.
5. `declared_validation_cases` — every probe matches its `expect`, with success and
   negative coverage per tool.
6. `confirmation_policy` — `confirm: false` yields `awaiting_confirmation` and
   leaves state untouched.
7. `representative_generation_contract` — the run-wide inputs hold and every template
   can produce a publishable row. First the category budget must keep at least one
   instance per template and the render contract must resolve (language, every text
   block in that language, system prompt, `tool_names_exempt`). Then the first
   deterministic instance of every template expands, binds an expected trace
   (including `from_result` dependencies), passes its tool schemas, renders without
   breaking a surface guard, replays twice deterministically, and passes its declared
   assertions. Gold means the pack generates, so a typoed argument, an unbound
   placeholder, a missing text block, a starved budget, a broken assertion, or a
   paraphrase that always drops its own rows is reported here instead of aborting a
   run or silently publishing fewer rows than the pack declares. Only the budget is
   settled over the templates generation may bind; a reserved template is still
   compiled, because a private held-out run compiles it from this same pack. When
   reserved fixtures remain in backend state, that private representative may use
   them. Under `fixtures_in_backend_state: false`, only template blocking is disabled
   and fixture reservations stay active, so replay uses exactly the state the oracle
   may observe.

Extra checks: `M1` mutation declaration (observed changes require `x-mutates`, and a
tool declaring it must change state in at least one successful probe), `D1`
determinism (repeat one observed success probe per tool with an identical
`RunContext`, comparing both result and final state), `D2` error shape (every
observed structured error carries a `code`),
`T1` timeout enforcement (a tool that never returns is killed on the same path pack
tools run through), `I1` isolation (`oracle_runtime.worker: process`). D1 and D2
replay the full `reset_before: false` prefix of a chained case rather than executing
that case out of context.

A check whose preconditions failed is recorded as `skipped`, never as `pass`:
check 5 and `M1` skip when check 3 failed, because probing a backend that disagrees
with `tools.json` would only report noise.

`gold` requires every check to pass plus a backend, templates, and assertions.
A `skipped` check is not a pass, so it keeps the pack below gold.
`worker: thread` is a debug aid that runs pack code in-process and can never
reach gold: the pack sees the caller's environment, and a tool that hangs cannot
always be stopped. Only `worker: process` sanitizes the environment and enforces
hard deadlines, which is why a gold claim requires it. `stage=generate` refuses a
pack that is not `gold_eligible`, and it derives that verdict from the individual
checks rather than the summary flag. The report on disk is a human-readable artifact,
not a signed attestation, so `stage=generate` never trusts one written by an earlier
run: it reuses a verdict only when the same process produced it for the same pack and
config fingerprints, which is why `stage=all` validates once rather than twice.
The fingerprint is checked before and after validation and again before final output.
Pack code and concurrent writers must not modify files in the pack while a run is
active; drift aborts publication rather than stamping a report and benchmark from
different content.
`expt_name` is one directory name, not a path: every artifact hash and cache path is
stated relative to the run directory, so a separator or a `.` / `..` segment would move
the run somewhere the config does not name. `output_dir/expt_name` must also remain
outside the oracle pack root: otherwise generated reports and parquets would become pack
inputs and change the fingerprint they claim to describe.

## Generation Stages And Output

`stage=generate` consumes a gold-eligible report and runs, in order: `expand`
(bind slots into locked task instances), `state_machine` (order milestones into
turns and batch `call_group`s), `render` (verbatim surface plus guard checks),
`expected_trace` (derive `expected_tool_calls`), `schema_validation` (check calls
against the tool schemas), `executable_replay` (reset and replay each task twice,
then run its assertions), and `final_output` (assemble `messages`, write the
parquets and `run_manifest.json`).

Reusing an `expt_name` starts a new invocation: completed parquets and the old manifest
are invalidated before validation, parquet files are written through temporary files,
and the manifest is atomically installed last. A failed rerun therefore leaves no
completed manifest that could make stale output look current.

Each stage leaves one artifact under `stage_cache/`, keyed by `task_id` with one
row per task, so a run can be read stage by stage instead of guessed at:

| Stage | Artifact | Holds |
| --- | --- | --- |
| `expand` | `task_instances.parquet` | bound slots, the slot timeline, seed, required tools, assertions |
| `state_machine` | `conversation_plans.parquet` | ordered steps, turn counts, confirmation and correction flags |
| `render` | `rendered_conversations.parquet` | rendered turns, language, guard verdict and violations |
| `expected_trace` | `expected_traces.parquet` | derived `expected_tool_calls` per task |
| `schema_validation` | `schema_validated_traces.parquet` | per-task verdict, reject reason, failure detail |
| `executable_replay` | `replay_validated_tasks.parquet` | verdict, tool results, assertion outcomes |

Every table carries the same `task_id` set, so joining them shows exactly which
stage dropped a task. A task that fails to bind at `expected_trace` keeps a row
there with `derived: false` and appears in the later tables as a skip
(`reject_reason` / `reason: trace_not_derived`) rather than vanishing. Nested
content (slot bindings, plan steps, rendered turns, derived calls, failure lists)
is canonical JSON text, for the reason given under the two benchmark columns below.
`run_manifest.json` records a content hash per artifact — including the
validation report and prepare-normalized files — so a published row is
traceable to the intermediates behind it.

`task_generation.tasks_per_category` budgets a whole **category**, not one
template. Each template in the category first contributes one instance, then a
second, and so on until the budget is spent, so adding a template spreads the
budget instead of growing the set. A budget smaller than the number of templates
in a category is rejected rather than silently dropping the last templates. Slot
candidates are enumerated in fixture order and each template's product is capped
as it is built, so the same pack and config always produce the same instances —
but the cap keeps the earliest candidates, so raising it varies the last slot
first. Use narrower filters when you need spread across a specific slot.

Set `task_generation.candidate_tasks_per_category` above the publication budget
when Stage 11 needs inventory from which to satisfy difficulty or turn-mix
quotas. Stage 4 expands to that candidate ceiling; Stage 11 still enforces
`tasks_per_category`. The candidate ceiling must never be smaller than the
publication ceiling.

Generation rejects unsupported task-generation keys instead of recording
targets that no stage applies. Category budgets are consumed by expansion and
publication; mix targets and hard limits are consumed by Stage 11. The same rule
covers the rest of the config: a request no stage can honor — a paraphrase count,
a judge's drop authority, an unrecognized `surface_generation` key — stops the
run instead of being dropped in silence. Config values are type-strict as well: quoted strings do
not stand in for booleans or integers, and an empty list does not stand in for an
empty mapping. Timeout values must be positive finite numbers; YAML `NaN` and
infinities are rejected before they can reach deadline arithmetic.

### Slot Sources And Filters

A slot's `source` is `<kind>:<rest>`, and a bare value means `fixture`:

| Source | Binds |
| --- | --- |
| `fixture:<collection>.<field>` | `field` of every row of `collection` the `filter` matches. |
| `enum:<tool>.<parameter>` | the parameter's declared `enum` values. |
| `literal:[a, b]` / `literal:200000` | the listed values, or one value keeping its Python type. |
| `range:{min: 1, max: 5, step: 1}` | the inclusive integer range; step is non-zero and points from min toward max, including descending ranges. |
| `absent:<collection>` | the manifest's `absent_ids` for that collection. |

`filter` is a conjunction of comparisons over one fixture row, parsed as a Python
expression: a field name on the left, a literal on the right, `and` between clauses,
and the operators `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`. Because it is
parsed rather than split on operator text, a literal may contain anything —
`title == 'Pride and Prejudice'` compares the whole title. A field the row does not
carry makes its comparison false; `or`, chained comparisons, and a non-literal
right-hand side are rejected.

Calls that share a `call_group` must be declared consecutively; a template that
interleaves another milestone between them is rejected, because the group can then
no longer map to a single assistant completion. A call that declares no `call_group`
is a batch of its own and can never be absorbed into a declared group. The published
`call_group` numbers batches in the order their assistant turns issue them, starting
at 0, so declaring 0 and 5 publishes 0 and 1 while keeping the declared order.

A `dependent_call` argument names a value that only exists after an earlier call
has run, so the template points at that call's result instead of a slot:

```yaml
assistant_milestones:
  - {id: recent_list, type: tool_call, tool: list_recent_transactions, call_group: 0}
  - type: tool_call
    tool: get_transaction_status
    call_group: 1
    args:
      transaction_id: {from_result: {call: recent_list, path: "transactions.0.transaction_id"}}
  - {type: final_answer}
```

`expected_trace` opens one worker episode, executes the calls incrementally, reads
`path` out of each producing result (dotted fields, integer list indices), and locks
the extracted value into the next call; it does not restart and replay the whole
prefix for every dependency. `executable_replay` then re-derives the trace from fresh
resets, so a wrong path shows up as a failing task rather than a plausible row. Rules
the stage enforces:

- `call` must be a declared milestone `id` issued earlier, in a **strictly lower**
  `call_group` — a dependency cannot share a group with its producer.
- `path` must resolve to a scalar, and the producing call must not have returned
  an `error`. A path that misses for one instance drops that instance, with the
  reason recorded in `expected_traces.parquet`; a path that cannot work for any
  instance — indexing a list with a name, for example — fails the run. If more
  instances drop than you expect, narrow the template's slot filter.
- A `from_result` argument shadows a same-named slot, and the two directions must
  agree — a marker requires `turn_policy: dependent_call`, and that policy
  requires at least one marker, so the edge can never quietly degrade into
  slot-bound arguments.
- Pair the template with an assertion that the consumed id appeared in the earlier
  result; that is what keeps the anti-hallucinated-id meaning of this edge.

During executable evaluation, the published locked value is not replayed into
the expected downstream call. The runner projects the marker coordinates,
extracts the value again from the candidate episode's paired live producer
result, and compares the candidate's next call against that live-derived
expectation. The runner never edits candidate arguments. If the live result
cannot satisfy the declared producer, path, scalar type, and consumer schema,
the episode ends with dependency infrastructure evidence rather than falling
back to generation-time oracle bytes.

A `correction` template has a later user turn replace a value the user already
gave, through `user_simulator_turns[].slot_updates`:

```yaml
slots:
  amount_vnd: {source: "literal:[200000]", visible_in_first_turn: true}
assistant_milestones:
  - {id: confirm_original, type: ask_confirm}
  - {id: confirm_corrected, type: ask_confirm}
  - {type: tool_call, tool: create_transfer, args: {confirm: true}}
  - {type: final_answer}
user_simulator_turns:
  - after: confirm_original
    content_template: {vi: "Khoan đã, sửa số tiền thành {amount_vnd_corrected}đ nhé."}
    slot_updates:
      amount_vnd: {source: "literal:[1500000]", bind_as: amount_vnd_corrected}
  - after: confirm_corrected
    content_template: {vi: "Đúng rồi, xác nhận số tiền mới."}
```

Expansion binds both values, so each instance carries the original bindings, the
replacements **in the order the conversation delivers them** — milestone order, not
declaration order — and the values in force at the end; rendering and the expected
trace then walk the conversation and use whichever value is current at that point. `bind_as` is a surface-only alias for the replacement, so the turn can
name the new value while calls keep binding from the canonical slot key. Rules:

- A replacement resolves through a source of its own, which faces the same gold
  gate as the value it replaces and must be the same source kind.
- Only a slot the user already stated (`visible_in_first_turn: true`) can be
  corrected, and a replacement equal to the value in force when the turn lands is
  skipped as a no-op — the pair is dropped while the instance budget is being filled,
  so a template drawing both values from one collection still binds instances.
- `slot_updates` requires `turn_policy: correction` and that policy requires at
  least one replacement, in both directions.
- No call may consume the superseded value. The check follows the **slot**, not the
  value: a call is rejected when a corrected slot it read still held the old value at
  that point, so an unrelated argument that merely happens to equal the replaced value
  is left alone. `expected_trace` refuses it while deriving, and `schema_validation`
  re-checks the published arguments by slot name.
- A correction withdraws whatever the user confirmed before it. A confirmed
  mutation therefore needs a fresh `ask_confirm` and a fresh user reply after the
  replacement, otherwise the row fails validation the same way an unconfirmed
  mutation does.

`task_id` and the per-task `seed` are SHA-256 derivations over pack id/version,
`template_id`, sorted `fixture_refs`, canonical slot bindings (a correction's
replacements included), and `variant_index`, so ids are stable across machines.

A task is dropped rather than published when its surface breaks a guard, its
trace fails schema validation, the two replays diverge (`nondeterministic_replay`),
or an assertion fails. `run_manifest.json` reports `stage_counts` with one entry per
stage that can drop a task (`expanded`, `surface_passed`, `trace_derived`,
`trace_dropped`, `schema_passed`, `replay_passed`, `published`). Each count reports its
own stage only — a guard-rejected row that replayed still counts in `replay_passed` —
and `published` is where the stages meet, so a shortfall names the stage that caused it.
The manifest also summarizes dependent binding failures under `trace_drop_rejections`,
and counts guard rejections per template under `surface_guard_rejections`.
`benchmark_raw.parquet` holds every row that passed schema and replay;
`benchmark.parquet` is that set minus surface-guard rejections, so the two files match
only when no guard fired. A declared held-out policy is validated and included in the
pack fingerprint during prepare, then enforced twice during generation: expansion
never binds a reserved template or fixture row, and publication re-scans every row
against the same policy and stamps `held_out_hit`. Any hit aborts the run before a
parquet or manifest exists, and the offending task ids are listed in
`stage_cache/held_out_scan.json`. A reservation that leaves a slot with no bindable
row, starves a category's `tasks_per_category`, or withholds every template is a pack
error and stops generation with the shortfall named. The manifest records the policy
lineage, the rows scanned, and the templates and fixture rows Stage 4 withheld. Packs
without a held-out source keep `held_out_hit` null and record `evaluated: false`.
When `held_out.policy.fixtures_in_backend_state` is `false`, the same reserved fixture
rows are also removed from every generation replay and executable-evaluation oracle
reset. The full inventory remains available only to the trusted binding and publication
checks that prove those rows were withheld. Local Python episodes run from a temporary
pack mirror whose fixture file contains that same projection, so backend or assertion
code resolving `fixtures.json` relative to `__file__` cannot reopen the full inventory.
This mirror protects the pack-relative contract; it is not an OS sandbox for hostile
allowlisted code that deliberately names unrelated host paths. A value of `true`
retains the reserved rows in backend state for packs whose oracle behavior intentionally
depends on them.
Validation does not reject a probe merely because one argument equals a reserved id:
tool argument names do not prove which fixture, if any, the backend dereferences. If
the executed probe instead returns `not_found`, a mismatch records the matching
argument, collection, and canonical held-out fixture reference in the validation
report.
Each row exposes exactly the tool definitions named by its `tools_present`.
Rows from `lineage.policy: smoke_no_publication` retain the pack's validation
`tier` but set `gold_eligible: false`.

Two columns deviate from a naive Arrow struct, because a struct unions keys across
rows and would pad every call with nulls — turning "argument absent" into
"argument is null" and advertising parameters a tool does not accept:

- `expected_tool_calls[].arguments` is a `map<string, string>` whose values are
  canonical JSON. Decode with `row_schema.decode_arguments`.
- `tools` is canonical JSON text (a JSON Schema has per-tool keys). Decode with
  `row_schema.decode_tools`.

`expected_tool_calls[].turn_index` is the 0-based ordinal of the assistant message
that issues the call, counted across assistant messages — so the confirming call
of a `confirmation` task has `turn_index: 1`, after the `ask_confirm` message.

## Run It

```bash
# Validate a pack and print its report
python -m nemotron.steps.byob.scripts.validate_oracle_pack \
  --config src/nemotron/steps/byob/bfcl/config/tiny.yaml

# Full slice: prepare, validate, then generate the benchmark
uv run nemotron steps run byob/bfcl \
  -c src/nemotron/steps/byob/bfcl/config/tiny.yaml \
  stage=all family=bfcl
```

`config/tiny.yaml` (`tiny_library`) and `config/banking_vn.yaml` pin
`lineage.policy: smoke_no_publication`; neither is publication-eligible.
`banking_vn` is the reference domain pack: it declares a template for every
policy edge the pipeline supports (`single_turn`, `missing_slot`, `confirmation`,
`multi_tool`, `dependent_call`, `negative_path`, `clarify_only`, `irrelevant`) and
every template carries a distractor in `tools_present`. Point `config/default.yaml` at it for a
publication-oriented run.
`config/default.yaml` is the publication-oriented template. Its
`oracle_pack.manifest_path` is a `REPLACE_ME_*` placeholder so the template can never
publish an example domain by omission; point it at your own pack and the config runs
as-is, because its optional profile, paraphrase, and surface-judge roles are disabled.
Disabled roles are recorded with null identity and set
`generation_mode: template_only`, which does not affect gold eligibility. The
remaining `REPLACE_ME_*` entries only matter once the corresponding role is enabled.

Generation supports reference profiling, controlled paraphrasing, Stage 10
surface-quality validation, and optional Stage 11 semantic deduplication and
balancing. Stage 11 fails closed on backend/artifact errors and requires an
explicit abort-or-non-gold policy for unmet targets. Stage 12 supports
`exports.bfcl_json` and `exports.nemo_evaluator_bundle`; unknown export names and
settings owned by later stages are refused rather than ignored.

## Compatibility Exports and Recovery

Both compatibility writers receive the same canonical projection of
`benchmark.parquet`. Stage 12 reads enabled outputs back, checks row count, task
order, tool definitions, expected calls, ordering policy, and content hashes, and
writes `exports/export_validation_report.json`. `run_manifest.json` records every
format as enabled or disabled and pins the report and export hashes.

The NeMo bundle is input for the W5 native-function-calling adapter. Its
`evaluator.yaml` describes seed/replay/scoring semantics but is not a standalone
NeMo Evaluator 0.2.x task registration or Launcher config. Native tool calls need
an installed/containerized harness plus a tool resource service.

Stage 12 writes into `.stage12-*`, validates all payloads, then promotes parquet
and exports before moving `run_manifest.json` last. Treat the manifest as the
commit marker: without it, no adjacent parquet or export belongs to a published
run.

Recovery is fail-closed:

- **Schema mismatch:** do not edit generated rows; remove final Stage 12 payloads
  and regenerate with one code revision. Select consumers using the
  `schema_version` values in the manifest.
- **Unsupported call layout:** use the task id in the exception to fix the pack's
  tool identifiers or conversation shape. The BFCL JSON adapter requires
  Python-compatible function and argument names; the NeMo input contract requires
  an unambiguous seed/replay plan.
- **Hash or equivalence mismatch:** inspect
  `exports/export_validation_report.json` when it exists, then regenerate the
  whole publication. Never repair one exported file in place.
- **Interrupted publication:** if `run_manifest.json` is absent, rerun
  `stage=generate`. Startup removes abandoned `.stage12-*` directories and stale
  final payloads before the next attempt.

## Evaluating a Published Benchmark

Evaluation reads a published run; it never re-derives one. Its input is
`eval_config.yaml` (schema `1.1`), templated at
`bfcl/config/eval.default.yaml` and validated by
`runtime/benchmark_families/bfcl/eval/`. What a resulting score means is defined by
[`bfcl-eval-scoring-contract.md`](bfcl-eval-scoring-contract.md), which the config
references and content-hashes.

The config contract holds these rules:

- **Source is a manifest.** `source_run_manifest` names `run_manifest.json`, not a
  parquet. The published table, its declared content hash, the gold-eligibility
  verdict, and oracle kind are read from the manifest, so the config cannot claim
  something the run did not publish. Executable mode also requires
  `source_oracle.pack_manifest` and `source_oracle.resource`: the latter is the
  concrete `backend.py` or endpoint config. Their kind and pack id/version must
  match the source run, and both files are content-hashed. A
  `translation_manifest`, when present, must reference the same source run.
- **Nothing defaults.** Modes, scoring gates, runtime limits, decoding parameters,
  contamination policy, and output flags are all stated. Unknown keys, quoted
  booleans, quoted numbers, and leftover `REPLACE_ME_*` values are refused.
- **Candidate identity is weight identity.** `provider`/`model`/`base_url` name the
  serving route; `model_identity` must pin an immutable `revision` or
  `weights_digest`. Refs such as `main`, `latest`, or `refs/heads/*` are refused,
  and a revision without a digest must be a full 40–64 hexadecimal commit id.
  Model/revision case is preserved for case-sensitive registries. Two candidates
  may not share an alias or the same canonical weights.
- **Secrets stay in the environment.** `api.api_key_env` names a variable. A
  literal credential anywhere in the config is refused, and error messages redact
  values rather than echoing them. A variable that is not exported yet is an
  execution failure, not a parse failure.
- **Publication is gated.** `publication.requested: true` requires
  `schema_then_canonical` argument matching, call order and grouping respected, no
  LLM repair, `all_applicable_gates` task success, contamination enforced with
  `fail_run` over a `common_intersection`, and every eval artifact written.
  Executable publication also requires a gold-eligible source run. Relaxations are
  legal only for debug runs, which report every weakened field in
  `non_publication_reasons`.
- **Outputs stay out of the publication tree.** `outputs.output_dir` may not be,
  contain, or sit inside the generation run's directory, and may not point at
  another published tree or an existing regular file. The resolved-config writer
  only accepts targets below that output directory, including when its caller
  supplies a relative path.
- **One hash stands for the evaluation.** `eval_config_hash` covers referenced
  bytes, not paths: moving a checkout preserves it, while changing a candidate,
  revision, inference parameter, limit, scoring contract, or source run changes it.
  Eval inputs are excluded from generation lineage hashes, so scoring a new
  candidate never changes the identity of the benchmark.

A valid config only says what an operator named. `verify_eval_source()` then reads
that source back from disk and holds it to the record, before any candidate is
contacted, and returns the handle a runner must be given — there is no way to
obtain one without verification:

- **The manifest is still the manifest.** `run_manifest.json` is re-read, checked
  for every publication field and a schema this build can decode, and held to the
  hash the config resolved. Structure is reported before drift, because a file
  that is not a manifest and a manifest that changed need different fixes.
- **The tables are the published bytes.** Both `benchmark_raw.parquet` and
  `benchmark.parquet` are hashed and compared against every declaration the
  manifest makes about them, in the `publication` section, the `artifacts`
  section, and the resolved config. A symlink is refused: it can be re-pointed at
  another benchmark without changing anything the manifest records.
- **Publication semantics hold on disk.** `publication_contract` (`1.0`) is
  replayed over both files: the published table selects raw rows without
  rewriting truth, in the declared order, and ships no held-out row.
- **Every row is addressable.** The published rows decode under this build's
  benchmark schema into a unique task index in publication order. A row that
  cannot be decoded aborts verification rather than being skipped, since skipping
  it would change the task set. Task ids must work as a path component and a log
  token; non-ASCII letters are allowed, path separators, whitespace, control
  characters, and reserved names are not.
- **The oracle is the certified one.** For `executable` mode the pack fingerprint
  is recomputed across every file in the tree — a helper module the backend
  imports changes what the oracle does — and must equal the manifest's
  `pack.content_hash`. The resource that will run must be the one the pack's own
  `manifest.yaml` selects; no eval-side override is honored, because an override
  is indistinguishable from a substitution. A Python backend is imported in a
  throwaway process worker, under the eval config's own timeouts, to confirm it
  exposes `list_tools`, `reset`, `call_tool`, and `get_state`. An endpoint pack's
  pinned identity must equal the `endpoint_metadata` the source run recorded, and
  its declared CA bundle must be present. No live endpoint is contacted and no
  task is replayed: reachability is an execution-time question, and replay is the
  runner's work.
- **A translation preserves truth.** A translated benchmark must derive from this
  run, declare its language, table, and task-id hash, match its declared bytes,
  carry exactly the source task ids in publication order, and leave every field a
  scorer reads byte-identical under canonical JSON. Conversation text, intent,
  localized metadata, and `tools[].function.description` may change; tool names
  and complete parameter schemas may not. Translation manifests require a
  `model` block plus complete contamination scope and self-hash their semantic
  body; an unidentified translator is refused rather than treated as clean.
- **Every model that read a row is named.** The `models.*` roles the manifest
  records — `profile`, `paraphrase`, `surface_judge` — are read together with the
  rows each one touched: rows carrying a `profile_hash`, rows attributed to the
  paraphraser's canonical id, and the whole published surface for a judge that
  ran. A manifest that omits the block, names a role this build does not read,
  enables a role without naming a model, contradicts its own rows about
  `profile_influenced_surface`, or ships a paraphrased row no role claims is
  refused: a gap in this inventory would read as "no contamination found".
- **The evidence is written down.** A pass writes
  `source_verification_report.json` into `outputs.output_dir`, atomically, listing
  the checks that actually passed; a failure writes
  `source_verification_failure.json` under a different name so a diagnosis cannot
  be mistaken for a pass. The report's `verification_identity` hashes hashes, row
  counts, task ids, and pack fingerprints, and no path or timestamp: an intact
  tree keeps its identity when moved, and loses it when one byte changes.
- **The source is pinned twice.** `assert_source_unchanged()` recomputes every
  recorded hash, the pack fingerprint included, immediately before execution.
  Verification and use are separated in time, and that gap is where a source gets
  replaced — a regeneration into the same directory, a pack edited to make a
  failing task pass.

A verified source says which benchmark is being scored. `evaluate_contamination()`
then decides who may answer which rows of it, and returns the second handle a
runner is given:

- **A collision is evidence, not a flag.** Each candidate is compared against each
  exposure, and the result is recorded with the role, the model, and the exact
  task ids. Comparison weighs the strongest available evidence first: two weights
  digests settle it either way, then an equal operator canonical id, then an equal
  serving route, then a normalized model name plus revision. Names are compared
  case-insensitively with the registry prefix and punctuation removed, so a
  registry naming difference cannot be mistaken for a different model.
- **An unprovable separation is never guessed.** When neither side pinned enough
  to decide, the comparison is `unknown`. It does not exclude rows on suspicion
  and does not abort a debug run, but it always blocks publication — and when
  `publication.requested` is true it is refused here rather than producing a
  number that cannot be published.
- **The policy only narrows.** `fail_run` refuses the run on a match;
  `exclude_row` drops exactly the rows that exposure covered. If exclusion empties
  a candidate's set, or leaves no row every candidate can answer, the run stops:
  a benchmark whose surface models are the candidates cannot be salvaged by
  scoring zero rows.
- **The comparable set is decided here.** Under `common_intersection` every
  candidate answers the rows all of them may answer, in publication order, so two
  numbers are comparable by construction; `per_candidate` keeps each candidate's
  own set and is debug-only. `plan_identity` hashes the whole decision with
  candidates ordered by alias, and no path or timestamp.
- **The decision is written down and re-pinned.** A pass writes
  `contamination_report.json` into `outputs.output_dir` and removes any stale
  `contamination_failure.json`, and the other way round.
  `assert_plan_unchanged()` re-pins the source and re-derives the decision
  immediately before the first request, so a plan widened after authorization
  cannot be the plan a runner acts on.

The native candidate client transports one authorized assistant turn. It sends
only model-facing messages and OpenAI-compatible tools, preserves ordered
`message.tool_calls` and raw arguments without repair, retries only transient
transport failures, and writes a hash-verified request/attempt/completion
sequence to `candidate_io_cache.jsonl`. Completed calls replay without network
or credentials; an interrupted sequence fails closed. The client does not choose
what to ask next.

The conversation driver selects the row and call-group plan from a
canonical projection whose hash, row count, and complete task sequence match the
verified source, then replays the episode the pack's turn policy described. The
model is asked one assistant turn at a time, and a tool result your pack's replay
recorded is handed back only after a type=`function` call with a unique id matches
the trace, addressed to that candidate id. An intermediate text turn must equal
the published assistant text before the next user request is released. This
fail-closed rule is what makes a `missing_slot` or `ask_confirm` policy safe: a
model that calls straight through or emits unrelated prose never receives the
slot value it failed to ask for. Nothing else from the row enters a prompt. The
driver releases recorded results only; it executes no tool and derives no score.

Executable evaluation uses a separate live driver. It first projects an
`ExecutableTaskSpec` that binds the published row, candidate authorization,
source clock, pack fingerprint, oracle resource, template milestones, fixture
references, assertions, and pack-local mutation flags. The Python and endpoint
adapters then keep one reset/call/state/assertion session inside a task-local
process worker. Local `backend.py` and `assertions.py` are never imported into
the evaluator process; endpoint sessions are deleted on every normal and
exceptional exit. Calls execute once in candidate order, and only canonical live
results are returned under the candidate's call IDs. Recorded replay results,
expected calls, assertion metadata, and fixture values have no operation that
can add them to the live model-facing conversation. A timed-out mutating call is
never retried because its commit state is unknown. A returned mutating result is
classified as committed only when canonical state snapshots before and after the
call differ; equal snapshots prove no commit, and a missing snapshot remains
unknown. The runner reconstructs assertion `slots`, `slots_initial`, and
`slot_updates` from verified template metadata and published evidence: the
verbatim opening turn, the expected trace, the cited fixture rows, and typed
fixture, literal, enum, range, and absent-id values from the verified pack.
Because the opening turn renders pre-correction values, it selects the typed
candidate used for `slots_initial`, including values no tool argument names. A
paraphrased surface is never read back. A final slot no channel settles is
listed in `unresolved_slots`; unknown pre-correction and correction values are
listed separately in `unresolved_slots_initial` and `unresolved_slot_updates`
instead of being inferred from another phase. The isolated assertion runner
tracks reads of all three so an assertion that needs a missing value fails as
infrastructure rather than as a candidate error, while an unrelated assertion
runs normally.

`stage=generate` refuses `eval_config_path` and inline `eval` blocks because
generation does not consume evaluation settings. Candidate execution belongs to
the evaluation runtime, not to benchmark publication.
