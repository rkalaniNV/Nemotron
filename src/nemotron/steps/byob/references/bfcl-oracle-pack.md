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
`(*, state, trace, task, ctx)` and raise `AssertionError` on failure. Export them
through an `ASSERTIONS` dict or name them `assert_*`. Assertions run inside the
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
   declared `type` rejects.
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
   run or silently publishing fewer rows than the pack declares.

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

`tasks_per_category` is currently the only active `task_generation` control.
Generation rejects other keys (for example mix targets or turn limits) instead
of recording targets that no balancing stage applied. The same rule covers the rest
of the config: a request no stage can honor — a paraphrase count, a judge's drop
authority, an unrecognized `surface_generation` key — stops the run instead of
being dropped in silence. Config values are type-strict as well: quoted strings do
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
only when no guard fired. No held-out source is loaded
yet, so `held_out_hit` is null rather than false, and the
manifest's `held_out` block records `evaluated: false` instead of a drop that did not
happen.
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

Generation refuses a config that asks for work no stage performs — an enabled
profile, paraphrase, or surface-judge role, `model_paraphrase_enabled`,
`surface_quality_validation`, `semantic_deduplication_config`, or an export — so
a run never claims lineage or quality guarantees nothing produced.
