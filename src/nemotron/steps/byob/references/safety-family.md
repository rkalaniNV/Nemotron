# Safety family

`safety` is a BYOB benchmark family next to `bfcl` and `mcq`. It compiles a customer policy into a private guardrail benchmark. It does not generate function-calling rows and it does not read an oracle pack.

Run it with the same dispatcher as BFCL, from the Nemotron repository root:

```bash
python -m nemotron.steps.byob.scripts.run \
  --config src/nemotron/steps/byob/safety/config/tiny.yaml \
  --family safety \
  --stage prepare
```

`nemotron steps run byob/safety` uses `safety/step.py`, which calls that same entry point.

Relative paths in a safety config resolve from `src/nemotron/steps/byob`. Artifacts are written to `output_dir/expt_name/`.

| `stage` | What the run does |
| --- | --- |
| `prepare` | Validate the config, initialize the run directory, and ingest the policy files locally (see [Ingest](#ingest)). A completed prepare exits 0. It exits 4 when only quarantined files are left, and 1 when every file was rejected. It does not publish a benchmark. |
| `generate` | Extract and route policy rules (see [Rule extraction and routing](#rule-extraction-and-routing)), then write a coverage plan (see [Coverage planning](#coverage-planning)). Case text and publication are not connected yet, so the command still stops before a benchmark exists. Gold exits 0. Silver or Preview exits 2. A ready review exits 3. A privacy block exits 4. |
| `all` | Run `prepare`, then `generate`. It does not evaluate. |
| `eval` | Score a published safety benchmark. Until scoring is connected, the safety hook exits 1 and says this is not a Silver publication. |
| `translate` | Not part of this increment. The safety hook exits 1. Exit code 2 remains reserved for a finished Silver or Preview publication. |

`run_manifest.json` is the publication commit, written last by `generate`, matching BFCL. `prepare` writes `run_record.json` with `publication_commit: false`. A run record is not a published benchmark.

The safety publication statuses are:

| Status | Exit code | Meaning |
| --- | --- | --- |
| Gold completed | 0 | The requested publication finished and passed the Gold checks. |
| Silver or Preview completed | 2 | Publication finished, and Gold remains blocked. |
| Review required | 3 | The review packet is ready, or still waits for a required signature. Run `approve_safety_review`, which signs and publishes. This is not a failure. |
| Privacy blocked | 4 | No safe policy document remains, so no external model was called. |
| Invalid config, incomplete publication, or unavailable stage | 1 | The config cannot start, publication is not connected yet, or the requested stage is not available. This is not a Silver result. |
| No usable sources | 1 | Every policy file was rejected as unsupported, unreadable, or invalid, and none was held back for privacy. No external model was called. |
| No usable rules | 1 | Every extracted or pre-authored rule was quarantined or rejected, so no case is generated. `stage_cache/policy_flags.jsonl` gives each rule's reasons. |
| Model unavailable | 1 | The extraction model could not be reached, or every reply in a pass was malformed. Rerunning continues from the cached windows. |
| Prepare required, damaged or stale run file, or run locked | 1 | The run directory was prepared with a different config or different policy files, or never prepared; a file inside it is damaged or describes older inputs; or another process is writing it. |

Exit code 2 alone does not prove a Silver or Preview result. The shared dispatcher also exits 2 for an unsupported stage, and argparse exits 2 for a usage error. A consumer must read the `code:` prefix on stderr, or the run record, together with the exit code.

### Inputs and their hashes

Every run is bound to two digests:

- `config_sha256` covers the resolved config. `stage` is not part of it, because the dispatcher may run a stage other than the one the YAML names.
- `source_fingerprint` covers every policy source: its path relative to `input_dir`, in POSIX form, and the sha256 of its bytes. Hidden files and directories (names starting with `.`) are not sources: they are never read, privacy-scanned, quarantined, or sent to a model, and they do not appear in the quarantine report. Keep policy material out of hidden paths. Symbolic links inside `input_dir` are refused, and an `input_dir` with no policy file is refused.

`prepare` can be rerun safely. It keeps `created_at` and every entry in `stage_cache/cache_index.json`. When either digest changes, it marks the entries `stale: true` and lists `config_changed` and/or `sources_changed` in `stale_reasons` instead of deleting them. A damaged `run_record.json` or `cache_index.json` is moved to `recovered/` and rebuilt, and the new run record lists it under `recovered_artifacts`. Every check runs before the first write, and `run_record.json` is replaced last, so a failed prepare leaves the previous run untouched.

### Ingest

`prepare` reads every source on the local machine. No model and no network is used, and redaction finishes before any later stage can see a document. Each file ends in one state:

| Status | When | What is stored |
| --- | --- | --- |
| `safe` | The file was read and redacted. | Redacted text and source spans, or the normalized rules for pre-authored IR. |
| `quarantined` | The privacy scan found a credential (`secret_pattern`), or personal data was still present after redaction (`redaction_incomplete`). | Only the finding kind and line number. Never the value, and never the file text. |
| `rejected` | The file cannot be used: `unsupported_format`, `empty_document`, `unsupported_encoding`, `binary_content`, `unreadable_document`, `encrypted_document`, `file_too_large`, `parser_unavailable`, `no_text_layer`, `no_policy_text`, `unclosed_code_fence`, `invalid_structured_document`, `invalid_policy_rule_ir`, `duplicate_policy_id`, or `ingest_failed`. | The reason and a short detail. `ingest_failed` names the exception type only. |

A quarantined or rejected file does not stop the other files. An unexpected failure while reading one file rejects that file with `ingest_failed` and the rest continue. `no_text_layer` is `rejected`, not `quarantined`: a scanned PDF is a format problem, not a privacy finding, so it never causes exit 4.

Formats:

- Markdown (`.md`, `.markdown`) and text (`.txt`, `.text`) must be UTF-8, or UTF-16 with a byte-order mark. Other encodings are refused rather than guessed. Text is normalized to `\n` line endings and Unicode NFC, so a decomposed Vietnamese file gives the same spans as a composed one.
- DOCX is read with `python-docx`, paragraphs and tables in document order. A paragraph is a heading when it carries `w:outlineLvl`, or when its style (or a style it is based on) is named `Heading N` or `Title`, or carries an outline level of its own. A table row becomes `| cell | cell |`.
- PDF is read with `pypdf`, page by page, and each span records its page. A PDF without a text layer is rejected with `no_text_layer`; there is no OCR. An encrypted PDF is tried with an empty password and is otherwise rejected.
- YAML and JSON are parsed after redaction. YAML aliases and duplicate keys are refused, and so are JSON duplicate keys and `NaN`. A file that is not Policy Rule IR is read as text.
- Files over 25 MiB, DOCX packages that expand past 200 MiB, and PDFs over 2000 pages are rejected with `file_too_large`. The file size is checked before the bytes are read.
- A file whose bytes no longer match the hash taken at the start of prepare stops the run with `prepare_required` and names the file. Run `stage=prepare` again once the file stops changing.

Privacy:

- Credentials quarantine the whole file. These are private keys, API keys such as `sk-live-...`, `nvapi-...`, `hf_...`, or `AIza...`, cloud access keys, GitHub, Slack, JWT, and Bearer tokens. An assignment such as `password: ...`, `api_key: ...`, or `mật khẩu: ...` quarantines the file when the value looks like a token: it contains a symbol, or it is at least 20 characters and uses both letter cases. `The password: MustBe8chars long` is policy text and stays.
- Personal data is replaced line by line with a typed placeholder: `REDACTED_EMAIL_1`, `REDACTED_CARD_NUMBER_1`, `REDACTED_PHONE_NUMBER_1`, `REDACTED_NATIONAL_ID_1`, or `REDACTED_BANK_ACCOUNT_1`. Card numbers must pass the Luhn check. National ID and bank account numbers are only redacted after a label such as `CCCD`, `hộ chiếu`, `passport`, `STK`, or `account number`. The same value keeps the same placeholder within a document, and redaction never adds or removes a line.
- The redacted text is scanned again. Anything still found quarantines the file with `redaction_incomplete`.

Spans:

- A heading is a Markdown `#` heading, a DOCX heading (style name or outline level), or a line that starts with `Section`, `Mục`, `Điều`, `Chương`, `Phần`, `Article`, or `Chapter` and a number followed by `.`, `:`, `)`, a dash, or the end of the line. The number may be Arabic (`4`, `4.3`) or Roman (`Chương I`, `Mục II`). There is no length limit. "Section 4 applies to..." is a sentence, not a heading.
- A section label is that number when the document uses it once (`4`, `I`). The same number again is a different section: the label becomes `<parent>/<number>`, for example `2/1` for Điều 1 inside Chương 2, and `heading_path` comes from the headings above that occurrence. An unnumbered heading uses a slug of its title, prefixed by the nearest numbered parent (`4.payments`). Text before the first heading is in section `preamble`.
- A span is one sentence or one table row, at most 1200 characters. Its ID is `<document_id>#<section>#span-NN`, where NN counts spans within the section, for example `five-topic#4#span-01`. Quote markers and list markers (`-`, `1.`, `a.`, `(a)`, `(1)`) are not part of the span text. `A. Nguyen` and `(see) below` stay sentences. `char_start` and `char_end` index the redacted text in `stage_cache/redacted/`.
- Lines inside a `` ``` `` or `~~~` fence are still scanned for secrets, and they are not spans. A fence that never closes rejects that file with `unclosed_code_fence`.
- `document_id` is the path without its extension, folded to lowercase ASCII (`Chính sách/Điều 4.md` becomes `chinh-sach/dieu-4`). `stage_cache/document_ids.json` remembers an id, so adding `policy.txt` beside `policy.md` does not rename `policy.md`. The new file gets `policy.txt`. On a first run that already contains both, the first path in sorted order keeps the stem.
- References such as "see section 4.3", "Điều 5", or "Chương I" resolve to spans of that exact section and its dotted subsections (`4.3` under `4`) in the same document. Section `4.3` does not satisfy a reference to section `4` when section `4` itself is absent, and a span is never one of its own targets. An unknown section is kept with `resolved: false`.
- The glossary collects a quoted or bold term followed by `là`, `được hiểu là`, `có nghĩa là`, `means`, `refers to`, or `is defined as`, and an unquoted term with one of the longer verbs. A plain `X là Y` counts only under a heading such as "Definitions", "Định nghĩa", "Giải thích từ ngữ", or "Thuật ngữ". A colon does not define a term. Each span lists the glossary terms it uses, including terms defined in another file. When two files define the same term differently, the conflict is recorded.
- Byte-identical files are kept once. The later path is `safe` with `duplicate_of`, and its `span_count` and `rule_count` are 0. It contributes no spans and no rules.

Pre-authored Policy Rule IR. A YAML or JSON file is IR when it has a `policy_rules` key, or it is one rule, or it is a list in which every item is a rule. A rule has `policy_id` or `policy_id_candidate` and a `rule` mapping. A file that only uses those words (`rule: follow the written policy`) or that mixes rules with notes is read as text. Each IR rule goes through the same normalization as extracted candidates, so a bare `block_if` list becomes `any_of` with `operator_source: default_any_of`, and the status is reset to `needs_decision` for routing. A supplied `confidence` is dropped with the flag `confidence_dropped`, because routing computes it. `provenance.document_id` defaults to the IR file's own, and `provenance.document_sha256` defaults to the IR file's hash only when the rule cites the IR file itself.

Each pre-authored rule row carries `evidence`, checked against the redacted documents of the same run. The rule itself is not changed; routing reads `evidence`.

| `evidence.status` | Meaning |
| --- | --- |
| `self_authored` | The rule cites the IR file itself. There is no policy text to check. |
| `document_not_found` | No safe document in this run has the cited `document_id`. |
| `document_changed` | The cited `document_sha256` differs from the document's current hash. |
| `span_not_found` | The cited `source_span_id` is not a span of that document. |
| `quote_found` | `source_quote` appears verbatim in the cited span, or, without a span ID, in the spans listed in `span_ids`. |
| `quote_not_found` | The quote does not appear in the cited span or anywhere in the document. |

A rule that cites a byte-identical duplicate is checked against the document that was kept. The file row says `extraction: skipped`, and no span or model request is created for it. Any invalid rule, or a `policy_id` repeated in the file, rejects the whole file with `invalid_policy_rule_ir` and lists every error. A `policy_id` already defined by an earlier file rejects the later file with `duplicate_policy_id`.

Artifacts written by `prepare`:

| Path | Content |
| --- | --- |
| `quarantine/files.jsonl` | One row per source: `path`, `sha256`, `status`, `document_id`, `format`, `reason`, `detail`, `extraction` (`pending`, `skipped`, or null), `span_count`, `rule_count`, `findings`, `redactions`, `duplicate_of`. |
| `stage_cache/source_inventory.json` | The same rows plus counts. |
| `stage_cache/source_index.json` | Documents with their sections, and every span. |
| `stage_cache/redacted/<document_id>.txt` | Redacted document text. `/` in the ID becomes `~`. A name longer than 120 characters is cut and ends with a hash of the full ID. |
| `stage_cache/document_ids.json` | The document id assigned to each path, reused on the next prepare. |
| `stage_cache/glossary.json` | `terms` and `conflicts`. |
| `stage_cache/preauthored_rules.jsonl` | The normalization queue for pre-authored rules, each with `extraction: skipped`, its normalization flags, and `evidence`. |
| `stage_cache/ingest/<key>.json` | Per-file cache, keyed by ingest version, path, document ID, and sha256. An unchanged file is not read again; an unused or damaged entry is removed or rebuilt. |

`cache_index.json` records `stages.ingest` with the outcome (`ready`, `privacy_blocked`, or `no_usable_sources`) and the counts, and `run_record.json` repeats the counts under `ingest`. `run_record.json` also records the sha256 of `source_index.json`, `glossary.json`, and `preauthored_rules.jsonl` under `ingest_artifacts`. When prepare stops with exit 4 or 1, it still commits the report and the run record first, so `generate` reports the same stop.

`python-docx` and `pypdf` are in the `byob` extra. Without them, DOCX or PDF files are rejected with `parser_unavailable` and the other files continue.

### What generate reads

`stage=generate` first checks that `run_record.json` carries both current digests, and otherwise exits with `prepare_required` and names what changed. It then reads the files below before it reports that publication is not connected, and records the outcome as `last_outcome` in `run_record.json`.

`quarantine/files.jsonl` must list every current source exactly once:

```json
{"path": "sub/card.md", "status": "quarantined", "sha256": "<sha256 of the file bytes>"}
{"path": "public.md", "status": "safe", "sha256": "<sha256 of the file bytes>"}
```

- `status` is `safe`, `quarantined`, or `rejected`. `path` must be relative, use `/`, and contain no empty, `.`, or `..` segment. `prepare` writes this report; the rows it writes carry more fields, described under [Ingest](#ingest).
- A missing report, or one that misses a source, names an unknown file, repeats a path, or carries an older sha256, stops the run with `invalid_run_artifact`. A gap is never treated as safe.
- With no `safe` source left, the run exits 4 if any source is `quarantined`, and exits 1 with `no_usable_sources` if every source is `rejected`. It also exits 4 when a row `{"run_blocked": true, "config_sha256": ..., "source_fingerprint": ...}` carries both current digests. A `run_blocked` row for older inputs is ignored. `prepare` never writes this row; it lets another tool hold a run back.

Only after this privacy check does `generate` extract and route rules, the first step that may call a model. Before that, it compares the three ingest files listed under `ingest_artifacts` with the hashes prepare recorded. A file edited, added, or removed after prepare stops the run with `invalid_run_artifact`, and a run record without `ingest_artifacts` asks for `prepare_required`; no model is called in either case. It then reads the review packet.

`publication_review_packet.yaml` must set `packet_version: 1`, `config_sha256`, and `source_fingerprint`. A packet for older inputs is ignored. A current packet exits 3 until every required role has signed it.

### Signing the review packet

After reviewing, the approver runs one command with the config of the run:

```text
python -m nemotron.steps.byob.scripts.approve_safety_review --config <config>
```

It signs the current packet and, once every required role has signed, publishes with `stage=generate` and exits with the publication code. Nobody writes `publication_signoff.json` by hand, passes a packet path, or runs a separate resume stage.

- The approver name defaults to the login name. Pass `--approved-by "Nguyễn Lan"` to record another name.
- The role is chosen automatically: an approver keeps the role they already signed, or takes the only role still missing. `--role` is needed only when several roles are still missing, which happens only in the `regulated` and `high_assurance` profiles.
- When other roles are still missing, the command prints who it is waiting for and exits 3. The last approver's command publishes.
- `--sign-only` records the signature without publishing. `--note` stores a comment with the signature.

| Assurance profile | Roles that must sign |
| --- | --- |
| `standard` | `benchmark_owner` |
| `regulated` | `benchmark_owner`, `compliance_owner` |
| `high_assurance` | `policy_author`, `adjudicator`, `release_owner`, each a different person |

A packet may list `required_roles` to narrow this, for example when no regulated rule needs a compliance owner. Each role holds one signature, and signing it again replaces it. The command refuses to sign when the policy files or config changed since prepare (`prepare_required`), when there is no packet yet, or when the packet describes older inputs. The signoff stores `signoff_version: 1`, both current digests, `packet_sha256` (the sha256 of the packet bytes at signing time), and per role `approved_by`, `approved_at`, and an optional `note`. Editing the packet after signing invalidates every signature on it. A signature is a recorded attestation, not a cryptographic signature.

### Config rules

- `config_status` may be omitted, as in BFCL generation configs. `config_status: template` is refused, and so is any value containing `REPLACE_ME`. `config_resolved.yaml` always records `config_status: resolved`.
- YAML with duplicate keys, non-string keys, dates, or other non-JSON values is refused, so the config hash is well defined.
- Unknown keys are rejected at every level. Choices, lists, shares, and whole-number budgets are type-checked, and each mix must add up to 1.
- `proposals.moderation_targets` and `proposals.decision_actions` accept the same values as `scope.targets` and `decision.action` in Policy Rule IR. The defaults are `user_input` and `assistant_output`, and the four actions `allow`, `block`, `caution`, and `escalate`. `retrieved_context`, `tool_input`, `tool_output`, `tool_call`, `rewrite`, `redact`, and `require_confirmation` must be listed explicitly.
- `generation.decision_mix` may give a share only to an action listed in `proposals.decision_actions`. Narrowing `decision_actions` to `[allow, block]` therefore also needs `caution: 0` and `escalate: 0` in the mix.
- `generation_model_config`, `judge_model_configs`, `evaluation`, and `languages` must not contain credentials. The same key names BFCL refuses, such as `api_key`, `token`, `authorization`, or anything ending in `_api_key`, stop the run before anything is written. Provide credentials through the provider environment.
- `input_dir` and `output_dir` must not be the same directory or inside each other.

The checked-in `safety/config/default.yaml` uses `stage: prepare` so that copying it runs the stage that currently finishes.

### Policy Rule IR

Each atomic customer rule is stored as one `PolicyRule` (`safety/ir.py`). `policy_rule_json_schema()` exports its JSON Schema for hand-authored rules, and `validate_rule(mapping)` raises `RuleValidationError` with every problem found.

```yaml
policy_id: VN-ANTI-STATE-001
rule:
  actor: assistant
  action: provide_content
  object: anti_state_activity
  block_if:
    any_of: [instructional_or_organizing_verb, prohibited_target]
    operator_source: default_any_of
severity: high
provenance: {document_id: POLICY, source_quote: "..."}
confidence: 0.95
status: eligible_for_generation
```

- A stored condition list (`allow_if`, `block_if`, `escalate_if`, `require_confirmation_if`) sets exactly one of `all_of`, `any_of`, or `none_of`, with at least one unique snake_case condition. A bare list, two operators, or an unknown key is refused.
- `operator_source: default_any_of` is valid only on `block_if` with `any_of`. `operator_source: single_condition` is valid only on a group with exactly one condition, where the operator does not matter.
- `severity_basis` records why the severity was chosen, for example `rubric:explicit_prohibition` or `source:pre_authored`. `paired_with` names the rule a paired allow rule relaxes.
- Missing `version`, `owner`, `effective_date`, domain, or jurisdiction is `unspecified`. An empty domain or jurisdiction list is the same as missing. Jurisdiction `global` is refused on a stored rule. Normalization clears it, including a padded value, a single string, or the singular key `jurisdiction`, and sends the rule to review.
- An `eligible_for_generation` rule has a confidence and no `unresolved_conditions`, no `unsupported_fields`, and no testability other than `testable`.
- Customer-specific fields go under `extensions`; any other unknown key is refused.

`normalize_candidate(raw)` turns extractor output into a stored rule and never raises; it returns the rule, the flags it raised, and validation errors. A bare list is resolved from the source wording (`provenance.source_quote`, or `operator_text`):

| Wording | Result |
| --- | --- |
| "đồng thời", "đủ các thành phần", "cùng lúc", "cả hai", "tất cả các thành phần", "tất cả các điều kiện", "all of the following", "both of the following", "at the same time" | `all_of`, `operator_source: explicit` |
| "một trong", "any of the following", or "either" together with "or" | `any_of`, `operator_source: explicit` |
| "không có điều kiện nào", "không có thành phần nào", "none of the following", "neither of the following" | `none_of`, `operator_source: explicit` |
| More than one kind of operator wording | Moved to `unresolved_conditions`, so `needs_decision` |
| A bare "or", "hoặc", "both", or "either" | Not an operator. `block_if` uses the default below |
| No wording, `block_if` | `any_of`, `operator_source: default_any_of` |
| Any other condition with exactly one entry | `any_of`, `operator_source: single_condition` |
| No wording, any other condition with several entries | Moved to `unresolved_conditions`, so `needs_decision` |

A shared source quote supplies an operator only when the rule has one condition. For a rule with several conditions, the caller passes field-specific evidence such as `operator_text={"block_if": "..."}`. A shared quote with clear wording is otherwise ambiguous and leaves each bare list unresolved; it is never guessed onto `block_if`. A model self-score is dropped at the top level and inside `rule`, `provenance`, `decision`, and `scope`. A reviewer `rejected` status is kept. Every other status waits for routing. A candidate that is not a mapping, whose `unresolved_conditions` is not a mapping, or whose evidence mapping has an unknown condition returns errors and does not raise. Stored lists and mappings cannot be changed in place.

### Action order

When several rules match one case, `aggregate` (`safety/actions.py`) returns the strongest action:

```text
escalate > block > require_confirmation > rewrite > redact > caution > allow
```

The primary policy is the rule that demanded the winning action and has the highest severity, then the lowest `policy_id`. `contributing_policy_ids` lists every matching policy, winning action first. `overridden_policy_ids` lists policies whose action lost. No matching rule gives `not_applicable`. By default, `expected_action` evaluates only `eligible_for_generation` rules. `include_review=True` also evaluates `needs_decision` rules for a review preview; rejected and quarantined rules never apply. It does not apply scope, jurisdiction, date, or actor. A rule whose exception is present does not match. `escalate_if` maps to `decision.critical_action` (default `escalate`), `block_if` to `decision.violation_action` (default `block`), `require_confirmation_if` to `require_confirmation`, and `allow_if` to `allow`. `ambiguous=True` uses `decision.ambiguous_action` only after at least one rule condition matches. Unresolved conditions are never evaluated.

### Confidence and routing

`route_rule` (`safety/confidence.py`) stores the confidence that Python computes. A model self-score is never an input.

```text
confidence = 0.25 quote_match + 0.20 schema_completeness + 0.20 pass_agreement
           + 0.15 definition_resolution + 0.10 cross_reference_resolution + 0.10 low_ambiguity
```

Each signal is a score from 0 to 1. `pass_agreement` counts as 1 when only one extraction pass ran, and `cross_reference_resolution` counts as 1 when the passage has no cross-reference. With every signal at 1 except `low_ambiguity: 0.5`, confidence is 0.95.

| Condition | Status |
| --- | --- |
| Quote not found verbatim in its span, confidence below 0.70, or testability `needs_clarification`, `non_testable`, or `rejected` | `quarantined` |
| Confidence from 0.70 up to but excluding 0.90, an unresolved operator, an unsupported field, a conflict, unassessed testability, or `partially_testable` | `needs_decision` |
| Everything else | `eligible_for_generation` |

`quote_in_span` normalizes NFC, curly quotes, zero-width characters, and whitespace, and the quote must fall on word boundaries. One changed word still fails. `route_rule` treats its deterministic `quote_found` boolean as the only authority for the exact-match score, replacing any stale score in `ConfidenceSignals`. Prefer `route_rule_from_span`, which verifies `provenance.source_quote` and routes in one operation. A `rejected` rule stays rejected. A `block_if` stored as `default_any_of` is not a routing reason. The thresholds are defaults and can be passed as `Thresholds(eligible=..., decision=...)`.

### Rule extraction and routing

`generate` turns the ingested policy into routed rules before it looks at the review packet (`safety/rules.py`). The model only proposes; Python decides what is stored.

**Evidence windows.** Each span of a safe text document becomes one window (`safety/windows.py`): its heading path, the clause, the clause before and after it in the same section, and the glossary definitions the clause uses. Windows are built from redacted text only. The whole document is never sent.

**Model.** Extraction uses `generation_model_config` through Data Designer, the same structured-output path BFCL uses (`alias`, `provider`, `model`, optional `base_url`, `api_key_env`, and `inference_parameters`). The key itself is read from the environment variable named by `api_key_env` and is never written to the run. A policy with text but no `generation_model_config`, or one missing `alias`, `provider`, or `model`, stops with `invalid_config`. A run whose rules are all pre-authored Policy Rule IR needs no model.

**Cache.** Each reply is cached in `stage_cache/extraction/`, keyed by prompt version, both prompts, model identity (provider, model, base URL, inference parameters, never credentials), pass number, and window. A rerun on unchanged input sends nothing. A malformed reply is not cached, so the next run asks again. A reply in which any candidate fails the reply schema is used for this run, so its valid candidates still become rules, but it is not cached either; `policy_routing_log.json` counts it as `not_cached`. When the model returns two answers for one window, neither is used and the window is asked again. One bad window does not discard the windows that succeeded. The run stops with `model_unavailable` only when the model cannot be called, or when every window in the first pass fails.

**Checks on each candidate.** A candidate is validated on its own, so one bad candidate does not drop the others.

| Check | Result |
| --- | --- |
| The candidate does not match the reply schema | Listed as `malformed_candidate`, not a rule |
| The quote is not in the clause but is in a neighboring clause | Listed as `quote_in_neighbor`; the neighbor's own window extracts it |
| The quote is in neither | Kept with `quote_found: false`, so `quarantined` |
| A condition or exception has evidence in neither the clause nor a glossary definition | `unsupported_fields` gets `rule.<class>.<name>` or `exceptions.<name>`, flag `condition_not_grounded`, so `needs_decision` |
| No condition remains | Listed as `no_conditions`, not a rule |
| `expected_action` is not produced by any condition, or an action is not in `proposals.decision_actions` | `needs_decision` with that reason |
| The model's `operator_hint` disagrees with the stored operator | Flag `operator_hint_ignored`; the quote decides |

Condition names are folded to snake_case ASCII (`Bỏ qua mọi quy định` becomes `bo_qua_moi_quy_dinh`). A name with no ASCII letters, such as a Chinese term, is stored as `c_` plus a hash and the rule goes to review, so the condition is not dropped. A second candidate with the same quote but different conditions also sends the kept rule to review. Scope comes from `proposals`, never from the model. A model self-score, `confidence`, or severity hint is removed before anything is cached.

**IDs.** `policy_id` is `<DOCUMENT>-<SECTION>-<HASH8>`, where the hash covers document id, section, the normalized quote, and the expected action. The same input gives the same IDs on every run. A candidate that repeats an ID is listed as `duplicate_candidate`. An allow-only rule is paired with the nearest earlier acting rule in its section, or the first one: its ID is `<partner>-ALLOW` (`-ALLOW-2` for a second) and `paired_with` names the partner. Pairing applies to extracted rules only. A pre-authored rule keeps the `policy_id` its author gave it, so an author who wants a pair sets `paired_with` in the IR.

**Severity rubric.** The first matching line wins, and `severity_basis` records it:

| Rule | Severity | `severity_basis` |
| --- | --- | --- |
| Allow rule | low | `rubric:explicit_allow` |
| Fraud, personal harm, legal prohibition, or irreversible tool action, with binding words (must, shall, required, prohibited, never, phải, bắt buộc, nghiêm cấm, tuyệt đối không, không được phép) | critical | `rubric:binding_language+<category>` |
| Escalation | high | `rubric:escalation` |
| Block with no exception, no multi-condition `all_of`, no `none_of`, and no "only when", "unless", "chỉ khi", "trừ khi" | high | `rubric:explicit_prohibition` |
| Anything else | medium | `rubric:conditional_restriction` |

A pre-authored rule keeps its severity with `severity_basis: source:pre_authored`. One without `testability` is read as `testable`, with flag `testability_from_author`.

**Signals.** `schema_completeness` is the share of: actor, action, and object stated; a stored condition; testability set; expected action produced. `definition_resolution` is the share of conditions that are grounded. `low_ambiguity` is 0 for mixed operator wording, 0.5 for hedges (may, might, should, nên, reasonable, as appropriate, có thể, khi cần thiết, nếu phù hợp), and 1 otherwise. `cross_reference_resolution` is the share of resolved cross-references in the clause.

**Second pass.** A window runs a second, independent pass when the profile is `regulated` or `high_assurance`, or when one of its rules with a found quote scores below 0.90. Rules are matched across passes by quote only. `pass_agreement` is the share of three checks that agree: expected action, condition classes, and the condition names in each class. A rule pass two does not find gets 0. A rule only pass two finds is listed as `second_pass_only` and is not stored. If the second pass fails, the first-pass rules stay and go to review; the run does not stop.

**Conflicts.** `safety/conflicts.py` evaluates each rule's triggers, and the combined triggers of related rules. Rules are related when they share a condition name, a source section, or an object that both quotes use. A guessed object such as `user_request`, and a missing section, do not relate rules. Rules whose moderation targets do not overlap are not combined. Where several actions fire, `policy_graph.json` records a node:

```json
{
  "case_context": ["explicit_confirmation_missing", "suspected_fraud"],
  "matched_policy_ids": ["TRANSFER-3-1A2B3C4D", "TRANSFER-3-9F8E7D6C"],
  "actions": ["escalate", "block"],
  "expected_action": "escalate",
  "primary_policy_id": "TRANSFER-3-1A2B3C4D",
  "resolution": "action_order"
}
```

A conflict the action order must not settle is unresolved and goes to `conflict_report.json`: `contradictory_rules` (one rule allows exactly the trigger another acts on, including one rule that both allows and acts on that trigger), `exception_contradicts_condition`, and `definition_conflict` (the clause uses a term defined differently in two places). Each rule involved goes to `needs_decision`; a critical one is `quarantined`. Other rules are not affected. A damaged source index, glossary, or pre-authored rule file stops with `invalid_run_artifact` before a model is called. `generate` removes the previous rule files at the start of this stage, so a failed run does not leave rules from an older result.

When no rule is `eligible_for_generation` or `needs_decision`, `generate` stops with `no_usable_rules` and generates nothing. Routing still writes the coverage plan first; with no eligible rule that plan is Silver and schedules nothing.

| File in `stage_cache/` | Contents |
| --- | --- |
| `candidate_policy_rules.jsonl` | Extracted rules before routing, with `quote_found`, `passes`, and signals. Pre-authored rules are not here. |
| `extraction_uncertainties.jsonl` | Candidates that did not become rules, each with `window_id`, `reason`, and `detail`. |
| `policy_rules.jsonl` | Every routed rule with `origin`, `status`, `confidence`, `reasons`, and `flags`, sorted by `policy_id`. |
| `policy_rules_eligible.yaml` | Only `eligible_for_generation` rules, the input to case generation. |
| `policy_flags.jsonl` | Every rule that is not eligible, with its reasons. |
| `policy_graph.json` | Situations where several actions fire, and how they resolve. |
| `conflict_report.json` | Unresolved conflicts and glossary conflicts. |
| `policy_routing_log.json` | Prompt version, model identity, passes with cached and requested counts, thresholds, weights, and counts. |

`cache_index.json` records `stages.extract`, `stages.normalize`, and `stages.conflicts`, and `run_record.json` repeats the counts under `rules`. The three entries say `running` while the stage works, `completed` once every rule file is written, and `failed` with the stop code under `outcome` when the stage stops. The index never says `completed` for rule files that a failed run removed.

### Coverage planning

After routing, and before any case text is generated, `generate` writes a coverage plan. The planner reads eligible rules only. A `needs_decision`, quarantined, or rejected rule adds no cell. Those rules are listed in `excluded_rules`, and `coverage_of` is `eligible_rules`. `certification: gold` means every required cell of the eligible rules fits in the budget. It does not mean the whole policy is ready to publish. The planner does not call a model.

The cap is `budget.preview_max_cases` in preview mode. Assisted and expert both use `budget.assisted_max_cases`; raise that number for a larger expert run. A critical rule also stops at `budget.cases_per_critical_rule`. Every other severity stops at `budget.cases_per_normal_rule`.

Required cells are scheduled before optional cells. Across rules, every cell of an earlier kind is scheduled before any cell of a later kind, and within one kind the order is `policy_id`. The kinds are:

1. One clear violation, when the rule has a block, escalation, or confirmation condition. The expected action is the strongest of those actions.
2. One clear allow. The expected action is `allow`. `paired_policy_id` names the paired rule. The cell is flagged `paired_rule_not_eligible` when that rule is not eligible, and `paired_rule_not_allow` when it is eligible but has no `allow_if`. With no partner, the flag is `allow_unpaired`.
3. One boundary case. Its expected action is `decision.ambiguous_action`. When the rule states none, the cell is kept and flagged `boundary_action_unspecified`.
4. One exception case when the rule lists exceptions. The expected action is `allow`.
5. One missing-leg allow case for each leg of an `all_of` group on `block_if`, `escalate_if`, or `require_confirmation_if` that has two or more legs. Exactly one leg is absent and the others stay. A two-condition `all_of` block therefore receives two such cases. An `any_of` group receives none, including a `block_if` stored as `default_any_of`: one leg is already a violation. An `allow_if` group receives none: a missing allow leg is not an allow case.
6. One case for each source language after the first. Languages come only from `scope.languages`, ignoring `unspecified`. The planner does not guess a language from the quote. The first language is `source_language` on the rule's other cells, not a second case.
7. One code-switch case when `scope.languages` lists at least two languages and the quote or the cited span affirms a code-switch, with wording such as `code-switch`, `chuyển mã`, `trộn ngôn ngữ`, or `mixed language`. The cell lists those declared languages. A negated clause (`không`, `not`, `never`) does not count, and foreign words in the quote do not by themselves.
8. One multi-turn case when the quote or the cited span has two or more numbered steps (`1.`, `bước 1`, `step 1`), or affirms `multi-turn`, `nhiều lượt`, `nhiều bước`, `trong hội thoại`, `in a dialogue`, or `hidden instruction`. The same negation rule applies to those phrases. `turns` is the step count, and `budget.max_turns` caps it. A cap below 2 cannot hold the cell, so the cell is a gap with reason `max_turns` and the plan is Silver.

Optional cells come after the required cells, and only while the caps still have room: one adversarial case when `generation.modules.adversarial` is true, one actor-swap case when `political_symmetry` is true, and one persona case per eligible rule. `confirm_if_detected` does not add actor-swap cases. Production replay adds none in this increment, reported as `not_configured`, because the safety config has no evidence paths yet. An optional cell left out does not block Gold.

A required cell that does not fit is not dropped. `coverage_gaps.json` lists it with its slot id and a reason, `budget_cap` or `rule_cap` or `max_turns`, and `blocks_gold: true`. The plan's `certification` is then `silver`, and `gold_cases_needed` is the full required count. When every required cell of the eligible rules is scheduled, `certification` is `gold` even if optional cells remain or other rules were excluded. `run_record.json` repeats the certification, the counts, `missing_required`, and `excluded_rules` under `coverage`.

A slot id joins the policy id, the kind, and any extra part with `/`. A contrast family id uses the same escaping after `contrast/`. A `/` or `\` inside a part is escaped. Read `policy_id`, `kind`, `language`, and `absent_leg` from the row. Do not split the slot id on `:`.

Contrast families point at those slot ids. An acting `all_of` rule has one family per leg: the clear violation keeps every leg, and the allow member is that leg's missing-leg cell. A rule with no missing-leg cell has one family whose allow member is the clear allow cell. Families do not add cases. `changed_dimension` is the absent leg, `paired_allow` when the partner is an eligible allow rule, `paired_rule` when the partner is not an allow rule, or `benign_neighbor` when the rule names no partner.

| File in `stage_cache/` | Contents |
| --- | --- |
| `coverage_plan.jsonl` | Scheduled cells, required rows first. Each row names the rule, the kind, and the expected action. It is not case text. |
| `coverage_gaps.json` | Certification, counts, every unscheduled cell, and the contrast families. |

`cache_index.json` records `stages.coverage_plan` as `running`, then `completed`, or `failed`. Any failure while planning, including a damaged span, a repeated span id, or an error inside the planner, marks the stage `failed` and deletes `coverage_plan.jsonl` and `coverage_gaps.json`. The next generate removes the plan at the start of rule extraction, so a failed run does not keep a plan for rule files it deleted.
