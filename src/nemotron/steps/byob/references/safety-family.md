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
| `generate` | Extract and route policy rules, plan coverage, generate scenarios, validate and split the dataset, and create the review packet. An unsigned or incomplete review exits 3. Resuming with the signed current packet publishes Gold with exit 0, or Silver/Preview with exit 2. If `quality_gates.on_failure` is `fail` and the minima are still unmet after review, generate exits 1 and does not publish. |
| `all` | Run `prepare`, then `generate`. It does not evaluate. |
| `eval` | Score a published safety benchmark. Path 1 calls a direct guard adapter. Path 2 evaluates a complete Guardrails configuration from a live endpoint or recorded log. The two paths write separate metric files and exit 0 when those files are written. Evaluation is not a Gold or Silver publication. An unpublished run exits 1. |
| `translate` | Not part of this increment. The safety hook exits 1. Exit code 2 remains reserved for a finished Silver or Preview publication. |

`run_manifest.json` is the publication commit. `generate` writes compiled files, stores the publication claim in `run_record.json` with `publication_commit: false`, writes the manifest, then sets the commit flag. A crash after the manifest is recovered by verifying that claim. `prepare` writes `run_record.json` with `publication_commit: false`. A run record is not a published benchmark.

The safety publication statuses are:

| Status | Exit code | Meaning |
| --- | --- | --- |
| Gold completed | 0 | The requested publication finished and passed the Gold checks. |
| Silver or Preview completed | 2 | Publication finished, and Gold remains blocked. |
| Review required | 3 | The review packet is ready, or still waits for a required signature. Run `approve_safety_review`, which signs and publishes. This is not a failure. |
| Privacy blocked | 4 | No safe policy document remains, so no external model was called. |
| Invalid config, incomplete publication, or unavailable stage | 1 | The config cannot start, a required artifact is missing or changed, or the requested stage is unavailable. This is not a Silver result. |
| No usable sources | 1 | Every policy file was rejected as unsupported, unreadable, or invalid, and none was held back for privacy. No external model was called. |
| No usable rules | 1 | Every extracted or pre-authored rule was quarantined or rejected, so no case is generated. `stage_cache/policy_flags.jsonl` gives each rule's reasons. |
| Model unavailable | 1 | The extraction model could not be reached, or every reply in a pass was malformed. Rerunning continues from the cached windows. |
| Minimum dataset quality not met | 1 | `quality_gates.on_failure` is `fail`, and the configured minima are still unmet after review. The review packet remains; no benchmark is published. |
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

- Credentials quarantine the whole file. These are private keys, API keys such as `sk-live-...`, `nvapi-...`, `hf_...`, or `AIza...`, cloud access keys, GitHub, Slack, JWT, and Bearer tokens. An assignment such as `password: ...`, `api_key: ...`, `mật khẩu: ...`, `mot de passe: ...`, or `密码: ...` quarantines the file when the value looks like a token: it contains a symbol, or it is at least 20 characters and uses both letter cases. `The password: MustBe8chars long` is policy text and stays.
- Personal data is replaced line by line with a typed placeholder: email, payment card, phone, national ID, bank account, IBAN, or a labeled date of birth. Card numbers pass Luhn and IBANs pass ISO 13616 MOD-97 before redaction. National ID, bank account, and birth dates require a contextual label, so an unlabeled number such as `123-45-6789` stays in the text. Those labels, and birth-date labels, cover common English, Vietnamese, French, Spanish, German, Chinese, Japanese, and Korean forms. The same value keeps the same placeholder within a document, and redaction never adds or removes a line. Free-form names and postal addresses still require customer-side redaction because deterministic regex cannot distinguish them from policy prose safely.
- The redacted text is scanned again. Anything still found quarantines the file with `redaction_incomplete`.

Spans:

- A heading is a Markdown `#` heading, a DOCX heading (style name or outline level), or a line that starts with a section keyword (`Section`, `Mục`, `Điều`, `Chapitre`, `Artículo`, `Kapitel`, `Article`, `Chapter`, and the other listed forms) and a number followed by `.`, `:`, `)`, a dash, or the end of the line. The number may be Arabic (`4`, `4.3`) or Roman (`Chương I`, `Mục II`). There is no length limit. "Section 4 applies to..." is a sentence, not a heading.
- A section label is that number when the document uses it once (`4`, `I`). The same number again is a different section: the label becomes `<parent>/<number>`, for example `2/1` for Điều 1 inside Chương 2, and `heading_path` comes from the headings above that occurrence. An unnumbered heading uses a slug of its title, prefixed by the nearest numbered parent (`4.payments`). Latin letters in that slug are ASCII-folded; letters with no ASCII form, such as CJK, are kept. Text before the first heading is in section `preamble`.
- A span is one sentence or one table row, at most 1200 characters. Sentence boundaries support Latin punctuation and `。！？`; a following lowercase sentence is not merged into the previous one. Its ID is `<document_id>#<section>#span-NN`, where NN counts spans within the section, for example `five-topic#4#span-01`. Quote markers and list markers (`-`, `1.`, `a.`, `(a)`, `(1)`) are not part of the span text. `A. Nguyen` and `(see) below` stay sentences. `char_start` and `char_end` index the redacted text in `stage_cache/redacted/`.
- Lines inside a `` ``` `` or `~~~` fence are still scanned for secrets, and they are not spans. A fence that never closes rejects that file with `unclosed_code_fence`.
- `document_id` is the path without its extension, folded to lowercase ASCII (`Chính sách/Điều 4.md` becomes `chinh-sach/dieu-4`). `stage_cache/document_ids.json` remembers an id, so adding `policy.txt` beside `policy.md` does not rename `policy.md`. The new file gets `policy.txt`. On a first run that already contains both, the first path in sorted order keeps the stem.
- References such as "see section 4.3", "Điều 5", or "Chương I" resolve to spans of that exact section and its dotted subsections (`4.3` under `4`) in the same document. Section `4.3` does not satisfy a reference to section `4` when section `4` itself is absent, and a span is never one of its own targets. An unknown section is kept with `resolved: false`.
- Headings are markdown headings, an outline level, a numbered keyword such as "section" or "điều", or a language-neutral form such as `第4条` and `§ 2`. The keyword list is not required for those structural forms.
- The glossary collects a quoted term followed by a colon, a quoted or bold term followed by `là` or a listed definition verb (`means`, `signifie`, `是指`, …), and an unquoted term with one of the longer verbs. Unsegmented verbs such as `是指` or `とは` do not need spaces around them. A plain `X là Y` counts only under a heading such as "Definitions", "Định nghĩa", "Glosario", or "术语". A bare colon after an unquoted word does not define a term. Each span lists the glossary terms it uses, including terms defined in another file. An unsegmented key needs at least two alphanumeric characters, and a longer matching compound suppresses a shorter matching key. When two files define the same term differently, the conflict is recorded.
- Byte-identical files are kept once. The later path is `safe` with `duplicate_of`, and its `span_count` and `rule_count` are 0. It contributes no spans and no rules.

Pre-authored Policy Rule IR. A YAML or JSON file is IR when it has a `policy_rules` key, or it is one rule, or it is a list in which every item is a rule. A rule has `policy_id` or `policy_id_candidate` and a `rule` mapping. A file that only uses those words (`rule: follow the written policy`) or that mixes rules with notes is read as text. Each IR rule goes through the same normalization as extracted candidates. A one-condition bare `block_if` list becomes `any_of` with `operator_source: default_any_of`. A longer bare list stays unresolved unless the source states one clear operator frame. The status is reset to `needs_decision` for routing. A pre-authored id stays the id its author supplied. Its expected action is the strongest action its conditions produce, so rewrite, redact, caution, and confirmation are not collapsed to block. A supplied `confidence` is dropped with the flag `confidence_dropped`, because routing computes it. `provenance.document_id` defaults to the IR file's own, and `provenance.document_sha256` defaults to the IR file's hash only when the rule cites the IR file itself.

Each pre-authored rule row carries `evidence`, checked against the redacted documents of the same run. Ingest does not rewrite the authored IR. Routing reads `evidence`, and when the quote matches exactly one source span it binds that span onto the stored provenance so accepted cases can point at a source span.

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

`stage=generate` first checks that `run_record.json` carries both current digests, and otherwise exits with `prepare_required` and names what changed. It then runs the deterministic pipeline, creates or resumes the review packet, and records the outcome as `last_outcome` in `run_record.json`.

`quarantine/files.jsonl` must list every current source exactly once:

```json
{"path": "sub/card.md", "status": "quarantined", "sha256": "<sha256 of the file bytes>"}
{"path": "public.md", "status": "safe", "sha256": "<sha256 of the file bytes>"}
```

- `status` is `safe`, `quarantined`, or `rejected`. `path` must be relative, use `/`, and contain no empty, `.`, or `..` segment. `prepare` writes this report; the rows it writes carry more fields, described under [Ingest](#ingest).
- A missing report, or one that misses a source, names an unknown file, repeats a path, or carries an older sha256, stops the run with `invalid_run_artifact`. A gap is never treated as safe.
- With no `safe` source left, the run exits 4 if any source is `quarantined`, and exits 1 with `no_usable_sources` if every source is `rejected`. It also exits 4 when a row `{"run_blocked": true, "config_sha256": ..., "source_fingerprint": ...}` carries both current digests. A `run_blocked` row for older inputs is ignored. `prepare` never writes this row; it lets another tool hold a run back.

Only after this privacy check does `generate` extract and route rules, the first step that may call a model. Before that, it compares the three ingest files listed under `ingest_artifacts` with the hashes prepare recorded. A file edited, added, or removed after prepare stops the run with `invalid_run_artifact`, and a run record without `ingest_artifacts` asks for `prepare_required`; no model is called in either case. It then writes the coverage plan, the scenario seeds, and the checked dataset split, and reads the review packet.

`publication_review_packet.yaml` sets `packet_version: 1`, `config_sha256`, and `source_fingerprint`. It binds the routed and eligible rules, conflict report, coverage gaps, contrast families, validation report, deduplicated cases, split manifest, and decision sheet by sha256. A packet for older inputs is rebuilt. A current packet exits 3 until every review card is resolved and every required role has signed it.

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

A packet may list `required_roles` to narrow this, for example when no regulated rule needs a compliance owner. Each role holds one signature, and signing it again replaces it. The command refuses to sign when the policy files or config changed since prepare (`prepare_required`), when there is no packet yet, when the packet describes older inputs, or after publication. The signoff stores `signoff_version: 1`, both current digests, `packet_sha256` (the sha256 of the packet bytes at signing time), and per role `approved_by`, `approved_at`, and an optional `note`. Editing the packet or any bound content after signing refuses publication. A signature is a recorded attestation, not a cryptographic signature.

### Config rules

- `config_status` may be omitted, as in BFCL generation configs. `config_status: template` is refused, and so is any value containing `REPLACE_ME`. `config_resolved.yaml` always records `config_status: resolved`.
- YAML with duplicate keys, non-string keys, dates, or other non-JSON values is refused, so the config hash is well defined.
- Unknown keys are rejected at every level. Choices, lists, shares, and whole-number budgets are type-checked, and each mix must add up to 1.
- `proposals.moderation_targets` and `proposals.decision_actions` accept the same values as `scope.targets` and `decision.action` in Policy Rule IR. The default targets are `user_input` and `assistant_output`. The default actions are every Policy Rule IR action. Narrow the lists when a run should not label those targets or actions.
- `languages` is a mapping with `source_language` and `target_languages`, matching the main safety design. It fills the corresponding proposal fields; if both forms are present, they must agree. A source language outside the built-in contextual privacy profiles (`de`, `en`, `es`, `fr`, `ja`, `ko`, `vi`, and `zh`) must provide all four locale label lists under `privacy.additional_labels` before ingest. The required kinds are `credential`, `national_id`, `bank_account`, and `date_of_birth`; values are literal labels, never regular expressions.
- `generation.decision_mix` may give a share only to an action listed in `proposals.decision_actions`. Narrowing `decision_actions` to `[allow, block]` therefore also needs `caution: 0` and `escalate: 0` in the mix.
- `quality_gates` is optional. It checks minimum counts only after label validation, exact deduplication, and splitting; it never generates duplicate filler cases. Supported gates are `min_total_cases`, `min_development_cases`, `min_private_test_cases`, the scalar `min_cases_per_policy`, and per-label mappings `min_cases_per_action` and `min_cases_per_language`. `on_failure` is `silver` (default) or `fail`. Counts must be non-negative whole numbers. Action keys are canonical Policy Rule IR actions, compared case-insensitively. Language keys are Unicode NFC-normalized and compared case-insensitively; a code-switched case counts for each of its language tags, including comma-separated leftovers. Duplicate keys that fold to the same action or language are refused. An unmet gate is recorded in `split_manifest.json`, shown in `review.html`, copied to `quality_report.json`, and blocks Gold with publication reason `minimum_dataset_quality_not_met` after coverage and split completeness have already been applied. `on_failure: silver` still publishes that Silver benchmark. `on_failure: fail` still writes the review packet, then refuses publication after sign-off (exit 1). Preview stays Preview. Before review, generate may backfill unscheduled **distinct** coverage cells into room left when scheduled rows did not survive validation or exact-dedup. It skips `translation_required`, `span_missing`, and `max_turns` holds, skips a cell whose identity already exists on the plan or in the surviving dataset, skips every cell of a policy whose cases were rejected for a rule-level reason (`quote_not_in_span`, `span_missing`, `evidence_mismatch`, `evidence_missing`), and never quote-copies the source sentence to inflate counts. A round that does not lower the total gate shortfall (for example because a completed contrast family rejected a row that had passed, or the new split moved cases) is rolled back and backfill stops. Required cells already on the plan are not dropped to make room. Minima that the remaining distinct cells and caps cannot satisfy stay unmet.
- `generation_model_config`, `judge_model_configs`, `evaluation`, and `languages` must not contain credentials. The same key names BFCL refuses, such as `api_key`, `token`, `authorization`, or anything ending in `_api_key`, stop the run before anything is written. Provide credentials through the provider environment.
- `input_dir` and `output_dir` must not be the same directory or inside each other.

The checked-in `safety/config/default.yaml` uses `stage: prepare`, so copying it first performs local ingestion and privacy checks before any model-backed generation.

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
- `operator_source: model_evidence` means extraction grounded the model's per-condition operator in an exact source quote. `default_any_of` is valid only on `block_if` with `any_of`. `single_condition` is valid only on a group with exactly one positive condition, where `all_of` and `any_of` have the same meaning.
- `severity_basis` records why the severity was chosen, for example `rubric:explicit_prohibition` or `source:pre_authored`. `paired_with` names the rule a paired allow rule relaxes.
- Missing `version`, `owner`, `effective_date`, domain, or jurisdiction is `unspecified`. An empty domain or jurisdiction list is the same as missing. Jurisdiction `global` is refused on a stored rule. Normalization clears it, including a padded value, a single string, or the singular key `jurisdiction`, and sends the rule to review.
- An `eligible_for_generation` rule has a confidence and no `unresolved_conditions`, no `unsupported_fields`, and no testability other than `testable`.
- Customer-specific fields go under `extensions`; any other unknown key is refused.

`normalize_candidate(raw)` normalizes pre-authored or programmatic input and never raises; it returns the rule, the flags it raised, and validation errors. Its legacy bare-list interface can resolve source wording from `provenance.source_quote` or `operator_text`:

| Wording | Result |
| --- | --- |
| "đồng thời", "all of the following", "toutes les conditions suivantes", "同时成立", and the other listed AND frames | `all_of`, `operator_source: explicit` |
| "một trong các", "any of the following", "以下任一条件", and the other listed OR frames | `any_of`, `operator_source: explicit` |
| "không có điều kiện nào", "none of the following", "aucune des conditions suivantes", and the other listed NONE frames | `none_of`, `operator_source: explicit` |
| More than one kind of operator wording | Moved to `unresolved_conditions`, so `needs_decision` |
| A bare "or", "hoặc", "ou", "oder", "both", "either", "cả hai", or "at the same time" | Not an operator |
| No wording, `block_if` with one condition | `any_of`, `operator_source: default_any_of` |
| No wording, `block_if` with several conditions | Moved to `unresolved_conditions`, so `needs_decision` |
| Any other condition with exactly one entry | `any_of`, `operator_source: single_condition` |
| No wording, any other condition with several entries | Moved to `unresolved_conditions`, so `needs_decision` |

A shared source quote supplies an operator only when the rule has one condition. For a rule with several conditions, a programmatic caller passes field-specific evidence such as `operator_text={"block_if": "..."}`. Model extraction uses the stricter per-condition protocol described below and never applies this legacy `default_any_of` fallback. A model self-score is dropped at the top level and inside `rule`, `provenance`, `decision`, and `scope`. A reviewer `rejected` status is kept. Every other status waits for routing. A candidate that is not a mapping, whose `unresolved_conditions` is not a mapping, or whose evidence mapping has an unknown condition returns errors and does not raise. Stored lists and mappings cannot be changed in place.

### Action order

When several rules match one case, `aggregate` (`safety/actions.py`) returns the strongest action:

```text
escalate > block > require_confirmation > rewrite > redact > caution > allow
```

The primary policy is the rule that demanded the winning action and has the highest severity, then the lowest `policy_id`. `contributing_policy_ids` lists every matching policy, winning action first. `overridden_policy_ids` lists policies whose action lost. No matching rule gives `not_applicable`. By default, `expected_action` evaluates only `eligible_for_generation` rules. `include_review=True` also evaluates `needs_decision` rules for a review preview; rejected and quarantined rules never apply. The low-level function expects callers to pass applicable rules. Coverage and scenario labeling filter participants by actor, action, object, domain, jurisdiction, language, audience, and moderation target, treating `unspecified` or an empty scope as a wildcard. A rule whose exception is present does not match. `escalate_if` maps to `decision.critical_action` (default `escalate`), `block_if` to `decision.violation_action` (default `block`), `require_confirmation_if` to `require_confirmation`, and `allow_if` to `allow`. `ambiguous=True` uses `decision.ambiguous_action` only after at least one rule condition matches. `ambiguous_rules` does that for the named policy ids and leaves every other rule on its normal action. Unresolved conditions are never evaluated.

### Confidence and routing

`route_rule` (`safety/confidence.py`) stores the confidence that Python computes. A model self-score is never an input.

```text
confidence = 0.36 quote_match + 0.28 schema_completeness + 0.28 pass_agreement
           + 0.08 operator_explicitness
```

Each signal is a score from 0 to 1. `pass_agreement` counts as 1 when only one extraction pass ran. With every signal at 1 except `operator_explicitness: 0.5`, confidence is 0.96. `explicit`, grounded `model_evidence`, and `single_condition` score 1. A multi-leg `default_any_of` block scores 0. Evidence that a condition appears in the source is a pass-or-fail check, not a weighted signal. Section labels are not scored.

| Condition | Status |
| --- | --- |
| Quote not found verbatim in its span, confidence below 0.70, or testability `needs_clarification`, `non_testable`, or `rejected` | `quarantined` |
| Confidence from 0.70 up to but excluding 0.90, an unresolved operator, an unsupported field, a conflict, unassessed testability, or `partially_testable` | `needs_decision` |
| Everything else | `eligible_for_generation` |

`quote_in_span` normalizes NFC, curly quotes, zero-width characters, and whitespace. A quote in a spaced script must fall on word boundaries. A quote of at least two letters in a script that does not separate words may match as a substring. One changed word still fails. `route_rule` treats its deterministic `quote_found` boolean as the only authority for the exact-match score, replacing any stale score in `ConfidenceSignals`. Prefer `route_rule_from_span`, which verifies `provenance.source_quote` and routes in one operation. A `rejected` rule stays rejected. A `block_if` stored as `default_any_of` is not a routing reason. The thresholds are defaults and can be passed as `Thresholds(eligible=..., decision=...)`.

### Rule extraction and routing

`generate` turns the ingested policy into routed rules before it looks at the review packet (`safety/rules.py`). The model only proposes; Python decides what is stored.

**Evidence windows.** Each span of a safe text document becomes one window (`safety/windows.py`): its heading path, the clause, the clause before and after it in the same section, and the glossary definitions the clause uses. Windows are built from redacted text only. The whole document is never sent. A window already cited by a pre-authored rule with `quote_found` is not sent to the model.

**Model.** Extraction uses `generation_model_config` through Data Designer, the same structured-output path BFCL uses (`alias`, `provider`, `model`, optional `base_url`, `api_key_env`, and `inference_parameters`). The key itself is read from the environment variable named by `api_key_env` and is never written to the run. A policy with text clauses that no pre-authored rule cites, and no `generation_model_config`, or one missing `alias`, `provider`, or `model`, stops with `invalid_config`. A run whose remaining clauses are empty needs no model.

**Cache.** Each reply is cached in `stage_cache/extraction/`, keyed by prompt version, both prompts, model identity (provider, model, base URL, inference parameters, never credentials), pass number, and window. A rerun on unchanged input sends nothing. A malformed reply is not cached, so the next run asks again. A reply in which any candidate fails the reply schema is used for this run, so its valid candidates still become rules, but it is not cached either; `policy_routing_log.json` counts it as `not_cached`. When the model returns two answers for one window, neither is used and the window is asked again. One bad window does not discard the windows that succeeded. The run stops with `model_unavailable` only when the model cannot be called, or when every window in the first pass fails.

**Checks on each candidate.** A candidate is validated on its own, so one bad candidate does not drop the others.

| Check | Result |
| --- | --- |
| The candidate does not match the reply schema | Listed as `malformed_candidate`, not a rule |
| The quote is not in the clause but is in a neighboring clause | Listed as `quote_in_neighbor`; the neighbor's own window extracts it |
| The quote is in neither | Kept with `quote_found: false`, so `quarantined` |
| A condition or exception has evidence in neither the clause, a neighboring clause, nor a structural definition | `unsupported_fields` gets `rule.<class>.<name>` or `exceptions.<name>`, flag `condition_not_grounded`, so `needs_decision` |
| No condition remains | Listed as `no_conditions`, not a rule |
| `expected_action` is not produced by any condition, or an action is not in `proposals.decision_actions` | `needs_decision` with that reason |
| A multi-condition class omits `condition_operators`, or its evidence is not an exact part of `source_quote` | Moved to `unresolved_conditions`, so `needs_decision` |
| Grounded model evidence names an operator and deterministic wording recognizes the same operator | Stored with `operator_source: model_evidence`; flag `operator_keyword_confirmed` |
| Deterministic wording recognizes one different operator | Moved to `unresolved_conditions`; flag `operator_evidence_conflict` |
| Deterministic wording sees more than one operator frame | Moved to `unresolved_conditions`; flag `operator_keyword_ambiguous` |
| `binding_evidence`, `conditional_evidence`, `multi_turn_evidence`, or `code_switch_evidence` is not an exact part of `source_quote` | The hint is ignored; flag `signal_not_grounded`. It does not add `unsupported_fields` |
| Deterministic wording does not recognize the language | The grounded model operator is kept; absence of a keyword match does not overrule it |
| Legacy `operator_hint` is present | Flag `legacy_operator_hint_ignored`; only per-condition grounded evidence is authoritative |

The structured model output uses `condition_operators.<class>.operator` plus `evidence`, copied exactly from `source_quote`. One positive condition is canonicalized to `any_of` with `single_condition`; a proposed single-condition `none_of` still needs grounded evidence because it reverses the meaning. All-Latin condition names are folded to snake_case ASCII (`Bỏ qua mọi quy định` becomes `bo_qua_moi_quy_dinh`). A name containing another script keeps its Unicode letters, numbers, and mixed-script suffixes. Punctuation-only or digit-leading text is stored as `c_` plus a hash and goes to review. Two source names that normalize to the same identifier also go to review instead of silently losing a leg. Exact evidence in an unsegmented non-ASCII script can ground without whitespace word boundaries. Multiple atomic rules from one quote remain separate when their conditions or operators differ. A clause may also return rule-specific domain, jurisdiction, language, audience, and moderation target values with exact quote evidence; grounded values override run scope for that rule. Run-level `proposals.domain` and `jurisdiction` are fallbacks only when they are not `unspecified`; absent dimensions are stored using Policy Rule IR's canonical `unspecified` wildcard. Ungrounded scope is unsupported and goes to review. A model self-score, `confidence`, or severity hint is removed before anything is cached.

**IDs.** `policy_id` is `<DOCUMENT>-<SECTION>-<HASH8>`, where the hash covers document id, section, normalized quote, expected action, condition classes, operators, and condition names. Two atomic rules extracted from the same quote therefore keep distinct stable IDs, while an identical candidate is still a duplicate. An allow-only rule is paired automatically only when its section has exactly one acting rule: its ID is `<partner>-ALLOW` (`-ALLOW-2` for a second) and `paired_with` names the partner. With several possible acting partners, the allow rule remains unpaired and goes to review instead of being attached by document order. Pairing applies to extracted rules only. A pre-authored rule keeps the `policy_id` its author gave it.

**Severity rubric.** The first matching line wins, and `severity_basis` records it:

| Rule | Severity | `severity_basis` |
| --- | --- | --- |
| Allow rule | low | `rubric:explicit_allow` |
| An open, source-language `risk_category`, a grounded `signals.binding_evidence`, and a `critical_risk_basis` in the four rubric classes | critical | `rubric:binding_evidence+<basis>` |
| Escalation | high | `rubric:escalation` |
| Block with no exception, no `none_of`, and no grounded `signals.conditional_evidence` | high | `rubric:explicit_prohibition` |
| Anything else | medium | `rubric:conditional_restriction` |

A pre-authored rule keeps its severity with `severity_basis: source:pre_authored`. One without `testability` is read as `testable`, with flag `testability_from_author`.

**Signals.** `schema_completeness` is the share of: actor, action, and object stated; a stored condition; testability set; expected action produced. `operator_explicitness` is the share of condition groups stored as `explicit`, grounded `model_evidence`, or `single_condition`. A one-leg `default_any_of` also scores 1 for the legacy normalization path. A multi-leg `default_any_of` and an unresolved group score 0. The scorer does not match hedge words. A condition whose evidence is missing from the clause, a neighboring clause, and any structural definition is `needs_decision` through `unsupported_fields`, not through the confidence weights. A structural definition is a table row with two or more cells, or a bold term followed by a colon or dash. The verb glossary is sent to the model as context and is not a grounding source. Section labels are not part of confidence.

**Second pass.** A window runs a second, independent pass when the profile is `regulated` or `high_assurance`, or when one of its rules with a found quote scores below 0.90. Rules are matched across passes by quote only. `pass_agreement` is the share of four checks that agree: expected action, condition classes, condition names, and the operator of each class. An operator disagreement explicitly sends the first-pass rule to review, even when the weighted confidence remains high. A rule pass two does not find gets 0. A rule only pass two finds is listed as `second_pass_only` and is not stored. If the second pass fails, the first-pass rules stay and go to review; the run does not stop.

**Conflicts.** `safety/conflicts.py` evaluates each rule's triggers, including the empty minimal witness for `none_of`, and the combined triggers of related rule pairs. Rules are related when they share a condition name, a source section, or an object that both quotes use. They are combined only when actor, action, object, domain, jurisdiction, language, audience, and moderation target can overlap; unspecified fields are wildcards. Where several actions fire, `policy_graph.json` records a node:

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

Required cells are scheduled before optional cells. Every policy receives its first anchor before additional cells. Remaining ties minimize the error against `generation.decision_mix`, `case_family_mix`, and `turn_mix`, while policy counts stay round-robin; stable hashes replace lexical policy-id ordering as the final tie-break. The required kinds are:

1. One clear violation, when the rule has a block, escalation, or confirmation condition. Its facts are the legs of the strongest action. A `none_of` group contributes absent legs. The expected action is aggregated only across eligible rules that overlap the anchor rule's event dimensions.
2. One clear allow. When the rule has `allow_if`, its facts are the legs that satisfy that group. Otherwise the facts are empty, plus any `none_of` leg that would otherwise fire. The expected action is `allow` when no rule matches, and the aggregated action when one does. `paired_policy_id` names the paired rule. The cell is flagged `paired_rule_not_eligible` when that rule is not eligible, and `paired_rule_not_allow` when it is eligible but has no `allow_if`. With no partner, the flag is `allow_unpaired`.
3. One boundary case. When the rule names `decision.ambiguous_action`, the facts are the clear-violation facts and that rule is evaluated as ambiguous. When the rule names none, the cell is kept with a null action and flagged `boundary_action_unspecified`.
4. One exception case when the rule lists exceptions. The facts are the clear-violation facts plus the exception names. The expected action is `allow` when those facts match no rule.
5. One missing-leg case for each leg of a multi-leg `all_of` acting group. When that group is not already the clear violation, the plan also adds one `all_legs` case. Every `any_of` leg and every `none_of` acting condition class also gets a minimal `condition_path` case unless it is identical to the clear violation. Secondary block, escalation, and confirmation paths are therefore required even when another condition class owns the strongest action.
6. One pending case for each additional declared language and each `proposals.target_languages` entry that is not already the rule's source language. Extracted rules store only `proposals.source_language` in `scope.languages` unless the clause itself grounds a language. Target languages are translation work, not languages the clause declared. A target-language cell is a blocking `translation_required` gap until real translated text exists.
7. One pending code-switch case when the rule stores grounded `signals.code_switch` and `scope.languages` lists at least two languages. It is a `translation_required` gap. A phrase such as "code-switch" does not create the cell.
8. One multi-turn case when the rule stores grounded `signals.multi_turn_turns`. `turns` is that count, and `budget.max_turns` caps it. A numbered list or a phrase such as "multi-turn" does not create the cell. A cap below 2 cannot hold the cell, so the cell is a gap with reason `max_turns` and the plan is Silver. A rule that cites a `source_span_id` missing from the source index stops the plan. A rule with no `source_span_id` holds every required cell as `span_missing` and the plan is Silver, so coverage cannot read Gold when later stages cannot point at a source span.

Optional cells come after the required cells, and only while the caps still have room: one adversarial case when `generation.modules.adversarial` is true, one actor-swap case when `political_symmetry` is true, and one persona case when `generation.modules.persona` is true. `confirm_if_detected` does not add actor-swap cases. Production replay adds none in this increment, reported as `not_configured`, because the safety config has no evidence paths yet. An optional cell left out does not block Gold.

A required cell that does not fit is not dropped. `coverage_gaps.json` lists it with its slot id and a reason: `budget_cap`, `rule_cap`, `max_turns`, `translation_required`, or `span_missing`, with `blocks_gold: true`. The plan's `certification` is then `silver`, and `gold_cases_needed` is the full required count. When every required cell of the eligible rules is scheduled, `certification` is `gold` even if optional cells remain or other rules were excluded. `run_record.json` repeats the certification, the counts, `missing_required`, and `excluded_rules` under `coverage`. After the dataset stage, if quality gates are unmet and surviving rows are below the case cap, generate may promote remaining generatable gap cells onto the plan, mark those rows `backfill: true`, and rebuild scenarios and the dataset. The backfill identity is `(policy, language, action, kind, present_legs, absent_legs)`; a matching cell is a quote-copy and is skipped. The case cap is applied to surviving rows, so a rejected plan row can be replaced by a different cell without dropping required cells that are already scheduled.

A slot id joins the policy id, the kind, and any extra part with `/`. A contrast family id uses the same escaping after `contrast/`. A `/` or `\` inside a part is escaped. Read `policy_id`, `kind`, `language`, and `absent_leg` from the row. Do not split the slot id on `:`.

Contrast families point at those slot ids. An acting `all_of` group has one family per leg. The block member is the case that contains every leg of that group: the clear violation when that case is exactly the group, and the group's `all_legs` case otherwise. The allow member is that leg's missing-leg cell. A rule with no missing-leg cell has one family whose allow member is the clear allow cell. Families do not add cases beyond those cells. `changed_dimension` is the absent leg, `paired_allow` when the partner is an eligible allow rule, `paired_rule` when the partner is not an allow rule, or `benign_neighbor` when the rule names no partner.

| File in `stage_cache/` | Contents |
| --- | --- |
| `coverage_plan.jsonl` | Scheduled cells, required rows first. Each row names the rule, the kind, and the expected action. It is not case text. |
| `coverage_gaps.json` | Certification, counts, every unscheduled cell, and the contrast families. |

`cache_index.json` records `stages.coverage_plan` as `running`, then `completed`, or `failed`. Any failure while planning, including a damaged span, a repeated span id, or an error inside the planner, marks the stage `failed` and deletes `coverage_plan.jsonl` and `coverage_gaps.json`. The next generate removes the plan at the start of rule extraction, so a failed run does not keep a plan for rule files it deleted.

### Scenario generation

After the coverage plan, `generate` writes one conversation for each scheduled cell. The planner's slot is the case id. The stage copies the rule, the expected action, and the source quote from the eligible rule. It does not call a model and it does not add a condition the rule does not already name.

A missing-leg cell removes exactly one leg of the rule's `all_of` group and keeps the others, in the group's order, then adds any `none_of` leg the cell is not holding absent. The contrast family points at that same case id. The conversation is not written a second time. When both members of a family are present, the block case must contain every leg of that group and the allow case must remove only the family's named leg. The stored action must be the aggregated action of the case's facts. A family that does not have that shape stops the build. A member slot that is neither in the plan nor in the gap list stops the build. A case whose action is `allow` is listed in `overrefusal_cases.jsonl`. That file stores the case id only.

A `none_of` violation records those legs in `absent_legs`. The allow case for a block rule keeps the same legs present, so the `none_of` group does not fire. An allow rule's clear-allow case uses the `allow_if` legs. An exception case copies the rule's exception names into the facts. Every scored turn's text is the source quote; legs and exceptions stay in the case fields. A multi-turn cell has one scored user turn for each planned turn, and each turn carries the same quote. Every scored turn carries the cell's expected action. A boundary whose action is unspecified stores null on its scored turn. The generator does not fill that action in. A boundary that names an ambiguous action uses the clear-violation facts. Every other plan row needs an expected action from the action order.

`run_record.json` stores the counts under `scenarios`. This command writes the seeds. A plan row that names an unknown policy, an unknown action, or legs the rule does not have stops the build, and the scenario stage is `failed`. A proposal handed to the builder may replace a seed only when its rule, quote, action, and legs match that seed, in the same order, and every turn uses that seed's sentence. A proposal with an unknown policy id, a changed meaning, or a missing-leg set that drops another required leg or keeps the removed leg is written to `scenario_rejections.jsonl` and does not become a case. A slot with no proposal still receives its seed.

| File in `stage_cache/` | Contents |
| --- | --- |
| `scenario_candidates.jsonl` | One accepted case per scheduled slot: policy id, source evidence, legs, and a decision on every scored turn. |
| `scenario_rejections.jsonl` | Rejected proposals and plan cells that do not match an eligible rule. |
| `contrast_families.jsonl` | The plan's contrast families, each member pointing at one case id. |
| `overrefusal_cases.jsonl` | Pointers to accepted allow cases. The conversation stays in `scenario_candidates.jsonl`. |

`cache_index.json` records `stages.scenarios` as `running`, then `completed`, or `failed`. Any failure deletes the four scenario files and leaves the coverage plan in place. The next generate removes those files at the start of rule extraction.

### Dataset selection

After the scenario seeds, `generate` checks every accepted case, drops exact duplicate rows, and writes the development and private-test splits. This stage does not call a model. The preliminary label is the aggregated action of the case's facts. A stored action that differs is rejected. The case keeps the action it arrived with.

A case is accepted only when its policy id is an eligible rule and its evidence names that rule's source span. The span must be in `source_index.json`, and the quote must appear in that span. A rule that cites no span cannot place a case in the dataset. Every turn's text is the rule's source quote. Legs, absent legs, and exceptions stay in the case fields, so the turn is not wrapped in one language's scaffolding. A rewritten turn is rejected with `turn_text_mismatch`. Every scored turn carries the case action. A multi-turn case has a decision on each scored turn. A boundary whose action is unspecified stays unlabeled.

When one member of a complete contrast family fails, the other members are rejected with `contrast_pair_broken`. A family counts only when its members change the facts. A missing-leg family must remove exactly the named leg. Two copies of the same facts are duplicates, so the family does not keep both rows. A family that still has an unscheduled member is not a pair, and the member that was accepted can stay. Two families that share a case stay together: if that shared case is rejected, both families are rejected. A persona, multi-turn case, exception, or labeled boundary that still carries the removed violation's conditions is rejected with `attack_removed`. An unlabeled boundary has empty facts, so it can stay.

An exact duplicate has the same policy, the same action, the same language, the same kind, the same legs, and the same canonical turn text. The language is the case's own language, or the rule's first declared language when the case names none. The lowest case id is kept, and the dropped id is listed in `split_manifest.json`. Members of one real contrast family are not collapsed. A row with a different action, or a different kind such as a translation, code-switch, actor swap, adversarial case, persona, or missing leg, stays as its own row. Semantic deduplication is not configured.

The split keeps a whole group on one side. A group is a real contrast family, the cases that repeat its violation, and any case whose conditions still contain that violation's conditions. An exception and a boundary that uses the violation's facts stay with the violation. An unlabeled boundary stays in its own group. A clear allow stays with the violation when the contrast family names it. Groups are processed largest first and greedily assigned to minimize imbalance in case count, policy, action, kind, and language; case IDs only break deterministic ties. Both lists are non-empty when two or more groups exist. `balance.by_split` records the achieved distribution. One group stays in development, `complete` is false, and `reason` is `one_group`. When no case is accepted, both lists are empty and `reason` is `no_cases`. A public test split is not assigned.

`run_record.json` stores the counts under `dataset`. Case ids for both splits are in `stage_cache/split_manifest.json`.

Optional minimum dataset gates are configured independently from the maximum generation budget:

```yaml
quality_gates:
  min_total_cases: 100
  min_development_cases: 40
  min_private_test_cases: 40
  min_cases_per_policy: 4
  min_cases_per_action: {allow: 20, block: 20}
  min_cases_per_language: {vi: 20}
  on_failure: silver
```

The gate result contains the configured requirements, `on_failure`, observed total/split/slice counts, and one structured failure per unmet minimum. The per-policy gate covers every eligible policy that still remains after review, including a policy with zero retained cases. A withheld policy is not scored against `min_cases_per_policy`. A damaged requirement type aborts instead of coercing booleans or strings through `int()`. Minimums do not override `budget`. If surviving rows left room under the caps, generate backfills distinct unscheduled cells rather than copying quotes. If the remaining distinct cells still cannot satisfy the minima, `on_failure: silver` publishes Silver so the shortfall is explicit, and `on_failure: fail` leaves the review packet in place and does not publish.

| File in `stage_cache/` | Contents |
| --- | --- |
| `label_checks.jsonl` | One row per candidate: the stored action, the aggregated action, and whether they agree. |
| `validated_cases.jsonl` | Cases that passed the checks, including rows later removed as duplicates. |
| `rejected_cases.jsonl` | Cases rejected by a check, a broken contrast pair, or exact deduplication. |
| `validation_report.json` | Counts, rejection reasons, and the deterministic judge summary. The judge method is `deterministic` and `model` is null. |
| `deduplicated_cases.jsonl` | The rows that enter the split. |
| `split_manifest.json` | `development` and `private_test` case ids, the groups, and the dropped duplicate ids. |

`cache_index.json` records `stages.dataset` as `running`, then `completed`, or `failed`. Any failure deletes the six dataset files and leaves the coverage plan and the scenario files in place. The next generate removes those files at the start of rule extraction.

### Review packet and decisions

After dataset selection, `generate` writes three files in the run directory:

| File | Contents |
| --- | --- |
| `review.html` | A local, escaped HTML summary. Every unresolved rule, unresolved conflict, blocking gap, and excluded rule is listed. Only representative cases are capped globally by `budget.max_review_items`; cases are selected round-robin across policies with a stable hash tie-break. |
| `review_decisions.csv` | One row per rule or conflict card. Decisions are `pre_accepted`, `pending`, `approve`, `reject`, or `edit`. |
| `publication_review_packet.yaml` | The complete review claim, coverage result, cards, representative examples, and sha256 bindings for every publication input, including the decision sheet and `review.html`. |

Eligible rules begin as `pre_accepted`; rules that need a decision and unresolved conflicts begin as `pending`. The reviewer records `approve`, `reject`, or `edit`. `pre_accepted` is only valid on an eligible rule; writing it on a pending card is refused. An `edit` requires a non-empty note in this increment and withholds the affected rule, as does `reject`. Withholding one member removes every member of its contrast family. Conflict and glossary cards use a content hash, not a list index, so a decision cannot attach to a different conflict when the set changes. Unknown or repeated card ids, unknown decisions, CSV metadata that does not match the current card, a malformed CSV, and stale decision rows stop the run instead of being ignored.

The next `generate` rebuilds deterministic stages, imports the decision sheet, and writes a canonical packet. If this changes the packet, existing signatures no longer match and the current packet must be signed again. A signed packet with a pending card still exits 3. Approved unresolved rules or conflicts remain documented limitations because approval alone does not invent a condition operator or conflict resolution; such a publication is Silver.

### Benchmark compilation

When the decision sheet is complete and the required signatures match the current packet, `generate` compiles:

| File | Contents |
| --- | --- |
| `benchmark.jsonl` | Every retained case with `split: development` or `split: private_test`. |
| `benchmark_development.jsonl` | Retained development cases. |
| `benchmark_private_test.jsonl` | Retained private-test cases. |
| `quality_report.json` | Certification, coverage reason, case counts, gaps, withheld policies and cases, unresolved review items, split result, and provenance digests. |
| `run_manifest.json` | Publication commit. It binds the packet, decision sheet, review page, signoff, benchmark files, and quality report. `files` must list every compiled output. |

Compilation refuses duplicate case ids, unknown split ids, a case missing from both splits, overlap between splits, or any file whose digest differs from the signed packet. Gold requires assisted or expert mode, complete Gold coverage, all configured minimum dataset quality gates, a complete non-empty development/private split, no withheld cases, and no unresolved rule or conflict. A documented coverage gap, unmet minimum, incomplete split, review limitation, or reviewer removal publishes Silver. Preview mode publishes Preview without requiring a signoff and remains non-comparable.

After publication, another `generate` is idempotent: it verifies the manifest, the run-record claim, and the Gold/Silver result implied by the signed packet, then returns that status. Editing the certification in the manifest cannot turn Silver into Gold. Editing or deleting the packet, decisions, HTML page, signoff, benchmark, or quality report is detected and refused. Running `prepare` starts a new review cycle and removes the previous publication commit, compiled files, and evaluator metrics.

### Evaluation

`stage=eval` scores a published benchmark. It does not publish, and it does not use the Gold or Silver exit codes. The private-test split is preferred; the combined benchmark is used only when the private split is empty. It writes:

| File | Contents |
| --- | --- |
| `evaluator/adapter_smoke.json` | Credential-free contract preflight for each live adapter, using only a fixed synthetic message. |
| `evaluator/path1_metrics.json` | Direct guard-model scoring through a scripted or live adapter. |
| `evaluator/path2_metrics.json` | Complete Guardrails-configuration scoring from a log or live endpoint. |

The metric files are never the same path. Before reading or sending private-test rows, each live adapter must parse one fixed synthetic contract-smoke request. The smoke artifact records only status, canonical decision, latency, credential-free target identity, and rail count—not prompt text or secrets. A failed smoke stops evaluation.

Both paths report precision/recall, false-positive and false-negative rates, binary macro F1, exact-action accuracy, parse reliability, per-policy and per-language slices, throughput, and p50/p95/p99 latency. Path 2 also records which rails ran and how long they took. Timeout, rate limit, authentication failure, malformed reply, model error, and unsupported capability are counted separately; none is treated as allow or block. Events are scored per labelled turn. Requests contain model-facing messages and identifiers, never the expected action.

The offline adapters remain useful for deterministic CI:

```yaml
evaluation:
  path1:
    adapter: test
    responses:
      "case-1#1": {decision: block, latency_ms: 12}
  path2:
    adapter: mocked_guardrails
    log: rails.jsonl
```

`adapter: test` reads an inline `responses` mapping or a JSON object named by `file`. A mocked Guardrails log must contain exactly one JSONL row for every scored event. Relative `log` and `file` paths resolve from the config directory, then from the run directory.

For a live OpenAI-compatible guard, configure the complete chat-completions URL:

```yaml
evaluation:
  path1:
    adapter: openai_compatible
    endpoint: https://guard.example.com/v1/chat/completions
    model: customer-guard
    evaluation_mode: native_guard
    system_prompt: null
    api_key_env: CUSTOMER_GUARD_API_KEY
    timeout_seconds: 30
    max_retries: 2
    capabilities: [user_input, assistant_output]
    decision_map:
      safe: allow
      unsafe: block
```

Path 1 records one explicit `evaluation_mode` in metrics and smoke metadata:

- `native_guard`: send the conversation unchanged; `system_prompt` must be null, empty, or omitted.
- `fixed_system_prompt`: prepend the same required non-empty `system_prompt` to every event.
- `canonical_classifier_prompt`: use the pipeline's built-in JSON-classification prompt. This is the backward-compatible default when both fields are omitted.

When `evaluation_mode` is omitted, the adapter infers it compatibly: an omitted `system_prompt` means `canonical_classifier_prompt`, null/empty means `native_guard`, and non-empty means `fixed_system_prompt`. Metrics store `system_prompt_sha256`, never the prompt text, so runs can prove they used the same fixed prompt without disclosing it.

Override `content_path` when the OpenAI envelope stores content elsewhere, and `request_options` for non-identity generation parameters. The response may be a canonical action, a JSON object such as `{"decision":"block"}`, or an exact label declared in `decision_map`. Labels such as `safe`/`unsafe` or locale-specific verbs are never assumed; they must be listed in `decision_map`. Set `decision_format: first_line` when the model returns a decision token followed by extra lines. Parsing uses Unicode-normalized exact labels, never substring keyword matching. Optional `smoke_message` replaces the language-neutral contract probe.

For a live NeMo Guardrails service:

```yaml
evaluation:
  path2:
    adapter: nemo_guardrails
    endpoint: http://localhost:8000/v1/chat/completions
    config_id: vai-safety
    model: customer-chat-model
    api_key_env: GUARDRAILS_API_KEY
    decision_path: output_data.decision
    rails_path: log.activated_rails
    decision_map:
      deny: block
      pass: allow
```

The default `chat_completions` API format sends `messages`, optional `model`, and a nested `guardrails` object containing `config_id` and `options`. Set `api_format: legacy_chat_completions` for older servers that expect top-level `config_id`, `messages`, and `options`. By default, `options` requests activated-rail logs. `decision_path` and `rails_path` are dotted paths into the response. Only the value at `decision_path` is classified; a natural-language refusal is recorded as malformed rather than inferred as block or allow.

Path 2 live metrics use `evaluation_mode: guardrails_configuration`; replayed logs use `recorded_guardrails_log`. The deterministic Path 1 test adapter uses `scripted_responses`.

Credentials are read only from environment-variable names in `api_key_env` or `headers_env`; secret values are not written to metrics or error messages. A missing path is written as `not_run` with the reason. An unknown adapter, invalid configuration, damaged log, or mismatched event list stops evaluation.

Run either live or offline evaluation after publication:

```bash
PYTHONPATH=src python -m nemotron.steps.byob.scripts.run \
  --config /absolute/path/to/customer.yaml \
  --family safety \
  --stage eval
```

The week 2 end-to-end gate runs this path without network access. The complete fixture is pre-authored Policy Rule IR that cites a short generic source document, plus a quarantined secret. Generate does not call an extraction model because every source clause is already cited. The first generate writes the review packet and exits 3. Applying approved review decisions publishes Gold with exit 0. An intentionally small case budget publishes Silver with exit 2 and lists the omitted coverage on the review page and in the quality report. Both metric files are readable JSON and contain scored events.
