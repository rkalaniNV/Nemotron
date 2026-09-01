# BFCL workflow acceptance matrix

This file transcribes the acceptance criteria of the product workflow document into the
repository so that each one can name a test. Without the transcription the criteria live
only in a binary outside version control, and "backed by a named test" cannot be checked.

Source document: `BFCL_LLM_Generated_Workflow_Overview.docx`, section 13,
"Production-Readiness Acceptance Criteria (Pending Until Proven)".
Transcribed digest: `sha256:a0b5630aafdcb8184447f4737ec0e28dcff0b73872ce11579a3c8ee9d4fa1e1e`.

The digest identifies the revision this matrix was transcribed from. When the document
changes, recompute it, re-read section 13, and update the rows below. The digest is
recorded as provenance only; the document is not in this repository, so no test can
verify it.

## How this file is enforced

`test_bfcl_authoring_documentation.py::test_workflow_acceptance_criteria_are_backed_by_named_tests`
parses the table below and fails when a criterion names no test, or names a test
function that does not exist in `tests/steps/byob/`. Owning tests are listed as
`file.py::test_name`. A criterion may not be marked as owned by a file alone.

Ownership means the named test fails if the criterion is violated. It does not mean the
criterion is fully proven; the plan's `## Definition of done for the whole plan` section
records where coverage is still partial.

## Criteria

| ID | Criterion (section 13, verbatim) | Owning tests |
| --- | --- | --- |
| AC-1 | Existing manual Oracle Packs and generation tests remain unchanged and passing. | `test_bfcl_stages.py::test_manual_oracle_packs_require_no_flow_two_adapter_metadata`, `test_bfcl_stages.py::test_manual_gold_tier_ignores_adapter_and_certification_fields` |
| AC-2 | The LLM authoring pipeline emits only canonical pack files accepted by `load_pack()`. | `test_bfcl_mcp_release_review.py::test_freeze_atomically_seals_the_canonical_pack_and_lineage`, `test_bfcl_authoring_e2e.py::test_real_local_guided_publication_runs_stage_all` |
| AC-3 | The frozen pack is independently revalidated and reaches Gold before generation, and its source adapter is certified at the A2 tier by independently verified probe evidence. | `test_bfcl_authoring_release.py::test_v2_freeze_requires_a2_certification`, `test_bfcl_authoring_e2e.py::test_guided_publish_completes_stage_all_with_fresh_gold`, `test_bfcl_authoring_release.py::test_publication_refuses_fresh_validation_for_a_different_pack` |
| AC-4 | No target-model output, score, or identity is used as generation evidence. | `test_bfcl_authoring_refusals.py::test_refusal_schema_cannot_carry_model_output_or_scores` |
| AC-5 | Before production default rollout, held-out status must be explicit: required partitions are selected and redacted before authoring; not_applicable requires a reviewed reason; undeclared absence blocks before the first model call. | `test_bfcl_pack_drafting.py::test_missing_held_out_state_stops_before_first_model_call`, `test_bfcl_pack_drafting.py::test_v2_refuses_missing_held_out_proof_before_model_or_output`, `test_bfcl_authoring_held_out.py::test_clean_redaction_proof_is_signed_and_evidence_bound` |
| AC-6 | Every model call has a request hash, response hash, prompt version, model canonical ID, seed, and inference settings. | `test_bfcl_pack_drafting.py::test_a_full_drafting_run_writes_drafts_provenance_and_compiled_assertions`, `test_bfcl_pack_drafting.py::test_a_second_run_is_served_from_the_immutable_cache` |
| AC-7 | No automatic semantic repair may exist. Before production rollout, every model correction must be a new immutable, approved revision and must not modify schemas, oracle behavior, source identity, or fixtures. | `test_bfcl_authoring_cli.py::test_answer_commits_a_revision_that_must_be_reauthorized_and_reapproved`, `test_bfcl_authoring_questions.py::test_answers_create_a_new_evidence_revision_without_mutating_parent`, `test_bfcl_authoring_revisions.py::test_complete_revision_is_manifest_bound_and_immutable` |
| AC-8 | Nondeterminism, timeout, identity drift, evidence mismatch, unsafe probing, schema/oracle mismatch, and unclear business meaning fail closed. | `test_bfcl_authoring_revisions.py::test_resume_refuses_source_identity_drift`, `test_bfcl_authoring_revisions.py::test_resume_refuses_bound_artifact_drift`, `test_bfcl_authoring_refusals.py::test_refused_session_can_create_revision_only_with_bound_authorization` |
| AC-9 | Human approval binds exact source evidence, validation report, adapter identity, pack fingerprint, and freeze inputs; stale approval cannot authorize changed bytes. | `test_bfcl_authoring_revisions.py::test_resume_refuses_stale_approval`, `test_bfcl_authoring_generalized_review.py::test_generalized_review_verifies_common_trust_records_for_every_adapter`, `test_bfcl_authoring_generalized_review.py::test_review_refuses_a_validation_report_the_bound_config_did_not_write` |
| AC-10 | A generated tiny reference pack completes the existing stage=all workflow and produces verified final artifacts. | `test_bfcl_authoring_e2e.py::test_real_local_guided_publication_runs_stage_all` |
| AC-11 | The nine-run pilot is retained as descriptive rollout evidence; target-model evaluation and broader-domain replication remain required for causal or default-UX claims. | `test_bfcl_mcp_ablation_rollout.py::test_descriptive_protocol_cannot_be_promoted_to_causal`, `test_bfcl_mcp_ablation_rollout.py::test_missing_runs_force_descriptive_decision` |
