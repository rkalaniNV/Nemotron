# Skill Authoring Checklist

Use this checklist when creating or reviewing any workflow skill under `src/nemotron/steps/`.

The checklist combines:
- the Agent Skills open specification
- OpenAI Codex skill guidance
- Anthropic Claude Code skill guidance

Mark every item before treating a skill as ready.

## 1. Scope And Coherence

- [ ] The skill covers one coherent job, not a bundle of loosely related tasks.
- [ ] The skill is the right abstraction level for workflow use.
- [ ] Facts that should load in every session are **not** in the skill; those belong in project-wide agent context.
- [ ] Reusable procedures and playbooks **are** in the skill.

## 2. Required Packaging

- [ ] The skill lives in its own directory named after the skill.
- [ ] The directory contains `SKILL.md` at the root.
- [ ] Supporting code lives in `scripts/` when runtime behavior is needed.
- [ ] Heavy reference material lives in `references/`.
- [ ] Templates, example configs, and golden cases live in `assets/`.

## 3. Frontmatter

- [ ] `name` exists and matches the directory name.
- [ ] `name` uses lowercase letters, numbers, and hyphens only.
- [ ] `description` exists and clearly states what the skill does and when to use it.
- [ ] `description` is specific enough to trigger the skill correctly.
- [ ] `compatibility` is present only when environment requirements matter.
- [ ] `metadata` is sparse and useful.
- [ ] `when_to_use` is present when extra trigger guidance will improve routing.

Optional, use only when they help:
- [ ] `allowed-tools`
- [ ] `disable-model-invocation`
- [ ] `user-invocable`
- [ ] `argument-hint`
- [ ] `context: fork`
- [ ] `agent`
- [ ] model / effort overrides

## 4. SKILL.md Content

- [ ] `SKILL.md` is short enough to load cheaply.
- [ ] `SKILL.md` focuses on what the agent would otherwise get wrong.
- [ ] The main body is procedural, not generic advice.
- [ ] A clear default path is given.
- [ ] There is a `Gotchas` section for non-obvious failure cases.
- [ ] There is a validation section or equivalent self-check loop.
- [ ] The skill tells the agent when to load additional files.
- [ ] The skill provides defaults rather than a long menu of equal options.

## 5. Progressive Disclosure

- [ ] Long manifests, guides, and deep detail are not in `SKILL.md`.
- [ ] `SKILL.md` points to specific `references/` files for deeper detail.
- [ ] Supporting files are one hop away from `SKILL.md`, not deeply chained.
- [ ] Examples or templates are available when output shape matters.

## 6. Execution And Verification

- [ ] Runnable code is in `scripts/`, not embedded as long inline snippets.
- [ ] The skill gives the agent a way to verify success.
- [ ] Tests, screenshots, expected outputs, or golden cases exist where appropriate.
- [ ] A validator or test covers the skill layout and frontmatter.
- [ ] At least one real execution path has been exercised when the skill owns runtime behavior.

## 7. Trigger Quality

- [ ] The `description` and `when_to_use` text include the key request language users will actually say.
- [ ] The trigger is narrow enough to avoid false positives.
- [ ] The trigger is broad enough to catch real requests.
- [ ] The skill does not shadow another skill with overlapping purpose.

## 8. Workflow Integration

- [ ] Input types and output types are explicit.
- [ ] The skill names the runtime entrypoint or adapter module.
- [ ] The skill documents the expected artifacts.
- [ ] Shared workflow vocabulary is updated if the skill changes:
  - [ ] `types.toml`
- [ ] Skill-local routing and validation files are updated if the skill changes:
  - [ ] `patterns/index.yaml`
  - [ ] `eval/golden_cases.yaml`
  - [ ] `eval/skill_cases.yaml`

## 9. Cross-Tool Notes

For Codex:
- [ ] The skill can work as a focused playbook with optional supporting files.
- [ ] If Codex-specific metadata is ever needed, keep it additive rather than replacing the open format.

For Claude Code:
- [ ] Decide explicitly whether the skill should be auto-invokable by the model.
- [ ] Use invocation-control fields only when side effects or timing matter.
- [ ] Prefer skills over bloating always-loaded agent context with procedures.

## 10. Ready-To-Ship Gate

- [ ] Local asset tests pass.
- [ ] Any runtime smoke test passes.
- [ ] The skill follows this checklist without major exceptions.
- [ ] Any exceptions are documented in the PR or module notes.
