# SKILL: Sovereign AI Onboarding for Nemotron

## Purpose

Help new end users choose a productive sovereign AI customization path for Nemotron, validate infrastructure readiness, recommend baseline evaluation of Nemotron Nano V3 30B, and produce a first-day plan without committing planning artifacts.

This is the novice-safe entry point for sovereign customization. It is Nemotron-only and use it before jumping into stage-specific execution when the user is still figuring out what to run.

## When to Use

Use this skill when the user is new to the Nemotron customization workflow, says they want to build a sovereign model, or needs help choosing the right first step.

Also use it when the user needs help deciding whether they should start with customization instead of full training, or when cluster profile readiness is still unclear.

## Scope

- Support only `Slurm`, `Lepton`, and `Run:AI`
- Keep the interaction diagnostic rather than dogmatic
- Assume meaningful evaluation usually requires HPC-style remote execution
- Produce a local plan at `src/nemotron/customization_recipes/nemotron/planning/sai-onboarding-plan.md`
- Do not commit the generated plan and do not encourage the agent to commit it

## Intake

Ask these four questions:

1. Launcher: `Slurm`, `Lepton`, or `Run:AI`
2. Hardware shape, for example `8xH100 on 4 nodes`
3. Desired outcomes (multi-select):
   - `language adaptation`
   - `domain adaptation`
   - preserve or verify instruction-following quality
4. Autonomy mode: `guided`, `balanced`, or `autorun`

Default autonomy mode to `balanced` if the user does not choose one.

Do not widen intake beyond these four questions at the start. If more detail is needed, ask follow-up questions only after diagnosing the initial path.

## Diagnosis

If the user says they want a sovereign model, diagnose whether customization is the correct first move.

Redirect toward customization when:

- the real goal is language adaptation
- the real goal is domain adaptation
- the team lacks pretraining-scale data
- evaluation maturity is low

State the redirect in diagnostic terms. Example:

> Based on your adaptation goal, available data, and current evaluation readiness, customization is the correct first move.

Do not frame full training as the default. If the user is really asking for pretraining-scale work, explain the tradeoff clearly and note that this onboarding skill is meant to get them onto the highest-leverage sovereign customization path first.

## Infrastructure Gate

Treat infrastructure readiness as the first hard gate.

If there is no working `env.toml` or equivalent profile, stop other progress and use the closest reference under `planning/references/` to prepare a proposed draft.

Before writing `env.toml`, show the full proposed contents and get user confirmation.

Clearly separate:

- values inferred from the user
- values that require admin confirmation
- values that can only be validated at submission time

When choosing the closest reference, use only:

- `planning/references/slurm/`
- `planning/references/lepton/`
- `planning/references/runai/`

Do not guess that infrastructure is ready just because the user has GPUs. Launcher access, writable remote paths, container availability, and profile correctness matter more than raw hardware.

## `env.toml` Drafting Rules

When a draft is needed:

1. Pick the nearest launcher-specific sample under `planning/references/`
2. Fill in only values that can be inferred safely from the intake and follow-up answers
3. Mark unknown or risky fields plainly
4. Show the full proposed file contents before writing anything
5. Wait for explicit confirmation
6. Write the approved draft only after confirmation

If important fields are still unknown, keep the draft partial and label the missing items instead of inventing values.

## Evaluation Recommendation

Recommend baseline evaluation of Nemotron Nano V3 30B before tuning.

Explain that this is the default teaching path because it:

- establishes a baseline before customization
- helps preserve or verify instruction-following quality
- exposes infrastructure issues before expensive tuning work

This recommendation is strong by default, but it is not a hard block. If the user opts out, record that choice in the onboarding plan and continue with the best feasible customization path.

## Autonomy

- `guided`: explain and wait before meaningful execution
- `balanced`: run preflight checks, dry-runs, and small real runs when the environment appears usable
- `autorun`: proceed more aggressively, while still requiring explicit confirmation before writing `env.toml` or submitting major remote jobs

Use `balanced` as the default teaching mode. Regardless of mode, do not submit major remote jobs without checking that the environment appears usable first.

## Output Plan

Write a local markdown plan to `src/nemotron/customization_recipes/nemotron/planning/sai-onboarding-plan.md`.

The plan should include:

- the four intake answers
- the diagnosed recommendation and rationale
- whether customization was recommended over full training
- infrastructure readiness status
- whether `env.toml` exists, needs drafting, or needs admin validation
- whether baseline evaluation of Nemotron Nano V3 30B was accepted or declined
- a first-day sequence of concrete next steps
- exact next commands or dry-runs when enough information is available

The plan is a local runtime artifact. Do not commit the generated plan or encourage the agent to commit it.

## Working Style

Prefer the following sequence:

1. Ask the four intake questions
2. Diagnose the path
3. Gate on infrastructure readiness
4. Recommend baseline evaluation of Nemotron Nano V3 30B
5. Propose `env.toml` contents if needed, but only after showing the full draft and getting confirmation
6. Produce the local first-day plan

If the environment is not ready, do not pretend that tuning or evaluation execution is the next step. The next step is infrastructure completion.
