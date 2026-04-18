# Sovereign AI Customization

Adapt a Nemotron base model to your language, domain, and use case without training from scratch.

The Sovereign AI customization pipeline takes Nemotron Nano V3 30B as a starting point and produces a deployment-ready model tailored to your organization's language and domain. The full pipeline covers data preparation, continued pretraining, supervised fine-tuning, reinforcement learning, benchmark construction, evaluation, and quantization.

## Is Customization Right For You?

Most new users who say they want to "build a sovereign model" are better served by customization than by foundation-model training from scratch.

Customization is the right move when:

- Your goal is language adaptation (e.g., Swahili, Arabic, Bahasa Indonesia, Sinhala)
- Your goal is domain adaptation (e.g., health, legal, agriculture, public administration)
- You lack pretraining-scale data (tens or hundreds of billions of tokens)
- Your evaluation maturity is still developing

Full training from scratch is rarely the highest-leverage first step for a new sovereign AI program.

If you are not sure which path fits your situation, start with the [Sovereign AI Onboarding](./sovereign-ai-onboarding.md) skill. It asks four questions, diagnoses the right first move, and produces a concrete first-day plan.

## Evaluate Before You Tune

**Run Stage 5 evaluation on the base model before any customization work.** This baseline is the only way to know whether your tuning had a positive effect, no effect, or a negative effect.

If your use case has a matching benchmark — a domain MCQ set, a language-specific MMLU subset, or an existing industry evaluation — use it. If it does not, run general instruction-following benchmarks (MMLU, ARC Challenge, HellaSwag) as a proxy. These measure whether the model can still follow instructions and reason after tuning, which is a minimum bar regardless of domain.

The recommended sequence is:

```
Stage 5 (baseline) → Stage 1 → Stage 2 → Stage 5 (post-tuning) → compare
```

Without the pre-tuning baseline, you cannot tell whether a post-tuning score of 68% is an improvement, a regression, or where the model already was. With it, the comparison is unambiguous.

## The Customization Pipeline

```
[baseline eval] → stage0  →  stage1  →  stage2  →  stage3  →  stage4  →  stage5  →  stage6
(before tuning)   Data Prep   CPT        SFT         RL         BYOB       Eval        Quantize
                  (translate  (language  (SDG +      (DPO/      (MCQ       (standard   (INT4/
                   /curate)    /domain    instruct    GRPO)       benchmark  + BYOB)     FP8)
                               inject)    tuning)                 build)
```

| Stage | Name | Purpose |
|-------|------|---------|
| — | **Baseline evaluation** | Run Stage 5 on the untuned base model before any customization |
| 0 | Data Preparation | Translate English data into the target language before CPT |
| 1 | Continued Pretraining (CPT) | Inject language and domain knowledge into the base model |
| 2 | Supervised Fine-Tuning (SFT) | Teach instruction following in the target language and domain |
| 3 | Reinforcement Learning (RL) | Align preferences, improve reasoning quality |
| 4 | Build Your Own Benchmark (BYOB) | Generate domain MCQ evaluation sets from your own text |
| 5 | Evaluation | Measure model quality on standard and sovereign benchmarks |
| 6 | Quantization | Compress the model for deployment (INT4 AWQ, FP8) |

You do not need to run every stage. A common first path is baseline eval → Stage 1 (CPT) → Stage 2 (SFT) → Stage 5 (post-tuning eval), with Stage 0 added when you have English data that needs translation and Stage 6 added when deploying to production.

### Choosing Benchmarks

| Situation | Recommended benchmarks |
|-----------|----------------------|
| Domain-specific evaluation set exists | Run it alongside MMLU and HellaSwag |
| No domain benchmark yet | Run MMLU, ARC Challenge, HellaSwag against the base model; build a domain benchmark with Stage 4 (BYOB) in parallel |
| Language-specific MMLU subset exists | Include it; compare target-language and English scores before and after |
| No language benchmark exists | Use general English benchmarks to guard instruction-following quality; note the gap explicitly in your plan |

The key rule: **something measured before tuning is always better than nothing measured after.** A general benchmark you ran on the base model is far more useful than a domain benchmark you only ran on the tuned model.

## Infrastructure

The customization pipeline runs on:

- **Slurm** — HPC batch clusters
- **Lepton** — DGX Cloud managed compute
- **Run:AI** — Kubernetes GPU orchestration

A working `env.toml` cluster profile is required before any remote execution. Reference samples for each launcher live in `src/nemotron/customization_recipes/nemotron/planning/references/`.

## Recommended Starting Point

New users should start with the [Sovereign AI Onboarding](./sovereign-ai-onboarding.md) skill rather than jumping directly into stage execution. The onboarding skill:

1. Asks four intake questions (launcher, hardware shape, desired outcomes, autonomy mode)
2. Diagnoses whether customization or full training is the right first move
3. Gates on infrastructure readiness and helps draft `env.toml` if needed
4. Recommends baseline evaluation of Nemotron Nano V3 30B before any tuning
5. Produces a tailored first-day execution plan

Advanced users who already have a working cluster profile and know which stages to run can go directly to the stage execution skills.

## Source

The customization recipes live at:

```
src/nemotron/customization_recipes/nemotron/
  SKILL.md                   ← main execution-oriented pipeline skill
  planning/SKILL.md          ← novice-safe onboarding skill (start here)
  stage0_data_prep/
  stage1_cpt/
  stage2_sft/
  stage3_rl/
  stage4_byob/
  stage5_eval/
  stage6_quantization/
```
