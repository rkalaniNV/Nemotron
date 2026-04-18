# Sovereign AI Onboarding

The Sovereign AI Onboarding skill is the recommended starting point for any organization beginning a sovereign AI program with Nemotron. It helps new users choose the right first step, validates infrastructure readiness, and produces a concrete first-day execution plan — without committing anything or running anything significant until you have confirmed it.

## What It Does

The onboarding skill walks you through four questions, then:

1. **Diagnoses the right path.** If your real goal is language or domain adaptation, it will say so explicitly and explain why customization is the correct first move — rather than training from scratch.
2. **Gates on infrastructure.** A working `env.toml` cluster profile is required before any remote execution can proceed. If one does not exist, the skill identifies the closest reference for your launcher and drafts a proposed file for your review and approval.
3. **Recommends baseline evaluation first.** Before any tuning work, the skill recommends running baseline evaluation of Nemotron Nano V3 30B. This establishes a pre-tuning baseline, exposes infrastructure issues early, and makes later fine-tuning outcomes accountable.
4. **Produces a first-day plan.** The plan is written locally to `src/nemotron/customization_recipes/nemotron/planning/sai-onboarding-plan.md`. It captures your intake answers, the diagnosed recommendation, infrastructure status, and the exact next commands to run.

## How To Start

Open Claude Code in the repo root and paste one of the prompts below — or write your own. The skill will ask its four questions and then proceed at the pace you set.

The four intake questions are:

| # | Question | Supported answers |
|---|----------|-------------------|
| 1 | Launcher | `Slurm`, `Lepton`, or `Run:AI` |
| 2 | Hardware shape | e.g. `8xH100 on 4 nodes` |
| 3 | Desired outcomes | language adaptation, domain adaptation, preserve/verify instruction-following quality |
| 4 | Autonomy mode | `guided` (explain and wait), `balanced` (default), `autorun` (proceed unless a real gate is hit) |

## Sample Prompts

The prompts below are examples of how real users have started sovereign AI programs. Copy one as a starting point, or write your own. The more context you give about your organization, use case, language, infrastructure, and how much guidance you want, the better the onboarding skill can tailor the plan.

### Novice — New to the Repo, Prefer a Careful Walkthrough

These prompts are suitable when you or your team are still orienting to the repository and want the skill to explain its reasoning at each step.

---

**Ministry of Public Health, Sri Lanka — Sinhala and English clinic assistant**

> I'm with the Ministry of Public Health in Sri Lanka. We're exploring whether Nemotron could support a Sinhala and English assistant for rural clinic guidance, patient-intake support, and health worker reference. We launch on Slurm and have 8xH100 on 4 nodes. I'm new to this repo, so please walk me through the options carefully and explain why you recommend one path over another before you run anything important.

---

**Department of Agriculture, Kenya — Swahili and English crop advisory**

> I work for the Department of Agriculture in Kenya. We're considering a Swahili and English model for crop advisory, pest diagnosis guidance, and extension-worker support. We use Run:AI with 8xH100 on 4 nodes. I do not want to take a wrong turn, so help me understand what is realistic for our setup and stop to explain the tradeoffs as you go.

---

**National Archives Authority, Jordan — Arabic document assistant**

> I'm from the National Archives Authority of Jordan. We want to see whether a sovereign Arabic document assistant could help with records search, summarization, and triage for internal staff. We run on Lepton with 8xH100 on 16 nodes. Assume I need a fairly careful walkthrough because I'm still learning how this repo is organized and what the safest first step is.

---

**Provincial Department of Justice, Quebec — French and English legal research**

> I'm with the Provincial Department of Justice in Quebec. We're interested in French and English adaptation for internal legal research and policy memo drafting, but I'm not sure whether we should fine-tune, evaluate first, or do something else. We use Slurm with 8xH100 on 4 nodes. Please guide me conservatively and make the reasoning explicit.

---

**Digital Services Office, Ministry of Education, Morocco — Darija, Arabic, and French curriculum assistant**

> I work in the Digital Services Office for the Ministry of Education in Morocco. We want to prototype a Darija, Arabic, and French assistant for curriculum lookup, school policy Q&A, and teacher support. We use Run:AI with 8xH100 on 16 nodes. I'd like help getting oriented without making assumptions, and I want you to pause before writing or launching anything significant.

---

### Advanced — Ready to Move, Want Momentum

These prompts are suitable when you have already oriented to the repository or are comfortable with the workflow and want the skill to proceed as far as it responsibly can before stopping.

---

**Estonian Tax and Customs Board — Estonian, Russian, and English tax guidance**

> I'm with the Estonian Tax and Customs Board. We already expect domain adaptation in Estonian, Russian, and English for tax guidance and case triage, and we run on Slurm with 8xH100 on 16 nodes. You can move quickly: validate what you can, line up the most sensible baseline evaluation path for Nemotron Nano V3 30B, and only stop me when you hit a real decision or approval boundary.

---

**Department of Water Resources, Andhra Pradesh — Telugu and English hydrology support**

> I'm from the Department of Water Resources in Andhra Pradesh. Our target is Telugu and English adaptation for reservoir operations support, hydrology report summarization, and field engineer Q&A. We use Lepton with 8xH100 on 4 nodes. Please take initiative, use the closest working references, and do as much of the setup and preflight work as you responsibly can before asking me to step in.

---

**National Cyber Defense Secretariat, Chile — Spanish incident response assistant**

> I'm with the National Cyber Defense Secretariat in Chile. We want a Spanish domain-adapted model for incident summarization, analyst copiloting, and playbook retrieval, and we need to verify instruction-following quality doesn't regress. We run on Run:AI with 8xH100 on 16 nodes. Treat me like an operator who wants momentum: do the checks, prepare the right path, and surface only the points where explicit approval is actually necessary.

---

**Ministry of Labor, UAE — Arabic and English labor regulation assistant**

> I work for the Ministry of Labor in the UAE. We want Arabic and English customization for labor regulation Q&A, inspection support, and complaint triage. We already believe fine-tuning is the likely answer. We're on Slurm with 8xH100 on 4 nodes. I'm comfortable with you driving the process end-to-end as long as you show me what you're about to write or submit before crossing a real boundary.

---

**National Land Registry, Indonesia — Bahasa Indonesia land records assistant**

> I'm with the National Land Registry of Indonesia. We want Bahasa Indonesia plus domain adaptation for parcel record lookup, deed summarization, and clerk assistance. We use Lepton with 8xH100 on 16 nodes. Don't just give me a reading list; actively steer us toward the first viable evaluation and customization steps, reuse the nearest working templates, and keep going unless there's a concrete reason to stop.

---

## What the Skill Will Not Do

- It will not commit generated plans or `env.toml` drafts.
- It will not write `env.toml` without first showing you the full proposed contents and getting your confirmation.
- It will not submit major remote jobs without confirming the environment appears usable.
- It does not support launchers other than Slurm, Lepton, and Run:AI in v1.

## After Onboarding

Once the onboarding skill has produced your first-day plan, you can move into stage execution using the main customization skill at `src/nemotron/customization_recipes/nemotron/SKILL.md`. The plan produced by onboarding includes the exact next commands to hand off to that skill.
