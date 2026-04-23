# Nemotron Steps

`steps/` is a container for workflow-specific skills used during customization.

Each workflow step should be packaged as its own skill directory with:
- `SKILL.md` for activation metadata and concise instructions
- `scripts/` for runnable code when needed
- `references/` for manifests, guides, and deep detail
- `assets/` for config templates and golden cases
- `patterns/` for skill-specific routing hints
- `eval/` for skill-specific golden cases and validation prompts

Shared artifacts that apply across multiple step skills stay at the container level:
- [types.toml](types.toml)
- [SKILL_CHECKLIST.md](SKILL_CHECKLIST.md)

Current step skills:
- [translation](translation)
- [byob](byob)
