# Curator git-mount pin

BYOB and `translate/translation` mount NeMo Curator from upstream at a fixed
commit so semantic deduplication, the experimental translation stage, and the
FAITH evaluator behave identically across runs. This page is the single
canonical lookup for that pin.

## Current alias

| Alias | Value | Track |
|---|---|---|
| `nemotron-curator-2026-05` | `d10cd6ffe9f5ac4cbb176d7b3ada698f22633aea` | post-`v1.1.0`, approaching upstream `v1.2.0` |

The literal SHA in every config matches `nemotron-curator-2026-05`. Treat the
alias name as the stable reference; the SHA is the implementation detail.

## Environment override

Every Curator mount in this repo resolves through `NEMOTRON_CURATOR_PIN`:

```yaml
run:
  env:
    mounts:
      - ${auto_mount:git+https://github.com/NVIDIA-NeMo/Curator.git@${oc.env:NEMOTRON_CURATOR_PIN,d10cd6ffe9f5ac4cbb176d7b3ada698f22633aea},/opt/Curator}
```

Override at run time without editing any config:

```bash
# pin to an upstream release tag once it ships
export NEMOTRON_CURATOR_PIN=v1.2.0

# or pin to a fork / SHA you are validating
export NEMOTRON_CURATOR_PIN=<sha-or-tag>
```

The fallback (the literal SHA) stays inside the YAML so reproducibility does
not depend on the env var being exported.

## Files that share the pin

The same alias is referenced from:

- [`src/nemotron/steps/byob/config/default.yaml`](../config/default.yaml)
- [`src/nemotron/steps/byob/config/tiny.yaml`](../config/tiny.yaml)
- [`src/nemotron/steps/byob/config/translate.yaml`](../config/translate.yaml)
- [`src/nemotron/steps/byob/config/cpu_smoke.yaml`](../config/cpu_smoke.yaml)
- [`src/nemotron/steps/byob/references/guide.md`](guide.md)
- [`src/nemotron/steps/translate/translation/config/default.yaml`](../../translate/translation/config/default.yaml)

## Bumping the pin

1. Pick a target — preferably an upstream release tag
   (`https://github.com/NVIDIA-NeMo/Curator/releases`). Avoid floating
   branches like `main`.
2. Run the BYOB CPU smoke (`config/cpu_smoke.yaml`, see
   [cpu-smoke-path.md](cpu-smoke-path.md)) and the translation use-case
   (`use-case-examples/translate-corpus/`) with `NEMOTRON_CURATOR_PIN=<new>`.
3. Update **this file** first: change the alias name (mint a new
   `nemotron-curator-YYYY-MM` row) and the literal SHA in the table.
4. Update the literal SHA fallback in every file under "Files that share
   the pin" using a single search-and-replace.
5. Update `pyproject.toml`'s `nemo-curator @ git+...@main` lines only if the
   uv lock needs to track the new commit; the runtime mount is what most
   step configs care about.

Pinning the commit through a named alias keeps reproducibility legible to
agents: a `step.toml [reference]` lookup can now cite
`references/curator-pin.md` as the single source of truth instead of every
configuration file separately.
