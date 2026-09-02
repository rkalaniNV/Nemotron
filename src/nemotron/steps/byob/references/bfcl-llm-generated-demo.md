# Walking the LLM-generated lane end to end

`scripts/bfcl_llm_generated_demo.py` runs the whole Flow 2 lane in one command: a reviewed
source package is certified by live probes, an authoring model drafts what that source can
support, the drafts plus reviewed semantics become a candidate pack, and the pack is
validated, reviewed, frozen, published into a benchmark, and scored by a real evaluation
run. It exists so the lane can be watched, not so anything can be claimed: every gate it
crosses is owned by a test named in
[bfcl-unified-authoring-plan.md](bfcl-unified-authoring-plan.md), and the demo is evidence
of nothing on its own.

For setup, live-model configuration, per-stage artifacts, an intentionally
failing scorer run, and evaluation against an independent candidate endpoint,
see the
[detailed LLM-generated Oracle Pack flow](bfcl-llm-generated-oracle-pack-flow.md).

```shell
export BFCL_LLM_DEMO_ROOT="${TMPDIR:-/tmp}/bfcl-llm-generated-demo"
uv run python scripts/bfcl_llm_generated_demo.py --workdir "$BFCL_LLM_DEMO_ROOT"
```

The run takes a few minutes, most of it real probe sessions and two unmocked validation
passes. It refuses a workdir that already exists, because a lane that reuses state is not
the lane being demonstrated.

## What is simulated, and what is not

Two things stand in for people, and both announce themselves in the output.

The authoring model is scripted by default, so the demo needs no credentials: the drafting
prompts are built and cached exactly as they would be, and canned stage answers come back.
`--author-model live` sends those same prompts to a real endpoint instead.

The four human review points are answered from constants, each printed as
`[simulated human review]` with the decision a reviewer would have been making:

| Step | What a person decides |
| --- | --- |
| Authorize | whether this source may be exposed to a model at all |
| Approve evidence | whether the normalized bundle describes the source they reviewed |
| Reviewed semantics | slot bindings, turn policies, and per-language user turns, which no draft schema may express |
| Release checklist | semantics, descriptions, held-out policy, assumptions, validation evidence, certification, pre-model authorization, and answered questions |

Nothing else is faked. Intake probes a real package in a real worker, certification
signatures are real, both validation passes run unmocked and derive their own tier, freeze
really seals the pack read-only, and evaluation runs the shipped scorer.

## The nine steps

1. **Intake.** `bfcl_author author` probes the source package and certifies it at A2,
   requiring a held-out decision before the first model call.
2. **Authorize.** The model-exposure boundary, bound to the resolved configuration digest.
3. **Approve evidence.** The evidence boundary, bound to both bundle digests.
4. **Draft.** `bfcl_author draft` produces the coverage plan, validation cases, task
   templates, and assertion specs, cached immutably for replay.
5. **Assemble.** `bfcl_author assemble` binds drafts, evidence, and the reviewed
   supplement into a candidate pack, and records every digest.
6. **Validate.** Unmocked `prepare_bfcl` derives the tier; the demo stops unless the pack
   is Gold-eligible.
7. **Review, approve, freeze.** The review packet, the release approval it pins, and the
   frozen pack the approval seals.
8. **Publish.** A fresh Gold validation, then benchmark generation, committed by
   `run_manifest.json`.
9. **Evaluate.** A real evaluation run against a candidate served on loopback.

## Reading the evaluation

The demo's candidate answers from the benchmark's own recorded assistant turns, keyed by
what the user last said and how many tool results have come back. A clean run therefore
scores 1.0 across every metric, which is the point: it shows the lane produced a benchmark
that is passable, with assertions that fire and traces that complete.

A scorer that only ever reports success proves nothing, so the same published benchmark
can be re-scored with one task sabotaged — the candidate answers it with text where a call
was expected:

```shell
uv run python scripts/bfcl_llm_generated_demo.py --workdir "$BFCL_LLM_DEMO_ROOT" \
    --stage eval --wrong-answer-task <task_id>
```

That run drops `task_success_rate` and names the failure codes with their attribution, so
the difference between a benchmark that cannot fail and a candidate that did is visible in
one comparison. Each `--stage eval` run writes a fresh `eval-N` directory, because a
committed eval report belongs to the run that wrote it.

Frozen release artifacts are read-only by design, so removing a finished workdir needs
write permission restored first (`chmod -R u+w`).
