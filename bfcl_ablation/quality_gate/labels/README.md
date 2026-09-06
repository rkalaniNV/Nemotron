# A7 human review labels

Human review is the independent evidence for content claims that executable replay
cannot establish. A7 never replaces missing labels with an LLM judgement.

Generate a deterministic review queue:

```bash
PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py \
  --emit-label-template bfcl_ablation/results/A7/human_labels.template.yaml
```

Copy the template, declare reviewers, and add one label per reviewer to each item.
Then run:

```bash
PYTHONPATH=src:. python3 bfcl_ablation/run_a7.py \
  --labels path/to/completed_labels.yaml
```

## Rubric

- `intent_preserved`: the candidate asks for the same outcome as the reference,
  including required tool family, literals, confirmation policy, and missing-slot
  behavior.
- `acceptable_for_benchmark`: the item is clear enough to publish and does not leak a
  tool name, expected result, or hidden oracle fact.
- `required_tools`: what the request itself requires. Reviewers must not inspect the
  pack's expected trace before answering.
- `turn_policy`: the interaction shape implied by the request, such as
  `single_turn`, `missing_slot`, `confirmation`, or `clarify_only`.
- `severity`: `critical` when the wording changes the scored behavior, `major` for a
  material but recoverable ambiguity, `minor` for quality-only defects, otherwise
  `none`.

Reviewers should work independently. If they disagree, preserve both labels and add an
`adjudication` with a short rationale. A7 checks completeness against the emitted
queue: an omitted item or too few reviewers is `INCONCLUSIVE`, never an implicit pass.
