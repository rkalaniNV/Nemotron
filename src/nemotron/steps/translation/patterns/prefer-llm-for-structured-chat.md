# Pattern: prefer-llm-for-structured-chat

## Orient

Structured chat data, wildcard field paths, and tool-calling transcripts usually need schema-aware handling and better formatting fidelity.

## Recommend

Prefer the LLM backend in `nemotron.steps.translation`.

## Verify

Check that the input uses nested chat fields such as `messages.*.content` and that valid JSON tool payloads must stay unchanged.

## Boundaries

If throughput dominates and the data is plain text, this pattern should not override `prefer-nmt-for-large-corpora`.
