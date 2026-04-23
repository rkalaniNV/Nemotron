# Benchmark Schema

Expected BYOB input rows contain:
- `question`
- `options`
- `answer`

`options` may be either:
- a dict such as `{ "A": "...", "B": "..." }`
- a list of option strings

The adapter preserves the original shape on output after the generic translation step returns translated rows.
