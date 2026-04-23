# Data And Artifacts

Input data can be JSONL or Parquet.

The checked-in step uses Curator readers and writers rather than custom pandas loader loops. Input directories should be homogeneous: JSONL or Parquet, not both together.

The default output is Curator-written JSONL shards, but the step can also write Parquet when `output_format=parquet`.

Use `text_field` for plain text rows.
Use wildcard paths such as `messages.*.content` for chat data.

If valid JSON objects or arrays appear inside translatable fields, Curator preserves them instead of translating them.

If the source dataset is one very large file, the agent may generate a temporary splitter before running translation:
- split the source into homogeneous temporary shards
- preserve row order and schema
- run the generic translation step on the shard directory
- merge translated shards afterward only if the user needs one final artifact

For schema-sensitive datasets, preserve stable record IDs before splitting so validation and reassembly remain deterministic.
