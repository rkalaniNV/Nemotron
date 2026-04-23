# Compute And Backends

The runtime is single-node and mostly CPU-bound unless the chosen backend is an LLM service or a local NMT server.

Curator's LLM path uses `AsyncOpenAIClient` against any OpenAI-compatible endpoint, including local vLLM, local NIM, or hosted services.

The checked-in step relies on Curator IO stages by default. If a user dataset needs row-level chunking inside a single giant file, the agent can generate a one-off helper around the generic step rather than expanding the checked-in driver. Keep that helper local to the run unless the same chunking pattern becomes common enough to deserve a reusable utility.

Google and AWS backends are direct translation-service adapters.

The local NMT backend is best when:
- the corpus is large
- latency matters less than throughput
- the data is mostly plain text
