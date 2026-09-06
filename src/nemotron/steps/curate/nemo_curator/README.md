# Lightweight Text Curation (NeMo Curator)

Use `curate/nemo_curator` to turn raw JSONL or a Hugging Face snapshot into
`filtered_jsonl` that feeds translation, pretraining prep, or SFT prep. See
`../README.md` for the broader curation journey.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs or code.

## Pipeline Shape

The step is intentionally small:

```text
JsonlReader -> optional FastText language filter -> optional WordCountFilter ->
optional MultilingualDomainClassifier -> optional approved policy -> JsonlWriter
```

If `dataset` is set, the HF snapshot is downloaded first and `input_glob`
should point into that local snapshot. Crawling, full extraction, and
deduplication are out of scope here — use a dedicated Curator recipe for
those.

## Applying A Profiled Policy

`curate/profile` measures what a threshold would remove. This step is where an
*approved* one runs. The two are deliberately not one command: a distribution
says what a gate removes, never whether removing it is right.

```yaml
heuristic_filters:
  approved_policy: ./policy/approved_policy.yaml
mode: both
```

The `candidate_policies.yaml` that `curate/profile` writes carries
`approved: false` and is **refused here**. Promote it deliberately — filling in
the `approval` block with approver, date, and evidence — or set
`heuristic_filters.allow_unvalidated_policy: true`, which proceeds and logs a
warning naming the policy on every run.

Signal names in a policy resolve through the closed allowlist in
`../runtime/registry.py`. A policy file is a document people paste between
machines, so it can never name an import path.

`mode` decides what reaches the writer:

| `mode` | Rows | Columns added |
|---|---|---|
| `filter` | rejected rows dropped | none — scores discarded after use |
| `annotate` | all kept | `__<signal>` for every row |
| `both` | rejected rows dropped | `__<signal>` for the survivors |

Use `annotate` when you want to re-threshold later without re-reading the
corpus, and `filter` to keep the historical column set exactly.

Nemotron does not bundle production language packs. When a policy uses
pack-backed signals, set `heuristic_filters.langpack_dir` to the reviewed pack
root used by this run. The policy carries its content hash; the loaded word
lists and character set must match that hash or the run stops.

`heuristic_filters.langpack_content_hash` may additionally pin the expected
hash in the run config, but it does not replace `langpack_dir`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` to verify the reader/writer path with no
filters. In a project overlay, developers usually change:

- `input_glob` and `output_dir`: local JSONL paths or paths inside a
  downloaded HF snapshot.
- `text_field`: source text column.
- `dataset`: set `null` for local files; set repo fields for HF snapshots.
- `language_codes` and `models.fasttext_langid`: enable language gating.
- `quality_filters`: set both `min_words` and `max_words` together when using
  word count.
- `domains`, `models.hf_cache_dir`, and `ray.num_cpus`.
- `domain_score_field`: optionally retain the complete class-probability vector
  in model label order. This field is not a scalar confidence value.

On small CPU Lepton runs, set `NEMOTRON_CURATOR_RAY_NUM_CPUS=4` through the
env profile when the YAML does not include `ray.num_cpus`.

Related pattern: [data-quality-before-quantity.md](../../patterns/data-quality-before-quantity.md).

## Run It

Smoke first to validate the reader/writer path with filters disabled:

```bash
uv run nemotron steps run curate/nemo_curator -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run curate/nemo_curator \
  -c <project>/config/curate.yaml \
  input_glob='<project>/data/raw/*.jsonl' \
  output_dir=<project>/data/filtered
```

For a Lepton execution profile, add `-r lepton_curate` and inspect logs with
`uv run lep log get -j curate-nemo-curator-step-<id> --limit 300`.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py)
- Configs: `config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Don't enable every optional filter on the first run. Start with `tiny` or
  local JSONL plus no filters, then add language, word-count, and domain gates.
- Inspect intermediate JSONL when output is empty or tiny — a filter is usually
  too aggressive.
- Split very large input files before reading; OOMs come from oversized
  partitions.
