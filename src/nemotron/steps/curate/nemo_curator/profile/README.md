# Corpus Filtering Impact Profile

Use `curate/profile` to find out what a filtering threshold would actually do to
*your* corpus before you apply it.

Use this README for workflow and pitfalls; use `step.toml` for the exact
artifact, parameter, strategy, and error manifest before editing configs.

## The Problem It Addresses

Today a user sets `min_words: 50` because that number was in the sample config.
Nobody knows whether it removes 8% of their corpus or 80%. Curator's heuristic
defaults were calibrated on English-language work, and its own non-English
preset handles the mismatch by deleting three filters rather than retuning them.

This step measures instead of guessing:

```
word_count                 min 50           retains 91.9%   corpus p5 = 41 words
non_alpha_numeric          threshold 0.25   retains  5.8%   corpus p50 = 0.46  <-
repeating_duplicate_ngrams threshold 0.2    retains 97.2%
```

The marked line is the one nobody gets today. Right now you see a suspiciously
small output and cannot tell which gate caused it — the `empty_or_tiny_output`
failure the category documents but cannot diagnose.

## What It Does Not Establish

**A retention curve is descriptive.** It answers *how much does this threshold
remove*. It does not answer *is what it removes bad*. A corpus can have a small
but valuable tail, or a large body of spam, and a distribution cannot tell them
apart.

So the report's vocabulary is deliberate — *candidate* threshold, not correct;
*retention-stable* band, not feasible. And the output is not executable:
`candidate_policies.yaml` carries `approved: false`. Promotion to an
`approved_policy.yaml` is a separate act that records who approved it and on
what evidence.

`min_keep_rate` and `max_keep_rate` are **analysis constraints you chose**, not
properties discovered in the data. They bound which part of a curve gets
reported.

## Two Views Of Every Figure

Sampling the same number of documents from a large source and a small one is
unbiased within each source and skewed at corpus level. Both views are reported
and both are labelled:

| View | Weighting | Answers |
|---|---|---|
| `macro` | each source equally | is this threshold reasonable for each source? |
| `micro` | each sampled document by how many it stands for | how much of the corpus does it remove? |

Any figure that did not say which view it came from could not be acted on.

## Language Packs

`language` is a BCP-47 tag and has **no default**. A wrong default produces
plausible numbers for the wrong language, which is worse than refusing to start.

A pack is data — word lists, a character set, boilerplate patterns, a fold map —
plus a declaration of what can meaningfully be measured for that language. See
[../data/langpacks/SPEC.md](../data/langpacks/SPEC.md) to author one.

Nemotron ships one **opt-in English reference pack** under
`../data/langpacks/en/`, sourced from Snowball and Unicode CLDR 48 with pinned
content hashes and license texts. It supplies score inputs, not filtering
thresholds, and neither the language nor its directory is selected by default.
Run `-c en` to choose it explicitly. For another language, set `langpack_dir`
to a reviewed pack root owned by your corpus workflow.

The `x-test-*` packs under `tests/steps/curate/fixtures/langpacks/` validate the
implementation but are not installed, user defaults, or claims of supported
languages.

The capability declaration does real work. Vietnamese tone marks strip to
degraded but readable text, so a diacritic ratio measures something. Devanagari
matras are obligatory vowels; stripping them yields nonsense, so the `hi` pack
declares neither `diacritic_ratio` nor `stopword_ratio_folded` and both are
**absent from the Hindi report**, with a recorded warning, rather than computed
on a false premise.

Naming a signal the pack cannot support is an error. Leaving `signals` empty
skips it with a warning — the difference is that a named signal is a question
the caller asked.

## Signals

A closed allowlist in `../runtime/registry.py`. Config names a signal, never an
import path.

Curator's filters do not share one shape — some are upper bounds, some lower,
and `word_count` and `mean_word_length` gate from both sides at once. A
two-sided gate produces a **retention surface**, not a curve: the retention of a
lower bound depends on where the upper bound sits, so reporting a single line
would fix one bound at an unstated value and attribute all of the effect to the
other.

Several Curator filters are deliberately excluded, with the reason recorded in
`registry.EXCLUDED` — English-hardcoded word lists, binary scores that cannot be
swept, and four repetition filters whose parameter names and `keep_document`
comparisons disagree.

**The Unicode signal restores a gate, it does not fix a broken default.**
Curator's `heuristic_filter_non_english_pipeline.yaml` already omits
`NonAlphaNumericFilter`, `CommonEnglishWordsFilter` and
`WordsWithoutAlphabetsFilter` — the three its English cascade uses and that assume
ASCII. That omission is correct, and it leaves a non-English corpus with no
character-composition gate and no vocabulary gate. `unicode_alpha_numeric` and
`stopword_ratio` are the script-aware replacements for the first two.

**And the Unicode handling is not a Vietnamese fix.** `UnicodeAwareNonAlphaNumericFilter`
accepts Unicode categories L, N and *all* of M. Writing it as `\p{Mn}` — the
natural thing to reach for with only Vietnamese to test against — passes every
Vietnamese case and rejects correct Devanagari outright, because matras split
across `Mn` and `Mc` with `Mc` in the majority. It also accepts ZWJ and ZWNJ,
which are category `Cf` and obligatory inside Indic conjuncts; treating them as
junk moved a correct Hindi paragraph from 0.218 to 0.265, across the 0.25
default.

**Direction is verified, not trusted.** The sweep compares scores to thresholds
directly rather than constructing a filter at every grid point. That shortcut is
only sound if the registry's stated direction matches the filter's own
`keep_document`, so it is checked against the real implementation on real scores
before any figure is produced. A mismatch stops the run.

## Run It

Run the packaged English smoke profile with no model download:

```bash
uv run nemotron steps run curate/profile -c en
```

For your own corpus, profile the *unfiltered* input and supply the pack
explicitly:

```bash
uv run nemotron steps run curate/profile \
  input_glob='./output/raw_jsonl/**/*.jsonl' \
  output_dir=./output/profile \
  source_field=source id_field=id \
  language=vi langpack_dir=./langpacks
```

Four files land in `output_dir`:

| File | Contents |
|---|---|
| `profile_summary.md` | **Read this one.** The same measurements rendered for a person |
| `profile_report.json` | Quantiles, retention curves and surfaces, co-occurrence, policy simulation, what each Curator default would keep |
| `candidate_policies.yaml` | Proposed threshold sets, `approved: false` |
| `sample_manifest.json` | Seed, per-source `(population, sampled, weight)`, and the sampled key hashes |

## Reading The Summary

`profile_summary.md` has four parts, in the order you need them.

**Language composition.** What languages the corpus is actually in, scored with
the same FastText model the filter uses. No signal below it can answer this:
languages that share a script are indistinguishable to a script ratio. The gap is
not academic — naming the wrong language in `steps.filter.language_codes` removes
almost everything and still reports success, because the row counts reconcile
either way. Confidence is reported as its own distribution, because which
languages to keep and how sure the model must be are two separate decisions.
Absent a model the section is omitted rather than guessed.

**One section per signal.** Quantiles, a sparkline, then a `gate at` table:

```text
| to keep | set `max:` | actually keeps | drops |
| ~99% | 0.9365 | 99.03% | 193 docs |
| ~95% | 0.8730 | 95.97% | 807 docs |
| ~90% | 0.8254 | 91.46% | 1,708 docs |
| ~80% | 0.7778 | 80.32% | 3,936 docs |
```

Every row is **this gate applied alone**. The 80% row exists because a
concentrated signal makes 99/95/90 resolve to the same grid point and the table
then repeats one number three times; it is the floor because
`band_search.min_keep_rate` defaults there.

Which side of a signal is desirable is deliberately not stated. It does not
generalise: a ratio that marks noise in one language is ordinary in another.
`max` means an upper bound and documents above it are dropped, `min` a lower
bound, `interval` a two-sided pair — and which key a signal takes comes from the
registry, not from you.

**Policy simulation.** What all the gates cost *together*. A policy is a
conjunction, so its cost is the union of the rejects, and the union is smaller
than the sum by however many documents more than one gate rejects — on a
20,000-document Vietnamese sample the per-gate figures summed to 2,718 while the
union was 1,808. Each gate also reports what it is alone responsible for:
`foreign_script_ratio` rejected 194 documents there and not one of them
uniquely, so deleting that gate would have changed nothing.

Order-dependent and order-independent figures are separated, because only one of
them is a property of the policy. `fails alone`, `shared`, `unique` and
`keeps without` do not move when stages are reordered; `first to reject` does.
That last one is the shape `curation_ledger.json` records, which is why its
per-reason counts understate what a gate rejects.

**The approve block.** Paste-ready YAML with each threshold's retention on the
same line, so the two cannot be mis-paired. Take thresholds from here rather
than from `candidate_policies.yaml`: a band's `threshold_low` and
`threshold_high` are two *alternative* single bounds, not a range, and choosing
the wrong end is accepted without a warning. The block covers one-sided signals
only — `word_count`, `mean_word_length` and `token_count` gate from both sides
and are written by hand with both bounds.

The sample is reproducible from `(seed, max_total_docs)` — hash-bottom-k, never
Python's salted `hash()`.

## Repository Layout

- Manifest: [step.toml](step.toml)
- Runner: [step.py](step.py) delegating to `../scripts/run_profile.py`
- Signals: `../runtime/registry.py`
- Measurement: `../runtime/profiling.py`
- Sampling: `../runtime/determinism.py`
- Policy schema: `../runtime/policy.py`

## Guardrails

- Profile the input you want to filter, not the already-filtered output — the
  gates you are trying to understand have already run on the latter.
- `source_field` matters. Without it each shard becomes its own "source" and the
  per-source figures describe shards; the report says so, but the numbers are
  easy to misread.
- Do not quote a co-occurrence figure without its operating point. Overlap is
  only defined at a specific threshold per signal, and every entry in the report
  carries the thresholds it was computed at.
