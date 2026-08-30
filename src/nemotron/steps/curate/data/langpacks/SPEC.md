# Language Pack Specification

A language pack is **data**. Nothing in `runtime/` knows any language exists —
character sets, word lists, patterns and fold maps all arrive from here. Adding
a language means adding a directory, not editing code.

## Layout

```
data/langpacks/<bcp47-tag>/
├── pack.toml        the manifest
├── stopwords.txt    one function word per line
├── charset.txt      one character per line
└── boilerplate.txt  one regular expression per line
```

Blank lines and lines starting with `#` are ignored in every `.txt` file, so
each can carry its own provenance header.

The directory name is a **BCP-47 tag** (`vi`, `hi`, `pt-BR`), not an ISO 639-1
code. Private-use tags (`x-…`) are valid and are what test fixtures use.

## `pack.toml`

```toml
[pack]
pack_id      = "vi-generic"   # identifies the pack; may differ from the tag
language_tag = "vi"           # BCP-47
version      = "2.0"
schema       = 1

[sources]
stopwords   = { file = "stopwords.txt",   n = 287, origin = "…", license = "…" }
charset     = { file = "charset.txt",     script = "Latn" }
boilerplate = { file = "boilerplate.txt", n = 31,  origin = "…" }

[orthography]
sentence_terminators = [".", "!", "?"]

[fold_map]
"đ" = "d"

[capabilities]
supports = ["script_ratio", "stopword_ratio", "boilerplate_hits", "sentence_end_ratio"]
```

## Capabilities

`supports` is the load-bearing field. It declares what this language can
meaningfully be measured for, and a signal whose capability is undeclared is
**absent from the report** rather than computed on a false premise.

| Capability | Needs | Enables |
|---|---|---|
| `script_ratio` | `charset` | `script_ratio`, `latin_ratio`, `foreign_script_ratio` |
| `diacritic_ratio` | `charset`, `fold_map` | `diacritic_ratio` |
| `stopword_ratio` | `stopwords` | `stopword_ratio` |
| `stopword_ratio_folded` | `stopwords`, `fold_map` | `stopword_ratio_folded` |
| `boilerplate_hits` | `boilerplate` | `boilerplate_hits` |
| `sentence_end_ratio` | `sentence_terminators` | `sentence_end_ratio` |

Declaring a capability without the data behind it is rejected at load. It would
otherwise fill a report with zeroes, which reads as a finding about the corpus
rather than a hole in the pack.

**Do not declare a capability that does not apply to your language.** Vietnamese
tone marks strip to degraded but readable text, so a diacritic ratio measures
something real. Devanagari matras are obligatory vowels; stripping them yields
nonsense, so the `hi` pack declares neither `diacritic_ratio` nor
`stopword_ratio_folded`. That absence is the correct answer, not a gap to fill.

## `fold_map`

Only for marks the language treats as removable. The runtime already drops
Unicode combining marks; the map is for characters that do **not** decompose —
Vietnamese `đ` is the reason it exists.

Folding merges distinct words together (`mà`, `má`, `mã`, `mạ` all become `ma`).
That collision count is reported alongside any folded figure. It is a property
of the language, not a defect in your word list.

## `charset`

Every character that counts as this language's own script, including marked
forms. If you list only base letters, correct text scores as foreign.

For an abugida, include the matras — and note that they split across Unicode
categories `Mn` (nonspacing) and `Mc` (spacing combining). Omitting `Mc` is the
single most common way to break an Indic or South-East Asian pack: a filter
written against `Mn` alone passes every Latin-script test and rejects correct
Devanagari outright.

## `sentence_terminators`

Defaults to `.`, `!`, `?`. Override it for any script that ends sentences
otherwise — Hindi and Sanskrit use the danda `।` (U+0964), which is Unicode
category `Po` and matches none of the defaults.

## Before you ship one

1. `load()` it and check `describe()` reports the counts you expect.
2. Run `curate/profile` against a corpus in that language with `signals: []` and
   confirm the skipped-capability warnings say what you intended.
3. Score correct text through `UnicodeAwareNonAlphaNumericFilter` and confirm it
   is kept at the 0.25 default. If it is not, the charset is incomplete.
4. Score text in **both** NFC and NFD. Some scripts differ materially between
   them; some do not. A pack tested in only one form is tested in neither.

## What packs are not for

Thresholds. A pack says what *can* be measured for a language, never what a good
value is — that depends on the corpus, and finding it is what `curate/profile`
is for. Numbers chosen for one corpus do not transfer to another in the same
language.
