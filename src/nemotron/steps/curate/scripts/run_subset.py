# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for ``steps/curate/subset``.

Four phases, in this order for a reason:

``scan``
    Read every document once, count its tokens with the configured tokenizer,
    and record its stratum inputs. Token counts are cached keyed by
    ``(id, tokenizer, revision)`` because re-tokenizing a corpus to change a
    budget is the slowest possible way to answer a cheap question.

``plan``
    Compute every tier's per-stratum quota **before writing anything**, and
    write ``plan.json``. A plan that can be read and rejected before a corpus is
    materialized is worth more than one inferred from the output afterwards.

``materialize``
    Write one output directory per tier.

``verify``
    Check ``ids(N1) ⊆ ids(N2)`` for every pair on the ids actually written, and
    fail the run if it does not hold. Verifying the plan rather than the output
    would check the arithmetic and not the artifact.

Tokenization goes through Curator's ``TokenCountFilter``. It has no ``revision``
parameter, but forwards ``transformers_init_kwargs`` verbatim to
``AutoTokenizer.from_pretrained``, which is where the pin is applied. The
resolved name and revision go into every tier's manifest: two subsets counted
under different revisions are not comparable and must not be presented as an
ablation pair.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.runtime import integrity, subset
from nemotron.steps.curate.runtime import manifest as run_manifest

logger = logging.getLogger("curate.subset")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "subset" / "config" / "default.yaml"

#: Cache schema. Bumped when a change would alter a cached count's meaning.
CACHE_VERSION = 1


class ConfigError(ValueError):
    """The run cannot start as configured."""


# -- input --------------------------------------------------------------------


def resolve_inputs(input_glob: str | list[str]) -> list[str]:
    """One resolver, shared with every other curate step.

    Previously a bare ``glob.glob``, which accepted only globs: a directory —
    the spelling ``curate/audit`` and ``curate/ingest`` both take — resolved to
    nothing and the run died claiming the corpus was empty.
    """
    resolved = integrity.expand_inputs(input_glob)
    if not resolved:
        raise ConfigError(f"input_glob matched no files: {input_glob!r}")
    return resolved


def iter_records(
    paths: list[str], damage: dict[str, int] | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Shared with every other step. See :func:`integrity.iter_records`."""
    return integrity.iter_records(paths, damage)




# -- tokenization -------------------------------------------------------------


class TokenCounter:
    """Curator's ``TokenCountFilter``, pinned to a revision and cached on disk.

    The cache is keyed by ``(tokenizer, revision, document id)``. A cache that
    ignored the revision would answer a later run with counts from a different
    tokenizer and there would be nothing in the output to show it.
    """

    def __init__(self, name: str, revision: str, cache_path: Path | None) -> None:
        self.name = name
        self.revision = revision
        self.cache_path = cache_path
        self._counts: dict[str, int] = {}
        self._filter = None
        self.hits = 0
        self.misses = 0

        if cache_path and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("cache_version") == CACHE_VERSION
                and cached.get("tokenizer") == name
                and cached.get("revision") == revision
            ):
                self._counts = dict(cached.get("counts") or {})
            else:
                logger.warning(
                    "token cache at %s was written for a different tokenizer or schema and is ignored",
                    cache_path,
                )

    def _load(self):
        if self._filter is None:
            from nemo_curator.stages.text.filters.token.token_count import TokenCountFilter

            # hf_model_name, not tokenizer: the latter takes a *loaded*
            # AutoTokenizer. Passing a model-name string there is accepted by the
            # constructor and then fails per document as
            # ``str.encode(encoding=<the document>)`` -> LookupError, and, worse,
            # leaves transformers_init_kwargs unread, so the revision pin — the
            # whole point of recording one — would be silently inert.
            self._filter = TokenCountFilter(
                hf_model_name=self.name,
                transformers_init_kwargs={"revision": self.revision},
            )
            if hasattr(self._filter, "load_tokenizer"):
                self._filter.load_tokenizer()
        return self._filter

    def count(self, doc_id: str, text: str) -> int:
        if doc_id in self._counts:
            self.hits += 1
            return self._counts[doc_id]
        self.misses += 1
        value = int(self._load().score_document(text))
        self._counts[doc_id] = value
        return value

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "tokenizer": self.name,
                    "revision": self.revision,
                    "counts": self._counts,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class WordCounter:
    """Whitespace words, for runs that cannot load a tokenizer.

    Available so the step is testable and so a smoke run does not need model
    weights. Every artifact records ``unit: words``, because a budget in words
    is not a budget in tokens and two subsets measured in different units are
    not an ablation pair.
    """

    name = "whitespace"
    revision = "n/a"
    hits = 0
    misses = 0

    def count(self, doc_id: str, text: str) -> int:  # noqa: ARG002
        return len(text.split())

    def save(self) -> None:
        return


def build_counter(cfg: dict) -> tuple[Any, str]:
    block = cfg.get("tokenizer")
    if block in (None, False):
        return WordCounter(), "words"
    if not isinstance(block, dict) or not block.get("name"):
        raise ConfigError("tokenizer must be a mapping with a name, or null to count words")
    if not block.get("revision"):
        raise ConfigError(
            "tokenizer.revision is required. Token counts from two revisions are not "
            "comparable, and a subset that cannot name its revision cannot be reused as "
            "an ablation baseline. Set it to null to count whitespace words instead."
        )
    cache = cfg.get("token_cache")
    return (
        TokenCounter(block["name"], str(block["revision"]), Path(cache) if cache else None),
        "tokens",
    )


# -- phases -------------------------------------------------------------------


def scan(cfg: dict, paths: list[str], counter: Any) -> tuple[list[subset.ScanRow], dict[str, Any]]:
    id_field = cfg.get("id_field")
    if not id_field:
        raise ConfigError(
            "id_field is required. A subset is a set of document ids; without one, nesting "
            "cannot be stated. Curator's AddId is positional and does not survive resharding."
        )
    text_field = cfg.get("text_field") or "text"
    source_field = cfg.get("source_field")
    score_field = cfg.get("quality_score_field")

    damage: dict[str, int] = {}
    rows: list[subset.ScanRow] = []
    skipped_no_text = 0
    skipped_no_id = 0

    for path, record in iter_records(paths, damage):
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            skipped_no_text += 1
            continue
        raw_id = record.get(id_field)
        if raw_id is None or str(raw_id).strip() == "":
            skipped_no_id += 1
            continue
        doc_id = str(raw_id)
        score = record.get(score_field) if score_field else None
        rows.append(
            subset.ScanRow(
                doc_id=doc_id,
                source=str(record.get(source_field) or path) if source_field else path,
                tokens=counter.count(doc_id, text),
                score=float(score) if score is not None else None,
            )
        )

    stats = {
        "files": len(paths),
        "documents": len(rows),
        "skipped_missing_text": skipped_no_text,
        "skipped_missing_id": skipped_no_id,
        "unparsable_lines": sum(damage.values()),
        "unparsable_by_shard": dict(sorted(damage.items())),
        "source_field_used": bool(source_field),
    }
    if not rows:
        raise ConfigError(
            f"no usable documents in {len(paths)} file(s): "
            f"{skipped_no_text} without {text_field!r}, {skipped_no_id} without {id_field!r}, "
            f"{sum(damage.values())} unparsable line(s)"
        )
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested stratified token-budget subsets")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    started_at = run_manifest.utc_now()

    try:
        run(cfg, started_at)
    except ConfigError as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subset.NestingViolation as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
    except subset.SubsetError as exc:
        print(f"curate/subset: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def run(cfg: dict, started_at: str | None = None) -> dict[str, Any]:
    started_at = started_at or run_manifest.utc_now()
    budgets = cfg.get("token_budgets") or []
    if not isinstance(budgets, list) or not budgets:
        raise ConfigError("token_budgets must be a non-empty list of positive integers")

    paths = resolve_inputs(cfg["input_glob"])
    counter, unit = build_counter(cfg)
    output_dir = Path(cfg.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, scan_stats = scan(cfg, paths, counter)
    counter.save()

    plan = subset.build_plan(
        rows,
        [int(b) for b in budgets],
        seed=int(cfg.get("seed") or 0),
        score_field=cfg.get("quality_score_field"),
        length_bands=tuple(cfg.get("length_bands") or subset.DEFAULT_LENGTH_BANDS),
    )
    plan_doc = plan.to_dict()
    plan_doc["unit"] = unit
    plan_doc["tokenizer"] = {"name": counter.name, "revision": counter.revision}
    (output_dir / "plan.json").write_text(
        json.dumps(plan_doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for warning in plan.warnings:
        logger.warning(warning)

    results = subset.materialize(plan, rows)

    # Verify on the ids that were selected, before writing, then again on what
    # each tier actually contains. The first catches a planning defect, the
    # second catches a writing one, and they are different failures.
    problems = subset.verify_nesting(results)
    if problems:
        raise subset.NestingViolation(
            "the tiers do not nest, so they cannot support the ablation they exist for: "
            + "; ".join(problems)
        )

    id_field = cfg["id_field"]
    text_field = cfg.get("text_field") or "text"
    tiers: list[dict[str, Any]] = []
    written_ids: dict[int, set[str]] = {}

    for budget in plan.budgets:
        result = results[budget]
        wanted = set(result.doc_ids)
        tier_dir = output_dir / f"budget_{budget}_{unit}"
        tier_dir.mkdir(parents=True, exist_ok=True)
        damage: dict[str, int] = {}
        # Collected from the write itself, not copied from the plan. The second
        # verification below exists to catch a WRITING defect; handing it
        # `wanted` made it re-check the arithmetic the first verification had
        # already checked, so a tier that silently failed to write a document
        # still reported nesting_verified: true.
        actually_written: set[str] = set()
        with (tier_dir / "subset.jsonl").open("w", encoding="utf-8") as handle:
            for _, record in iter_records(paths, damage):
                raw_id = record.get(id_field)
                if raw_id is None:
                    continue
                if str(raw_id) in wanted and isinstance(record.get(text_field), str):
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    actually_written.add(str(raw_id))
        written = len(actually_written)
        written_ids[budget] = actually_written

        entry = result.to_dict()
        entry["unit"] = unit
        entry["documents_written"] = written
        entry["output"] = str(tier_dir)
        if written != len(wanted):
            entry["write_mismatch"] = len(wanted) - written
            logger.warning(
                "tier %s: planned %d documents but wrote %d", budget, len(wanted), written
            )
        tiers.append(entry)

    written_problems = subset.verify_nesting(
        {b: subset.TierResult(b, sorted(ids), 0, 0, {}, 0, []) for b, ids in written_ids.items()}
    )
    if written_problems:
        raise subset.NestingViolation("written tiers do not nest: " + "; ".join(written_problems))

    report = {
        "schema_version": subset.SCHEMA_VERSION,
        "step_id": "curate/subset",
        "started_at": started_at,
        "completed_at": run_manifest.utc_now(),
        "config_hash": run_manifest.config_hash(cfg),
        "unit": unit,
        "tokenizer": {"name": counter.name, "revision": counter.revision},
        "token_cache": {"hits": counter.hits, "misses": counter.misses},
        "seed": plan.seed,
        "corpus": {"total_tokens": plan.total_tokens, **scan_stats},
        "stratification": {
            "score_field": plan.score_field,
            "length_bands": list(plan.length_bands),
            "strata": len(plan.strata),
        },
        "tiers": tiers,
        "nesting_verified": True,
        "warnings": list(plan.warnings),
        "interpretation": (
            "Tiers nest by construction: each is a prefix of one fixed per-stratum ordering. "
            "Tokens delivered are at most the budget; any difference is reported as "
            "token_shortfall and is never made up from another stratum."
        ),
    }
    (output_dir / "subset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"curate/subset: wrote {len(tiers)} tier(s), plan.json and subset_report.json to {output_dir}"
    )
    return report


if __name__ == "__main__":
    main()
