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

"""CLI for ``steps/curate/decontamination``.

Three passes, cheapest first, because each one can make the next unnecessary for
the documents it already resolved:

``group``
    Exact source-document identity across the split — canonical URL, then
    namespaced id, then normalised-text hash. No GPU, no similarity. Catches the
    leak similarity cannot: a page rewritten enough that its shingles no longer
    overlap is still the same source document.

``candidates``
    Curator's ``FuzzyDeduplicationWorkflow`` over the union of both splits.
    GPU-backed, and the reason this step declares ``gpus_per_node = 1`` while
    the other three curate steps declare zero. LSH proposes; it does not decide.

``verify``
    Exact Jaccard on every candidate pair, at the *same* shingling the
    candidates were generated with. Then removal — from the training split only.

The union trick matters: Curator's fuzzy dedup deduplicates one corpus against
itself, so both splits are fed in together with a marker and only cross-split
pairs are kept. Same-split pairs are ordinary duplicates and are not this step's
business.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from nemotron.steps.curate.runtime import decon, grouping, integrity
from nemotron.steps.curate.runtime import manifest as run_manifest

logger = logging.getLogger("curate.decontamination")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "decontamination" / "config" / "default.yaml"

SPLIT_FIELD = "__split"
TRAIN, HOLDOUT = "train", "holdout"


class ConfigError(ValueError):
    """The run cannot start as specified."""


def resolve_inputs(pattern: str | list[str], label: str) -> list[str]:
    """One resolver, shared with every other curate step.

    Previously a bare ``glob.glob``, which accepted only globs: a directory —
    the spelling ``curate/audit`` and ``curate/ingest`` both take — resolved to
    nothing and the run died claiming the split was empty.
    """
    resolved = integrity.expand_inputs(pattern)
    if not resolved:
        raise ConfigError(f"{label} matched no files: {pattern!r}")
    return resolved


def read_split(paths: list[str], id_field: str, text_field: str, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    unparsable = 0
    for path in paths:
        with Path(path).open("rb") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    unparsable += 1
                    continue
                doc_id = record.get(id_field)
                if doc_id is None or str(doc_id).strip() == "":
                    raise ConfigError(
                        f"{label}: a record has no {id_field!r}. Decontamination reports which "
                        "documents were removed and why, which needs a stable identifier."
                    )
                doc_id = str(doc_id)
                if doc_id in seen:
                    raise ConfigError(
                        f"{label}: {id_field}={doc_id!r} appears more than once. A removal "
                        "report keyed on a non-unique id cannot say which document it removed."
                    )
                seen.add(doc_id)
                if not isinstance(record.get(text_field), str):
                    continue
                records.append(record)
    if unparsable:
        logger.warning("%s: skipped %d unparsable line(s)", label, unparsable)
    if not records:
        raise ConfigError(f"{label}: no records with a {text_field!r} string")
    return records


def candidate_pairs(cfg: dict, train: list[dict], holdout: list[dict], id_field: str, text_field: str):
    """Cross-split candidate pairs from Curator's fuzzy deduplication workflow.

    Imported lazily and only when the GPU pass is enabled, so the module — and
    every test in this file — loads on a host without ``cudf``.
    """
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow

    work_dir = Path(cfg.get("work_dir") or "./cache/decontamination")
    union_dir = work_dir / "union"
    union_dir.mkdir(parents=True, exist_ok=True)
    union_path = union_dir / "union.jsonl"

    split_of: dict[str, str] = {}
    with union_path.open("w", encoding="utf-8") as handle:
        for split, records in ((HOLDOUT, holdout), (TRAIN, train)):
            for record in records:
                key = f"{split}:{record[id_field]}"
                split_of[key] = split
                handle.write(
                    json.dumps(
                        {id_field: key, text_field: record[text_field], SPLIT_FIELD: split},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    minhash = cfg.get("minhash") or {}
    workflow = FuzzyDeduplicationWorkflow(
        cache_path=str(work_dir / "cache"),
        output_path=str(work_dir / "out"),
        input_path=str(union_path),
        input_filetype="jsonl",
        text_field=text_field,
        perform_removal=False,
        seed=int(minhash.get("seed", 42)),
        char_ngrams=int(minhash.get("char_ngrams", decon.DEFAULT_SHINGLE_SIZE)),
        num_bands=int(minhash.get("num_bands", 20)),
        minhashes_per_band=int(minhash.get("minhashes_per_band", 13)),
    )
    workflow.run()

    return read_cross_split_pairs(work_dir / "out", split_of, id_field)


def read_cross_split_pairs(output_dir: Path, split_of: dict[str, str], id_field: str):
    """Keep only pairs that straddle the split.

    Same-split matches are ordinary duplicates. Removing them here would silently
    turn a decontamination run into a deduplication run, which is a different
    decision with a different owner.
    """
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(Path(output_dir).rglob("*.parquet"))]
    if not frames:
        return []
    edges = pd.concat(frames, ignore_index=True)

    columns = list(edges.columns)
    left_col, right_col = columns[0], columns[1]
    pairs = []
    for left, right in zip(edges[left_col], edges[right_col], strict=True):
        left, right = str(left), str(right)
        left_split, right_split = split_of.get(left), split_of.get(right)
        if left_split == right_split or None in (left_split, right_split):
            continue
        train_key = left if left_split == TRAIN else right
        holdout_key = right if left_split == TRAIN else left
        pairs.append((train_key.split(":", 1)[1], holdout_key.split(":", 1)[1]))
    return sorted(set(pairs))


def run(cfg: dict, started_at: str | None = None) -> dict[str, Any]:
    started_at = started_at or run_manifest.utc_now()
    id_field = cfg.get("id_field")
    if not id_field:
        raise ConfigError("id_field is required so the report can name what it removed")
    text_field = cfg.get("text_field") or "text"
    threshold = float(cfg.get("threshold", 0.8))
    if not 0.0 < threshold <= 1.0:
        raise ConfigError(f"threshold must be in (0, 1], got {threshold}")

    train = read_split(resolve_inputs(cfg["train_glob"], "train_glob"), id_field, text_field, TRAIN)
    holdout = read_split(
        resolve_inputs(cfg["holdout_glob"], "holdout_glob"), id_field, text_field, HOLDOUT
    )

    norm = decon.Normalization(**(cfg.get("normalization") or {}))
    shingle_kind = cfg.get("shingle_kind") or decon.DEFAULT_SHINGLE_KIND
    minhash = cfg.get("minhash") or {}
    shingle_size = int(minhash.get("char_ngrams", decon.DEFAULT_SHINGLE_SIZE))

    train_text = {str(r[id_field]): r[text_field] for r in train}
    holdout_text = {str(r[id_field]): r[text_field] for r in holdout}
    holdout_before = list(holdout_text)

    # -- pass 1: exact source identity, no GPU -------------------------------
    groups = grouping.cross_split_groups(
        train, holdout, left_source=TRAIN, right_source=HOLDOUT,
        cfg=grouping.GroupKeyConfig(text_field=text_field),
    )
    if not groups["comparable"]:
        # Zero shared groups here would be a structural certainty, not a
        # measurement: the two sides key off different fields, so their key
        # spaces are disjoint and no pair can ever match. Refuse rather than
        # report the clean result that guarantees.
        raise ConfigError(
            "the train and holdout splits identify documents by different fields — train uses "
            f"{groups['left_key_fields']}, holdout uses {groups['right_key_fields']}. Group keys "
            "are namespaced by the field that produced them, so these two sets can never "
            "intersect and the exact-identity pass would report no overlap whatever the data "
            "contains. Give both splits the same identifying field (commonly url, or the same "
            "id space), or prepare the holdout from the same corpus the train split was ingested "
            "from — ingest mints new ids, so a holdout built from the raw corpus no longer "
            "shares an id space with the ingested one."
        )

    group_removals = {
        left_id for entry in groups["shared_groups"] for left_id in entry["left_ids"] if left_id
    }

    # -- pass 2: candidates ---------------------------------------------------
    if cfg.get("skip_similarity"):
        candidates: list[tuple[str, str]] = []
        similarity_note = (
            "skip_similarity was set: only exact source-document identity was checked. "
            "Near-duplicate overlap was NOT measured on this run."
        )
    else:
        candidates = candidate_pairs(cfg, train, holdout, id_field, text_field)
        similarity_note = ""

    # -- pass 3: verify, then remove -----------------------------------------
    pairs = decon.verify_pairs(
        candidates,
        train_text,
        holdout_text,
        threshold=threshold,
        shingle_size=shingle_size,
        shingle_kind=shingle_kind,
        norm=norm,
    )
    verified = decon.removals(pairs, threshold)
    remove = set(verified) | group_removals

    output_dir = Path(cfg.get("output_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    with (output_dir / "train_decontaminated.jsonl").open("w", encoding="utf-8") as handle:
        for record in train:
            if str(record[id_field]) in remove:
                continue
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1

    # The holdout is never written, so this cannot fail here — it is asserted
    # because the guarantee is what the step is for, and a future edit that
    # started writing it should fail loudly rather than quietly.
    decon.assert_holdout_untouched(holdout_before, list(holdout_text))

    unverifiable = [p for p in pairs if not p.verifiable]
    report = {
        "schema_version": decon.SCHEMA_VERSION,
        "step_id": "curate/decontamination",
        "started_at": started_at,
        "completed_at": run_manifest.utc_now(),
        "config_hash": run_manifest.config_hash(cfg),
        "threat_model": (
            "document near-duplicate (A). Detects a training document largely the same as a "
            "holdout document. Does NOT detect a short benchmark question embedded in a long "
            "training document — that is substring contamination and needs a different algorithm."
        ),
        # What was removed depends on which passes ran, so the claim has to as
        # well. Stating "near-duplicate overlap removed" after skip_similarity
        # turned that measurement off describes a pass that did not happen.
        "claim": (
            (
                "exact source-document overlap detected and removed from the training split. "
                "skip_similarity was set, so near-duplicate overlap was NOT measured and this "
                "run makes no claim about it. This is also not a statement that the holdout "
                "is uncontaminated."
            )
            if cfg.get("skip_similarity")
            else (
                "near-duplicate overlap detected and removed from the training split. This is "
                "not a statement that the holdout is uncontaminated."
            )
        ),
        "parameters": {
            "threshold": threshold,
            "shingle_kind": shingle_kind,
            "shingle_size": shingle_size,
            "normalization": norm.to_dict(),
            "minhash": dict(minhash),
        },
        "splits": {
            "train": {
                "documents": len(train),
                "fingerprint": decon.corpus_fingerprint(train_text),
            },
            "holdout": {
                "documents": len(holdout),
                "fingerprint": decon.corpus_fingerprint(holdout_text),
                "modified": False,
            },
        },
        "group_overlap": {
            "shared_group_count": groups["shared_group_count"],
            "train_documents_affected": len(group_removals),
            "groups": groups["shared_groups"][:50],
            "note": (
                "Exact source-document identity across the split. Independent of similarity: "
                "a page rewritten past the threshold is still the same source document."
            ),
        },
        "similarity": {
            "candidate_pairs": len(candidates),
            "verified_duplicates": len(verified),
            "unverifiable_pairs": len(unverifiable),
            "removed_pairs": [p.to_dict() for p in sorted(verified.values(), key=lambda p: -p.similarity)[:200]],
            "unverifiable": [p.to_dict() for p in unverifiable[:50]],
            "note": similarity_note
            or (
                "Candidates come from LSH, which contains false positives by construction. "
                "Every removal below rests on an exact Jaccard computed at the same shingling "
                "the candidates were generated with."
            ),
        },
        "result": {
            "train_documents_in": len(train),
            "train_documents_removed": len(remove),
            "train_documents_out": kept,
            "removed_by_group_identity_only": len(group_removals - set(verified)),
            "removed_by_similarity_only": len(set(verified) - group_removals),
            "removed_by_both": len(group_removals & set(verified)),
        },
    }
    (output_dir / "decontamination_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if unverifiable:
        logger.warning(
            "%d candidate pair(s) could not be verified and were NOT removed; see the report",
            len(unverifiable),
        )
    print(
        f"curate/decontamination: removed {len(remove)} of {len(train)} training documents; "
        f"wrote train_decontaminated.jsonl and decontamination_report.json to {output_dir}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove training documents that near-duplicate a holdout split"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    try:
        run(cfg, run_manifest.utc_now())
    except ConfigError as exc:
        print(f"curate/decontamination: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except decon.HoldoutModified as exc:
        print(f"curate/decontamination: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
