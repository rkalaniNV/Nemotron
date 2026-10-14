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

"""Step-owned preparation for the released Lightning 3.5 RLVR JSONL.

The public Lightning blend is a repository containing ``rlvr.jsonl`` rather
than a regular Hub dataset split. Some rows mask their question through
``_hf_question_placeholder`` and must be restored from the public DAPO and
Skywork datasets. This module intentionally lives with the step so the public
``data_prep/rl_prep -c lightning35`` workflow does not depend on a model recipe
implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend

QUESTION_PLACEHOLDER_KEY = "_hf_question_placeholder"
DAPO_QUESTION_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
SKYWORK_QUESTION_DATASET = "Skywork/Skywork-OR1-RL-Data"
QUESTION_PLACEHOLDER_SOURCES: tuple[tuple[str, str], ...] = (
    (DAPO_QUESTION_DATASET, "train"),
    (SKYWORK_QUESTION_DATASET, "math"),
)
RELEASED_BLEND_TRANSFORM_SCHEMA = 2
NEMO_GYM_RUNTIME_METADATA_KEYS = frozenset(
    {
        "_ng_rollout_index",
        "_ng_task_index",
    }
)
_DAPO_INSTRUCTION_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
_DAPO_INSTRUCTION_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'


@dataclass(frozen=True)
class LocalSplitResult:
    """Paths and row counts produced by the released-blend preparation."""

    train_path: str
    val_path: str | None
    train_rows: int
    val_rows: int
    run_dir: str
    manifest_path: str


def _write_local_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _normalize_agent_names(allowed_agent_names: Iterable[str] | None) -> tuple[str, ...] | None:
    if allowed_agent_names is None:
        return None

    cleaned: set[str] = set()
    for name in allowed_agent_names:
        if not isinstance(name, str):
            raise TypeError(f"allowed_agent_names must contain only strings, got {type(name).__name__}")
        if name.strip():
            cleaned.add(name.strip())
    if not cleaned:
        raise ValueError("allowed_agent_names must contain at least one non-empty name")
    return tuple(sorted(cleaned))


def _load_json_object(line: str, input_path: Path, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {error.msg}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object at {input_path}:{line_number}, got {type(record).__name__}")
    return record


def _responses_api_agent_name(record: Mapping[str, Any]) -> str | None:
    """Return a policy-served NeMo-Gym agent name, or None for other rows."""
    agent_ref = record.get("agent_ref")
    agent_type = agent_ref.get("type") if isinstance(agent_ref, dict) else None
    agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else None
    if agent_type != "responses_api_agents" or not isinstance(agent_name, str) or not agent_name.strip():
        return None
    return agent_name.strip()


def split_local_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    val_holdout: int = 100,
    sample: int | None = None,
    force: bool = False,
    allowed_agent_names: Iterable[str] | None = None,
) -> LocalSplitResult:
    """Filter and split a restored JSONL, then atomically publish its manifest."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")
    if sample is not None and sample <= 0:
        raise ValueError(f"sample/max_rows must be greater than zero, got {sample}")
    if val_holdout <= 0:
        raise ValueError(f"val_holdout must be greater than zero, got {val_holdout}")
    if sample is not None and sample <= val_holdout:
        raise ValueError(
            f"sample/max_rows ({sample}) must be greater than val_holdout "
            f"({val_holdout}) so both training and validation data are produced"
        )

    normalized_agent_names = _normalize_agent_names(allowed_agent_names)
    allowed_agent_name_set = set(normalized_agent_names) if normalized_agent_names is not None else None

    stat = input_path.stat()
    run_config: dict[str, object] = {
        "input_path": str(input_path),
        "input_mtime_ns": stat.st_mtime_ns,
        "input_size": stat.st_size,
        "released_blend_transform_schema": RELEASED_BLEND_TRANSFORM_SCHEMA,
        "val_holdout": val_holdout,
        "sample": sample,
    }
    if normalized_agent_names is not None:
        run_config["allowed_agent_names"] = normalized_agent_names
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    run_hash = config_hash if not force else f"{config_hash}_{time.time_ns()}"
    run_dir = output_dir / "runs" / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    stale_manifest_for_run = False
    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_hash") == run_hash:
                stale_manifest_for_run = True
                cached_train = existing.get("train", "")
                cached_val = existing.get("val")
                cached_train_bytes = existing.get("train_bytes")
                cached_val_bytes = existing.get("val_bytes")
                train_is_current = (
                    bool(cached_train)
                    and isinstance(cached_train_bytes, int)
                    and Path(cached_train).is_file()
                    and Path(cached_train).stat().st_size == cached_train_bytes
                )
                val_is_current = isinstance(cached_val_bytes, int) and (
                    (not cached_val and cached_val_bytes == 0)
                    or (
                        bool(cached_val)
                        and Path(cached_val).is_file()
                        and Path(cached_val).stat().st_size == cached_val_bytes
                    )
                )
                if train_is_current and val_is_current:
                    return LocalSplitResult(
                        train_path=existing.get("train", ""),
                        val_path=existing.get("val") or None,
                        train_rows=existing.get("train_rows", 0),
                        val_rows=existing.get("val_rows", 0),
                        run_dir=str(run_dir),
                        manifest_path=str(manifest_path),
                    )
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            pass
    if stale_manifest_for_run:
        manifest_path.unlink(missing_ok=True)

    _write_local_json_atomic(run_dir / "config.json", run_config)

    source_rows = 0
    total_rows = 0
    eligible_agent_counts: Counter[str] = Counter()
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            source_rows += 1
            if allowed_agent_name_set is None:
                total_rows += 1
                continue
            record = _load_json_object(line, input_path, line_number)
            agent_name = _responses_api_agent_name(record)
            if agent_name not in allowed_agent_name_set:
                continue
            eligible_agent_counts[agent_name] += 1
            total_rows += 1

    if total_rows == 0:
        if normalized_agent_names is None:
            raise ValueError(f"Input JSONL has no rows to split: {input_path}")
        raise ValueError(f"No rows in {input_path} matched allowed_agent_names={list(normalized_agent_names)}")

    effective_total = min(total_rows, sample) if sample is not None else total_rows
    if effective_total <= val_holdout:
        limit_detail = f" after sample/max_rows={sample}" if sample is not None else ""
        raise ValueError(
            "Cannot create both training and validation splits: "
            f"{effective_total} eligible rows{limit_detail}, but "
            f"val_holdout={val_holdout}. Increase sample/max_rows or reduce "
            "val_holdout so at least one training row remains."
        )
    train_end = effective_total - val_holdout

    train_dir = run_dir / "train"
    val_dir = run_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_file = train_dir / "train.jsonl"
    val_file = val_dir / "val.jsonl"

    train_count = 0
    val_count = 0
    written_agent_counts: Counter[str] = Counter()
    temporary_train: Path | None = None
    temporary_val: Path | None = None
    try:
        with (
            input_path.open(encoding="utf-8") as input_file,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=train_dir,
                prefix=f".{train_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as train_output,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=val_dir,
                prefix=f".{val_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as val_output,
        ):
            temporary_train = Path(train_output.name)
            temporary_val = Path(val_output.name)
            selected_index = 0
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                agent_name = None
                if allowed_agent_name_set is not None:
                    record = _load_json_object(line, input_path, line_number)
                    agent_name = _responses_api_agent_name(record)
                    if agent_name not in allowed_agent_name_set:
                        continue
                if selected_index >= effective_total:
                    break
                output_line = line if line.endswith("\n") else f"{line}\n"
                if selected_index < train_end:
                    train_output.write(output_line)
                    train_count += 1
                else:
                    val_output.write(output_line)
                    val_count += 1
                if agent_name is not None:
                    written_agent_counts[agent_name] += 1
                selected_index += 1
            for output_file in (train_output, val_output):
                output_file.flush()
                os.fsync(output_file.fileno())

        os.replace(temporary_train, train_file)
        temporary_train = None
        if val_count > 0:
            os.replace(temporary_val, val_file)
            temporary_val = None
        else:
            temporary_val.unlink(missing_ok=True)
            temporary_val = None
            val_file.unlink(missing_ok=True)
    finally:
        if temporary_train is not None:
            temporary_train.unlink(missing_ok=True)
        if temporary_val is not None:
            temporary_val.unlink(missing_ok=True)

    train_path = str(train_file.resolve())
    val_path = str(val_file.resolve()) if val_count > 0 else None
    manifest: dict[str, object] = {
        "train": train_path,
        "val": val_path or "",
        "test": "",
        "mode": "released_jsonl_split",
        "source": str(input_path),
        "run_hash": run_hash,
        "train_rows": train_count,
        "val_rows": val_count,
        "train_bytes": train_file.stat().st_size,
        "val_bytes": val_file.stat().st_size if val_count > 0 else 0,
        "source_rows": source_rows,
        "released_blend_transform_schema": RELEASED_BLEND_TRANSFORM_SCHEMA,
    }
    if normalized_agent_names is not None:
        manifest.update(
            {
                "allowed_agent_names": list(normalized_agent_names),
                "matched_rows": total_rows,
                "filtered_rows": source_rows - total_rows,
                "agent_counts": dict(sorted(written_agent_counts.items())),
                "eligible_agent_counts": dict(sorted(eligible_agent_counts.items())),
            }
        )
    _write_local_json_atomic(manifest_path, manifest)

    return LocalSplitResult(
        train_path=train_path,
        val_path=val_path,
        train_rows=train_count,
        val_rows=val_count,
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
    )


def strip_dapo_instruction_wrapper(text: str) -> str:
    stripped = text
    if _DAPO_INSTRUCTION_PREFIX in stripped:
        stripped = stripped.split(_DAPO_INSTRUCTION_PREFIX, 1)[1]
    if _DAPO_INSTRUCTION_SUFFIX in stripped:
        stripped = stripped.rsplit(_DAPO_INSTRUCTION_SUFFIX, 1)[0]
    return stripped.strip()


def reconstruct_question_placeholder(placeholder: Mapping[str, Any], bare: str) -> str:
    if placeholder.get("mode") == "canonical":
        return f"{placeholder.get('lead', '')}{bare}{placeholder.get('trail', '')}"
    return f"{placeholder.get('prefix', '')}{bare}{placeholder.get('suffix', '')}"


def _unwrap_question_placeholder_answer(raw: Any) -> str:
    if not isinstance(raw, str):
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)

    stripped = raw.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return stripped
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
        return str(parsed)
    return stripped


def _question_placeholder_source_row(
    placeholder: Mapping[str, Any],
    sources: Mapping[tuple[str, str], Any],
) -> tuple[str, Mapping[str, Any]]:
    dataset = placeholder.get("dataset")
    split = placeholder.get("split")
    raw_row_idx = placeholder.get("row")
    reference = f"dataset={dataset!r}, split={split!r}, row={raw_row_idx!r}"

    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError(f"Placeholder has an invalid dataset ({reference})")
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"Placeholder has an invalid split ({reference})")
    if isinstance(raw_row_idx, int) and not isinstance(raw_row_idx, bool):
        row_idx = raw_row_idx
    elif isinstance(raw_row_idx, str) and raw_row_idx.strip().lstrip("-").isdigit():
        row_idx = int(raw_row_idx)
    else:
        raise ValueError(f"Placeholder has an invalid row index ({reference})")

    source_key = (dataset, split)
    if source_key not in sources or sources[source_key] is None:
        available = ", ".join(f"{source_dataset}/{source_split}" for source_dataset, source_split in sorted(sources))
        raise ValueError(f"No loaded source for placeholder ({reference}); available sources: {available or 'none'}")
    table = sources[source_key]
    try:
        source_rows = len(table)
    except TypeError as error:
        raise ValueError(f"Placeholder source is not indexable ({reference})") from error
    if row_idx < 0 or row_idx >= source_rows:
        raise ValueError(f"Placeholder row is out of bounds ({reference}); source has {source_rows} rows")
    try:
        source_row = table[row_idx]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read placeholder source row ({reference}): {error}") from error
    if not isinstance(source_row, Mapping):
        raise ValueError(f"Placeholder source row must be an object ({reference}), got {type(source_row).__name__}")
    return dataset, source_row


def restore_question_placeholder_row(
    row: dict[str, Any],
    sources: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    """Restore one masked Lightning question without mutating the input row."""
    if QUESTION_PLACEHOLDER_KEY not in row:
        return row

    placeholder = row[QUESTION_PLACEHOLDER_KEY]
    if not isinstance(placeholder, Mapping):
        raise ValueError(f"{QUESTION_PLACEHOLDER_KEY} must be an object, got {type(placeholder).__name__}")

    dataset, source_row = _question_placeholder_source_row(placeholder, sources)
    reference = (
        f"dataset={placeholder.get('dataset')!r}, split={placeholder.get('split')!r}, row={placeholder.get('row')!r}"
    )
    prompt = source_row.get("prompt")
    if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], Mapping):
        raise ValueError(f"Placeholder source row has no prompt[0] object ({reference})")
    content = prompt[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Placeholder source row has no non-empty prompt[0].content ({reference})")
    reward_model = source_row.get("reward_model")
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError(f"Placeholder source row has no reward_model.ground_truth ({reference})")

    bare = strip_dapo_instruction_wrapper(content) if dataset == DAPO_QUESTION_DATASET else content.strip()
    question = reconstruct_question_placeholder(placeholder, bare)
    answer = _unwrap_question_placeholder_answer(reward_model["ground_truth"])

    restored = deepcopy(row)
    restored.pop(QUESTION_PLACEHOLDER_KEY, None)
    restored["question"] = question
    restored["expected_answer"] = answer
    responses = restored.get("responses_create_params")
    if isinstance(responses, dict):
        inputs = responses.get("input")
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
            inputs[0]["content"] = question
    for matched in restored.get("matched_sources") or []:
        if isinstance(matched, dict) and "expected_answer" in matched:
            matched["expected_answer"] = answer
    return restored


def load_question_placeholder_sources() -> dict[tuple[str, str], Any]:
    from datasets import load_dataset

    return {(dataset, split): load_dataset(dataset, split=split) for dataset, split in QUESTION_PLACEHOLDER_SOURCES}


def _question_placeholder_restore_marker(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.restore.json")


def question_placeholder_restore_is_current(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int | None = None,
) -> bool:
    """Check that a restored JSONL and its atomic completion marker are current."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    marker_path = _question_placeholder_restore_marker(output_path)
    if not input_path.is_file() or not output_path.is_file() or not marker_path.is_file():
        return False

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        source_stat = input_path.stat()
        output_stat = output_path.stat()
    except (OSError, json.JSONDecodeError, TypeError):
        return False

    expected = {
        "schema": RELEASED_BLEND_TRANSFORM_SCHEMA,
        "source_path": str(input_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "max_rows": max_rows,
        "output_size": output_stat.st_size,
        "output_mtime_ns": output_stat.st_mtime_ns,
    }
    return all(marker.get(key) == value for key, value in expected.items()) and all(
        isinstance(marker.get(key), int) and marker[key] >= 0 for key in ("total_rows", "restored_rows")
    )


def _write_question_placeholder_restore_marker(
    input_path: Path,
    output_path: Path,
    *,
    max_rows: int | None,
    total_rows: int,
    restored_rows: int,
    stripped_runtime_metadata_counts: Mapping[str, int],
) -> None:
    source_stat = input_path.stat()
    output_stat = output_path.stat()
    marker: dict[str, object] = {
        "schema": RELEASED_BLEND_TRANSFORM_SCHEMA,
        "source_path": str(input_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "max_rows": max_rows,
        "output_size": output_stat.st_size,
        "output_mtime_ns": output_stat.st_mtime_ns,
        "total_rows": total_rows,
        "restored_rows": restored_rows,
        "stripped_runtime_metadata_counts": dict(sorted(stripped_runtime_metadata_counts.items())),
    }
    _write_local_json_atomic(_question_placeholder_restore_marker(output_path), marker)


def restore_question_placeholder_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sources: Mapping[tuple[str, str], Any] | None = None,
    max_rows: int | None = None,
) -> tuple[int, int]:
    """Restore every masked question and atomically replace the output JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded_sources = sources
    total = 0
    restored = 0
    stripped_runtime_metadata_counts: Counter[str] = Counter()
    temporary_path: Path | None = None
    try:
        with (
            input_path.open(encoding="utf-8") as input_file,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file,
        ):
            temporary_path = Path(output_file.name)
            for line_number, line in enumerate(input_file, start=1):
                if max_rows is not None and total >= max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {error.msg}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {input_path}:{line_number}, got {type(row).__name__}")
                if QUESTION_PLACEHOLDER_KEY in row:
                    try:
                        if loaded_sources is None:
                            loaded_sources = load_question_placeholder_sources()
                        candidate = restore_question_placeholder_row(row, loaded_sources)
                    except Exception as error:
                        raise ValueError(f"Failed to restore {input_path}:{line_number}: {error}") from error
                    if QUESTION_PLACEHOLDER_KEY in candidate:
                        raise ValueError(
                            f"Failed to restore {input_path}:{line_number}: "
                            f"{QUESTION_PLACEHOLDER_KEY} remains in the output row"
                        )
                    row = candidate
                    restored += 1
                removed_keys = NEMO_GYM_RUNTIME_METADATA_KEYS.intersection(row)
                for key in removed_keys:
                    row.pop(key)
                stripped_runtime_metadata_counts.update(removed_keys)
                output_file.write(json.dumps(row) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    _write_question_placeholder_restore_marker(
        input_path,
        output_path,
        max_rows=max_rows,
        total_rows=total,
        restored_rows=restored,
        stripped_runtime_metadata_counts=stripped_runtime_metadata_counts,
    )
    return total, restored


def _hf_repo_id(path: str) -> str:
    if not path.startswith("hf://"):
        raise ValueError(f"Released JSONL RL prep expects an hf:// dataset path, got {path!r}")
    return path.removeprefix("hf://")


def run_released_jsonl_blend(
    *,
    blend: DataBlend,
    output_dir: str | Path,
    val_holdout: int = 1000,
    sample: int | None = None,
    force: bool = False,
    jsonl_name: str = "rlvr.jsonl",
    allowed_agent_names: Iterable[str] | None = None,
) -> LocalSplitResult:
    """Download, restore, filter, split, and manifest the Lightning blend."""
    from huggingface_hub import snapshot_download

    if sample is not None and sample <= 0:
        raise ValueError(f"sample/max_rows must be greater than zero, got {sample}")
    if val_holdout <= 0:
        raise ValueError(f"val_holdout must be greater than zero, got {val_holdout}")
    if sample is not None and val_holdout > 0 and sample <= val_holdout:
        raise ValueError(
            f"sample/max_rows ({sample}) must be greater than val_holdout "
            f"({val_holdout}) so Lightning RL prep produces both train and validation data"
        )
    if blend.datasets is None or len(blend.datasets) != 1:
        raise ValueError(
            "Released JSONL RL prep expects exactly one dataset in the blend, "
            f"got {0 if blend.datasets is None else len(blend.datasets)}"
        )

    repo_id = _hf_repo_id(blend.datasets[0].path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_cache_key = hashlib.sha256(repo_id.encode()).hexdigest()[:16]
    snapshot_dir = output_dir / ".hf_snapshot" / repo_cache_key
    downloaded_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(snapshot_dir),
            allow_patterns=["*.jsonl"],
            force_download=force,
        )
    )
    snapshot_jsonl = downloaded_dir / jsonl_name
    if not snapshot_jsonl.is_file():
        found = sorted(path.name for path in snapshot_dir.glob("*.jsonl"))
        raise FileNotFoundError(
            f"{repo_id} snapshot at {snapshot_dir} is missing {jsonl_name}. Found: {found or 'no jsonl files'}"
        )

    restore_sample = None if allowed_agent_names is not None else sample
    sample_tag = f"_n{restore_sample}" if restore_sample is not None else ""
    restored_jsonl = output_dir / ".restored" / repo_cache_key / f"{Path(jsonl_name).stem}{sample_tag}.jsonl"
    if force or not question_placeholder_restore_is_current(
        snapshot_jsonl,
        restored_jsonl,
        max_rows=restore_sample,
    ):
        restore_question_placeholder_jsonl(
            snapshot_jsonl,
            restored_jsonl,
            max_rows=restore_sample,
        )

    return split_local_jsonl(
        restored_jsonl,
        output_dir,
        val_holdout=val_holdout,
        sample=sample if allowed_agent_names is not None else None,
        force=force,
        allowed_agent_names=allowed_agent_names,
    )


__all__ = [
    "DAPO_QUESTION_DATASET",
    "LocalSplitResult",
    "QUESTION_PLACEHOLDER_KEY",
    "SKYWORK_QUESTION_DATASET",
    "question_placeholder_restore_is_current",
    "restore_question_placeholder_jsonl",
    "restore_question_placeholder_row",
    "run_released_jsonl_blend",
    "split_local_jsonl",
]
