# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate the config-driven persona QASynth pipeline."""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from nemotron.steps.sdg.qasynth.runtime.answers import generate_answers
from nemotron.steps.sdg.qasynth.runtime.io import canonical_hash, read_jsonl, redact_config, require_file, write_jsonl
from nemotron.steps.sdg.qasynth.runtime.lexical import deduplicate as lexical_deduplicate
from nemotron.steps.sdg.qasynth.runtime.semantic import deduplicate_embeddings, embed_questions
from nemotron.steps.sdg.qasynth.runtime.sft import build_sft_records, prepare_answer_seed, sample_aligned_datasets

STAGES = (
    "questions",
    "lexical_dedup",
    "semantic_dedup",
    "answer_seed",
    "answers",
    "build_sft",
    "sample",
)


def _stage_list(value: Any) -> list[str]:
    """Accept YAML lists and the bracket-list strings produced by CLI overrides."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return list(value or [])


def _identity_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove execution controls that may legitimately change between resumes."""
    identity = redact_config(config)
    run = identity.get("run") or {}
    for key in ("stages", "resume", "overwrite"):
        run.pop(key, None)
    return identity


def validate_config(config: dict[str, Any]) -> None:
    run = config.get("run") or {}
    if not run.get("experiment_name"):
        raise ValueError("run.experiment_name is required")
    stages = _stage_list(run.get("stages"))
    if not stages:
        raise ValueError("run.stages must select 'all' or at least one named stage")
    unknown = set(stages) - set(STAGES) - {"all"}
    if unknown:
        raise ValueError(f"Unknown QASynth stages: {sorted(unknown)}")
    if "all" in stages and len(stages) != 1:
        raise ValueError("run.stages may contain 'all' or named stages, not both")
    languages = config.get("languages") or {}
    if not languages:
        raise ValueError("at least one language must be configured")
    for name, language in languages.items():
        if not language.get("locale") or not language.get("display_name"):
            raise ValueError(f"language {name!r} requires locale and display_name")
    models = config.get("models") or {}
    for group in ("question_models", "answer_models"):
        selected = config.get(group) or []
        if not selected:
            raise ValueError(f"{group} must select at least one model")
        missing = set(selected) - set(models)
        if missing:
            raise ValueError(f"{group} references unknown models: {sorted(missing)}")
    if len(config["answer_models"]) < 3:
        raise ValueError("persona QASynth requires at least three answer models for voting")
    response_teachers = (config.get("sft") or {}).get("response_teachers") or []
    if not response_teachers:
        raise ValueError("sft.response_teachers must select at least one answer model")
    missing_teachers = set(response_teachers) - set(config["answer_models"])
    if missing_teachers:
        raise ValueError(
            "sft.response_teachers references models not selected in answer_models: "
            f"{sorted(missing_teachers)}"
        )
    for key, model in models.items():
        for field in ("model", "endpoint", "api_key_env"):
            if not model.get(field):
                raise ValueError(f"model {key!r} requires {field}")
        if not str(model["api_key_env"]).replace("_", "").isalnum():
            raise ValueError(f"model {key!r} api_key_env must be an environment-variable name")
    semantic = config.get("semantic_dedup") or {}
    if not 0 <= float(semantic.get("threshold", 0.965)) <= 1:
        raise ValueError("semantic_dedup.threshold must be between zero and one")


def _records_from_result(result: Any) -> list[dict[str, Any]]:
    dataset = result.load_dataset() if hasattr(result, "load_dataset") else getattr(result, "dataset", None)
    if dataset is None:
        raise ValueError("Data Designer returned no dataset")
    if isinstance(dataset, list):
        return dataset
    if hasattr(dataset, "to_pandas"):
        dataset = dataset.to_pandas()
    if hasattr(dataset, "to_dict"):
        return dataset.to_dict(orient="records")
    raise TypeError(f"Unsupported Data Designer dataset type: {type(dataset).__name__}")


def generate_questions(
    *,
    model_key: str,
    model: dict[str, Any],
    language_key: str,
    language: dict[str, Any],
    count: int,
    artifact_root: Path,
    dataset_name: str,
    resume: bool,
    random_seed: int,
    max_parallel: int,
    buffer_size: int,
) -> list[dict[str, Any]]:
    import data_designer.config as dd
    from data_designer.config.models import ModelProvider
    from data_designer.interface import DataDesigner

    from nemotron.steps.sdg.plugins.qasynth.config import QASynthMCQConfig

    inference = dict((model.get("question") or {}).get("inference") or {})
    provider_name = f"qasynth-{model_key}"
    provider = ModelProvider(
        name=provider_name,
        endpoint=model["endpoint"],
        provider_type=model.get("provider_type", "openai"),
        api_key=model["api_key_env"],
        extra_headers=model.get("extra_headers"),
    )
    model_config = dd.ModelConfig(
        alias="question_model",
        model=model["model"],
        provider=provider_name,
        skip_health_check=bool(model.get("skip_health_check", False)),
        inference_parameters=dd.ChatCompletionInferenceParams(**inference),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.add_column(
        dd.SamplerColumnConfig(
            name="persona",
            drop=False,
            sampler_type=dd.SamplerType.PERSON,
            params=dd.PersonSamplerParams(locale=language["locale"], with_synthetic_personas=True),
        )
    )
    builder.add_column(
        QASynthMCQConfig(
            name="conversation",
            drop=False,
            model_alias="question_model",
            language=language["display_name"],
            random_seed=random_seed,
        )
    )
    client = DataDesigner(artifact_path=artifact_root, model_providers=[provider])
    client.set_run_config(
        dd.RunConfig(
            non_inference_max_parallel_workers=max_parallel,
            buffer_size=buffer_size,
            disable_early_shutdown=True,
        )
    )
    kwargs: dict[str, Any] = {"config_builder": builder, "num_records": count, "dataset_name": dataset_name}
    if "resume" in inspect.signature(client.create).parameters:
        kwargs["resume"] = dd.ResumeMode.IF_POSSIBLE if resume else dd.ResumeMode.NEVER
    result = client.create(**kwargs)
    records = _records_from_result(result)
    for record in records:
        record["source_model"] = model_key
        record["language"] = language_key
    return records


class QASynthPipeline:
    """Run selected QASynth stages against one immutable experiment configuration."""

    def __init__(self, config: dict[str, Any]) -> None:
        config.setdefault("run", {})["stages"] = _stage_list((config.get("run") or {}).get("stages"))
        validate_config(config)
        self.config = config
        run = config["run"]
        self.root = Path(run["output_root"]).expanduser().resolve() / run["experiment_name"]
        self.resume = bool(run.get("resume", True))
        self.overwrite = bool(run.get("overwrite", False))
        self.summary: dict[str, Any] = {"stages": {}}

    def run(self) -> None:
        stages = list(self.config["run"].get("stages") or ["all"])
        selected = list(STAGES) if stages == ["all"] else stages
        self._prepare_experiment()
        functions: dict[str, Callable[[], None]] = {
            "questions": self._questions,
            "lexical_dedup": self._lexical,
            "semantic_dedup": self._semantic,
            "answer_seed": self._answer_seed,
            "answers": self._answers,
            "build_sft": self._build_sft,
            "sample": self._sample,
        }
        for stage in STAGES:
            if stage in selected:
                print(f"[qasynth] starting stage: {stage}", flush=True)
                functions[stage]()
                self._write_summary()

    def _prepare_experiment(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state_path = self.root / "run.json"
        config_hash = canonical_hash(_identity_config(self.config))
        if state_path.exists():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != config_hash and not self.overwrite:
                raise ValueError(
                    f"Experiment {self.root} was created with a different config; choose a new experiment_name "
                    "or set run.overwrite=true"
                )
        try:
            dd_version = version("data-designer")
        except PackageNotFoundError:
            dd_version = "not-installed"
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        state = {
            "config_hash": config_hash,
            "config": redact_config(self.config),
            "data_designer_version": dd_version,
            "nemotron_commit": commit,
            "sovereign_source_commit": "88afa6e30ff123dd0abeba3a77555e369600c036",
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_summary(self) -> None:
        (self.root / "summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _questions(self) -> None:
        cfg = self.config
        stats: dict[str, int] = {}
        for model_key in cfg["question_models"]:
            for language_key, language in cfg["languages"].items():
                output = self.root / "questions" / model_key / language_key / "records.jsonl"
                if self.resume and not self.overwrite and output.exists():
                    stats[f"{model_key}/{language_key}"] = len(read_jsonl(output))
                    continue
                records = generate_questions(
                    model_key=model_key,
                    model=cfg["models"][model_key],
                    language_key=language_key,
                    language=language,
                    count=int(cfg["question_generation"]["num_records"]),
                    artifact_root=output.parent / "artifacts",
                    dataset_name=f"{cfg['run']['experiment_name']}_{model_key}_{language_key}",
                    resume=self.resume and not self.overwrite,
                    random_seed=int(cfg["run"]["seed"]),
                    max_parallel=int(cfg["question_generation"]["max_parallel_requests"]),
                    buffer_size=int(cfg["question_generation"]["buffer_size"]),
                )
                stats[f"{model_key}/{language_key}"] = write_jsonl(output, records)
        self.summary["stages"]["questions"] = stats

    def _lexical(self) -> None:
        cfg = self.config["lexical_dedup"]
        stage_stats: dict[str, Any] = {}
        for language in self.config["languages"]:
            accepted: list[dict[str, Any]] = []
            per_model: dict[str, Any] = {}
            for model in self.config["question_models"]:
                path = require_file(self.root / "questions" / model / language / "records.jsonl", "lexical_dedup")
                records, stats = lexical_deduplicate(
                    read_jsonl(path),
                    language=language,
                    source_model=model,
                    threshold=float(cfg["threshold"]),
                    shingle_size=int(cfg["shingle_size"]),
                    permutations=int(cfg["permutations"]),
                    bands=int(cfg["bands"]),
                    seed=int(cfg["seed"]),
                )
                accepted.extend(records)
                per_model[model] = stats
            pooled_input = [
                {
                    "conversation": json.dumps(
                        {
                            "metadata": {
                                "parsed_question": {"question": row["question"], "choices": row["choices"]},
                                **row["metadata"],
                            }
                        }
                    )
                }
                for row in accepted
            ]
            pooled, pooled_stats = lexical_deduplicate(
                pooled_input,
                language=language,
                source_model="pooled",
                threshold=float(cfg["threshold"]),
                shingle_size=int(cfg["shingle_size"]),
                permutations=int(cfg["permutations"]),
                bands=int(cfg["bands"]),
                seed=int(cfg["seed"]),
            )
            # Restore source provenance by query identity from the first pass.
            source_by_identity = {(row["question"], tuple(row["choices"])): row["metadata"] for row in accepted}
            for row in pooled:
                row["metadata"] = source_by_identity[(row["question"], tuple(row["choices"]))]
            write_jsonl(self.root / "lexical" / f"{language}.jsonl", pooled)
            stage_stats[language] = {"models": per_model, "pooled": pooled_stats}
        self.summary["stages"]["lexical_dedup"] = stage_stats

    def _semantic(self) -> None:
        cfg = self.config["semantic_dedup"]
        stage_stats: dict[str, Any] = {}
        for language in self.config["languages"]:
            records = read_jsonl(require_file(self.root / "lexical" / f"{language}.jsonl", "semantic_dedup"))
            embeddings = embed_questions(
                records,
                model_name=cfg["model"],
                device=cfg["device"],
                batch_size=int(cfg["batch_size"]),
            )
            records, stats = deduplicate_embeddings(
                records,
                embeddings,
                threshold=float(cfg["threshold"]),
                method=cfg["method"],
                seed=int(cfg["seed"]),
                chunk_size=int(cfg["chunk_size"]),
            )
            write_jsonl(self.root / "semantic" / f"{language}.jsonl", records)
            stage_stats[language] = stats
        self.summary["stages"]["semantic_dedup"] = stage_stats

    def _answer_seed(self) -> None:
        stats: dict[str, int] = {}
        seed = int(self.config["run"]["seed"])
        for language in self.config["languages"]:
            records = read_jsonl(require_file(self.root / "semantic" / f"{language}.jsonl", "answer_seed"))
            stats[language] = write_jsonl(
                self.root / "answer_seed" / f"{language}.jsonl", prepare_answer_seed(records, seed)
            )
        self.summary["stages"]["answer_seed"] = stats

    def _answers(self) -> None:
        cfg = self.config["answer_generation"]
        stats: dict[str, Any] = {}
        for model_key in self.config["answer_models"]:
            model = {**self.config["models"][model_key], **(self.config["models"][model_key].get("answer") or {})}
            for language in self.config["languages"]:
                records = read_jsonl(require_file(self.root / "answer_seed" / f"{language}.jsonl", "answers"))
                key = f"{model_key}/{language}"
                stats[key] = asyncio.run(
                    generate_answers(
                        records,
                        language=language,
                        model=model,
                        output_path=self.root / "answers" / model_key / language / "answers.jsonl",
                        failure_path=self.root / "answers" / model_key / language / "failures.jsonl",
                        resume=self.resume and not self.overwrite,
                        max_parallel=int(cfg["max_parallel_requests"]),
                        timeout=float(cfg["timeout"]),
                        max_retries=int(cfg["max_retries"]),
                    )
                )
        self.summary["stages"]["answers"] = stats

    def _build_sft(self) -> None:
        cfg = self.config["sft"]
        stats: dict[str, Any] = {}
        for language, language_cfg in self.config["languages"].items():
            answers = {
                model: read_jsonl(
                    require_file(self.root / "answers" / model / language / "answers.jsonl", "build_sft")
                )
                for model in self.config["answer_models"]
            }
            purity = language_cfg["assistant_devanagari_fraction"]
            for teacher in cfg["response_teachers"]:
                records, result = build_sft_records(
                    answers,
                    response_model=teacher,
                    language=language,
                    agreement=cfg["agreement"],
                    min_devanagari_fraction=float(purity["min"]),
                    max_devanagari_fraction=float(purity["max"]),
                )
                write_jsonl(self.root / "sft" / teacher / f"{language}.jsonl", records)
                stats[f"{teacher}/{language}"] = result
        self.summary["stages"]["build_sft"] = stats

    def _sample(self) -> None:
        cfg = self.config["sampling"]
        datasets = {
            teacher: {
                language: read_jsonl(
                    require_file(self.root / "sft" / teacher / f"{language}.jsonl", "sample")
                )
                for language in self.config["languages"]
            }
            for teacher in self.config["sft"]["response_teachers"]
        }
        outputs, stats = sample_aligned_datasets(
            datasets,
            sample_per_language=int(cfg["per_language"]),
            seed=int(cfg["seed"]),
            reasoning_off_fraction=float(cfg["reasoning_off_fraction"]),
            answer_variant=cfg["answer_variant"],
        )
        for teacher, records in outputs.items():
            write_jsonl(self.root / "final" / f"{teacher}.jsonl", records)
        self.summary["stages"]["sample"] = {**stats, "teachers": {key: len(value) for key, value in outputs.items()}}
