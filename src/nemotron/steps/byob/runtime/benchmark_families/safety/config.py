# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety-family config. Relative paths resolve from the BYOB root, as BFCL does."""

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from nemotron.steps.byob.runtime.benchmark_families.safety.eval_contract import (
    PATH1_ADAPTERS,
    PATH1_EVALUATION_MODES,
    PATH2_ADAPTERS,
)
from nemotron.steps.byob.runtime.benchmark_families.safety.ir import ACTIONS, DEFAULT_TARGETS, TARGETS

BYOB_ROOT = Path(__file__).resolve().parents[3]
STAGES = frozenset({"prepare", "generate", "translate", "eval", "all"})
MODES = frozenset({"preview", "assisted", "expert"})
PROFILES = frozenset({"standard", "regulated", "high_assurance"})
MODULE_STRING_FLAGS = frozenset({"confirm_if_detected", "auto_if_evidence_exists"})
SECOND_JUDGE_TRIGGERS = frozenset({"disagreement", "critical_boundary", "low_confidence"})
PRIVACY_LABEL_KINDS = frozenset({"credential", "national_id", "bank_account", "date_of_birth"})
# The same vocabularies Policy Rule IR stores, so a config can name every target and action a rule can.
MODERATION_TARGETS = frozenset(TARGETS)
DECISION_ACTIONS = frozenset(ACTIONS)
DEFAULT_DECISION_ACTIONS = ACTIONS

DEFAULT_BUDGET = {
    "preview_max_cases": 100,
    "assisted_max_cases": 500,
    "cases_per_critical_rule": 12,
    "cases_per_normal_rule": 6,
    "max_review_items": 40,
    "max_turns": 6,
}
DEFAULT_GENERATION = {
    "decision_mix": {"allow": 0.40, "block": 0.40, "caution": 0.10, "escalate": 0.10},
    "case_family_mix": {"clear": 0.35, "boundary": 0.25, "exception": 0.15, "contrast": 0.25},
    "turn_mix": {"single_turn": 0.80, "multi_turn": 0.20},
    "paraphrases_per_cell": 1,
    "modules": {
        "adversarial": False,
        "political_symmetry": "confirm_if_detected",
        "production_replay": "auto_if_evidence_exists",
        "persona": False,
        "paraphrase": True,
    },
}
DEFAULT_PROPOSALS = {
    "source_language": "unspecified",
    "target_languages": [],
    "moderation_targets": list(DEFAULT_TARGETS),
    "decision_actions": list(DEFAULT_DECISION_ACTIONS),
    "domain": "unspecified",
    "country": "unspecified",
    "jurisdiction": "unspecified",
    "language_scripts": {},
}
_SCRIPT_NAME = re.compile(r"^[A-Za-z]{2,24}$")
DEFAULT_ASSURANCE = {"profile": "standard"}
ASSURANCE_KEYS = frozenset({"profile", "review_sample_rate", "second_judge_on"})
GENERATION_KEYS = frozenset({"decision_mix", "case_family_mix", "turn_mix", "paraphrases_per_cell", "modules"})
# Same Curator knobs the MCQ family uses. Safety drops marked near-dups by default.
DEFAULT_SEMANTIC_DEDUP = {
    "enabled": True,
    "model_identifier": "sentence-transformers/all-MiniLM-L6-v2",
    "n_clusters": 1,
    "eps": 0.07,
    "remove_duplicates": True,
}
SEMANTIC_DEDUP_KEYS = frozenset(DEFAULT_SEMANTIC_DEDUP)
EVALUATION_KEYS = frozenset({"path1", "path2"})
ENDPOINT_EVAL_KEYS = frozenset(
    {
        "adapter",
        "endpoint",
        "base_url",
        "api_key_env",
        "headers_env",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "max_concurrency",
        "capabilities",
        "decision_map",
        "decision_format",
        "smoke_message",
    }
)
PATH1_TEST_KEYS = frozenset({"adapter", "responses", "file"})
PATH1_LIVE_KEYS = ENDPOINT_EVAL_KEYS | frozenset(
    {"model", "evaluation_mode", "system_prompt", "request_options", "content_path"}
)
PATH2_MOCK_KEYS = frozenset({"adapter", "log", "file"})
PATH2_LIVE_KEYS = ENDPOINT_EVAL_KEYS | frozenset(
    {
        "config_id",
        "model",
        "api_format",
        "options",
        "decision_path",
        "rails_path",
    }
)
MODEL_IDENTITY_KEYS = ("alias", "provider", "model", "canonical_id")
# Same credential names BFCL refuses, so neither family writes a secret into a run directory.
MODEL_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
    }
)
MODEL_SECRET_SUFFIXES = ("_api_key", "_password", "_secret", "_access_token", "_auth_token")
PLACEHOLDER_MARKER = "REPLACE_ME"
_MIX_TOLERANCE = 1e-6
_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Read by the shared dispatcher, not by the safety config.
DISPATCH_KEYS = frozenset({"skip_until"})
TOP_LEVEL_KEYS = DISPATCH_KEYS | {
    "family",
    "expt_name",
    "stage",
    "config_status",
    "mode",
    "input_dir",
    "output_dir",
    "assurance",
    "budget",
    "generation",
    "quality_gates",
    "proposals",
    "languages",
    "privacy",
    "generation_model_config",
    "judge_model_configs",
    "evaluation",
    "require_plan_confirmation",
    "semantic_deduplication_config",
}
QUALITY_GATE_FAILURE_MODES = frozenset({"silver", "fail"})
QUALITY_GATE_KEYS = frozenset(
    {
        "min_total_cases",
        "min_development_cases",
        "min_cases_per_policy",
        "min_cases_per_action",
        "min_cases_per_language",
        "on_failure",
    }
)


class ConfigError(ValueError):
    """The safety config cannot start a run."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        super().__init__("\n".join(messages))


@dataclass(frozen=True)
class ResolvedConfig:
    """Config after defaults are filled and paths are anchored to the BYOB root."""

    family: str
    expt_name: str
    stage: str
    mode: str
    input_dir: Path
    output_dir: Path
    run_dir: Path
    assurance: dict[str, Any]
    budget: dict[str, Any]
    generation: dict[str, Any]
    proposals: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # stage is omitted: the dispatcher may run a different stage than the YAML names.
        data = {
            "family": self.family,
            "expt_name": self.expt_name,
            "config_status": "resolved",
            "mode": self.mode,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "assurance": self.assurance,
            "budget": self.budget,
            "generation": self.generation,
            "proposals": self.proposals,
        }
        data.update(self.extras)
        return data


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys at every depth."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping", node.start_mark, "found an unhashable key", key_node.start_mark
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_resolved(config_path: Path) -> ResolvedConfig:
    """Read and validate one safety config."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError([f"Config file does not exist: {config_path}"]) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError([f"Config file cannot be read as UTF-8 text: {config_path}: {exc}"]) from exc
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - _UniqueKeyLoader extends SafeLoader
    except yaml.YAMLError as exc:
        raise ConfigError([f"Config file is not valid YAML: {exc}"]) from exc
    if not isinstance(loaded, dict):
        raise ConfigError(["Config file must contain a YAML mapping."])
    return resolve_config(loaded)


def resolve_config(raw: dict[str, Any]) -> ResolvedConfig:
    """Validate a mapping and fill the safety defaults."""
    messages: list[str] = []
    _json_safe(raw, "config", messages)
    _reject_placeholders(raw, "config", messages)
    if messages:
        raise ConfigError(messages)

    unknown = sorted(str(key) for key in set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        messages.append("Unknown config keys: " + ", ".join(unknown) + ".")
    if raw.get("family") != "safety":
        messages.append(f'family must be "safety"; received {raw.get("family")!r}.')
    config_status = raw.get("config_status")
    if config_status == "template":
        messages.append(
            "config_status is template, so this file is not runnable. Copy it, fill in every value, "
            "then set config_status: resolved or remove the line."
        )
    elif config_status is not None and config_status != "resolved":
        messages.append(f'config_status must be "resolved" or omitted; received {config_status!r}.')

    expt_name = raw.get("expt_name")
    if not isinstance(expt_name, str) or _EXPERIMENT_NAME.fullmatch(expt_name) is None:
        messages.append(
            "expt_name must be one file name made of letters, numbers, dots, underscores, "
            "or hyphens, and it cannot be '.' or '..'."
        )
        expt_name = "invalid"

    stage = _choice(raw.get("stage", "prepare"), STAGES, "stage", messages)
    mode = _choice(raw.get("mode", "assisted"), MODES, "mode", messages)

    input_dir = _directory(raw.get("input_dir"), "input_dir", messages)
    output_dir = _path(raw.get("output_dir"), "output_dir", messages)
    if input_dir is not None and output_dir is not None and _overlaps(input_dir, output_dir):
        messages.append(
            "input_dir and output_dir must not be the same directory or inside each other, "
            "because ingest would read the run's own outputs."
        )
    assurance = _assurance(raw.get("assurance", {}), messages)
    budget = _budget(raw.get("budget", {}), messages)
    generation = _generation(raw.get("generation", {}), messages)
    proposals = _proposals(raw.get("proposals", {}), messages)
    _apply_languages(raw.get("languages"), proposals, messages)
    _check_decision_mix(generation["decision_mix"], proposals["decision_actions"], messages)
    extras = _passthrough(raw, messages)
    _check_privacy_language(proposals, extras, messages)
    _check_budget_and_gates(mode, budget, generation, extras.get("quality_gates"), messages)

    if messages or input_dir is None or output_dir is None:
        raise ConfigError(messages)
    return ResolvedConfig(
        family="safety",
        expt_name=expt_name,
        stage=stage,
        mode=mode,
        input_dir=input_dir,
        output_dir=output_dir,
        run_dir=output_dir / expt_name,
        assurance=assurance,
        budget=budget,
        generation=generation,
        proposals=proposals,
        extras=extras,
    )


def _json_safe(value: Any, path: str, messages: list[str]) -> None:
    """Config values must survive JSON hashing unchanged: no dates, binary, or non-string keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                messages.append(f"{path} has a non-string key {key!r}; quote it in YAML.")
                continue
            _json_safe(child, f"{path}.{key}", messages)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _json_safe(child, f"{path}[{index}]", messages)
    elif isinstance(value, float):
        if not math.isfinite(value):
            messages.append(f"{path} must be a finite number.")
    elif value is not None and not isinstance(value, (str, int, bool)):
        messages.append(f"{path} has unsupported YAML type {type(value).__name__}; quote it as a string.")


def _reject_placeholders(value: Any, path: str, messages: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{path}.{key}", messages)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{path}[{index}]", messages)
    elif isinstance(value, str) and PLACEHOLDER_MARKER in value:
        messages.append(f"{path} still contains a {PLACEHOLDER_MARKER} placeholder.")


def _choice(value: Any, allowed: frozenset[str], name: str, messages: list[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        messages.append(f"{name} must be one of {', '.join(sorted(allowed))}; received {value!r}.")
        return sorted(allowed)[0]
    return value


def _path(value: Any, name: str, messages: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        messages.append(f"{name} must be a path.")
        return None
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = BYOB_ROOT / path
    return path.resolve()


def _directory(value: Any, name: str, messages: list[str]) -> Path | None:
    path = _path(value, name, messages)
    if path is not None and not path.is_dir():
        messages.append(f"{name} does not exist or is not a directory: {path}")
        return None
    return path


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _unknown_keys(name: str, value: dict[str, Any], allowed: set[str] | frozenset[str], messages: list[str]) -> None:
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        messages.append(f"Unknown {name} keys: {', '.join(unknown)}. Allowed keys: {', '.join(sorted(allowed))}.")


def _mapping(value: Any, name: str, messages: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        messages.append(f"{name} must be a mapping.")
        return None
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _apply_languages(value: Any, proposals: dict[str, Any], messages: list[str]) -> None:
    """Apply the documented language block without changing source into target work."""
    if value is None:
        return
    supplied = _mapping(value, "languages", messages)
    if supplied is None:
        return
    allowed = {"source_language", "target_languages"}
    _unknown_keys("languages", supplied, allowed, messages)
    if "source_language" in supplied:
        source = supplied["source_language"]
        if not isinstance(source, str) or not source.strip():
            messages.append("languages.source_language must be a non-empty string.")
        else:
            configured = proposals.get("source_language")
            if isinstance(configured, str) and configured.casefold() not in ("", "unspecified", source.casefold()):
                messages.append("languages.source_language and proposals.source_language must match.")
            else:
                proposals["source_language"] = source.strip()
    if "target_languages" in supplied:
        targets = supplied["target_languages"]
        _string_list(targets, "languages.target_languages", messages)
        if isinstance(targets, list) and all(isinstance(item, str) and item.strip() for item in targets):
            codes = [item.strip() for item in targets]
            configured = proposals.get("target_languages")
            if configured and [str(item).casefold() for item in configured] != [item.casefold() for item in codes]:
                messages.append("languages.target_languages and proposals.target_languages must match.")
            else:
                proposals["target_languages"] = codes


def _string_list(value: Any, name: str, messages: list[str], allowed: frozenset[str] | None = None) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        messages.append(f"{name} must be a list of non-empty strings.")
        return
    if len(set(value)) != len(value):
        messages.append(f"{name} must not repeat an entry.")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            messages.append(
                f"{name} has unsupported entries {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}."
            )


def _assurance(value: Any, messages: list[str]) -> dict[str, Any]:
    supplied = _mapping(value, "assurance", messages)
    if supplied is None:
        return dict(DEFAULT_ASSURANCE)
    _unknown_keys("assurance", supplied, ASSURANCE_KEYS, messages)
    merged = {**DEFAULT_ASSURANCE, **supplied}
    _choice(merged["profile"], PROFILES, "assurance.profile", messages)
    if "review_sample_rate" in merged:
        rate = merged["review_sample_rate"]
        if not _number(rate) or not 0 <= rate <= 1:
            messages.append("assurance.review_sample_rate must be a share between 0 and 1.")
    if "second_judge_on" in merged:
        _string_list(merged["second_judge_on"], "assurance.second_judge_on", messages, SECOND_JUDGE_TRIGGERS)
    return merged


def _generation(value: Any, messages: list[str]) -> dict[str, Any]:
    supplied = _mapping(value, "generation", messages)
    if supplied is None:
        supplied = {}
    _unknown_keys("generation", supplied, GENERATION_KEYS, messages)
    modules = dict(DEFAULT_GENERATION["modules"])
    module_values = _mapping(supplied.get("modules", {}), "generation.modules", messages)
    if module_values is not None:
        _unknown_keys("generation.modules", module_values, set(modules), messages)
        modules.update(module_values)
    for name, flag in modules.items():
        if not isinstance(flag, bool) and not (isinstance(flag, str) and flag in MODULE_STRING_FLAGS):
            messages.append(
                f"generation.modules.{name} must be true, false, or one of "
                f"{', '.join(sorted(MODULE_STRING_FLAGS))}; received {flag!r}."
            )
    variants = supplied.get("paraphrases_per_cell", DEFAULT_GENERATION["paraphrases_per_cell"])
    if isinstance(variants, bool) or not isinstance(variants, int) or not 1 <= variants <= 5:
        messages.append("generation.paraphrases_per_cell must be a whole number from 1 through 5.")
        variants = DEFAULT_GENERATION["paraphrases_per_cell"]
    mixes = {
        key: _mix(
            f"generation.{key}",
            DEFAULT_GENERATION[key],
            supplied.get(key, {}),
            messages,
            DECISION_ACTIONS if key == "decision_mix" else None,
        )
        for key in ("decision_mix", "case_family_mix", "turn_mix")
    }
    return {**mixes, "paraphrases_per_cell": variants, "modules": modules}


def _check_decision_mix(mix: Any, actions: Any, messages: list[str]) -> None:
    """Every action the mix asks for must be one the benchmark labels with."""
    if not isinstance(mix, dict) or not isinstance(actions, list):
        return
    enabled = {action for action in actions if isinstance(action, str)}
    planned = sorted(action for action, share in mix.items() if _number(share) and share > 0 and action not in enabled)
    if planned:
        messages.append(
            f"generation.decision_mix gives a share to {', '.join(planned)}, which proposals.decision_actions "
            "does not enable. Add the action to decision_actions or set its share to 0."
        )


def _budget(value: Any, messages: list[str]) -> dict[str, Any]:
    supplied = _mapping(value, "budget", messages)
    if supplied is None:
        return dict(DEFAULT_BUDGET)
    _unknown_keys("budget", supplied, set(DEFAULT_BUDGET), messages)
    merged = {**DEFAULT_BUDGET, **supplied}
    for key, number in merged.items():
        minimum = 1 if key == "max_turns" else 0
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            messages.append(f"budget.{key} must be a whole number {minimum} or greater.")
    return merged


def _check_budget_and_gates(
    mode: str,
    budget: Mapping[str, Any],
    generation: Mapping[str, Any],
    quality_gates: Any,
    messages: list[str],
) -> None:
    """Refuse minima and mix targets that the configured maxima cannot satisfy."""
    if any(isinstance(budget.get(key), bool) or not isinstance(budget.get(key), int) for key in DEFAULT_BUDGET):
        return
    cap_key = "preview_max_cases" if mode == "preview" else "assisted_max_cases"
    case_cap = int(budget[cap_key])
    mix = generation.get("turn_mix") if isinstance(generation.get("turn_mix"), Mapping) else {}
    mix_share = mix.get("multi_turn") if isinstance(mix, Mapping) else 0
    if (
        isinstance(mix_share, (int, float))
        and not isinstance(mix_share, bool)
        and mix_share > 0
        and int(budget["max_turns"]) < 2
    ):
        messages.append("generation.turn_mix.multi_turn is greater than 0, so budget.max_turns must be at least 2.")
    if not isinstance(quality_gates, Mapping):
        return
    min_total = _optional_gate_count(quality_gates, "min_total_cases")
    min_dev = _optional_gate_count(quality_gates, "min_development_cases")
    min_policy = _optional_gate_count(quality_gates, "min_cases_per_policy")
    if min_total is None or min_dev is None or min_policy is None:
        return
    for label, amount in (
        ("min_total_cases", min_total),
        ("min_development_cases", min_dev),
    ):
        if amount > case_cap:
            messages.append(
                f"quality_gates.{label} is {amount}, which exceeds budget.{cap_key} ({case_cap}). "
                "Lower the minimum or raise the cap."
            )
    if min_policy > case_cap:
        messages.append(
            f"quality_gates.min_cases_per_policy is {min_policy}, which exceeds "
            f"budget.{cap_key} ({case_cap}). Lower the minimum or raise the cap."
        )
    action_mins = quality_gates.get("min_cases_per_action")
    if isinstance(action_mins, Mapping):
        action_total = 0
        valid = True
        for amount in action_mins.values():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                valid = False
                break
            action_total += amount
        if valid and action_total > case_cap:
            messages.append(
                f"quality_gates.min_cases_per_action sums to {action_total}, which exceeds "
                f"budget.{cap_key} ({case_cap}). Lower the minima or raise the cap."
            )
    language_mins = quality_gates.get("min_cases_per_language")
    if isinstance(language_mins, Mapping):
        for name, amount in language_mins.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                continue
            if amount > case_cap:
                messages.append(
                    f"quality_gates.min_cases_per_language.{name} is {amount}, which exceeds "
                    f"budget.{cap_key} ({case_cap}). Lower the minimum or raise the cap."
                )
                break


def _optional_gate_count(gates: Mapping[str, Any], key: str) -> int | None:
    if key not in gates:
        return 0
    amount = gates[key]
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        return None
    return amount


def _mix(
    name: str,
    defaults: dict[str, float],
    value: Any,
    messages: list[str],
    allowed: frozenset[str] | None = None,
) -> dict[str, Any]:
    supplied = _mapping(value, name, messages)
    if supplied is None:
        return dict(defaults)
    _unknown_keys(name, supplied, allowed or set(defaults), messages)
    merged = {**defaults, **supplied}
    shares = []
    for key, share in merged.items():
        if not _number(share) or not 0 <= share <= 1:
            messages.append(f"{name}.{key} must be a share between 0 and 1.")
        else:
            shares.append(share)
    if len(shares) == len(merged) and abs(sum(shares) - 1) > _MIX_TOLERANCE:
        messages.append(
            f"{name} shares must add up to 1; they add up to {sum(shares):g}. "
            "When you change one share, list every share."
        )
    return merged


def _proposals(value: Any, messages: list[str]) -> dict[str, Any]:
    supplied = _mapping(value, "proposals", messages)
    if supplied is None:
        return dict(DEFAULT_PROPOSALS)
    _unknown_keys("proposals", supplied, set(DEFAULT_PROPOSALS), messages)
    merged = {**DEFAULT_PROPOSALS, **supplied}
    for key in ("source_language", "domain", "country", "jurisdiction"):
        if not isinstance(merged[key], str) or not merged[key].strip():
            messages.append(f"proposals.{key} must be a non-empty string.")
    _string_list(merged["target_languages"], "proposals.target_languages", messages)
    _string_list(merged["moderation_targets"], "proposals.moderation_targets", messages, MODERATION_TARGETS)
    _string_list(merged["decision_actions"], "proposals.decision_actions", messages, DECISION_ACTIONS)
    merged["language_scripts"] = _language_scripts(merged.get("language_scripts"), messages)
    return merged


def _language_scripts(value: Any, messages: list[str]) -> dict[str, list[str]]:
    """Optional Unicode script overrides keyed by language prefix. Unknown prefixes stay fail-closed."""
    if value in (None, {}):
        return {}
    supplied = _mapping(value, "proposals.language_scripts", messages)
    if supplied is None:
        return {}
    normalized: dict[str, list[str]] = {}
    for raw_key, raw_values in supplied.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            messages.append("proposals.language_scripts keys must be language prefixes.")
            continue
        prefix = re.split(r"[-_]", raw_key.strip().casefold(), maxsplit=1)[0]
        _string_list(raw_values, f"proposals.language_scripts.{raw_key}", messages)
        if not isinstance(raw_values, list) or not all(isinstance(item, str) and item.strip() for item in raw_values):
            continue
        scripts: list[str] = []
        for item in raw_values:
            name = item.strip().upper()
            if name == "CJK":
                name = "HAN"
            if _SCRIPT_NAME.fullmatch(name) is None:
                messages.append(f"proposals.language_scripts.{raw_key} has an invalid script {item!r}.")
                continue
            if name not in scripts:
                scripts.append(name)
        if scripts:
            normalized[prefix] = scripts
    return normalized


def _reject_model_secrets(value: Any, path: str, messages: list[str]) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_model_secrets(child, f"{path}[{index}]", messages)
        return
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        lowered = str(key).lower().replace("-", "_")
        if lowered in MODEL_SECRET_KEYS or lowered.endswith(MODEL_SECRET_SUFFIXES):
            messages.append(f"{path}.{key} looks like a secret; provide credentials through the provider environment.")
            continue
        _reject_model_secrets(child, f"{path}.{key}", messages)


def _model_config(value: Any, path: str, messages: list[str]) -> None:
    config = _mapping(value, path, messages)
    if config is None:
        return
    for key in MODEL_IDENTITY_KEYS:
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            messages.append(f"{path}.{key} must be a non-empty string.")
    parameters = config.get("inference_parameters", {})
    if _mapping(parameters, f"{path}.inference_parameters", messages) is None:
        return
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        if key in parameters and (not _number(parameters[key]) or not low <= parameters[key] <= high):
            messages.append(f"{path}.inference_parameters.{key} must be a number from {low} to {high}.")
    if "max_tokens" in parameters:
        tokens = parameters["max_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            messages.append(f"{path}.inference_parameters.max_tokens must be a whole number 1 or greater.")


def _passthrough(raw: dict[str, Any], messages: list[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if "languages" in raw:
        extras["languages"] = raw["languages"]
    if "privacy" in raw:
        privacy = _privacy(raw["privacy"], messages)
        if privacy is not None:
            extras["privacy"] = privacy
    if "evaluation" in raw:
        evaluation = _evaluation(raw["evaluation"], messages)
        if evaluation is not None:
            extras["evaluation"] = evaluation
    if "quality_gates" in raw:
        quality_gates = _quality_gates(raw["quality_gates"], messages)
        if quality_gates is not None:
            extras["quality_gates"] = quality_gates
    if (
        "generation_model_config" in raw
        and _mapping(raw["generation_model_config"], "generation_model_config", messages) is not None
    ):
        extras["generation_model_config"] = raw["generation_model_config"]
    if "judge_model_configs" in raw:
        judges = raw["judge_model_configs"]
        if not isinstance(judges, list):
            messages.append("judge_model_configs must be a list.")
        else:
            for index, judge in enumerate(judges):
                _model_config(judge, f"judge_model_configs[{index}]", messages)
            extras["judge_model_configs"] = judges
    if "generation_model_config" in extras:
        _model_config(extras["generation_model_config"], "generation_model_config", messages)
    if "require_plan_confirmation" in raw:
        if not isinstance(raw["require_plan_confirmation"], bool):
            messages.append("require_plan_confirmation must be true or false.")
        else:
            extras["require_plan_confirmation"] = raw["require_plan_confirmation"]
    if "semantic_deduplication_config" in raw:
        semantic = _semantic_dedup(raw["semantic_deduplication_config"], messages)
        if semantic is not None:
            extras["semantic_deduplication_config"] = semantic
    for key, value in extras.items():
        _reject_model_secrets(value, key, messages)
    return extras


def _semantic_dedup(value: Any, messages: list[str]) -> dict[str, Any] | None:
    """MCQ-compatible Curator settings used after exact dedup to drop near-duplicate scored turns."""
    supplied = _mapping(value, "semantic_deduplication_config", messages)
    if supplied is None:
        return None
    _unknown_keys("semantic_deduplication_config", supplied, SEMANTIC_DEDUP_KEYS, messages)
    merged = {**DEFAULT_SEMANTIC_DEDUP, **supplied}
    enabled = merged["enabled"]
    if not isinstance(enabled, bool):
        messages.append("semantic_deduplication_config.enabled must be true or false.")
    model = merged["model_identifier"]
    if not isinstance(model, str) or not model.strip():
        messages.append("semantic_deduplication_config.model_identifier must be a non-empty string.")
    else:
        merged["model_identifier"] = model.strip()
    clusters = merged["n_clusters"]
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 1:
        messages.append("semantic_deduplication_config.n_clusters must be a whole number 1 or greater.")
    eps = merged["eps"]
    if not _number(eps) or not 0 < eps < 1:
        messages.append("semantic_deduplication_config.eps must be a number between 0 and 1.")
    remove = merged["remove_duplicates"]
    if not isinstance(remove, bool):
        messages.append("semantic_deduplication_config.remove_duplicates must be true or false.")
    return merged


def _quality_gates(value: Any, messages: list[str]) -> dict[str, Any] | None:
    """Validate minimum counts applied after validation, deduplication, and splitting."""
    gates = _mapping(value, "quality_gates", messages)
    if gates is None:
        return None
    _unknown_keys("quality_gates", gates, QUALITY_GATE_KEYS, messages)
    checked: dict[str, Any] = {}
    if "on_failure" in gates:
        mode = gates["on_failure"]
        if mode not in QUALITY_GATE_FAILURE_MODES:
            messages.append(f"quality_gates.on_failure must be silver or fail; received {mode!r}.")
        else:
            checked["on_failure"] = mode
    else:
        checked["on_failure"] = "silver"
    for key in (
        "min_total_cases",
        "min_development_cases",
        "min_cases_per_policy",
    ):
        if key not in gates:
            continue
        amount = gates[key]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            messages.append(f"quality_gates.{key} must be a non-negative whole number.")
        else:
            checked[key] = amount
    for key in ("min_cases_per_action", "min_cases_per_language"):
        if key not in gates:
            continue
        minimums = _mapping(gates[key], f"quality_gates.{key}", messages)
        if minimums is None:
            continue
        normalized: dict[str, int] = {}
        seen: dict[str, str] = {}
        for name, amount in minimums.items():
            if not isinstance(name, str) or not name.strip():
                messages.append(f"quality_gates.{key} keys must be non-empty strings.")
                continue
            label = name.strip()
            if key == "min_cases_per_action":
                canonical = label.casefold()
                if canonical not in DECISION_ACTIONS:
                    messages.append(
                        f"quality_gates.{key} has unsupported action {label!r}; "
                        f"allowed: {', '.join(sorted(DECISION_ACTIONS))}."
                    )
                    continue
                stored = canonical
            else:
                stored = unicodedata.normalize("NFC", label)
                canonical = stored.casefold()
            if canonical in seen:
                messages.append(f"quality_gates.{key} repeats {label!r}; it already lists {seen[canonical]!r}.")
                continue
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                messages.append(f"quality_gates.{key}.{label} must be a non-negative whole number.")
                continue
            seen[canonical] = stored
            normalized[stored] = amount
        checked[key] = normalized
    return checked


def _evaluation(value: Any, messages: list[str]) -> dict[str, Any] | None:
    evaluation = _mapping(value, "evaluation", messages)
    if evaluation is None:
        return None
    _unknown_keys("evaluation", evaluation, EVALUATION_KEYS, messages)
    path1 = evaluation.get("path1")
    if path1 is not None:
        _evaluation_path(path1, "evaluation.path1", PATH1_ADAPTERS, "test", messages)
    path2 = evaluation.get("path2")
    if path2 is not None:
        _evaluation_path(path2, "evaluation.path2", PATH2_ADAPTERS, "mocked_guardrails", messages)
    return dict(evaluation)


def _evaluation_path(
    value: Any,
    path: str,
    adapters: frozenset[str],
    default_adapter: str,
    messages: list[str],
) -> None:
    config = _mapping(value, path, messages)
    if config is None:
        return
    if not config:
        messages.append(f"{path} must not be empty; omit it when this path should not run.")
        return
    adapter = config.get("adapter", default_adapter)
    if not isinstance(adapter, str) or adapter not in adapters:
        messages.append(f"{path}.adapter must be one of {', '.join(sorted(adapters))}; received {adapter!r}.")
        return
    if path.endswith("path1"):
        allowed = PATH1_TEST_KEYS if adapter == "test" else PATH1_LIVE_KEYS
    else:
        allowed = PATH2_MOCK_KEYS if adapter == "mocked_guardrails" else PATH2_LIVE_KEYS
    _unknown_keys(path, config, allowed, messages)
    if adapter in {"test", "mocked_guardrails"}:
        _offline_eval_path(config, path, adapter, messages)
    else:
        _live_eval_path(config, path, adapter, messages)


def _offline_eval_path(config: Mapping[str, Any], path: str, adapter: str, messages: list[str]) -> None:
    if adapter == "test":
        inline = config.get("responses")
        file_name = config.get("file")
        if (inline is None) == (file_name is None):
            messages.append(f"{path} must name exactly one of responses or file.")
        if inline is not None and not isinstance(inline, dict):
            messages.append(f"{path}.responses must be a mapping.")
    else:
        file_name = config.get("log") or config.get("file")
        if file_name is None:
            messages.append(f"{path} must name log or file.")
    if file_name is not None and (not isinstance(file_name, str) or not file_name.strip()):
        messages.append(f"{path}.file/log must be a non-empty path string.")


def _live_eval_path(config: Mapping[str, Any], path: str, adapter: str, messages: list[str]) -> None:
    endpoint = config.get("endpoint") or config.get("base_url")
    if not isinstance(endpoint, str) or not endpoint.strip():
        messages.append(f"{path} requires endpoint or base_url.")
    if adapter == "openai_compatible":
        model = config.get("model")
        if not isinstance(model, str) or not model.strip():
            messages.append(f"{path}.model must be a non-empty string.")
        _path1_evaluation_mode(config, path, messages)
    for key in ("api_key_env", "content_path", "decision_path", "rails_path", "config_id", "smoke_message"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            messages.append(f"{path}.{key} must be a non-empty string.")
    if (
        "system_prompt" in config
        and config["system_prompt"] is not None
        and not isinstance(config["system_prompt"], str)
    ):
        messages.append(f"{path}.system_prompt must be a string or null.")
    for key in ("headers_env", "decision_map", "request_options", "options"):
        if key in config and not isinstance(config[key], dict):
            messages.append(f"{path}.{key} must be a mapping.")
    if "capabilities" in config:
        _string_list(config["capabilities"], f"{path}.capabilities", messages, MODERATION_TARGETS)
    decision_map = config.get("decision_map")
    if isinstance(decision_map, dict):
        for label, action in decision_map.items():
            if not isinstance(label, str) or not label.strip() or action not in DECISION_ACTIONS:
                messages.append(f"{path}.decision_map must map non-empty labels to canonical safety actions.")
                break
    for key, minimum, maximum in (
        ("timeout_seconds", 0, None),
        ("retry_backoff_seconds", 0, None),
        ("max_retries", 0, 10),
        ("max_concurrency", 1, 64),
    ):
        if key not in config:
            continue
        number = config[key]
        valid = _number(number) and number >= minimum and (maximum is None or number <= maximum)
        if key in {"max_retries", "max_concurrency"}:
            valid = valid and isinstance(number, int)
        if not valid or (key == "timeout_seconds" and number == 0):
            upper = f" through {maximum}" if maximum is not None else " or greater"
            messages.append(f"{path}.{key} must be a number from {minimum}{upper}.")


def _path1_evaluation_mode(config: Mapping[str, Any], path: str, messages: list[str]) -> None:
    mode = config.get("evaluation_mode")
    prompt_is_set = "system_prompt" in config
    prompt = config.get("system_prompt")
    normalized_prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
    if mode is None:
        return
    if not isinstance(mode, str) or mode not in PATH1_EVALUATION_MODES:
        messages.append(f"{path}.evaluation_mode must be one of {', '.join(sorted(PATH1_EVALUATION_MODES))}.")
        return
    if mode == "native_guard" and normalized_prompt is not None:
        messages.append(f"{path}.native_guard must not configure a non-empty system_prompt.")
    elif mode == "fixed_system_prompt" and normalized_prompt is None:
        messages.append(f"{path}.fixed_system_prompt requires a non-empty system_prompt.")
    elif mode == "canonical_classifier_prompt" and prompt_is_set and normalized_prompt is not None:
        messages.append(
            f"{path}.canonical_classifier_prompt uses the built-in prompt; "
            "remove system_prompt or select fixed_system_prompt."
        )


def _privacy(value: Any, messages: list[str]) -> dict[str, Any] | None:
    supplied = _mapping(value, "privacy", messages)
    if supplied is None:
        return None
    _unknown_keys("privacy", supplied, {"additional_labels"}, messages)
    labels = _mapping(supplied.get("additional_labels", {}), "privacy.additional_labels", messages)
    if labels is None:
        return None
    _unknown_keys("privacy.additional_labels", labels, PRIVACY_LABEL_KINDS, messages)
    normalized: dict[str, list[str]] = {}
    for kind, values in labels.items():
        _string_list(values, f"privacy.additional_labels.{kind}", messages)
        if isinstance(values, list) and all(isinstance(item, str) and item.strip() for item in values):
            normalized[kind] = [item.strip() for item in values]
    return {"additional_labels": normalized}


def _check_privacy_language(
    proposals: Mapping[str, Any],
    extras: Mapping[str, Any],
    messages: list[str],
) -> None:
    source = proposals.get("source_language")
    if not isinstance(source, str) or source.strip().casefold() == "unspecified":
        return
    privacy = extras.get("privacy")
    labels = privacy.get("additional_labels") if isinstance(privacy, dict) else None
    supplied = (
        {kind for kind, values in labels.items() if isinstance(labels, dict) and isinstance(values, list) and values}
        if isinstance(labels, dict)
        else set()
    )
    missing = sorted(PRIVACY_LABEL_KINDS - supplied)
    if missing:
        messages.append(
            f"source language {source!r} needs privacy.additional_labels for " + ", ".join(missing) + " before ingest."
        )
