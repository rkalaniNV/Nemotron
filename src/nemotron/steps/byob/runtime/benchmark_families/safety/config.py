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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from nemotron.steps.byob.runtime.benchmark_families.safety.ir import ACTIONS, DEFAULT_TARGETS, TARGETS

BYOB_ROOT = Path(__file__).resolve().parents[3]
STAGES = frozenset({"prepare", "generate", "translate", "eval", "all"})
MODES = frozenset({"preview", "assisted", "expert"})
PROFILES = frozenset({"standard", "regulated", "high_assurance"})
MODULE_STRING_FLAGS = frozenset({"confirm_if_detected", "auto_if_evidence_exists"})
SECOND_JUDGE_TRIGGERS = frozenset({"disagreement", "critical_boundary", "low_confidence"})
# The same vocabularies Policy Rule IR stores, so a config can name every target and action a rule can.
MODERATION_TARGETS = frozenset(TARGETS)
DECISION_ACTIONS = frozenset(ACTIONS)
DEFAULT_DECISION_ACTIONS = ("allow", "block", "caution", "escalate")

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
    "modules": {
        "adversarial": False,
        "political_symmetry": "confirm_if_detected",
        "production_replay": "auto_if_evidence_exists",
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
}
DEFAULT_ASSURANCE = {"profile": "standard"}
ASSURANCE_KEYS = frozenset({"profile", "review_sample_rate", "second_judge_on"})
GENERATION_KEYS = frozenset({"decision_mix", "case_family_mix", "turn_mix", "modules"})
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
    "proposals",
    "languages",
    "generation_model_config",
    "judge_model_configs",
    "evaluation",
    "require_plan_confirmation",
}


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
    _check_decision_mix(generation["decision_mix"], proposals["decision_actions"], messages)
    extras = _passthrough(raw, messages)

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
        return {key: dict(section) for key, section in DEFAULT_GENERATION.items()}
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
    return {**mixes, "modules": modules}


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
    return merged


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
    for key in ("languages", "evaluation", "generation_model_config"):
        if key in raw and _mapping(raw[key], key, messages) is not None:
            extras[key] = raw[key]
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
    for key, value in extras.items():
        _reject_model_secrets(value, key, messages)
    return extras
