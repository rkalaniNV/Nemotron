"""Project one authorized published row into a source-bound executable task.

The projection deliberately has two surfaces. ``script.seed_messages`` and
``script.tools`` are the only benchmark-owned values the live driver may place in
a candidate request. Expected calls, fixture references, milestones, assertion
names, and pack-local tool policy stay on the runner side of that boundary.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import OraclePackRef
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_projection import (
    build_conversation_script,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ExecutableAuthorizationError,
    ExecutableProjectionError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedEvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    assert_source_unchanged,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    ContentHash,
    FrozenDict,
    freeze_json,
    json_equal,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.fixture_filter import (
    evaluate_filter,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    project_model_facing_tools,
    resolve_declared_pack_paths,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.expand import (
    primary_key_for,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.templating import (
    placeholder_names,
    substitute,
)

EXECUTABLE_PROJECTION_VERSION: Final = "1.0"
_SLOT_REFERENCE = re.compile(r"^\{([^{}]+)\}$")


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class ExecutableToolPolicy(_Frozen):
    """Pack-local execution policy omitted from the model-facing tool schema."""

    function_name: StrictStr
    mutates: StrictBool = False
    requires_confirmation: StrictBool = False

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "mutates": self.mutates,
            "requires_confirmation": self.requires_confirmation,
        }


class ExecutableTaskSpec(_Frozen):
    """One candidate-authorized task with verified oracle and source lineage."""

    schema_version: Literal["1.0"] = EXECUTABLE_PROJECTION_VERSION
    task_id: StrictStr
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    source_verification_identity: ContentHash
    oracle_verification_identity: ContentHash
    source_content_hash: ContentHash
    oracle_clock: StrictStr
    seed: int
    fixture_refs: tuple[StrictStr, ...] = ()
    script: ConversationScript
    success_assertions: tuple[StrictStr, ...] = ()
    turn_policy: StrictStr
    assistant_milestones: tuple[FrozenDict, ...] = ()
    tool_policies: tuple[ExecutableToolPolicy, ...]
    assertion_task: FrozenDict

    @model_validator(mode="before")
    @classmethod
    def _freeze_runner_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        milestones = list(value.get("assistant_milestones") or ())
        validate_json_value(milestones, label="executable assistant milestones")
        value["assistant_milestones"] = tuple(freeze_json(item) for item in milestones)
        assertion_task = value.get("assertion_task")
        validate_json_value(assertion_task, label="executable assertion task")
        value["assertion_task"] = freeze_json(assertion_task)
        return value

    @model_validator(mode="after")
    def _coherent(self) -> ExecutableTaskSpec:
        if self.script.task_id != self.task_id:
            raise ValueError("the executable script belongs to this task")
        if self.script.source_verification_identity != self.source_verification_identity:
            raise ValueError("the executable script comes from this verified source")
        if any(
            call.recorded_result
            for turn in self.script.turns
            for call in turn.calls
        ):
            raise ValueError(
                "an executable script carries expected call structure but no recorded gold result"
            )
        names = [policy.function_name for policy in self.tool_policies]
        if len(set(names)) != len(names):
            raise ValueError("executable tool policies name each exposed tool once")
        exposed = {
            str((tool.get("function") or {}).get("name"))
            for tool in self.script.tools
        }
        if set(names) != exposed:
            raise ValueError("executable tool policies cover exactly the model-facing tools")
        if not self.oracle_clock.strip():
            raise ValueError("an executable task carries the source run's frozen oracle clock")
        return self

    def tool_policy(self, function_name: str) -> ExecutableToolPolicy | None:
        for policy in self.tool_policies:
            if policy.function_name == function_name:
                return policy
        return None

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "candidate_alias": self.candidate_alias,
            "canonical_model_identity": self.canonical_model_identity,
            "plan_identity": self.plan_identity,
            "eval_config_hash": self.eval_config_hash,
            "scoring_policy_hash": self.scoring_policy_hash,
            "source_verification_identity": self.source_verification_identity,
            "oracle_verification_identity": self.oracle_verification_identity,
            "source_content_hash": self.source_content_hash,
            "oracle_clock": self.oracle_clock,
            "seed": self.seed,
            "fixture_refs": list(self.fixture_refs),
            "script_hash": self.script.script_hash,
            "success_assertions": list(self.success_assertions),
            "turn_policy": self.turn_policy,
            "assistant_milestones": [
                thaw_json(milestone) for milestone in self.assistant_milestones
            ],
            "tool_policies": [policy.semantic_payload() for policy in self.tool_policies],
            "assertion_task": thaw_json(self.assertion_task),
        }

    @property
    def task_spec_hash(self) -> str:
        return _sha256_json(self.semantic_payload())


def _manifest(source: VerifiedEvalSource) -> dict[str, Any]:
    try:
        document = json.loads(source.source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutableProjectionError(
            "source_run_manifest",
            f"cannot be read as canonical JSON: {type(exc).__name__}",
            expected="the verified run_manifest.json",
            recovery="restore the verified publication and run source verification again",
        ) from exc
    if not isinstance(document, dict):
        raise ExecutableProjectionError(
            "source_run_manifest",
            "is not a JSON object",
            actual=type(document).__name__,
            expected="the verified run manifest object",
            recovery="restore the verified publication",
        )
    return document


def _pack_metadata(
    source: VerifiedEvalSource,
    *,
    row: CanonicalExportRow,
) -> tuple[
    tuple[ExecutableToolPolicy, ...],
    tuple[FrozenDict, ...],
    dict[str, Any],
    dict[str, Any],
]:
    oracle = source.oracle
    assert oracle is not None
    try:
        paths = resolve_declared_pack_paths(
            OraclePackRef(manifest_path=oracle.pack_manifest_path),
            (oracle.pack_root,),
        )
        tools = json.loads(paths.tools_path.read_text(encoding="utf-8"))
        templates = yaml.safe_load(paths.templates_path.read_text(encoding="utf-8")) or []
        pack_manifest = yaml.safe_load(paths.manifest_path.read_text(encoding="utf-8")) or {}
        fixtures = (
            json.loads(paths.fixtures_path.read_text(encoding="utf-8"))
            if paths.fixtures_path is not None
            else {}
        )
    except Exception as exc:
        raise ExecutableProjectionError(
            "source_oracle.pack",
            f"cannot recover runner-only metadata: {type(exc).__name__}",
            expected="the complete verified oracle pack",
            recovery="restore the pack revision and verify the source again",
        ) from exc
    if (
        not isinstance(tools, list)
        or not isinstance(templates, list)
        or not isinstance(pack_manifest, dict)
        or not isinstance(fixtures, dict)
    ):
        raise ExecutableProjectionError(
            "source_oracle.pack",
            "has invalid runner metadata containers",
            expected=(
                "manifest and fixtures objects, tools.json array, and "
                "task_templates.yaml array"
            ),
            recovery="restore the pack revision that source verification accepted",
        )
    full_by_name = {
        str((tool.get("function") or {}).get("name")): tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    policies: list[ExecutableToolPolicy] = []
    projected_by_name = {
        str((tool.get("function") or {}).get("name")): tool
        for tool in project_model_facing_tools(list(full_by_name.values()))
    }
    for model_tool in row.tools:
        name = str((model_tool.get("function") or {}).get("name"))
        full = full_by_name.get(name)
        if full is None or not json_equal(projected_by_name.get(name), model_tool):
            raise ExecutableProjectionError(
                f"task tools[{name}]",
                "do not match the verified pack's model-facing declaration",
                expected="the exact projected definition from the verified tools.json",
                recovery="re-publish the benchmark from the verified pack",
            )
        policies.append(
            ExecutableToolPolicy(
                function_name=name,
                mutates=full.get("x-mutates") is True,
                requires_confirmation=full.get("x-requires-confirmation") is True,
            )
        )
    matching = [
        template
        for template in templates
        if isinstance(template, dict) and template.get("template_id") == row.template_id
    ]
    if len(matching) != 1:
        raise ExecutableProjectionError(
            f"task template {row.template_id}",
            "is not uniquely present in the verified pack",
            actual=len(matching),
            expected="exactly one matching template",
            recovery="restore the source pack revision",
        )
    milestones = matching[0].get("assistant_milestones") or []
    validate_json_value(
        milestones,
        label=f"template {row.template_id} assistant milestones",
    )
    template = matching[0]
    validate_json_value(template, label=f"template {row.template_id}")
    assertion_bindings = _assertion_bindings(
        row,
        template,
        milestones,
        pack_manifest=pack_manifest,
        tools=tools,
        fixtures=fixtures,
    )
    return (
        tuple(policies),
        tuple(freeze_json(item) for item in milestones),
        assertion_bindings,
        template,
    )


def _fixture_references(row: CanonicalExportRow) -> dict[str, list[Any]]:
    """Group the row's fixture references by the collection each one cites."""
    values: dict[str, list[Any]] = {}
    for reference in row.fixture_refs:
        try:
            decoded = json.loads(reference)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, list) and len(decoded) == 2 and isinstance(decoded[0], str):
            values.setdefault(decoded[0], []).append(decoded[1])
    return values


def _unique_json_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(json_equal(value, prior) for prior in unique):
            unique.append(value)
    return unique


def _declared_values(
    definition: dict[str, Any],
    *,
    fixture_references: dict[str, list[Any]],
    pack_manifest: dict[str, Any],
    tools: list[dict[str, Any]],
    fixtures: dict[str, Any],
) -> list[Any]:
    """Return typed values this verified pack source could have bound."""
    source = definition.get("source")
    if not isinstance(source, str):
        return []
    # Expansion treats a source without a kind prefix as the original fixture
    # shorthand. Projection must accept the same verified template language.
    if ":" not in source:
        source = f"fixture:{source}"
    if source.startswith("fixture:"):
        collection, separator, field = source.removeprefix("fixture:").partition(".")
        rows = fixtures.get(collection)
        if not separator or not isinstance(rows, list):
            return []
        try:
            key = primary_key_for(pack_manifest, collection, rows)
        except ValueError:
            return []
        # ``fixture_ref`` renders the cited primary id with ``str``, so a numeric
        # key is published as text. Rejoining on the same rendering is what makes
        # the reference readable for every key type, not only string ids.
        references = {str(reference) for reference in fixture_references.get(collection, [])}
        return _unique_json_values(
            [
                row[field]
                for row in rows
                if isinstance(row, dict)
                and field in row
                and evaluate_filter(row, definition.get("filter"))
                and str(row.get(key)) in references
            ]
        )
    if source.startswith("literal:"):
        raw = source.removeprefix("literal:").strip()
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = raw
        return _unique_json_values(list(parsed) if isinstance(parsed, list) else [parsed])
    if source.startswith("enum:"):
        tool_name, separator, parameter = source.removeprefix("enum:").partition(".")
        if not separator:
            return []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or function.get("name") != tool_name:
                continue
            schema = function.get("parameters")
            properties = schema.get("properties") if isinstance(schema, dict) else None
            parameter_schema = properties.get(parameter) if isinstance(properties, dict) else None
            values = parameter_schema.get("enum") if isinstance(parameter_schema, dict) else None
            return _unique_json_values(list(values)) if isinstance(values, list) else []
        return []
    if source.startswith("range:"):
        try:
            bounds = ast.literal_eval(source.removeprefix("range:").strip())
        except (SyntaxError, ValueError):
            return []
        if not isinstance(bounds, dict):
            return []
        start, end, step = bounds.get("min"), bounds.get("max"), bounds.get("step", 1)
        if (
            any(not isinstance(value, int) or isinstance(value, bool) for value in (start, end, step))
            or step == 0
            or (end - start) * step < 0
        ):
            return []
        return list(range(start, end + (1 if step > 0 else -1), step))
    if source.startswith("absent:"):
        collection = source.removeprefix("absent:").strip()
        declared = (pack_manifest.get("absent_ids") or {}).get(collection)
        values = [declared] if not isinstance(declared, list) else declared
        return _unique_json_values(values) if declared is not None else []
    return []


def _sole(values: list[Any]) -> tuple[bool, Any]:
    return (True, values[0]) if len(values) == 1 else (False, None)


def _surface_bindings(
    row: CanonicalExportRow,
    template: dict[str, Any],
    *,
    declared: dict[str, list[Any]],
) -> dict[str, Any]:
    """Read slot values back out of the opening turn the row published.

    The reverse operation enumerates only typed values authorized by the verified
    pack. It never parses a surface fragment into a guessed type. Multiple
    assignments that render the same sentence contribute only bindings on which
    every assignment agrees.
    """
    if row.metadata.get("surface_source") != "template":
        return {}
    language = row.metadata.get("language")
    block = template.get("user_turn_templates")
    if not isinstance(block, dict) or not isinstance(language, str):
        return {}
    pattern_text = block.get(language)
    rendered = next((message.content for message in row.messages if message.role == "user"), None)
    if not isinstance(pattern_text, str) or not isinstance(rendered, str):
        return {}
    names = placeholder_names(pattern_text)
    unique_names = list(dict.fromkeys(names))
    if not unique_names or any(not declared.get(name) for name in unique_names):
        return {}
    combinations = 1
    for name in unique_names:
        combinations *= len(declared[name])
    if combinations > 10_000:
        # Fail closed rather than making projection cost depend exponentially on
        # an unconstrained pack declaration.
        return {}
    matching: list[dict[str, Any]] = []
    for values in itertools.product(*(declared[name] for name in unique_names)):
        assignment = dict(zip(unique_names, values, strict=True))
        if substitute(pattern_text, assignment) == rendered:
            matching.append(assignment)
    if not matching:
        return {}
    bound: dict[str, Any] = {}
    for name in unique_names:
        values = _unique_json_values([assignment[name] for assignment in matching])
        known, value = _sole(values)
        if known:
            bound[name] = value
    return bound


def _committed_values(
    row: CanonicalExportRow,
    milestones: list[Any],
    slot_specs: dict[str, Any],
) -> dict[str, Any]:
    """Recover the slot values the published expected trace actually committed."""
    direct: dict[str, list[Any]] = {}
    for expected in row.expected_tool_calls:
        for argument_name, argument_value in expected.arguments.items():
            if argument_name in slot_specs:
                direct.setdefault(argument_name, []).append(argument_value)

    tool_milestones = [
        milestone
        for milestone in milestones
        if isinstance(milestone, dict) and milestone.get("type") == "tool_call"
    ]
    expected_calls = list(row.expected_tool_calls)
    pairs: list[tuple[dict[str, Any], Any]] = []
    paired_milestones: set[int] = set()
    paired_calls: set[int] = set()

    # An explicit call_group is the same structural identity exported on each
    # expected call. Within a group, milestone order is position_in_group order.
    groups = {
        milestone.get("call_group")
        for milestone in tool_milestones
        if isinstance(milestone.get("call_group"), int)
    }
    for group in groups:
        milestone_indexes = [
            index
            for index, milestone in enumerate(tool_milestones)
            if milestone.get("call_group") == group
        ]
        call_indexes = sorted(
            [
                index
                for index, expected in enumerate(expected_calls)
                if expected.call_group == group
            ],
            key=lambda index: expected_calls[index].position_in_group,
        )
        if len(milestone_indexes) != len(call_indexes):
            continue
        if any(
            tool_milestones[milestone_index].get("tool")
            != expected_calls[call_index].function_name
            for milestone_index, call_index in zip(
                milestone_indexes, call_indexes, strict=True
            )
        ):
            continue
        for milestone_index, call_index in zip(
            milestone_indexes, call_indexes, strict=True
        ):
            pairs.append(
                (tool_milestones[milestone_index], expected_calls[call_index])
            )
            paired_milestones.add(milestone_index)
            paired_calls.add(call_index)

    # A unique remaining tool name is also an unambiguous structural match.
    for milestone_index, milestone in enumerate(tool_milestones):
        if milestone_index in paired_milestones:
            continue
        candidates = [
            call_index
            for call_index, expected in enumerate(expected_calls)
            if call_index not in paired_calls
            and expected.function_name == milestone.get("tool")
        ]
        if len(candidates) == 1:
            call_index = candidates[0]
            pairs.append((milestone, expected_calls[call_index]))
            paired_calls.add(call_index)

    authoritative: dict[str, list[Any]] = {}
    for milestone, expected in pairs:
        declared_args = milestone.get("args")
        if not isinstance(declared_args, dict):
            continue
        for argument_name, template_value in declared_args.items():
            reference = (
                _SLOT_REFERENCE.fullmatch(template_value)
                if isinstance(template_value, str)
                else None
            )
            if reference is not None and argument_name in expected.arguments:
                name = reference.group(1)
                if name in slot_specs:
                    authoritative.setdefault(name, []).append(
                        expected.arguments[argument_name]
                    )

    values: dict[str, Any] = {}
    for name in slot_specs:
        candidates = authoritative.get(name) or direct.get(name) or []
        unique = _unique_json_values(candidates)
        known, value = _sole(unique)
        if known:
            values[name] = value
    return values


def _declared_updates(template: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    simulator_turns = template.get("user_simulator_turns")
    if not isinstance(simulator_turns, list):
        return []
    updates: list[tuple[int, str, dict[str, Any]]] = []
    for entry_index, simulator_turn in enumerate(simulator_turns):
        if not isinstance(simulator_turn, dict):
            continue
        definitions = simulator_turn.get("slot_updates")
        if not isinstance(definitions, dict):
            continue
        for name, definition in definitions.items():
            if isinstance(name, str) and isinstance(definition, dict):
                updates.append((entry_index, name, definition))
    return updates


def _assertion_bindings(
    row: CanonicalExportRow,
    template: dict[str, Any],
    milestones: list[Any],
    *,
    pack_manifest: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    fixtures: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the assertion-facing generation task without guessing.

    Pack assertions receive bound ``slots``, ``slots_initial``, and
    ``slot_updates``, but the published row does not restate that generation
    object. Executable evaluation recovers it from what the row did publish: the
    verbatim opening surface, the expected trace, the cited fixtures, and the
    template's own declarations.

    A value that cannot be recovered unambiguously is left out and named in
    ``unresolved_slots`` rather than guessed. The isolator tracks reads of those
    missing slots and turns a dependent assertion verdict into infrastructure
    evidence, never a candidate failure. An assertion that does not read one — a
    pack's "no tool was called" check, say — retains its ordinary verdict.
    """

    raw_specs = template.get("slots")
    slot_specs = (
        {
            name: spec
            for name, spec in raw_specs.items()
            if isinstance(name, str) and isinstance(spec, dict)
        }
        if isinstance(raw_specs, dict)
        else {}
    )
    fixture_references = _fixture_references(row)
    source_context = {
        "fixture_references": fixture_references,
        "pack_manifest": pack_manifest or {},
        "tools": tools or [],
        "fixtures": fixtures or {},
    }
    declared = {
        name: _declared_values(spec, **source_context)
        for name, spec in slot_specs.items()
    }

    surface = _surface_bindings(row, template, declared=declared)
    committed = _committed_values(row, milestones, slot_specs)
    updates = _declared_updates(template)
    corrected = {name for _, name, _ in updates}

    initial: dict[str, Any] = {}
    slots: dict[str, Any] = {}
    for name in slot_specs:
        # The opening turn renders pre-correction values; the trace commits final
        # ones. For an uncorrected slot those are the same value.
        before_known = name in surface
        before = surface.get(name)
        if not before_known:
            before_known, before = _sole(declared[name])
        after_known = name in committed
        after = committed.get(name)
        if not after_known and name not in corrected:
            after = before
            after_known = before_known
        if not before_known and name not in corrected:
            before = after
            before_known = after_known
        if before_known:
            initial[name] = before
        if after_known:
            slots[name] = after

    grouped: dict[int, dict[str, Any]] = {}
    aliases: dict[int, dict[str, Any]] = {}
    unresolved_updates: dict[int, list[str]] = {}
    remaining = dict.fromkeys(corrected, 0)
    for _, name, _ in updates:
        remaining[name] += 1
    for entry_index, name, definition in updates:
        remaining[name] -= 1
        # The last correction of a slot leaves the value the trace committed.
        value_known, value = _sole(
            _declared_values(definition, **source_context)
        )
        if not value_known and remaining[name] == 0 and name in slots:
            value = slots.get(name)
            value_known = True
        if not value_known:
            unresolved_updates.setdefault(entry_index, []).append(name)
            continue
        grouped.setdefault(entry_index, {})[name] = value
        alias = definition.get("bind_as")
        if isinstance(alias, str) and alias:
            aliases.setdefault(entry_index, {})[alias] = value

    milestone_order = {
        milestone.get("id"): index
        for index, milestone in enumerate(milestones)
        if isinstance(milestone, dict) and isinstance(milestone.get("id"), str)
    }
    simulator_turns = template.get("user_simulator_turns") or []
    ordered = sorted(
        set(grouped) | set(unresolved_updates),
        key=lambda index: milestone_order.get(
            simulator_turns[index].get("after"),
            len(milestone_order) + index,
        ),
    )
    slot_updates = [
        {
            "entry_index": index,
            "values": grouped.get(index, {}),
            "aliases": aliases.get(index, {}),
        }
        for index in ordered
    ]
    unresolved_slot_updates = [
        {
            "update_index": update_index,
            "entry_index": entry_index,
            "slots": sorted(set(unresolved_updates.get(entry_index, []))),
        }
        for update_index, entry_index in enumerate(ordered)
        if unresolved_updates.get(entry_index)
    ]
    # A correction the pack states outright can settle a final value the expected
    # trace left ambiguous. Apply every delivered update in order so the last value
    # in force wins; a value recovered from the expected trace remains authoritative.
    for update in slot_updates:
        for name, value in update["values"].items():
            if name not in committed and name in initial:
                slots[name] = value

    return {
        "slots": slots,
        "slots_initial": initial,
        "slot_updates": slot_updates,
        "unresolved_slots": sorted(set(slot_specs) - set(slots)),
        "unresolved_slots_initial": sorted(set(slot_specs) - set(initial)),
        "unresolved_slot_updates": unresolved_slot_updates,
    }


def build_executable_task_spec(
    projection: CanonicalExportProjection,
    task_id: str,
    *,
    candidate_alias: str,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
) -> ExecutableTaskSpec:
    """Bind one published task to its authorized candidate and verified oracle."""

    assert_source_unchanged(source)
    if not source.executable or source.oracle is None or not source.gold_eligible:
        raise ExecutableAuthorizationError(
            "eval.executable_source",
            "is not a gold-eligible source verified for executable claims",
            actual={"executable": source.executable, "gold_eligible": source.gold_eligible},
            expected="trace_and_executable verification with a verified oracle",
            recovery="verify a gold-eligible source with executable mode enabled",
        )
    if (
        plan.source_verification_identity != source.verification_identity
        or plan.source_run_id != source.source_run_id
        or plan.source_task_ids_hash != source.task_index.task_ids_hash
        or plan.eval_config_hash != source.eval_config_hash
    ):
        raise ExecutableAuthorizationError(
            "eval.executable_plan",
            "was created for a different source or evaluation configuration",
            expected="a contamination plan produced from this VerifiedEvalSource",
            recovery="run contamination analysis again for this verified source",
        )
    try:
        eligibility = plan.candidate(candidate_alias)
    except KeyError as exc:
        raise ExecutableAuthorizationError(
            f"candidates[{candidate_alias}]",
            "is not present in the eligible evaluation plan",
            actual=candidate_alias,
            expected=f"one of {list(plan.candidate_aliases)}",
            recovery="use a candidate authorized by the contamination gate",
        ) from exc
    if task_id not in plan.evaluation_task_ids(candidate_alias):
        raise ExecutableAuthorizationError(
            f"candidates[{candidate_alias}]",
            "is not authorized to answer this task",
            actual=task_id,
            expected="one of the candidate's eligible task ids",
            recovery="iterate plan.evaluation_task_ids(candidate_alias)",
        )

    script = build_conversation_script(projection, task_id, source=source)
    # ``ConversationScript`` also serves trace replay, where each call owns a
    # recorded result. Executable driving needs its expected call structure and
    # deterministic user turns, but retaining those bytes in the live task handle
    # would make an accidental gold-result release possible. Strip them before
    # the task can cross into the live runtime.
    script = script.model_copy(
        update={
            "turns": tuple(
                turn.model_copy(
                    update={
                        "calls": tuple(
                            call.model_copy(update={"recorded_result": ""})
                            for call in turn.calls
                        )
                    }
                )
                for turn in script.turns
            )
        }
    )
    row = projection.row(task_id)
    oracle = source.oracle
    if (
        not row.gold_eligible
        or row.pack_id != oracle.pack_id
        or row.pack_version != oracle.pack_version
    ):
        raise ExecutableAuthorizationError(
            f"task {task_id}",
            "is not gold-eligible under the verified oracle pack",
            actual={
                "gold_eligible": row.gold_eligible,
                "pack_id": row.pack_id,
                "pack_version": row.pack_version,
            },
            expected=f"{oracle.pack_id} {oracle.pack_version}, gold eligible",
            recovery="evaluate a gold row published from the verified pack",
        )
    manifest = _manifest(source)
    clock = manifest.get("oracle_clock")
    if not isinstance(clock, str) or not clock.strip():
        raise ExecutableProjectionError(
            "source_run_manifest.oracle_clock",
            "is missing or empty",
            expected="the frozen ISO-8601 clock source verification accepted",
            recovery="restore the verified run manifest",
        )
    policies, milestones, assertion_bindings, template_metadata = _pack_metadata(
        source,
        row=row,
    )
    assertion_task = {
        **template_metadata,
        "assertion_task_schema_version": "1.0",
        "task_id": row.task_id,
        "template_id": row.template_id,
        "variant_index": row.variant_index,
        "seed": row.seed,
        "fixture_refs": list(row.fixture_refs),
        "intent": row.intent,
        "category": row.category,
        "difficulty": row.difficulty,
        "required_tools": list(row.required_tools),
        "tools_present": list(row.tools_present),
        "turn_policy": row.turn_policy,
        "is_multi_turn": row.is_multi_turn,
        "num_tool_calls": row.num_tool_calls,
        "call_order": row.call_order,
        "call_order_prefix": row.call_order_prefix,
        "system_prompt_id": row.system_prompt_id,
        "tier": row.tier,
        "gold_eligible": row.gold_eligible,
        "pack_id": row.pack_id,
        "pack_version": row.pack_version,
        "src": row.src,
        "metadata": row.metadata,
        "success_assertions": list(row.success_assertions),
        "tools": [thaw_json(tool) for tool in row.tools],
        "expected_tool_calls": [
            {
                "turn_index": call.turn_index,
                "call_group": call.call_group,
                "position_in_group": call.position_in_group,
                "function_name": call.function_name,
                "arguments": call.arguments,
            }
            for call in row.expected_tool_calls
        ],
        **assertion_bindings,
    }
    return ExecutableTaskSpec(
        task_id=row.task_id,
        candidate_alias=candidate_alias,
        canonical_model_identity=eligibility.canonical_model_identity,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        source_verification_identity=source.verification_identity,
        oracle_verification_identity=oracle.verification_identity,
        source_content_hash=source.evaluation_benchmark.content_hash,
        oracle_clock=clock,
        seed=row.seed,
        fixture_refs=row.fixture_refs,
        script=script,
        success_assertions=row.success_assertions,
        turn_policy=row.turn_policy,
        assistant_milestones=milestones,
        tool_policies=policies,
        assertion_task=assertion_task,
    )


__all__ = [
    "EXECUTABLE_PROJECTION_VERSION",
    "ExecutableTaskSpec",
    "ExecutableToolPolicy",
    "build_executable_task_spec",
]
