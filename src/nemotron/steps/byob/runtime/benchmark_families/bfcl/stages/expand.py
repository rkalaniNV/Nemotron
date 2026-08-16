"""Bind template slots against pack sources into locked task instances."""

from __future__ import annotations

import ast
import hashlib
import logging
from collections import Counter
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.fixture_filter import evaluate_filter
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import LoadedPack
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    TASK_INSTANCES,
    task_instance_row,
    task_instances_schema,
    write_stage_table,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import stage_cache_dir
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
    simulator_delivery_order,
)

logger = logging.getLogger(__name__)


def task_id_for(
    *,
    pack_id: str,
    pack_version: str,
    template_id: str,
    fixture_refs: list[str],
    slot_bindings: dict[str, Any],
    variant_index: int,
) -> str:
    payload = canonical_json(
        {
            "pack_id": pack_id,
            "pack_version": pack_version,
            "template_id": template_id,
            "fixture_refs": sorted(fixture_refs),
            "slot_bindings": slot_bindings,
            "variant_index": variant_index,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{pack_id}__{template_id}__{digest[0:16]}"


def task_seed_for(
    *,
    global_seed: int,
    pack_id: str,
    pack_version: str,
    template_id: str,
    fixture_refs: list[str],
    slot_bindings: dict[str, Any],
    variant_index: int,
) -> int:
    payload = canonical_json(
        {
            "global_seed": global_seed,
            "pack_id": pack_id,
            "pack_version": pack_version,
            "template_id": template_id,
            "fixture_refs": sorted(fixture_refs),
            "slot_bindings": slot_bindings,
            "variant_index": variant_index,
        }
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[0:8], byteorder="big", signed=False)


class ExpansionError(ValueError):
    """Raised when a template cannot be bound against the pack."""


def primary_key_for(manifest: dict[str, Any], collection: str, rows: list[dict[str, Any]]) -> str:
    """Resolve a collection's primary id field, declared or by convention.

    The key names the row a slot value came from, so guessing wrong attributes a task
    to the wrong record. Convention is only trusted when it is unambiguous — the
    collection's own ``<singular>_id`` or ``id`` — and a collection holding several
    ``*_id`` fields (its own plus a foreign key) must declare which one is its key.
    """
    declared = ((manifest.get("primary_keys") or {}) if isinstance(manifest, dict) else {}).get(collection)
    fields = next((list(row) for row in rows if isinstance(row, dict)), [])
    if isinstance(declared, str):
        if fields and declared not in fields:
            raise ExpansionError(
                f"manifest declares primary key {declared!r} for {collection!r}, which its rows do "
                f"not carry; rows hold {', '.join(fields)}"
            )
        return declared

    singular = collection[:-1] if collection.endswith("s") else collection
    for candidate in (f"{singular}_id", "id"):
        if candidate in fields:
            return candidate
    id_fields = [field for field in fields if field.endswith("_id")]
    if len(id_fields) == 1:
        return id_fields[0]
    raise ExpansionError(
        f"cannot tell which field identifies a row of {collection!r} (candidates: "
        f"{', '.join(id_fields) or 'none'}); declare primary_keys.{collection} in the manifest"
    )


def _tool_parameter(pack: LoadedPack, tool_name: str, param: str) -> dict[str, Any]:
    for tool in pack.tools:
        function = tool.get("function") or {}
        if function.get("name") == tool_name:
            return ((function.get("parameters") or {}).get("properties") or {}).get(param) or {}
    return {}


def _candidates(pack: LoadedPack, slot_name: str, slot: dict[str, Any]) -> list[tuple[Any, str | None]]:
    """Return ordered ``(value, fixture_ref)`` candidates for one slot."""
    source = str(slot.get("source") or "")
    kind, _, rest = source.partition(":")
    if not rest:
        kind, rest = "fixture", source

    if kind == "fixture":
        collection, _, field = rest.partition(".")
        rows = (pack.fixtures or {}).get(collection)
        if not isinstance(rows, list):
            raise ExpansionError(f"slot {slot_name!r} references unknown collection {collection!r}")
        key = primary_key_for(pack.manifest, collection, rows)
        matched = [
            row
            for row in rows
            if isinstance(row, dict) and field in row and evaluate_filter(row, slot.get("filter"))
        ]
        if not matched:
            raise ExpansionError(f"slot {slot_name!r} filter matched zero rows in {collection!r}")
        missing = [index for index, row in enumerate(matched) if row.get(key) is None]
        if missing:
            raise ExpansionError(
                f"rows {missing} of {collection!r} carry no {key!r}, so the tasks they bind could not "
                "record which record they came from"
            )
        return [(row[field], f"{collection}.{row[key]}") for row in matched]

    if kind == "enum":
        tool_name, _, param = rest.partition(".")
        values = _tool_parameter(pack, tool_name, param).get("enum")
        if not values:
            raise ExpansionError(f"slot {slot_name!r} has no enum on {tool_name}.{param}")
        return [(value, None) for value in values]

    if kind == "literal":
        raw = rest.strip()
        if raw.startswith("["):
            values = ast.literal_eval(raw)
        else:
            # A bare literal keeps its Python type, so literal:200000 binds an integer
            # and passes an integer-typed tool parameter. Anything unparsable is text.
            try:
                values = [ast.literal_eval(raw)]
            except (ValueError, SyntaxError):
                values = [raw]
        if not values:
            raise ExpansionError(f"slot {slot_name!r} declares an empty literal set")
        return [(value, None) for value in values]

    if kind == "range":
        bounds = ast.literal_eval(rest.strip()) if rest.strip().startswith("{") else None
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise ExpansionError(f"slot {slot_name!r} declares a degenerate range")
        start = bounds["min"]
        end = bounds["max"]
        step = bounds.get("step", 1)
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (start, end, step)
        ):
            raise ExpansionError(f"slot {slot_name!r} range min, max, and step must be integers")
        if step == 0:
            raise ExpansionError(f"slot {slot_name!r} range step must not be zero")
        if (end - start) * step < 0:
            raise ExpansionError(
                f"slot {slot_name!r} range step points away from max"
            )
        stop = end + (1 if step > 0 else -1)
        values = list(range(start, stop, step))
        if not values:
            raise ExpansionError(f"slot {slot_name!r} declares an empty range")
        return [(value, None) for value in values]

    if kind == "absent":
        collection = rest.strip()
        declared = (pack.manifest.get("absent_ids") or {}).get(collection)
        values = [declared] if isinstance(declared, str) else list(declared or [])
        if not values:
            raise ExpansionError(f"pack declares no absent id for collection {collection!r}")
        return [(value, None) for value in values]

    raise ExpansionError(f"slot {slot_name!r} uses unsupported source kind {kind!r}")


def _source_kind(source: Any) -> str:
    kind, _, rest = str(source or "").partition(":")
    return kind if rest else "fixture"


def declared_slot_updates(template: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """Return ``(simulator_turn_index, slot_name, definition)`` for every replacement."""
    declared: list[tuple[int, str, dict[str, Any]]] = []
    for index, entry in enumerate(template.get("user_simulator_turns") or []):
        updates = entry.get("slot_updates")
        if updates is None:
            continue
        if not isinstance(updates, dict):
            raise ExpansionError(
                f"template {template.get('template_id')!r} declares slot_updates that is not a mapping"
            )
        for slot_name, definition in updates.items():
            if not isinstance(definition, dict) or not definition.get("source"):
                raise ExpansionError(
                    f"slot_updates for {slot_name!r} must declare the source its replacement resolves through"
                )
            declared.append((index, str(slot_name), definition))
    return declared


def _check_correction_contract(
    template: dict[str, Any], slots: dict[str, Any], updates: list[tuple[int, str, dict[str, Any]]]
) -> None:
    """Enforce the rules that keep a correction meaningful and bindable."""
    template_id = template.get("template_id")
    policy = str(template.get("turn_policy"))
    if updates and policy != "correction":
        raise ExpansionError(
            f"template {template_id!r} replaces a slot mid-conversation but declares turn_policy "
            f"{policy!r}; declare correction so the edge stays visible to balancing and scoring"
        )
    if policy == "correction" and not updates:
        raise ExpansionError(
            f"template {template_id!r} is correction but declares no slot_updates; a correction is a "
            "later user turn that replaces a value the user already gave"
        )
    for _, slot_name, definition in updates:
        slot = slots.get(slot_name)
        if slot is None:
            raise ExpansionError(
                f"template {template_id!r} replaces undeclared slot {slot_name!r}"
            )
        if not slot.get("visible_in_first_turn"):
            raise ExpansionError(
                f"template {template_id!r} replaces hidden slot {slot_name!r}; a correction can only "
                "replace a value the user already stated"
            )
        if _source_kind(definition.get("source")) != _source_kind(slot.get("source")):
            raise ExpansionError(
                f"template {template_id!r} replaces {slot_name!r} through source kind "
                f"{_source_kind(definition.get('source'))!r} instead of "
                f"{_source_kind(slot.get('source'))!r}; a replacement must resolve the same way"
            )


def _grouped_updates(
    updates: list[tuple[int, str, dict[str, Any]]],
    values: dict[tuple[int, str], Any],
    delivery_order: dict[int, int],
) -> list[dict[str, Any]]:
    """Order bound replacements by the turn that delivers them, not by declaration.

    Two corrections declared in one order can reach the user in another, and the value
    in force is whichever landed last in the conversation.
    """
    grouped: dict[int, dict[str, Any]] = {}
    aliases: dict[int, dict[str, Any]] = {}
    for entry_index, slot_name, definition in updates:
        value = values[(entry_index, slot_name)]
        grouped.setdefault(entry_index, {})[slot_name] = value
        alias = definition.get("bind_as")
        if alias:
            aliases.setdefault(entry_index, {})[str(alias)] = value
    return [
        {
            "entry_index": entry_index,
            "values": grouped[entry_index],
            "aliases": aliases.get(entry_index, {}),
        }
        for entry_index in sorted(grouped, key=lambda index: delivery_order.get(index, index))
    ]


def expand_template(pack: LoadedPack, template: dict[str, Any], limit: int, global_seed: int) -> list[dict]:
    """Bind one template into at most ``limit`` deterministic task instances."""
    slots = template.get("slots") or {}
    slot_names = sorted(slots)
    updates = declared_slot_updates(template)
    _check_correction_contract(template, slots, updates)

    delivery_order = simulator_delivery_order(template)
    ordered_updates = sorted(
        updates,
        key=lambda update: delivery_order.get(update[0], update[0]),
    )
    update_keys = [(entry_index, name) for entry_index, name, _ in ordered_updates]
    keys: list[Any] = [*slot_names, *update_keys]
    candidate_lists = [_candidates(pack, name, slots[name]) for name in slot_names]
    candidate_lists.extend(
        _candidates(pack, f"{name} (correction)", definition)
        for _, name, definition in ordered_updates
    )

    combinations: list[list[tuple[Any, str | None]]] = [[]]
    for position, candidates in enumerate(candidate_lists):
        # Drop a replacement equal to the value it replaces before capping, or the cap
        # would keep only the pairs that correct nothing and the template would look
        # unbindable. When every candidate is a restatement, keep them: a sibling slot
        # in the same correction turn may still change.
        replaced = keys[position][1] if isinstance(keys[position], tuple) else None
        held_at = None
        if replaced is not None:
            # Compare with the most recent value for this slot in conversation order,
            # which may itself be an earlier correction rather than the initial slot.
            for prior_position in range(position - 1, -1, -1):
                prior_key = keys[prior_position]
                prior_name = prior_key[1] if isinstance(prior_key, tuple) else prior_key
                if prior_name == replaced:
                    held_at = prior_position
                    break
        next_combinations: list[list[tuple[Any, str | None]]] = []
        for combo in combinations:
            if held_at is None:
                usable = candidates
            else:
                differing = [
                    candidate
                    for candidate in candidates
                    if combo[held_at][0] != candidate[0]
                ]
                usable = differing if differing else candidates
            for candidate in usable:
                next_combinations.append(combo + [candidate])
        # Cap breadth as we go so a wide pack cannot build a huge product first.
        combinations = next_combinations[: max(limit, 1)]

    pack_id = str(pack.manifest.get("pack_id"))
    pack_version = str(pack.manifest.get("version"))
    tasks: list[dict[str, Any]] = []
    for combination in combinations[: max(limit, 1)]:
        raw = {key: value for key, (value, _) in zip(keys, combination, strict=True)}
        initial = {name: raw[name] for name in slot_names}
        replacements = {key: raw[key] for key in update_keys}
        bound_updates = _grouped_updates(ordered_updates, replacements, delivery_order)
        final = dict(initial)
        # A multi-slot correction turn is useful whenever any one slot changes. Treating
        # a single unchanged sibling as a no-op for the whole instance would discard
        # valid corrections that happen to restated another field.
        corrects_anything = False
        for update in bound_updates:
            for name, value in update["values"].items():
                if final.get(name) != value:
                    corrects_anything = True
            final.update(update["values"])
        if bound_updates and not corrects_anything:
            continue
        bindings = {
            **initial,
            **{f"{name}@correction{entry_index}": raw[(entry_index, name)] for entry_index, name in update_keys},
        }
        fixture_refs = sorted({ref for _, ref in combination if ref})
        tasks.append(
            {
                "slots_initial": initial,
                "slot_updates": bound_updates,
                "task_id": task_id_for(
                    pack_id=pack_id,
                    pack_version=pack_version,
                    template_id=str(template.get("template_id")),
                    fixture_refs=fixture_refs,
                    slot_bindings=bindings,
                    variant_index=0,
                ),
                "template_id": template.get("template_id"),
                "variant_index": 0,
                "seed": task_seed_for(
                    global_seed=global_seed,
                    pack_id=pack_id,
                    pack_version=pack_version,
                    template_id=str(template.get("template_id")),
                    fixture_refs=fixture_refs,
                    slot_bindings=bindings,
                    variant_index=0,
                ),
                "slots": final,
                "fixture_refs": fixture_refs,
                "pack_id": pack_id,
                "pack_version": pack_version,
                "intent": template.get("intent"),
                "category": template.get("category"),
                "difficulty": template.get("difficulty"),
                "turn_policy": template.get("turn_policy"),
                "call_order": template.get("call_order", "strict"),
                "call_order_prefix": template.get("call_order_prefix"),
                "required_tools": list(template.get("required_tools") or []),
                "tools_present": list(template.get("tools_present") or []),
                "success_assertions": list(template.get("success_assertions") or []),
                "edge_signatures": list(template.get("edge_signatures") or []),
                "mutates": bool(template.get("mutates", False)),
            }
        )
    if updates and not tasks:
        raise ExpansionError(
            f"template {template.get('template_id')!r} bound no correction instance because every "
            "replacement equalled the original; declare replacement values disjoint from the "
            "original source"
        )
    return tasks


def category_of(template: dict[str, Any]) -> str:
    """Return the balancing bucket a template expands into."""
    declared = template.get("category")
    return str(declared) if declared else str(template.get("template_id"))


def group_by_category(templates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket templates the way the expansion budget is spent."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for template in templates:
        grouped.setdefault(category_of(template), []).append(template)
    return grouped


def check_category_budgets(templates: list[dict[str, Any]], budget: int) -> None:
    """Raise when a category's budget cannot keep one instance per template.

    Validation calls this before granting gold so a pack does not learn about an
    unsatisfiable budget only once generation is underway.
    """
    for category, grouped in group_by_category(templates).items():
        if budget < len(grouped):
            raise ExpansionError(
                f"category {category!r} declares {len(grouped)} templates but "
                f"tasks_per_category is {budget}; raise the budget so every template "
                "keeps at least one instance"
            )


def _select_round_robin(pools: list[list[dict[str, Any]]], budget: int) -> list[dict[str, Any]]:
    """Take instances one per template per pass so every template stays represented."""
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < budget and any(depth < len(pool) for pool in pools):
        for pool in pools:
            if len(selected) >= budget:
                break
            if depth < len(pool):
                selected.append(pool[depth])
        depth += 1
    return selected


def run_expand(config: BfclConfig, pack: LoadedPack) -> list[dict[str, Any]]:
    """Expand every template and cache the locked task instances.

    ``tasks_per_category`` budgets each category, not each template, so adding a
    template to a category spreads that budget instead of growing the set.
    """
    budget = int(config.task_generation.get("tasks_per_category", 1) or 1)
    global_seed = int(config.random_seed or 0)

    by_category = group_by_category(pack.templates)
    check_category_budgets(pack.templates, budget)

    tasks: list[dict[str, Any]] = []
    for templates in by_category.values():
        pools = [expand_template(pack, template, budget, global_seed) for template in templates]
        tasks.extend(_select_round_robin(pools, budget))
    task_ids = [str(task["task_id"]) for task in tasks]
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise ExpansionError(f"expansion produced duplicate task_id values: {duplicates}")

    write_stage_table(
        stage_cache_dir(config) / TASK_INSTANCES,
        [task_instance_row(task) for task in tasks],
        task_instances_schema(),
    )
    logger.info(
        "BFCL expand produced %d task instances across %d categories",
        len(tasks),
        len(by_category),
    )
    return tasks
