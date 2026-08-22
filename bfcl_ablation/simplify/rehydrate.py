"""Expand an A1-authored pack back into a full pack the pipeline can run.

This is the code A1 proposes to add to the product. The pipeline itself is never
patched: rehydration writes a complete, ordinary oracle pack to disk, and the
unmodified generator reads it. That is what makes the A0/A1 comparison meaningful —
both arms go through exactly the same code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation.simplify import derive
from bfcl_ablation.simplify.milestones import compile_milestones, render_simulator_turns

COPIED_VERBATIM = ("backend.py", "assertions.py", "tools.json", "fixtures.json")

DEFAULT_CANONICAL_REPLIES = {
    "provide_slot": {"vi": "{slot} nhé."},
    "confirm": {"vi": "Tôi xác nhận thực hiện."},
    "correct": {"vi": "Khoan đã, sửa thành {slot}."},
}


# --------------------------------------------------------------------------------
# templates
# --------------------------------------------------------------------------------


def derived_template_fields(
    template: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, str]],
    languages: list[str],
) -> dict[str, Any]:
    """Return every template field A1 derives rather than reads."""
    milestones, simulator = compile_milestones(template, tools)
    fields: dict[str, Any] = {
        "assistant_milestones": milestones,
        "mutates": derive.derive_mutates(
            [str(t) for t in (template.get("required_tools") or [])], tools
        ),
        # The derivable value is the *default*, not an echo of what the template says.
        # Echoing it would let `shrink` conclude that `call_order: any` was derived,
        # drop it, and silently rebuild the template as `strict` — turning one parallel
        # call batch into two sequential ones. Only `strict` is inferable; declaring
        # that two calls may be issued together is a claim about the backend that no
        # schema carries.
        "call_order": "strict",
    }
    turns = render_simulator_turns(simulator, canonical, languages)
    if turns:
        fields["user_simulator_turns"] = turns
    return fields


def rehydrate_template(
    template: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, str]],
    languages: list[str],
) -> dict[str, Any]:
    """Merge derived fields into an authored template.

    An authored value always wins. A template that had to keep `assistant_milestones`
    because the compiler could not reproduce them is still a valid A1 pack — it just
    reports one fewer field as derivable, which is the measurement A1 exists to make.
    """
    full = dict(template)
    derived = derived_template_fields(template, tools, canonical, languages)
    for key, value in derived.items():
        full.setdefault(key, value)
    # `corrects` / `depends_on` are A1 authoring inputs, not pack schema.
    full.pop("corrects", None)
    full.pop("depends_on", None)
    return full


# --------------------------------------------------------------------------------
# validation cases
# --------------------------------------------------------------------------------


_collection_for_param = derive.collection_for_param


def needed_collections(
    templates: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
) -> tuple[set[str], set[str]]:
    """Return (collections needing a primary key, collections needing an absent id).

    Slots need a key to name the record they bound. Validation needs an absent id for
    every collection a *tool* can be called against, which is a wider set: a tool with
    no template binding an absent id still owes a not-found probe.
    """
    fixture_refs, absent_refs = derive.referenced_collections(templates)
    by_tools = derive.collections_used_by_tools(tools, derive.best_effort_primary_keys(fixtures))
    absent_needed = absent_refs | by_tools
    return fixture_refs | absent_needed, absent_needed


def generate_validation_cases(
    tools: dict[str, dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
    primary_keys: dict[str, str],
    absent_ids: dict[str, list[str]],
    seeds: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit the probes a tool schema plus fixtures already imply.

    Two per tool — one call that must succeed against a real record and one against
    an id nothing holds — plus, for a tool that declares it requires confirmation,
    the unconfirmed call and the mis-typed confirm flag. None of these state anything
    a human knows that the pack does not already declare.
    """
    cases: list[dict[str, Any]] = []
    for name in sorted(tools):
        spec = tools[name]
        seed = dict(seeds.get(name) or {})
        arguments: dict[str, Any] = {}
        absent_param: str | None = None

        for param in spec["required"]:
            collection = _collection_for_param(param, primary_keys)
            if collection and fixtures.get(collection):
                key = primary_keys[collection]
                arguments[param] = fixtures[collection][0][key]
                if absent_param is None and absent_ids.get(collection):
                    absent_param = param
            elif param in seed:
                arguments[param] = seed.pop(param)
            else:
                raise derive.DerivationError(
                    f"cannot build a validation call for {name!r}: parameter {param!r} maps to no "
                    f"fixture collection; declare it under seeds.{name}"
                )
        arguments.update(seed)
        if spec["requires_confirmation"]:
            arguments["confirm"] = True

        cases.append(
            {
                "id": f"success_{name}",
                "tool": name,
                "arguments": dict(arguments),
                "expect": {"result_class": "success", "error_code": None},
                "reset_before": True,
            }
        )
        if absent_param is not None:
            collection = _collection_for_param(absent_param, primary_keys)
            missing = dict(arguments)
            missing[absent_param] = absent_ids[str(collection)][0]
            cases.append(
                {
                    "id": f"miss_{name}",
                    "tool": name,
                    "arguments": missing,
                    "expect": {"result_class": "structured_error", "error_code": "not_found"},
                    "reset_before": True,
                }
            )
        if spec["requires_confirmation"]:
            unconfirmed = dict(arguments)
            unconfirmed["confirm"] = False
            cases.append(
                {
                    "id": f"confirm_false_{name}",
                    "tool": name,
                    "arguments": unconfirmed,
                    "expect": {"result_class": "awaiting_confirmation", "state_unchanged": True},
                    "reset_before": True,
                }
            )
            mistyped = dict(arguments)
            mistyped["confirm"] = "false"
            cases.append(
                {
                    "id": f"wrong_type_confirm_{name}",
                    "tool": name,
                    "arguments": mistyped,
                    "expect": {"result_class": "structured_error", "error_code": "invalid_argument"},
                    "reset_before": True,
                }
            )
    return cases


def merge_validation_cases(
    generated: list[dict[str, Any]], authored: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Authored probes win by id: a business outcome overrides the schema's guess.

    `list_recent_transactions` against an unknown account returns an empty list rather
    than `not_found`. No schema says so, so the authored case must replace the
    generated one rather than sit beside it.
    """
    by_id = {str(case["id"]): case for case in generated}
    for case in authored:
        by_id[str(case["id"])] = case
    return [by_id[key] for key in sorted(by_id)]


# --------------------------------------------------------------------------------
# whole pack
# --------------------------------------------------------------------------------


def rehydrate_pack(source: Path, target: Path) -> dict[str, Any]:
    """Write a complete pack at `target` from the A1-authored pack at `source`."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = yaml.safe_load((source / "manifest.yaml").read_text(encoding="utf-8")) or {}
    authored_templates = yaml.safe_load((source / "task_templates.yaml").read_text(encoding="utf-8")) or []
    raw_cases = yaml.safe_load((source / "validation_cases.yaml").read_text(encoding="utf-8")) or {}

    for name in COPIED_VERBATIM:
        shutil.copy2(source / name, target / name)

    tools_raw = json.loads((source / "tools.json").read_text(encoding="utf-8"))
    fixtures = json.loads((source / "fixtures.json").read_text(encoding="utf-8"))
    tools = derive.tool_index(tools_raw)
    languages = [str(x) for x in (manifest.get("languages") or ["vi"])]
    canonical = manifest.get("user_simulator_templates") or DEFAULT_CANONICAL_REPLIES

    key_needed, absent_needed = needed_collections(authored_templates, tools, fixtures)
    primary_keys = manifest.get("primary_keys") or derive.derive_primary_keys(fixtures, key_needed)
    absent_ids = manifest.get("absent_ids") or derive.derive_absent_ids(
        fixtures, primary_keys, absent_needed
    )

    templates = [rehydrate_template(t, tools, canonical, languages) for t in authored_templates]

    if isinstance(raw_cases, dict):
        seeds = raw_cases.get("seeds") or {}
        authored_cases = raw_cases.get("cases") or []
    else:
        seeds, authored_cases = {}, list(raw_cases)
    cases = merge_validation_cases(
        generate_validation_cases(tools, fixtures, primary_keys, absent_ids, seeds),
        authored_cases,
    )

    full_manifest = {k: v for k, v in manifest.items() if k != "user_simulator_templates"}
    full_manifest["paths"] = dict(derive.PATH_CONVENTION)
    full_manifest["primary_keys"] = primary_keys
    if absent_ids:
        full_manifest["absent_ids"] = absent_ids

    (target / "manifest.yaml").write_text(
        yaml.safe_dump(full_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "task_templates.yaml").write_text(
        yaml.safe_dump(templates, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "validation_cases.yaml").write_text(
        yaml.safe_dump(cases, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    return {
        "source": str(source),
        "target": str(target),
        "primary_keys_derived": manifest.get("primary_keys") is None,
        "primary_keys": primary_keys,
        "absent_ids_derived": manifest.get("absent_ids") is None,
        "absent_ids": absent_ids,
        "templates": len(templates),
        "validation_cases_generated": len(cases) - len(authored_cases),
        "validation_cases_authored": len(authored_cases),
    }
