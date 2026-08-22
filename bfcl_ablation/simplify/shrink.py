"""Produce the A1-authored pack from the A0 pack.

A field is dropped only when rehydration provably reproduces it. Anything the
derivation gets wrong stays authored and is reported as *not* derivable, so the LOC
reduction A1 claims is measured rather than asserted, and the `CUT_ANALYSIS` the plan
asks for falls out of the run instead of being argued in advance.

Two kinds of difference are treated differently:

  semantic  the derived value would change what the benchmark asserts. The field
            stays, and the finding is `not_derivable`.
  surface   the derived value changes only wording (`content_template`). The field
            goes, and the finding is `surface_changed` — A1's equality criterion
            covers task identity and expected calls, not phrasing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation.simplify import derive
from bfcl_ablation.simplify.rehydrate import (
    DEFAULT_CANONICAL_REPLIES,
    derived_template_fields,
    generate_validation_cases,
    needed_collections,
)

# Fields the A1 authoring surface no longer contains.
DERIVABLE_TEMPLATE_FIELDS = ("assistant_milestones", "user_simulator_turns", "mutates", "call_order")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _strip_surface(value: Any) -> Any:
    """Drop wording so a comparison sees structure only."""
    if isinstance(value, dict):
        return {k: _strip_surface(v) for k, v in value.items() if k != "content_template"}
    if isinstance(value, list):
        return [_strip_surface(v) for v in value]
    return value


def _normalize_handles(fields: dict[str, Any]) -> dict[str, Any]:
    """Rename milestone ids to their position before comparing.

    A milestone id is an internal handle: `user_simulator_turns.after` and
    `args.from_result.call` resolve through it, and nothing downstream ever sees it.
    Comparing the literal strings would report the compiler as wrong for choosing
    `call_0` where the author wrote `recent_list`, which is a naming difference, not a
    semantic one. An `after` that names a milestone *type* rather than an id is left
    alone, since no mapping entry matches it.
    """
    mapping = {
        str(milestone["id"]): f"m{index}"
        for index, milestone in enumerate(fields.get("assistant_milestones") or [])
        if isinstance(milestone, dict) and milestone.get("id")
    }

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: mapping.get(item, item)
                if key in {"id", "call", "after"} and isinstance(item, str)
                else rewrite(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    return {key: rewrite(value) for key, value in fields.items()}


def _extract_corrects(template: dict[str, Any]) -> dict[str, Any]:
    """Recover the one thing a correction needs that no schema implies."""
    corrects: dict[str, Any] = {}
    for entry in template.get("user_simulator_turns") or []:
        for name, definition in (entry.get("slot_updates") or {}).items():
            corrects[str(name)] = dict(definition)
    return corrects


def _extract_depends_on(template: dict[str, Any]) -> dict[str, Any]:
    """Recover which argument is read from an earlier result, and at what path."""
    milestones = template.get("assistant_milestones") or []
    ids: dict[str, str] = {}
    for index, milestone in enumerate(milestones):
        if str(milestone.get("type")) == "tool_call" and milestone.get("id"):
            ids[str(milestone["id"])] = str(milestone.get("tool"))
    depends: dict[str, Any] = {}
    for milestone in milestones:
        for param, value in (milestone.get("args") or {}).items():
            if isinstance(value, dict) and "from_result" in value:
                marker = value["from_result"]
                depends[str(param)] = {
                    "from_call": ids.get(str(marker.get("call")), str(marker.get("call"))),
                    "path": marker.get("path"),
                }
    return depends


def shrink_template(
    template: dict[str, Any],
    tools: dict[str, dict[str, Any]],
    canonical: dict[str, dict[str, str]],
    languages: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    authored = dict(template)
    findings: list[dict[str, Any]] = []
    template_id = str(authored.get("template_id"))

    candidate = dict(authored)
    candidate.pop("paraphrase", None)  # pack_loader defaults it to {}

    corrects = _extract_corrects(authored)
    if corrects:
        candidate["corrects"] = corrects
    depends_on = _extract_depends_on(authored)
    if depends_on:
        candidate["depends_on"] = depends_on

    try:
        derived = derived_template_fields(candidate, tools, canonical, languages)
    except derive.DerivationError as error:
        findings.append({"template_id": template_id, "field": "*", "verdict": "compiler_error", "detail": str(error)})
        return authored, findings

    want_all = _normalize_handles(derived)
    got_all = _normalize_handles({f: authored[f] for f in DERIVABLE_TEMPLATE_FIELDS if f in authored})

    for field in DERIVABLE_TEMPLATE_FIELDS:
        if field not in authored:
            # An absent field already costs nothing; `call_order` omitted means strict.
            candidate.pop(field, None)
            continue
        want = want_all.get(field)
        got = got_all[field]
        if _canonical(want) == _canonical(got):
            candidate.pop(field, None)
            findings.append({"template_id": template_id, "field": field, "verdict": "derived_exact"})
        elif _canonical(_strip_surface(want)) == _canonical(_strip_surface(got)):
            candidate.pop(field, None)
            findings.append(
                {
                    "template_id": template_id,
                    "field": field,
                    "verdict": "surface_changed",
                    "detail": "structure reproduced; wording now comes from the pack-wide canonical templates",
                }
            )
        else:
            findings.append(
                {
                    "template_id": template_id,
                    "field": field,
                    "verdict": "not_derivable",
                    "detail": f"authored={_canonical(got)[:220]} derived={_canonical(want)[:220]}",
                }
            )

    if str(authored.get("call_order") or "strict") == "strict":
        candidate.pop("call_order", None)
    return candidate, findings


def shrink_pack(source: Path, target: Path) -> dict[str, Any]:
    """Write the A1-authored pack at `target` from the A0 pack at `source`."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = yaml.safe_load((source / "manifest.yaml").read_text(encoding="utf-8")) or {}
    templates = yaml.safe_load((source / "task_templates.yaml").read_text(encoding="utf-8")) or []
    authored_cases = yaml.safe_load((source / "validation_cases.yaml").read_text(encoding="utf-8")) or []

    for name in ("backend.py", "assertions.py", "tools.json", "fixtures.json"):
        shutil.copy2(source / name, target / name)

    tools_raw = json.loads((source / "tools.json").read_text(encoding="utf-8"))
    fixtures = json.loads((source / "fixtures.json").read_text(encoding="utf-8"))
    tools = derive.tool_index(tools_raw)
    languages = [str(x) for x in (manifest.get("languages") or ["vi"])]
    canonical = manifest.get("user_simulator_templates") or DEFAULT_CANONICAL_REPLIES

    shrunk: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for template in templates:
        candidate, template_findings = shrink_template(template, tools, canonical, languages)
        shrunk.append(candidate)
        findings.extend(template_findings)

    # --- manifest ---------------------------------------------------------------
    key_needed, absent_needed = needed_collections(templates, tools, fixtures)
    manifest_findings: list[dict[str, Any]] = []
    slim = dict(manifest)

    if manifest.get("paths") == derive.PATH_CONVENTION:
        slim.pop("paths")
        manifest_findings.append({"field": "paths", "verdict": "derived_exact"})
    else:
        manifest_findings.append({"field": "paths", "verdict": "not_derivable", "detail": "paths deviate from the filename convention"})

    try:
        keys = derive.derive_primary_keys(fixtures, key_needed)
        declared = {k: v for k, v in (manifest.get("primary_keys") or {}).items() if k in keys}
        if keys == declared:
            slim.pop("primary_keys", None)
            manifest_findings.append({"field": "primary_keys", "verdict": "derived_exact", "detail": _canonical(keys)})
        else:
            manifest_findings.append({"field": "primary_keys", "verdict": "not_derivable", "detail": f"derived={_canonical(keys)} declared={_canonical(declared)}"})
    except derive.DerivationError as error:
        manifest_findings.append({"field": "primary_keys", "verdict": "not_derivable", "detail": str(error)})

    declared_absent = {k: v for k, v in (manifest.get("absent_ids") or {}).items() if k in absent_needed}
    try:
        minted = derive.derive_absent_ids(fixtures, manifest.get("primary_keys") or {}, absent_needed)
        if all(declared_absent.get(k, [None])[0] == v[0] for k, v in minted.items()):
            slim.pop("absent_ids", None)
            manifest_findings.append({"field": "absent_ids", "verdict": "derived_exact", "detail": _canonical(minted)})
        else:
            manifest_findings.append({"field": "absent_ids", "verdict": "not_derivable", "detail": f"derived={_canonical(minted)} declared={_canonical(declared_absent)}"})
    except derive.DerivationError as error:
        manifest_findings.append({"field": "absent_ids", "verdict": "not_derivable", "detail": str(error)})

    slim["user_simulator_templates"] = canonical

    # --- validation cases -------------------------------------------------------
    keys_for_cases = manifest.get("primary_keys") or derive.derive_primary_keys(fixtures, key_needed)
    absent_for_cases = manifest.get("absent_ids") or derive.derive_absent_ids(fixtures, keys_for_cases, absent_needed)
    seeds = _infer_seeds(authored_cases, tools, keys_for_cases, fixtures)
    generated = generate_validation_cases(tools, fixtures, keys_for_cases, absent_for_cases, seeds)
    generated_by_id = {str(c["id"]): _canonical(c) for c in generated}
    kept = [c for c in authored_cases if generated_by_id.get(str(c["id"])) != _canonical(c)]

    (target / "manifest.yaml").write_text(
        yaml.safe_dump(slim, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "task_templates.yaml").write_text(
        yaml.safe_dump(shrunk, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "validation_cases.yaml").write_text(
        yaml.safe_dump({"seeds": seeds, "cases": kept}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "template_findings": findings,
        "manifest_findings": manifest_findings,
        "validation_cases_authored_before": len(authored_cases),
        "validation_cases_authored_after": len(kept),
        "validation_cases_generated": len(generated),
    }


def _infer_seeds(
    cases: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
    primary_keys: dict[str, str],
    fixtures: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Keep only the argument values the generator cannot reach on its own.

    A parameter that names a fixture collection is filled from the fixtures; anything
    else — a destination account number, an amount, a rail — is a domain value the
    author has to supply once per tool.
    """
    from bfcl_ablation.simplify.rehydrate import _collection_for_param

    seeds: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = str(case.get("tool"))
        if not str(case.get("id", "")).startswith("success_") or name not in tools:
            continue
        spec = tools[name]
        seed: dict[str, Any] = {}
        for param, value in (case.get("arguments") or {}).items():
            if param == "confirm":
                continue
            collection = _collection_for_param(param, primary_keys)
            if param in spec["required"] and collection and fixtures.get(collection):
                continue
            seed[param] = value
        if seed:
            seeds[name] = seed
    return seeds
