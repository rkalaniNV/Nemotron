"""Deterministic derivation of pack fields an author currently writes by hand.

Every function here is pure inference over material the pack already declares —
tools.json, fixtures.json, and the template's own intent fields. No model is
involved, which is the whole point of A1: it establishes how much friction is
removable at zero generative risk, before any LLM enters the ladder.

Each deriver validates its own output. Inference that cannot be checked is worse
than the hand-written field it replaces, because it fails silently.
"""

from __future__ import annotations

from typing import Any

# Filename convention, replacing manifest.paths.
PATH_CONVENTION = {
    "tools": "tools.json",
    "backend": "backend.py",
    "fixtures": "fixtures.json",
    "templates": "task_templates.yaml",
    "assertions": "assertions.py",
    "validation_cases": "validation_cases.yaml",
}

_ID_SUFFIXES = ("_id", "_ref", "_code", "_key", "_no")


class DerivationError(ValueError):
    """Raised when a field cannot be inferred safely."""


# --------------------------------------------------------------------------------
# tools.json helpers
# --------------------------------------------------------------------------------


def tool_index(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for tool in tools:
        function = tool.get("function") or tool
        name = str(function.get("name"))
        parameters = function.get("parameters") or {}
        index[name] = {
            "required": [str(p) for p in (parameters.get("required") or [])],
            "properties": dict(parameters.get("properties") or {}),
            "mutates": bool(tool.get("x-mutates") or function.get("x-mutates")),
            "requires_confirmation": bool(
                tool.get("x-requires-confirmation") or function.get("x-requires-confirmation")
            ),
        }
    return index


def derive_mutates(required_tools: list[str], tools: dict[str, dict[str, Any]]) -> bool:
    """A template mutates iff a tool it must call is declared mutating.

    `mutates` restates a property of the tool contract. Deriving it also removes a
    way for the two to disagree.
    """
    return any(tools.get(str(name), {}).get("mutates") for name in required_tools)


# --------------------------------------------------------------------------------
# primary_keys
# --------------------------------------------------------------------------------


def derive_primary_key(collection: str, rows: list[dict[str, Any]]) -> str:
    """Infer which field identifies a row, then prove the choice.

    The production loader (`expand.primary_key_for`) already falls back to
    `<singular>_id` / `id`, and otherwise to the single `*_id` field present. That
    last rule is unsafe: a collection whose own key is not `*_id` but which carries
    one foreign key resolves to the foreign key. In banking_vn, `vietqr_payments`
    (key `payment_ref`, foreign key `transaction_id`) hits exactly that case.

    So the key must be *proved*, not guessed: a primary key is unique and never null
    across every row. Uniqueness is what disqualifies the foreign key.
    """
    if not rows:
        raise DerivationError(f"collection {collection!r} is empty; cannot infer its key")
    fields = [f for f in rows[0] if all(f in row for row in rows)]

    def usable(field: str) -> bool:
        values = [row.get(field) for row in rows]
        if any(value is None for value in values):
            return False
        try:
            return len({str(v) for v in values}) == len(values)
        except TypeError:
            return False

    singular = collection[:-1] if collection.endswith("s") else collection
    for candidate in (f"{singular}_id", "id"):
        if candidate in fields and usable(candidate):
            return candidate

    identifiers = [f for f in fields if f.endswith(_ID_SUFFIXES) and usable(f)]
    if len(identifiers) == 1:
        return identifiers[0]
    if not identifiers:
        raise DerivationError(
            f"collection {collection!r} carries no unique, non-null identifier field; "
            "declare primary_keys for it"
        )
    raise DerivationError(
        f"collection {collection!r} has several unique identifier fields "
        f"({', '.join(sorted(identifiers))}); declare primary_keys for it"
    )


def derive_primary_keys(
    fixtures: dict[str, list[dict[str, Any]]], needed: set[str]
) -> dict[str, str]:
    """Infer keys only for the collections some slot actually binds through.

    A collection nothing references — `fee_schedule` here — needs no key, and
    demanding one would fail a pack for an unused lookup table.
    """
    keys: dict[str, str] = {}
    for collection in sorted(needed):
        rows = fixtures.get(collection)
        if not isinstance(rows, list):
            raise DerivationError(f"slot references unknown collection {collection!r}")
        keys[collection] = derive_primary_key(collection, rows)
    return keys


# --------------------------------------------------------------------------------
# absent_ids
# --------------------------------------------------------------------------------


def derive_absent_ids(
    fixtures: dict[str, list[dict[str, Any]]],
    primary_keys: dict[str, str],
    needed: set[str],
    *,
    per_collection: int = 1,
) -> dict[str, list[str]]:
    """Mint ids guaranteed absent from every collection, matching the local format.

    CAVEAT, and it is the sharpest one in A1: an absent id is a *bound slot value*,
    so it enters `slot_bindings` and therefore the `task_id` hash. Generating one is
    only equivalence-preserving if the generated string equals what the author had
    written. The convention below (`<PREFIX>-ABSENT-<n>`, prefix taken from the
    collection's own ids) reproduces banking_vn, but that is a fact to verify per
    pack, not a property of the derivation. The A0/A1 task_id check is what proves it.
    """
    taken: set[str] = set()
    for rows in fixtures.values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    taken.update(str(v) for v in row.values() if isinstance(v, str))

    absent: dict[str, list[str]] = {}
    for collection in sorted(needed):
        rows = fixtures.get(collection) or []
        key = primary_keys.get(collection) or derive_primary_key(collection, rows)
        samples = [str(row[key]) for row in rows if isinstance(row, dict) and row.get(key) is not None]
        prefix = samples[0].split("-")[0] if samples and "-" in samples[0] else collection.upper()[:3]
        minted: list[str] = []
        index = 1
        while len(minted) < per_collection:
            candidate = f"{prefix}-ABSENT-{index}"
            index += 1
            if candidate in taken:
                continue
            taken.add(candidate)
            minted.append(candidate)
        absent[collection] = minted
    return absent


# --------------------------------------------------------------------------------
# what the templates need
# --------------------------------------------------------------------------------


def best_effort_primary_keys(fixtures: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Infer a key for every collection that has one, skipping those that do not.

    Used to answer "which collection does this tool parameter name?", where a lookup
    table with no identity — `fee_schedule` — is simply not an answer rather than an
    error.
    """
    keys: dict[str, str] = {}
    for collection, rows in fixtures.items():
        if not isinstance(rows, list) or not rows:
            continue
        try:
            keys[collection] = derive_primary_key(collection, rows)
        except DerivationError:
            continue
    return keys


def collection_for_param(param: str, primary_keys: dict[str, str]) -> str | None:
    """Map a tool parameter onto the collection whose records it identifies."""
    for collection, key in sorted(primary_keys.items()):
        if key == param:
            return collection
    for collection, key in sorted(primary_keys.items()):
        if param.endswith(key):
            return collection
    return None


def collections_used_by_tools(
    tools: dict[str, dict[str, Any]], primary_keys: dict[str, str]
) -> set[str]:
    """Collections a tool can be called against, so a not-found probe can be built.

    Validation coverage is per tool, not per slot: every tool needs a negative probe,
    including tools no template happens to bind an absent id through.
    """
    used: set[str] = set()
    for spec in tools.values():
        for param in spec["required"]:
            collection = collection_for_param(param, primary_keys)
            if collection:
                used.add(collection)
    return used


def referenced_collections(templates: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Return (fixture collections, absent-id collections) the templates bind through."""
    fixture_refs: set[str] = set()
    absent_refs: set[str] = set()

    def scan(slot: dict[str, Any]) -> None:
        source = str(slot.get("source") or "")
        kind, _, rest = source.partition(":")
        if not rest:
            kind, rest = "fixture", source
        if kind == "fixture":
            fixture_refs.add(rest.partition(".")[0])
        elif kind == "absent":
            absent_refs.add(rest.strip())

    for template in templates:
        for slot in (template.get("slots") or {}).values():
            scan(slot)
        for entry in template.get("user_simulator_turns") or []:
            for definition in (entry.get("slot_updates") or {}).values():
                scan(definition)
        for definition in (template.get("corrects") or {}).values():
            scan(definition if isinstance(definition, dict) else {"source": definition})
    return fixture_refs, absent_refs
