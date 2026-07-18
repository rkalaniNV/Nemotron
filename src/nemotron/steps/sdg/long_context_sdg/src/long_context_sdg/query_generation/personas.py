"""Projection helpers for Data Designer managed Nemotron personas."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .config import PersonaLocaleConfig
from .schemas import PersonaProjection


def persona_key(index: int, locale: PersonaLocaleConfig) -> str:
    return f"{index}:{locale.locale}"


def persona_weights(locales: list[PersonaLocaleConfig]) -> dict[str, float]:
    return {persona_key(index, locale): locale.weight for index, locale in enumerate(locales)}


def persona_column_name(key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:10]
    return f"managed_persona_{digest}"


def persona_config_by_key(
    locales: list[PersonaLocaleConfig],
) -> dict[str, PersonaLocaleConfig]:
    return {persona_key(index, locale): locale for index, locale in enumerate(locales)}


def _stable_source_id(row: dict[str, Any]) -> str:
    if row.get("uuid"):
        return str(row["uuid"])
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return "persona-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:20]]
    return str(value)[:500]


def project_persona(
    raw: dict[str, Any] | str,
    locale: PersonaLocaleConfig,
    *,
    seed: int,
) -> PersonaProjection:
    row = json.loads(raw) if isinstance(raw, str) else dict(raw)
    source_id = _stable_source_id(row)
    rng = random.Random(f"{seed}|{locale.locale}|{source_id}")
    available = {
        field: weight
        for field, weight in locale.narrative_fields.items()
        if weight > 0 and str(row.get(field, "")).strip()
    }
    if not available:
        raise ValueError(f"managed persona {source_id} has none of the configured narrative fields")
    fields = list(available)
    narrative_field = rng.choices(
        fields,
        weights=[available[field] for field in fields],
        k=1,
    )[0]
    attributes = {
        field: _compact_value(row[field])
        for field in locale.attribute_fields
        if field in row and row[field] not in (None, "")
    }
    return PersonaProjection(
        source_dataset=f"nemotron-personas/{locale.locale}",
        source_revision=locale.asset_revision,
        source_split=locale.locale,
        source_id=source_id,
        language=locale.language,
        narrative_field=narrative_field,
        narrative=str(row[narrative_field]).strip(),
        attributes=attributes,
    )
