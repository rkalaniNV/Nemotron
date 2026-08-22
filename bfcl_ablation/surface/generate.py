"""Generate paraphrase pools for a pack's opening user turns.

A0 showed the benchmark collapses to one sentence per template and that no config
knob moves that number. The only lever left is generating wording, so A2 asks a model
for many ways to say the same request and keeps everything else frozen.

Two things make this safe enough to measure:

  mechanical rejection   a variant is dropped before it can reach the pipeline if it
                         loses a slot placeholder, invents a literal the canonical
                         sentence never had, or names a tool. These are the same
                         guards `render.check_surface_guards` applies, run early so a
                         bad variant costs a rejection rather than a dropped task.
  a fixed variant 0      index 0 is the human-authored sentence verbatim, so N=1 is
                         bit-identical to A0 and every rung is measured against a
                         control that shares its code path.

What no guard here checks is whether the paraphrase still *asks the same thing*. That
is `intent_check`, and its catch rate is measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from bfcl_ablation.llm import LLMClient
from bfcl_ablation.surface import guards

PARAPHRASE_SYSTEM = """You rewrite Vietnamese banking chat messages.

Rules, all mandatory:
1. Keep the request identical in meaning. The customer must still be asking for
   exactly the same thing, about exactly the same object.
2. Copy every {placeholder} token character for character, braces included. Never
   translate, rename, reorder away, or drop one.
3. Introduce no new numbers, codes, dates, amounts or identifiers. If the original
   has no number, the rewrite has none.
4. Never name an internal function, API or tool.
5. Vary register, politeness, sentence order and vocabulary. Northern and Southern
   phrasing, terse and chatty, formal and casual are all wanted. Do not merely swap
   one word.
6. Natural Vietnamese as a real customer types it. Diacritics required."""

SHIFT_SYSTEM = """You write Vietnamese banking chat messages for a red-team test set.

You are given a customer message and a different thing the customer could ask about.
Write a message that asks for the NEW thing instead of the original one.

Rules, all mandatory:
1. The request must genuinely change. A reader must be able to tell that the
   customer now wants something else.
2. Copy every {placeholder} token character for character, braces included. Keep the
   same ones the original had, no more and no fewer.
3. Introduce no new numbers, codes, dates, amounts or identifiers.
4. Never name an internal function, API or tool. Describe the need in plain words.
5. Natural Vietnamese, diacritics required."""

# Requesting exactly `need` variants leaves no slack for the rejections below, and a
# top-up round costs a whole round trip. 1.6x is what the first pass measured as the
# accept rate with a little headroom.
_OVERSHOOT = 1.6
_MAX_ROUNDS = 3


def _normalize(text: str) -> str:
    """Fold a variant to the form duplicate detection compares.

    Case, whitespace and Unicode composition are not linguistic variation: two
    sentences differing only there would inflate `distinct_masked` without giving a
    model anything new to generalize over.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower().rstrip(" .!?")


def load_templates(pack_dir: Path) -> list[dict[str, Any]]:
    return yaml.safe_load((pack_dir / "task_templates.yaml").read_text(encoding="utf-8"))


def load_tools(pack_dir: Path) -> list[dict[str, Any]]:
    return json.loads((pack_dir / "tools.json").read_text(encoding="utf-8"))


def tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return [str((tool.get("function") or tool).get("name")) for tool in tools]


def _request(
    client: LLMClient,
    *,
    system: str,
    payload: dict[str, Any],
    count: int,
    seed: int,
) -> list[str]:
    def validate(parsed: Any) -> list[str]:
        items = parsed.get("variants") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list) or not items:
            raise ValueError("expected a non-empty list under 'variants'")
        return [str(item) for item in items if isinstance(item, str)]

    body = dict(payload)
    body["how_many"] = count
    return client.json_object(
        system=system,
        user=json.dumps(body, ensure_ascii=False, indent=2)
        + f'\n\nReturn {{"variants": [ ... {count} strings ... ]}}',
        validate=validate,
        max_tokens=400 * count + 900,
        seed=seed,
    )


def _accept(
    candidate: str,
    *,
    canonical: str,
    forbidden_tools: list[str],
    forbidden_phrases: list[str],
    seen: dict[str, str],
    allow_same_meaning: bool,
) -> str | None:
    """Return a rejection reason, or None when the candidate is usable.

    `allow_same_meaning` is False for paraphrases (a variant equal to one already in
    the pool buys nothing) and True for intent-shift decoys, which are allowed to look
    like each other because they are never published.
    """
    text = candidate.strip()
    if not text:
        return "empty"
    reason = guards.mechanical_rejection(
        text,
        canonical=canonical,
        forbidden_tools=forbidden_tools,
        forbidden_phrases=forbidden_phrases,
    )
    if reason is not None:
        return reason
    if not allow_same_meaning and _normalize(text) in seen:
        return "duplicate"
    return None


def paraphrase_pool(
    client: LLMClient,
    template: dict[str, Any],
    *,
    language: str,
    need: int,
    forbidden_tools: list[str],
) -> dict[str, Any]:
    """Build one template's ordered variant pool, index 0 being the authored sentence."""
    canonical = str((template.get("user_turn_templates") or {})[language])
    forbidden_phrases = [
        str(rule)
        for rule in (template.get("paraphrase") or {}).get("must_not_mention") or []
        if rule != guards.TOOL_NAME_RULE
    ]

    pool = [canonical]
    seen = {_normalize(canonical): canonical}
    rejections: list[dict[str, str]] = []

    for round_index in range(_MAX_ROUNDS):
        missing = need - len(pool)
        if missing <= 0:
            break
        candidates = _request(
            client,
            system=PARAPHRASE_SYSTEM,
            payload={
                "language": language,
                "canonical_message": canonical,
                "placeholders_that_must_survive": guards.placeholders(canonical),
                "forbidden_words": forbidden_tools + forbidden_phrases,
                "already_produced": pool[1:],
            },
            count=max(2, math.ceil(missing * _OVERSHOOT)),
            seed=1000 + round_index,
        )
        for candidate in candidates:
            if len(pool) >= need:
                break
            reason = _accept(
                candidate,
                canonical=canonical,
                forbidden_tools=forbidden_tools,
                forbidden_phrases=forbidden_phrases,
                seen=seen,
                allow_same_meaning=False,
            )
            if reason is not None:
                rejections.append({"reason": reason, "text": candidate, "round": str(round_index)})
                continue
            pool.append(candidate.strip())
            seen[_normalize(candidate)] = candidate.strip()

    return {
        "template_id": str(template.get("template_id")),
        "canonical": canonical,
        "variants": pool,
        "accepted": len(pool) - 1,
        "requested": need - 1,
        "rejections": rejections,
    }


def shift_pool(
    client: LLMClient,
    template: dict[str, Any],
    *,
    language: str,
    tools: list[dict[str, Any]],
    per_template: int,
    forbidden_tools: list[str],
) -> list[dict[str, Any]]:
    """Build deliberately intent-shifted decoys for this template.

    Each decoy names the tool it was steered towards, so a decoy that the checker
    resolves back to the original tool is a genuine miss rather than an unlabelled
    disagreement. Decoys that fail the mechanical guards are reported and dropped:
    a shift the cheap guards already catch says nothing about the semantic checker.
    """
    canonical = str((template.get("user_turn_templates") or {})[language])
    required = [str(name) for name in template.get("required_tools") or []]
    catalogue = {
        str((tool.get("function") or tool).get("name")): str((tool.get("function") or tool).get("description") or "")
        for tool in tools
    }
    candidates = [name for name in sorted(catalogue) if name not in required]
    # Taking the first entries alphabetically would steer almost every decoy at the two
    # mutating tools and measure the checker on one narrow kind of drift. Rotating by a
    # stable hash of the template spreads the targets while staying reproducible.
    offset = int(hashlib.sha256(str(template.get("template_id")).encode()).hexdigest(), 16)
    targets = [candidates[(offset + i) % len(candidates)] for i in range(min(per_template, len(candidates)))]

    out: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        candidates = _request(
            client,
            system=SHIFT_SYSTEM,
            payload={
                "language": language,
                "original_message": canonical,
                "placeholders_that_must_survive": guards.placeholders(canonical),
                "what_the_customer_now_wants": catalogue[target],
                "forbidden_words": forbidden_tools,
            },
            count=2,
            seed=5000 + index,
        )
        for candidate in candidates:
            reason = _accept(
                candidate,
                canonical=canonical,
                forbidden_tools=forbidden_tools,
                forbidden_phrases=[],
                seen={},
                allow_same_meaning=True,
            )
            out.append(
                {
                    "template_id": str(template.get("template_id")),
                    "text": candidate.strip(),
                    "steered_to": target,
                    "required_tools": required,
                    "guard_rejection": reason,
                }
            )
            break
    return out


def build(
    client: LLMClient,
    pack_dir: Path,
    *,
    language: str = "vi",
    need: int = 20,
    shifts_per_template: int = 2,
) -> dict[str, Any]:
    """Generate every pool the arm needs, concurrently, and report what was thrown away."""
    templates = load_templates(pack_dir)
    tools = load_tools(pack_dir)
    names = tool_names(tools)

    pools = client.map(
        [
            (lambda t=template: paraphrase_pool(client, t, language=language, need=need, forbidden_tools=names))
            for template in templates
        ]
    )
    shifts = client.map(
        [
            (
                lambda t=template: shift_pool(
                    client,
                    t,
                    language=language,
                    tools=tools,
                    per_template=shifts_per_template,
                    forbidden_tools=names,
                )
            )
            for template in templates
        ]
    )

    failed = [str(t.get("template_id")) for t, pool in zip(templates, pools) if pool is None]
    by_template = {pool["template_id"]: pool for pool in pools if pool is not None}
    flat_shifts = [entry for group in shifts if group for entry in group]

    rejection_reasons: dict[str, int] = {}
    for pool in by_template.values():
        for rejection in pool["rejections"]:
            rejection_reasons[rejection["reason"]] = rejection_reasons.get(rejection["reason"], 0) + 1

    return {
        "language": language,
        "requested_per_template": need - 1,
        "pools": by_template,
        "pool_sizes": {tid: len(pool["variants"]) for tid, pool in sorted(by_template.items())},
        "templates_without_pool": failed,
        "rejection_reasons": rejection_reasons,
        "rejections": [
            {"template_id": tid, **rejection}
            for tid, pool in sorted(by_template.items())
            for rejection in pool["rejections"]
        ],
        "shifts": flat_shifts,
    }


def write_variant_pack(source: Path, destination: Path, pools: dict[str, Any], index: int, language: str) -> Path:
    """Copy the pack with every template's opening turn set to its variant `index`.

    Rewriting the pack rather than the pipeline is what keeps A2 comparable to A0 and
    A1: the generator under test is byte-identical across arms, and the only input
    that moved is the sentence a human would have written.
    """
    import shutil

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    path = destination / "task_templates.yaml"
    templates = yaml.safe_load(path.read_text(encoding="utf-8"))
    for template in templates:
        pool = pools.get(str(template.get("template_id")))
        if pool is None:
            continue
        variants = pool["variants"]
        template["user_turn_templates"][language] = variants[index % len(variants)]
    path.write_text(yaml.safe_dump(templates, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination
