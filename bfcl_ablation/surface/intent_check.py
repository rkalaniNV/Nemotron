"""Round-trip check: does a generated sentence still ask for the same tool?

The mechanical guards in `guards.py` check that slot values survive, that a withheld
slot stays withheld and that no tool name leaks. None of them reads the sentence. A
Vietnamese request to see an account balance, paraphrased into a question about
whether the account is overdrawn, satisfies every one of them and would publish wrong
ground truth.

So a second model call reads the paraphrase and the tool catalogue, and nothing else
— no template id, no category, no required_tools, no expected answer — and names the
tools the request needs. Disagreement with the template's `required_tools` flags the
variant.

This is a model checking a model, which is exactly the dependence the plan's P1 warns
about. An unmeasured checker would be worthless, so `evaluate` scores it against
deliberately intent-shifted decoys and reports recall and false-alarm rate as A2
results in their own right. The measured numbers, not the design, are the argument.
"""

from __future__ import annotations

import json
from typing import Any

from bfcl_ablation.llm import LLMClient

CHECKER_SYSTEM = """You are a routing classifier for a Vietnamese bank's chat assistant.

You see one customer message and the assistant's full tool catalogue. Decide which
tools must be called to fulfil the message as written.

- Judge only what the message asks for. Do not guess at anything it does not say.
- Tokens in curly braces, like {account_id}, are redacted identifiers the customer did
  supply. Treat them as a concrete value of that kind.
- If the message needs several tools, list them all.
- If no tool applies, or the message is too vague to pick one and the assistant would
  have to ask a follow-up question first, return an empty list.

Return {"tools": ["name", ...], "why": "one short sentence"}."""


def catalogue(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for tool in tools:
        function = tool.get("function") or tool
        parameters = function.get("parameters") or {}
        entries.append(
            {
                "name": str(function.get("name")),
                "description": str(function.get("description") or ""),
                "arguments": sorted((parameters.get("properties") or {}).keys()),
            }
        )
    return sorted(entries, key=lambda e: e["name"])


def predict(client: LLMClient, message: str, tool_catalogue: list[dict[str, Any]]) -> dict[str, Any]:
    """Ask the checker which tools `message` needs."""
    known = {entry["name"] for entry in tool_catalogue}

    def validate(parsed: Any) -> dict[str, Any]:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tools"), list):
            raise ValueError("expected {'tools': [...]}")
        names = [str(name) for name in parsed["tools"]]
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(f"unknown tool names: {unknown}")
        return {"tools": sorted(set(names)), "why": str(parsed.get("why") or "")}

    return client.json_object(
        system=CHECKER_SYSTEM,
        user=json.dumps(
            {"customer_message": message, "tool_catalogue": tool_catalogue},
            ensure_ascii=False,
            indent=2,
        ),
        validate=validate,
        max_tokens=1200,
    )


def check_many(
    client: LLMClient,
    items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score a batch of `{text, required_tools, ...}` rows against the checker.

    A row whose call fails resolves to `agrees=None` and is excluded from the rates,
    so a transport error cannot masquerade as a caught intent shift.
    """
    entries = catalogue(tools)
    predictions = client.map([(lambda row=row: predict(client, row["text"], entries)) for row in items])

    scored = []
    for row, prediction in zip(items, predictions):
        expected = sorted(set(str(name) for name in row.get("required_tools") or []))
        if prediction is None:
            scored.append({**row, "predicted_tools": None, "agrees": None, "why": "checker call failed"})
            continue
        scored.append(
            {
                **row,
                "predicted_tools": prediction["tools"],
                "expected_tools": expected,
                "agrees": prediction["tools"] == expected,
                "why": prediction["why"],
            }
        )
    return scored


def disagreement_kind(row: dict[str, Any]) -> str | None:
    """Name the shape of a disagreement, because they are not equally dangerous.

    A checker that under-predicts is arguing about how many calls a request implies —
    a withheld slot it wants asked for first, a chained call it cannot see from the
    opening turn. A checker that names a *different* tool is saying the sentence now
    requests something else, which is the failure that would publish wrong ground
    truth. Collapsing both into one flag rate hides which one A2 actually produced.
    """
    if row.get("agrees") is not False:
        return None
    expected = set(row.get("expected_tools") or [])
    predicted = set(row.get("predicted_tools") or [])
    if predicted < expected:
        return "under_predicted"
    if expected < predicted:
        return "over_predicted"
    return "substituted"


def _kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = disagreement_kind(row)
        if kind is not None:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _by_key(rows: list[dict[str, Any]], key: str, lookup: dict[str, str] | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row.get(key))
        if lookup is not None:
            name = lookup.get(name, name)
        grouped.setdefault(name, []).append(row)
    out = {}
    for name, group in sorted(grouped.items()):
        usable = [row for row in group if row.get("agrees") is not None]
        flagged = [row for row in usable if row["agrees"] is False]
        out[name] = {
            "n": len(usable),
            "flagged": len(flagged),
            "rate": round(len(flagged) / len(usable), 4) if usable else None,
            "kinds": _kind_counts(flagged),
        }
    return out


def _rate(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    usable = [row for row in rows if row.get("agrees") is not None]
    hits = [row for row in usable if predicate(row)]
    return {
        "n": len(usable),
        "n_unscored": len(rows) - len(usable),
        "count": len(hits),
        "rate": round(len(hits) / len(usable), 4) if usable else None,
    }


def evaluate(
    client: LLMClient,
    *,
    canonical: list[dict[str, Any]],
    paraphrases: list[dict[str, Any]],
    shifts: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    policy_by_template: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure the checker on three populations and return its operating point.

    `canonical` is the human-authored sentence for each template. It is the calibration
    floor: whatever it flags is the checker disagreeing with a hand-written benchmark,
    not with a generated one, and that share is a lower bound on the false-alarm rate
    of the whole scheme.
    """
    scored_canonical = check_many(client, canonical, tools)
    scored_paraphrases = check_many(client, paraphrases, tools)
    scored_shifts = check_many(client, shifts, tools)

    flagged = lambda row: row["agrees"] is False  # noqa: E731 - one predicate, used twice
    return {
        "canonical_false_alarm": _rate(scored_canonical, flagged),
        "paraphrase_false_alarm": _rate(scored_paraphrases, flagged),
        "shift_recall": _rate(scored_shifts, flagged),
        "shift_recovered_target": _rate(
            scored_shifts,
            lambda row: row.get("predicted_tools") == [row.get("steered_to")],
        ),
        # A substituted prediction on a paraphrase is the only flag class that means
        # "this sentence now asks for something else". Reported separately so the
        # headline false-alarm rate cannot be read as the drift rate.
        "paraphrase_substitution": _rate(
            scored_paraphrases, lambda row: disagreement_kind(row) == "substituted"
        ),
        "paraphrase_flag_kinds": _kind_counts(scored_paraphrases),
        "paraphrase_by_template": _by_key(scored_paraphrases, "template_id"),
        "paraphrase_by_policy": _by_key(scored_paraphrases, "template_id", policy_by_template or {}),
        "canonical_flag_kinds": _kind_counts(scored_canonical),
        "shift_flag_kinds": _kind_counts(scored_shifts),
        "rows": {
            "canonical": scored_canonical,
            "paraphrases": scored_paraphrases,
            "shifts": scored_shifts,
        },
        "caveat": (
            "Recall is measured against decoys the same model family was asked to shift, "
            "so it bounds the checker's power against LLM-written drift and says nothing "
            "about drift a human author would introduce. False alarms on the canonical "
            "sentences are the floor: those templates are correct by construction, so "
            "every flag there is the checker being wrong."
        ),
    }
