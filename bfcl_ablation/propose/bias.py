"""Selection bias: what the model reached for when it was free to choose.

A3 is the first arm where the question is answerable at all. A0 and A1 have a fixed
template set, so their tool and entity distributions are a human's. Here the sampler
fixes the (category, policy) cell and the model chooses everything else, which makes
every remaining choice attributable.

Three kinds of bias are measured, and they are not the same thing:

  choice bias    over the pool the model could have picked from, which tools, records
                 and values it actually picked. Compared against a conditional-uniform
                 null — uniform *within each proposal's own category universe*, since a
                 model asked for a `dispute` task cannot be faulted for not calling
                 `get_card_limit`.
  failure bias   which cells the model could not fill correctly. A controlled sampler
                 removes the model's ability to avoid a hard policy by choosing an easy
                 one, so avoidance reappears as a differential drop rate, and that is
                 the number to read.
  spread bias    whether the accepted proposals concentrate on a few fixture rows. An
                 entity distribution is compared with the uniform one over the rows the
                 tasks could have bound, not over all rows a collection holds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _tvd(counts: dict[str, float], support: list[str]) -> float:
    """Total variation distance from uniform over `support`. 0 is uniform, 1 is a point mass."""
    if not support:
        return 0.0
    total = sum(counts.get(name, 0.0) for name in support)
    if total <= 0:
        return 1.0
    uniform = 1.0 / len(support)
    return round(0.5 * sum(abs(counts.get(name, 0.0) / total - uniform) for name in support), 4)


def _chi_square(observed: dict[str, float], expected: dict[str, float]) -> float:
    """Pearson statistic against the conditional-uniform null, reported without a p-value.

    No p-value: the proposals are not independent draws (one call may return several),
    and quoting significance for a statistic whose null is this rough would overclaim.
    The number is here to rank tools by how far they sit from the null, not to test one.
    """
    total = 0.0
    for name, count in expected.items():
        if count > 0:
            total += (observed.get(name, 0.0) - count) ** 2 / count
    return round(total, 3)


def tool_choice(
    accepted: list[dict[str, Any]],
    universes: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Observed tool usage against uniform choice inside each proposal's own universe."""
    observed: Counter = Counter()
    expected: dict[str, float] = defaultdict(float)
    off_universe: list[dict[str, str]] = []

    for proposal in accepted:
        required = [str(name) for name in proposal.get("required_tools") or []]
        universe = list(universes.get(str(proposal.get("category")), ()))
        observed.update(required)
        if not universe:
            continue
        share = len(required) / len(universe)
        for name in universe:
            expected[name] += share
        for name in required:
            if name not in universe:
                off_universe.append({"template_id": str(proposal.get("template_id")), "tool": name})

    support = sorted(set(observed) | set(expected))
    return {
        "observed": dict(sorted(observed.items())),
        "expected_uniform_within_category": {k: round(v, 2) for k, v in sorted(expected.items())},
        "ratio_observed_over_expected": {
            name: round(observed.get(name, 0) / expected[name], 2)
            for name in sorted(expected)
            if expected[name] > 0
        },
        "chi_square_vs_conditional_uniform": _chi_square(dict(observed), dict(expected)),
        "tvd_from_uniform_over_all_tools": _tvd({k: float(v) for k, v in observed.items()}, support),
        "tools_never_required": sorted(set(expected) - set(observed)),
        "off_universe_choices": off_universe,
    }


def entity_choice(
    tasks: list[dict[str, Any]],
    fixtures: dict[str, list[dict[str, Any]]],
    primary_keys: dict[str, str],
) -> dict[str, Any]:
    """Which fixture rows the accepted tasks actually bound, and how evenly."""
    bound: dict[str, Counter] = defaultdict(Counter)
    for task in tasks:
        for ref in task.get("fixture_refs") or []:
            collection, _, row_id = str(ref).partition(".")
            bound[collection][row_id] += 1

    per_collection: dict[str, Any] = {}
    for collection, rows in sorted(fixtures.items()):
        key = primary_keys.get(collection)
        ids = [str(row.get(key)) for row in rows if key and row.get(key) is not None]
        counts = bound.get(collection, Counter())
        if not ids:
            continue
        per_collection[collection] = {
            "rows": len(ids),
            "rows_bound": len([i for i in ids if counts.get(i)]),
            "bindings": int(sum(counts.values())),
            "tvd_from_uniform": _tvd({k: float(v) for k, v in counts.items()}, ids),
            "most_bound": [
                {"id": row_id, "tasks": n} for row_id, n in counts.most_common(3)
            ],
        }

    total_rows = sum(entry["rows"] for entry in per_collection.values())
    total_bound = sum(entry["rows_bound"] for entry in per_collection.values())
    return {
        "by_collection": per_collection,
        "entities_total": total_rows,
        "entities_bound": total_bound,
        "coverage": round(total_bound / total_rows, 4) if total_rows else None,
    }


def literal_choice(accepted: list[dict[str, Any]]) -> dict[str, Any]:
    """Values the model invented rather than read from a fixture.

    A literal is where a model's priors show most directly: nothing in the pack suggests
    200000 VND or `napas`, so a concentration here is the model's own.
    """
    values: dict[str, Counter] = defaultdict(Counter)
    filters: Counter = Counter()
    for proposal in accepted:
        for name, slot in (proposal.get("slots") or {}).items():
            source = str(slot.get("source") or "")
            if source.startswith("literal:"):
                values[str(name)][source.partition(":")[2].strip()] += 1
            if slot.get("filter"):
                filters[str(slot["filter"])] += 1
    return {
        "literal_slots": {
            name: {
                "proposals": int(sum(counter.values())),
                "distinct_value_sets": len(counter),
                "most_common": [{"value": v, "n": n} for v, n in counter.most_common(3)],
            }
            for name, counter in sorted(values.items())
        },
        "filters_written": int(sum(filters.values())),
        "distinct_filters": len(filters),
        "most_common_filters": [{"filter": f, "n": n} for f, n in filters.most_common(5)],
    }


def assertion_choice(accepted: list[dict[str, Any]], available: list[str]) -> dict[str, Any]:
    counts: Counter = Counter()
    for proposal in accepted:
        counts.update(str(name) for name in proposal.get("success_assertions") or [])
    return {
        "observed": dict(sorted(counts.items())),
        "never_used": sorted(set(available) - set(counts)),
        "tvd_from_uniform": _tvd({k: float(v) for k, v in counts.items()}, sorted(available)),
    }


def vacuous_gold(
    accepted: list[dict[str, Any]],
    tools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Find accepted tasks whose ground truth cannot fail, and those whose label looks wrong.

    `clarify_only` and `irrelevant` produce an empty expected trace by construction, and
    the only assertion available to them — `assert_no_tool_called` — passes exactly when
    the trace is empty. So the executable oracle validates nothing about them: replay,
    determinism and assertions all succeed no matter what the request says. Every gate in
    the arm reports these as accepted, which is why their accept rate has to be read
    separately from the rest.

    The label check is narrower and decidable: if some tool the template offers has all of
    its required parameters bound as visible slots, the request the customer made is one
    the assistant could have answered, so "decline" or "ask a clarifying question" is the
    wrong gold behaviour. It catches the clear cases only — a request that is answerable
    from context the slots do not name stays invisible to it.
    """
    empty_trace: list[str] = []
    answerable: list[dict[str, Any]] = []
    for template in accepted:
        policy = str(template.get("turn_policy"))
        if policy not in {"clarify_only", "irrelevant"}:
            continue
        template_id = str(template.get("template_id"))
        empty_trace.append(template_id)
        visible = {
            str(name)
            for name, slot in (template.get("slots") or {}).items()
            if slot.get("visible_in_first_turn") is not False
        }
        for name in template.get("tools_present") or []:
            required = set(tools.get(str(name), {}).get("required") or [])
            if required and required <= visible:
                answerable.append({"template_id": template_id, "policy": policy, "answerable_by": str(name)})
                break

    return {
        "unfalsifiable_templates": sorted(empty_trace),
        "unfalsifiable_count": len(empty_trace),
        "unfalsifiable_share_of_accepted": round(len(empty_trace) / len(accepted), 4) if accepted else None,
        "answerable_but_declined": answerable,
        "note": (
            "An unfalsifiable template is not necessarily a bad task; it is a task the "
            "executable oracle cannot check. `answerable_but_declined` is the decidable "
            "subset of those that are also mislabelled."
        ),
    }


def failure_bias(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Accept rate per policy and per category — where the model quietly could not deliver."""

    def table(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for outcome in outcomes:
            grouped[str(outcome[field])].append(outcome)
        rows: dict[str, Any] = {}
        for key, items in sorted(grouped.items()):
            accepted = sum(1 for item in items if item["status"] == "accepted")
            rows[key] = {
                "proposed": len(items),
                "accepted": accepted,
                "accept_rate": round(accepted / len(items), 4) if items else None,
                "drop_buckets": dict(
                    sorted(Counter(item["bucket"] for item in items if item["bucket"]).items())
                ),
            }
        return rows

    by_policy = table("policy")
    rates = [row["accept_rate"] for row in by_policy.values() if row["accept_rate"] is not None]
    return {
        "by_policy": by_policy,
        "by_category": table("category"),
        "accept_rate_spread": round(max(rates) - min(rates), 4) if rates else None,
    }


def compare_with_baseline(
    accepted: list[dict[str, Any]],
    baseline_templates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Put the model's tool and assertion mix next to the human pack's.

    Shares rather than counts, since the two packs are not the same size: A0 wrote 17
    templates, A3 proposes several times that.
    """

    def shares(templates: list[dict[str, Any]], field: str) -> dict[str, float]:
        counts: Counter = Counter()
        for template in templates:
            counts.update(str(name) for name in template.get(field) or [])
        total = sum(counts.values()) or 1
        return {name: round(count / total, 4) for name, count in sorted(counts.items())}

    result: dict[str, Any] = {}
    for field in ("required_tools", "success_assertions"):
        a3 = shares(accepted, field)
        a0 = shares(baseline_templates, field)
        keys = sorted(set(a3) | set(a0))
        result[field] = {
            "a3_share": a3,
            "a0_share": a0,
            "tvd_a3_vs_a0": round(0.5 * sum(abs(a3.get(k, 0.0) - a0.get(k, 0.0)) for k in keys), 4),
            "in_a3_only": sorted(set(a3) - set(a0)),
            "in_a0_only": sorted(set(a0) - set(a3)),
        }
    return result
