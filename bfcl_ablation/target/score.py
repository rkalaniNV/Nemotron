"""Score a rollout against the task's ground truth, and pair the two wordings.

Two independent verdicts are recorded per rollout, and they are **not** interchangeable:

  `ast_match`     the model's calls equal `expected_tool_calls` — name and arguments,
                  order honoured per the template's `call_order`. This is ground truth
                  as the pack declares it.
  `assertion`     the pack's own `success_assertions` accept the episode the model
                  produced. This is ground truth as the pack *checks* it.

A4 measured this pack's assertions at 0.610 false acceptance on argument-level
corruptions, so `assertion` alone would credit a model for reporting a fabricated
number. `ast_match` is therefore the headline and `assertion` is reported beside it as
the measure of how much the gap costs. Where they disagree, the disagreement is itself
the finding.

The paired statistic is McNemar's exact test. It conditions on the discordant pairs —
tasks one wording got right and the other got wrong — which is the right null for
"same tasks, same model, one thing changed". A two-sample proportion test would throw
away the pairing and lose most of the power at n=33.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any


def canonical_calls(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(name, canonical-argument-JSON) per call, comparable across encodings.

    `benchmark.parquet` encodes arguments as `[[key, canonical_json_value]]` pairs while
    a model returns a plain dict, so both sides are normalised to sorted-key JSON before
    anything is compared.
    """
    out: list[tuple[str, str]] = []
    for call in calls:
        args = call.get("arguments") or {}
        if isinstance(args, list):
            decoded: dict[str, Any] = {}
            for item in args:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    key, value = item
                    try:
                        decoded[str(key)] = json.loads(value) if isinstance(value, str) else value
                    except ValueError:
                        decoded[str(key)] = value
            args = decoded
        elif not isinstance(args, dict):
            args = {}
        out.append(
            (
                str(call.get("function_name") or call.get("name") or ""),
                json.dumps(args, sort_keys=True, ensure_ascii=False),
            )
        )
    return out


def ast_match(*, predicted: list[dict[str, Any]], expected: list[dict[str, Any]], call_order: str) -> bool:
    """Did the model make exactly the expected calls?

    `call_order: any` marks a template whose calls may be issued in one batch — A1
    proved that field is not derivable, so it is read, never inferred. For those the
    comparison is order-insensitive; everywhere else order is part of the claim.
    """
    got = canonical_calls(predicted)
    want = canonical_calls(expected)
    if len(got) != len(want):
        return False
    if call_order == "any":
        return sorted(got) == sorted(want)
    return got == want


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts.

    Under H0 each discordant pair is a fair coin, so the count is Binomial(b+c, 1/2).
    The exact form is used because b+c is small here; the chi-square approximation is
    not valid below ~25 discordant pairs and would overstate significance.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at 0/n and n/n, where Wald does not."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def pair(rows_a: dict[str, dict], rows_b: dict[str, dict], *, verdict: str) -> dict[str, Any]:
    """Compare two wordings over the tasks they share."""
    shared = sorted(set(rows_a) & set(rows_b))
    both = sum(1 for t in shared if rows_a[t][verdict] and rows_b[t][verdict])
    only_a = sum(1 for t in shared if rows_a[t][verdict] and not rows_b[t][verdict])
    only_b = sum(1 for t in shared if not rows_a[t][verdict] and rows_b[t][verdict])
    neither = sum(1 for t in shared if not rows_a[t][verdict] and not rows_b[t][verdict])
    n = len(shared)
    agree = both + neither
    acc_a = sum(1 for t in shared if rows_a[t][verdict])
    acc_b = sum(1 for t in shared if rows_b[t][verdict])
    return {
        "verdict": verdict,
        "n": n,
        "accuracy_a0": round(acc_a / n, 4) if n else None,
        "accuracy_a2": round(acc_b / n, 4) if n else None,
        "accuracy_a0_ci95": wilson(acc_a, n),
        "accuracy_a2_ci95": wilson(acc_b, n),
        "delta": round((acc_b - acc_a) / n, 4) if n else None,
        "paired_agreement": round(agree / n, 4) if n else None,
        "contingency": {
            "both_correct": both,
            "a0_only": only_a,
            "a2_only": only_b,
            "neither": neither,
        },
        "discordant": only_a + only_b,
        "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
        "flipped_task_ids": {
            "a0_correct_a2_wrong": [t for t in shared if rows_a[t][verdict] and not rows_b[t][verdict]],
            "a2_correct_a0_wrong": [t for t in shared if not rows_a[t][verdict] and rows_b[t][verdict]],
        },
    }


def by_group(rows_a: dict[str, dict], rows_b: dict[str, dict], *, key: str, verdict: str) -> dict[str, Any]:
    """Per-policy (or per-category) accuracy and delta.

    Reported per cell and never pooled: most cells hold one task, so a pooled figure
    would be dominated by `single_turn` and hide exactly the policies the ladder cares
    about. Cells this small cannot carry a significance claim, which is why the count
    is emitted alongside every rate.
    """
    shared = sorted(set(rows_a) & set(rows_b))
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id in shared:
        groups[str(rows_a[task_id].get(key))].append(task_id)

    out: dict[str, Any] = {}
    for name, ids in sorted(groups.items()):
        n = len(ids)
        a = sum(1 for t in ids if rows_a[t][verdict])
        b = sum(1 for t in ids if rows_b[t][verdict])
        flips_down = sum(1 for t in ids if rows_a[t][verdict] and not rows_b[t][verdict])
        flips_up = sum(1 for t in ids if not rows_a[t][verdict] and rows_b[t][verdict])
        out[name] = {
            "n": n,
            "a0_correct": a,
            "a2_correct": b,
            "accuracy_a0": round(a / n, 4),
            "accuracy_a2": round(b / n, 4),
            "delta": round((b - a) / n, 4),
            "agreement": round((n - flips_down - flips_up) / n, 4),
            "flipped_down": flips_down,
            "flipped_up": flips_up,
        }
    return out
