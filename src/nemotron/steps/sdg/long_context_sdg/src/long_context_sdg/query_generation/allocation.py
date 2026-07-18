"""Exact, deterministic marginal quota allocation."""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Mapping
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def largest_remainder(total: int, weights: Mapping[T, float]) -> dict[T, int]:
    """Allocate exactly ``total`` items according to relative weights."""
    positive = {key: weight for key, weight in weights.items() if weight > 0}
    if total < 0 or not positive:
        raise ValueError("quota allocation needs a nonnegative total and positive weights")
    weight_sum = sum(positive.values())
    exact = {key: total * weight / weight_sum for key, weight in positive.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(
        positive,
        key=lambda key: (-(exact[key] - allocated[key]), str(key)),
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def scheduled_labels(total: int, weights: Mapping[T, float], rng: random.Random) -> list[T]:
    labels = [key for key, count in largest_remainder(total, weights).items() for _ in range(count)]
    rng.shuffle(labels)
    return labels
