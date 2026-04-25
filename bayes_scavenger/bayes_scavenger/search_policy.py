from __future__ import annotations

import random
from typing import Mapping, Optional, Sequence

from bayes_scavenger.bayes_engine import BayesianSearchEngine

VALID_STRATEGIES = frozenset({"bayes", "random", "sequential"})


def choose_next_location(
    strategy: str,
    engine: BayesianSearchEngine,
    *,
    rng: Optional[random.Random] = None,
    sequence_order: Optional[Sequence[str]] = None,
    current_pose: Optional[tuple[float, float]] = None,
    waypoints: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> str:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported search strategy: {strategy}")

    if normalized_strategy == "bayes":
        return engine.choose_next_location(current_pose=current_pose, waypoints=waypoints)

    ordered_locations = tuple(sequence_order or engine.locations)
    if set(ordered_locations) != set(engine.locations):
        raise ValueError("sequence_order must contain exactly the configured locations")

    min_visits = min(engine.visited_counts.values())
    candidates = [
        location for location in ordered_locations if engine.visited_counts[location] == min_visits
    ]

    if normalized_strategy == "sequential":
        return candidates[0]

    chooser = rng or random.Random()
    return chooser.choice(candidates)
