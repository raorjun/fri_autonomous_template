import random

from bayes_scavenger.bayes_engine import BayesianSearchEngine
from bayes_scavenger.search_policy import choose_next_location


def test_sequential_policy_follows_configured_order() -> None:
    engine = BayesianSearchEngine(
        {"lab": 0.4, "lounge": 0.3, "kitchen": 0.3},
        {"lab": 0.8, "lounge": 0.8, "kitchen": 0.8},
        {"lab": 0.2, "lounge": 0.2, "kitchen": 0.2},
    )

    first = choose_next_location(
        "sequential",
        engine,
        sequence_order=["lounge", "lab", "kitchen"],
    )
    engine.update(first, detected=False)
    second = choose_next_location(
        "sequential",
        engine,
        sequence_order=["lounge", "lab", "kitchen"],
    )

    assert first == "lounge"
    assert second == "lab"


def test_random_policy_chooses_from_least_visited_candidates() -> None:
    engine = BayesianSearchEngine(
        {"lab": 0.4, "lounge": 0.3, "kitchen": 0.3},
        {"lab": 0.8, "lounge": 0.8, "kitchen": 0.8},
        {"lab": 0.2, "lounge": 0.2, "kitchen": 0.2},
    )
    engine.update("lab", detected=False)

    choice = choose_next_location(
        "random",
        engine,
        rng=random.Random(3),
        sequence_order=["lab", "lounge", "kitchen"],
    )

    assert choice in {"lounge", "kitchen"}
