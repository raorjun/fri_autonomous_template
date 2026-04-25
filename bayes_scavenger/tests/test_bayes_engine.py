from pathlib import Path

from bayes_scavenger.bayes_engine import BayesianSearchEngine
from bayes_scavenger.search_config import load_search_config


def test_negative_scan_pushes_belief_away_from_current_location() -> None:
    engine = BayesianSearchEngine(
        {"table": 0.6, "desk": 0.4},
        {"table": 0.8, "desk": 0.8},
        {"table": 0.2, "desk": 0.2},
        false_positive_rate=0.05,
    )

    engine.update("table", detected=False)

    assert engine.beliefs["table"] < engine.beliefs["desk"]
    assert engine.visited_counts["table"] == 1


def test_positive_scan_makes_current_location_most_likely() -> None:
    engine = BayesianSearchEngine(
        {"table": 0.5, "desk": 0.5},
        {"table": 0.9, "desk": 0.9},
        {"table": 0.1, "desk": 0.1},
        false_positive_rate=0.05,
    )

    engine.update("desk", detected=True)

    assert engine.beliefs["desk"] > 0.9


def test_next_choice_changes_after_negative_scan() -> None:
    engine = BayesianSearchEngine(
        {"table": 0.55, "desk": 0.45},
        {"table": 0.8, "desk": 0.8},
        {"table": 0.2, "desk": 0.2},
        false_positive_rate=0.05,
    )
    waypoints = {
        "table": {"x": 0.0, "y": 0.0},
        "desk": {"x": 1.0, "y": 0.0},
    }

    assert engine.choose_next_location(current_pose=(0.0, 0.0), waypoints=waypoints) == "table"

    engine.update("table", detected=False)

    assert engine.choose_next_location(current_pose=(0.0, 0.0), waypoints=waypoints) == "desk"


def test_config_loader_reads_false_positive_rate() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "search_config.yaml"
    config = load_search_config(str(config_path))

    assert config["likelihoods"]["false_positive_rate"] == 0.05
    assert config["detector"]["target_mode"] == "red"
    assert set(config["zones"]) == {"zone_1", "zone_2", "zone_3"}
