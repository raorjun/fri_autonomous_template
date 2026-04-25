from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from bayes_scavenger.bayes_engine import BayesianSearchEngine
from bayes_scavenger.search_policy import choose_next_location


@dataclass
class SearchStepRecord:
    step: int
    location: str
    detected: bool
    target_found: bool
    detection_probability: float
    total_distance_m: float
    beliefs: Dict[str, float]


@dataclass
class SearchTrialResult:
    strategy: str
    true_target: str
    found: bool
    steps_taken: int
    total_distance_m: float
    history: List[SearchStepRecord]


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def run_search_trial(
    config: dict,
    *,
    strategy: str,
    true_target: Optional[str],
    seed: int,
    max_steps: int,
    record_history: bool = True,
) -> SearchTrialResult:
    priors = config["priors"]
    waypoints = config["waypoints"]
    likelihoods = config["likelihoods"]
    search_cfg = config["search"]

    engine = BayesianSearchEngine(
        priors,
        likelihoods["positive_detection"],
        likelihoods["negative_detection"],
        false_positive_rate=likelihoods["false_positive_rate"],
        revisit_penalty=float(search_cfg["revisit_penalty"]),
    )

    rng = random.Random(seed)
    locations = list(priors.keys())
    if true_target is None:
        true_target = rng.choices(locations, weights=[priors[name] for name in locations], k=1)[0]
    if true_target not in priors:
        raise ValueError(f"Unknown target location: {true_target}")

    start_pose_cfg = search_cfg["start_pose"]
    current_pose = (float(start_pose_cfg["x"]), float(start_pose_cfg["y"]))
    total_distance_m = 0.0
    history: List[SearchStepRecord] = []
    sequence_order = tuple(search_cfg["sequence_order"])

    for step in range(1, max_steps + 1):
        next_location = choose_next_location(
            strategy,
            engine,
            rng=rng,
            sequence_order=sequence_order,
            current_pose=current_pose,
            waypoints=waypoints,
        )
        waypoint = waypoints[next_location]
        target_pose = (float(waypoint["x"]), float(waypoint["y"]))
        total_distance_m += _distance(current_pose, target_pose)
        current_pose = target_pose

        if next_location == true_target:
            detection_probability = engine.positive_likelihoods[next_location]
        else:
            detection_probability = engine.false_positive_rate

        detected = rng.random() < detection_probability
        engine.update(next_location, detected)
        target_found = detected and next_location == true_target

        if record_history:
            history.append(
                SearchStepRecord(
                    step=step,
                    location=next_location,
                    detected=detected,
                    target_found=target_found,
                    detection_probability=detection_probability,
                    total_distance_m=total_distance_m,
                    beliefs=dict(engine.beliefs),
                )
            )

        if target_found:
            return SearchTrialResult(
                strategy=strategy,
                true_target=true_target,
                found=True,
                steps_taken=step,
                total_distance_m=total_distance_m,
                history=history,
            )

    return SearchTrialResult(
        strategy=strategy,
        true_target=true_target,
        found=False,
        steps_taken=max_steps,
        total_distance_m=total_distance_m,
        history=history,
    )
