from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple


class BayesianSearchEngine:
    """Reusable Bayesian belief tracker for waypoint-based object search."""

    def __init__(
        self,
        priors: Mapping[str, float],
        positive_likelihoods: Mapping[str, float],
        negative_likelihoods: Mapping[str, float],
        *,
        false_positive_rate: float = 0.05,
        revisit_penalty: float = 0.15,
        distance_weight: float = 0.35,
    ) -> None:
        self._validate_location_keys(priors, positive_likelihoods, negative_likelihoods)
        self._validate_probability_map(positive_likelihoods, "positive_likelihoods")
        self._validate_probability_map(negative_likelihoods, "negative_likelihoods")
        if not 0.0 <= false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be in [0.0, 1.0)")
        if revisit_penalty < 0.0:
            raise ValueError("revisit_penalty must be non-negative")
        if distance_weight < 0.0:
            raise ValueError("distance_weight must be non-negative")

        self.locations = tuple(priors.keys())
        self.priors = self._normalize(priors)
        self.beliefs = dict(self.priors)
        self.positive_likelihoods = {
            location: float(value) for location, value in positive_likelihoods.items()
        }
        self.negative_likelihoods = {
            location: float(value) for location, value in negative_likelihoods.items()
        }
        self.false_positive_rate = float(false_positive_rate)
        self.revisit_penalty = float(revisit_penalty)
        self.distance_weight = float(distance_weight)
        self.visited_counts = {location: 0 for location in self.locations}

    @staticmethod
    def _validate_location_keys(*maps: Mapping[str, float]) -> None:
        key_sets = [set(mapping.keys()) for mapping in maps]
        if len({frozenset(keys) for keys in key_sets}) != 1:
            raise ValueError("priors and likelihood maps must use the same location keys")

    @staticmethod
    def _validate_probability_map(values: Mapping[str, float], name: str) -> None:
        for location, value in values.items():
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name}[{location!r}] must be in [0.0, 1.0]")

    @staticmethod
    def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
        total = sum(max(float(value), 0.0) for value in values.values())
        if total <= 0.0:
            uniform = 1.0 / max(len(values), 1)
            return {key: uniform for key in values}
        return {key: max(float(value), 0.0) / total for key, value in values.items()}

    def reset(self) -> None:
        self.beliefs = dict(self.priors)
        self.visited_counts = {location: 0 for location in self.locations}

    def update(self, scan_location: str, detected: bool) -> Dict[str, float]:
        if scan_location not in self.beliefs:
            raise KeyError(f"Unknown scan location: {scan_location}")

        updated: Dict[str, float] = {}
        off_location_negative_rate = 1.0 - self.false_positive_rate

        for hypothesis, prior in self.beliefs.items():
            if detected:
                if hypothesis == scan_location:
                    likelihood = self.positive_likelihoods[hypothesis]
                else:
                    likelihood = self.false_positive_rate
            else:
                if hypothesis == scan_location:
                    likelihood = self.negative_likelihoods[hypothesis]
                else:
                    likelihood = off_location_negative_rate
            updated[hypothesis] = prior * likelihood

        self.beliefs = self._normalize(updated)
        if not detected:
            self.visited_counts[scan_location] += 1
        return dict(self.beliefs)

    def score_location(
        self,
        location: str,
        *,
        current_pose: Optional[Tuple[float, float]] = None,
        waypoints: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> float:
        if location not in self.beliefs:
            raise KeyError(f"Unknown location: {location}")

        belief = self.beliefs[location]
        revisit_factor = max(0.05, 1.0 - self.revisit_penalty * self.visited_counts[location])
        score = belief * revisit_factor

        if current_pose is not None and waypoints is not None:
            waypoint = waypoints[location]
            dx = float(waypoint["x"]) - current_pose[0]
            dy = float(waypoint["y"]) - current_pose[1]
            distance = math.hypot(dx, dy)
            score /= 1.0 + self.distance_weight * distance

        return score

    def choose_next_location(
        self,
        *,
        current_pose: Optional[Tuple[float, float]] = None,
        waypoints: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> str:
        return max(
            self.locations,
            key=lambda location: self.score_location(
                location,
                current_pose=current_pose,
                waypoints=waypoints,
            ),
        )
