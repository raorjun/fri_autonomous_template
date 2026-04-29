from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _to_probability_map(values: Mapping[str, Any], name: str) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in values.items():
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name}[{key!r}] must be in [0.0, 1.0]")
        converted[str(key)] = numeric
    return converted


def _to_non_negative_map(values: Mapping[str, Any], name: str) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in values.items():
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"{name}[{key!r}] must be non-negative")
        converted[str(key)] = numeric
    return converted


def _to_string_list(values: Any, name: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    return [str(value) for value in values]


def load_search_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = _require_mapping(raw, "config")
    search_cfg = _require_mapping(config.get("search", {}), "search")
    detector_cfg = _require_mapping(config.get("detector", {}), "detector")
    yolo_cfg = _require_mapping(config.get("yolo", {}), "yolo")
    priors = _to_non_negative_map(_require_mapping(config.get("priors", {}), "priors"), "priors")
    zones_raw = config.get("zones", config.get("waypoints", {}))
    waypoints = _require_mapping(zones_raw, "zones")
    likelihoods = _require_mapping(config.get("likelihoods", {}), "likelihoods")
    positive_detection = _to_probability_map(
        _require_mapping(likelihoods.get("positive_detection", {}), "likelihoods.positive_detection"),
        "likelihoods.positive_detection",
    )
    negative_detection = _to_probability_map(
        _require_mapping(likelihoods.get("negative_detection", {}), "likelihoods.negative_detection"),
        "likelihoods.negative_detection",
    )

    location_keys = set(priors.keys())
    if not location_keys:
        raise ValueError("priors must not be empty")
    if set(waypoints.keys()) != location_keys:
        raise ValueError("waypoints and priors must define the same location keys")
    if set(positive_detection.keys()) != location_keys:
        raise ValueError("positive_detection and priors must define the same location keys")
    if set(negative_detection.keys()) != location_keys:
        raise ValueError("negative_detection and priors must define the same location keys")

    search_cfg.setdefault("target_label", detector_cfg.get("target_mode", "target"))
    search_cfg.setdefault("camera_topic", "/image_raw")
    search_cfg.setdefault("goal_topic", "/goal_pose")
    search_cfg.setdefault("navigation_mode", "action")
    search_cfg.setdefault("strategy", "bayes")
    search_cfg.setdefault("arrival_radius_m", 0.6)
    search_cfg.setdefault("scan_duration_sec", 3.0)
    search_cfg.setdefault("revisit_penalty", 0.15)
    search_cfg.setdefault("observation_confidence_threshold", 0.18)
    search_cfg.setdefault("auto_advance_sec", 8.0)
    search_cfg.setdefault("demo_force_detect_zone", "")
    search_cfg.setdefault("demo_force_detect_delay_sec", 1.0)
    search_cfg.setdefault("collapse_beliefs_on_found", False)
    search_cfg.setdefault("marker_topic", "/bayes/zones")
    search_cfg.setdefault("random_seed", 7)
    search_cfg.setdefault("sequence_order", list(location_keys))

    detector_cfg.setdefault("camera_topic", search_cfg["camera_topic"])
    detector_cfg.setdefault("target_mode", search_cfg["target_label"])
    detector_cfg.setdefault("min_area_px", 1800)
    detector_cfg.setdefault("blur_kernel", 5)
    detector_cfg.setdefault("red_1_lower", [0, 120, 70])
    detector_cfg.setdefault("red_1_upper", [10, 255, 255])
    detector_cfg.setdefault("red_2_lower", [170, 120, 70])
    detector_cfg.setdefault("red_2_upper", [180, 255, 255])
    detector_cfg.setdefault("blue_lower", [100, 120, 50])
    detector_cfg.setdefault("blue_upper", [140, 255, 255])
    detector_cfg.setdefault("green_lower", [40, 60, 50])
    detector_cfg.setdefault("green_upper", [85, 255, 255])

    yolo_cfg.setdefault("camera_topic", search_cfg["camera_topic"])
    yolo_cfg.setdefault("model_path", "yolov8n.pt")
    yolo_cfg.setdefault("target_label", search_cfg["target_label"])
    yolo_cfg.setdefault("confidence_threshold", search_cfg["observation_confidence_threshold"])
    yolo_cfg.setdefault("image_size", 640)
    yolo_cfg.setdefault("device", "cpu")

    false_positive_rate = float(likelihoods.get("false_positive_rate", 0.05))
    if not 0.0 <= false_positive_rate < 1.0:
        raise ValueError("likelihoods.false_positive_rate must be in [0.0, 1.0)")

    navigation_mode = str(search_cfg["navigation_mode"]).lower()
    if navigation_mode not in {"action", "topic"}:
        raise ValueError("search.navigation_mode must be either 'action' or 'topic'")
    search_cfg["navigation_mode"] = navigation_mode

    strategy = str(search_cfg["strategy"]).lower()
    if strategy not in {"bayes", "random", "sequential"}:
        raise ValueError("search.strategy must be one of: bayes, random, sequential")
    search_cfg["strategy"] = strategy

    search_cfg["demo_force_detect_zone"] = str(search_cfg["demo_force_detect_zone"]).strip()
    search_cfg["demo_force_detect_delay_sec"] = float(search_cfg["demo_force_detect_delay_sec"])
    if search_cfg["demo_force_detect_delay_sec"] < 0.0:
        raise ValueError("search.demo_force_detect_delay_sec must be non-negative")
    search_cfg["collapse_beliefs_on_found"] = bool(search_cfg["collapse_beliefs_on_found"])

    sequence_order = _to_string_list(search_cfg["sequence_order"], "search.sequence_order")
    if set(sequence_order) != location_keys:
        raise ValueError("search.sequence_order must contain exactly the configured locations")
    search_cfg["sequence_order"] = sequence_order

    if search_cfg["demo_force_detect_zone"] and search_cfg["demo_force_detect_zone"] not in location_keys:
        raise ValueError("search.demo_force_detect_zone must be one of the configured locations")

    return {
        "search": search_cfg,
        "detector": detector_cfg,
        "yolo": yolo_cfg,
        "priors": priors,
        "waypoints": waypoints,
        "zones": waypoints,
        "likelihoods": {
            "positive_detection": positive_detection,
            "negative_detection": negative_detection,
            "false_positive_rate": false_positive_rate,
        },
    }
