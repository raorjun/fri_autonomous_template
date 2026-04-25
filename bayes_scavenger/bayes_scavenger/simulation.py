from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

from bayes_scavenger.experiment import run_search_trial
from bayes_scavenger.search_config import load_search_config
from bayes_scavenger.search_policy import VALID_STRATEGIES


def _format_beliefs(beliefs: dict[str, float]) -> str:
    ordered = sorted(beliefs.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{name}={value:.3f}" for name, value in ordered)


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in values.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(values), 1)
        return {key: uniform for key in values}
    return {key: max(value, 0.0) / total for key, value in values.items()}


def run_simulation(
    config_path: str,
    *,
    strategy: str,
    true_target: Optional[str],
    seed: int,
    max_steps: int,
    history_csv: str,
) -> int:
    config = load_search_config(config_path)
    normalized_strategy = strategy.lower()
    if normalized_strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = run_search_trial(
        config,
        strategy=normalized_strategy,
        true_target=true_target,
        seed=seed,
        max_steps=max_steps,
        record_history=True,
    )

    print(f"Config: {Path(config_path).name}")
    print(f"Strategy: {normalized_strategy}")
    print(f"Hidden target location: {result.true_target}")
    if result.history:
        print(f"Initial beliefs: {_format_beliefs(_normalize(config['priors']))}")

    for record in result.history:
        outcome = "positive" if record.detected else "negative"
        print(
            f"Step {record.step}: scanned {record.location} | observation={outcome} "
            f"| p={record.detection_probability:.2f} | distance={record.total_distance_m:.2f}m"
        )
        print(f"Beliefs: {_format_beliefs(record.beliefs)}")

    if history_csv:
        output_path = Path(history_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "strategy",
                    "true_target",
                    "step",
                    "location",
                    "detected",
                    "target_found",
                    "detection_probability",
                    "total_distance_m",
                    "zone",
                    "belief",
                ]
            )
            for record in result.history:
                for zone, belief in record.beliefs.items():
                    writer.writerow(
                        [
                            normalized_strategy,
                            result.true_target,
                            record.step,
                            record.location,
                            record.detected,
                            record.target_found,
                            f"{record.detection_probability:.6f}",
                            f"{record.total_distance_m:.6f}",
                            zone,
                            f"{belief:.6f}",
                        ]
                    )

    if result.found:
        print(
            f"Target found at {result.true_target} in {result.steps_taken} steps after "
            f"{result.total_distance_m:.2f} meters."
        )
        return 0

    print("Stopped without a confirmed find. Increase --max-steps or retune the config.")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a text-mode robot search demo.")
    parser.add_argument(
        "--config",
        default="config/search_config.yaml",
        help="Path to the search YAML config.",
    )
    parser.add_argument(
        "--strategy",
        default="bayes",
        help="Search strategy: bayes, random, or sequential.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="True hidden target location. If omitted, one is sampled from the priors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for repeatable runs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of waypoint scans before stopping.",
    )
    parser.add_argument(
        "--history-csv",
        default="",
        help="Optional CSV path for per-step belief history.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        run_simulation(
            args.config,
            strategy=args.strategy,
            true_target=args.target,
            seed=args.seed,
            max_steps=args.max_steps,
            history_csv=args.history_csv,
        )
    )


if __name__ == "__main__":
    main()
