from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean

from bayes_scavenger.experiment import run_search_trial
from bayes_scavenger.search_config import load_search_config
from bayes_scavenger.search_policy import VALID_STRATEGIES


def _sample_target(config: dict, seed: int) -> str:
    priors = config["priors"]
    rng = random.Random(seed)
    locations = list(priors.keys())
    return rng.choices(locations, weights=[priors[name] for name in locations], k=1)[0]


def _write_summary_csv(path: str, rows: list[dict[str, float | int | str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_history_csv(path: str, strategy: str, trial_number: int, result) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "strategy",
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
                        trial_number,
                        strategy,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Bayesian, random, and sequential search strategies.")
    parser.add_argument("--config", default="config/search_config.yaml", help="Path to the search YAML config.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Monte Carlo trials to run.")
    parser.add_argument("--seed", type=int, default=7, help="Seed used to generate repeatable trials.")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum scans allowed per trial.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["bayes", "random", "sequential"],
        help="Strategies to evaluate.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path for aggregate strategy metrics.",
    )
    parser.add_argument(
        "--history-csv",
        default="",
        help="Optional path for detailed belief history from one selected trial.",
    )
    parser.add_argument(
        "--history-strategy",
        default="bayes",
        help="Which strategy to export to --history-csv.",
    )
    parser.add_argument(
        "--history-trial",
        type=int,
        default=1,
        help="1-indexed trial number to export to --history-csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_search_config(args.config)

    strategies = [strategy.lower() for strategy in args.strategies]
    invalid = [strategy for strategy in strategies if strategy not in VALID_STRATEGIES]
    if invalid:
        raise SystemExit(f"Unsupported strategies: {', '.join(invalid)}")

    trial_target_seeds = [args.seed + trial_index * 97 for trial_index in range(args.trials)]
    summary_rows: list[dict[str, float | int | str]] = []

    for strategy_index, strategy in enumerate(strategies):
        results = []
        for trial_number, target_seed in enumerate(trial_target_seeds, start=1):
            true_target = _sample_target(config, target_seed)
            result = run_search_trial(
                config,
                strategy=strategy,
                true_target=true_target,
                seed=target_seed + strategy_index * 10_000,
                max_steps=args.max_steps,
                record_history=bool(args.history_csv)
                and strategy == args.history_strategy.lower()
                and trial_number == args.history_trial,
            )
            results.append(result)

            if args.history_csv and strategy == args.history_strategy.lower() and trial_number == args.history_trial:
                _write_history_csv(args.history_csv, strategy, trial_number, result)

        success_rate = sum(1 for result in results if result.found) / max(len(results), 1)
        avg_steps_all = mean(result.steps_taken for result in results)
        avg_distance_all = mean(result.total_distance_m for result in results)
        found_results = [result for result in results if result.found]
        avg_steps_found = mean(result.steps_taken for result in found_results) if found_results else 0.0
        avg_distance_found = (
            mean(result.total_distance_m for result in found_results) if found_results else 0.0
        )

        summary_rows.append(
            {
                "strategy": strategy,
                "trials": len(results),
                "success_rate": round(success_rate, 4),
                "avg_steps_all": round(avg_steps_all, 4),
                "avg_steps_found": round(avg_steps_found, 4),
                "avg_distance_m_all": round(avg_distance_all, 4),
                "avg_distance_m_found": round(avg_distance_found, 4),
            }
        )

    for row in summary_rows:
        print(
            f"{row['strategy']}: success_rate={row['success_rate']:.2f} "
            f"avg_steps_all={row['avg_steps_all']:.2f} avg_distance_m_all={row['avg_distance_m_all']:.2f}"
        )

    if args.summary_csv and summary_rows:
        _write_summary_csv(args.summary_csv, summary_rows)


if __name__ == "__main__":
    main()
