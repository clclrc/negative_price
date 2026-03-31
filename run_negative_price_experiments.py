#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from negative_price_experiments.config import build_default_experiment_configs, build_default_transfer_config
from negative_price_experiments.pipeline import run_experiment, run_transfer_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run negative-price event prediction experiments.")
    parser.add_argument(
        "--data-file",
        default="ALL_COUNTRIES_2024_2025_WITH_ERA5.csv",
        help="Path to the merged power + ERA5 CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_outputs",
        help="Directory where experiment artifacts will be written.",
    )
    parser.add_argument(
        "--experiments",
        default="E1,E2,E3,E4",
        help="Comma-separated experiment IDs to run. Supported: E1,E2,E3,E4,E5,E6.",
    )
    parser.add_argument(
        "--skip-unavailable-models",
        action="store_true",
        help="Skip models whose dependencies are not installed instead of raising.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_ids = [item.strip().upper() for item in args.experiments.split(",") if item.strip()]
    output_dir = Path(args.output_dir).resolve()
    configs = build_default_experiment_configs(args.data_file)
    transfer = build_default_transfer_config(args.data_file)

    for experiment_id in experiment_ids:
        if experiment_id == "E6":
            run_transfer_experiment(transfer, output_dir=output_dir)
            continue
        if experiment_id not in configs:
            raise ValueError(f"Unsupported experiment id: {experiment_id}")
        run_experiment(
            configs[experiment_id],
            output_dir=output_dir,
            skip_unavailable_models=args.skip_unavailable_models,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
