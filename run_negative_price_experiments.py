#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from negative_price_experiments.config import build_default_experiment_configs, build_default_transfer_config
from negative_price_experiments.pipeline import run_experiment, run_transfer_experiment
from negative_price_experiments.progress import ProgressReporter, format_duration


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
        help="Comma-separated experiment IDs to run. Supported: E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15A,E15B,E16A,E16B,E17A,E17B,E18,E19,E20,E21,E22A,E22B,E23,E24,E25,E26,E27,E28,E29,E30,E31,E32,E33,E34,E35,E36,E37,E38,E39,E40.",
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
    data_file = Path(args.data_file).resolve()
    configs = build_default_experiment_configs(args.data_file)
    transfer = build_default_transfer_config(args.data_file)
    reporter = ProgressReporter()

    for experiment_index, experiment_id in enumerate(experiment_ids, start=1):
        experiment_started_at = reporter.now()
        reporter.log(
            (experiment_id,),
            (
                f"experiment {experiment_index}/{len(experiment_ids)} started "
                f"| data={data_file} | output={output_dir / experiment_id}"
            ),
        )
        if experiment_id == "E6":
            artifacts = run_transfer_experiment(transfer, output_dir=output_dir, reporter=reporter)
            artifact_dir = Path(artifacts["metrics_summary"]).resolve().parent
            reporter.log(
                (experiment_id,),
                (
                    f"experiment {experiment_index}/{len(experiment_ids)} completed "
                    f"| elapsed={format_duration(reporter.now() - experiment_started_at)} "
                    f"| artifacts={artifact_dir}"
                ),
            )
            continue
        if experiment_id not in configs:
            raise ValueError(f"Unsupported experiment id: {experiment_id}")
        artifacts = run_experiment(
            configs[experiment_id],
            output_dir=output_dir,
            skip_unavailable_models=args.skip_unavailable_models,
            reporter=reporter,
        )
        artifact_dir = Path(artifacts["metrics_summary"]).resolve().parent
        reporter.log(
            (experiment_id,),
            (
                f"experiment {experiment_index}/{len(experiment_ids)} completed "
                f"| elapsed={format_duration(reporter.now() - experiment_started_at)} "
                f"| artifacts={artifact_dir}"
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
