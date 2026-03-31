from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from negative_price_experiments.config import ExperimentConfig, TimeRange, WalkForwardFold, utc_ts
from negative_price_experiments.models import HAS_TORCH, HAS_XGBOOST
from negative_price_experiments.pipeline import run_experiment


def build_runner_frame() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1h", tz="UTC")
    for country, offset in [("AT", 0.0), ("BE", 1.0)]:
        for idx, ts in enumerate(times):
            signal = np.sin((idx + offset) / 8.0)
            rows.append(
                {
                    "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "country": country,
                    "time_zone": "UTC",
                    "is_weekend_local": float(ts.weekday() >= 5),
                    "is_holiday_local": 0.0,
                    "price": -5.0 if signal > 0.75 else 20.0 + signal,
                    "load": 100.0 + 10.0 * np.cos(idx / 6.0),
                    "temp_2m_c": 10.0 + signal,
                    "wind_speed_10m_ms": 4.0 + signal,
                    "shortwave_radiation_wm2": max(0.0, 200.0 * np.sin(idx / 12.0)),
                    "cloud_cover_pct": 40.0 + 20.0 * signal,
                    "precipitation_mm": max(0.0, signal),
                    "pressure_msl_hpa": 1000.0 + signal,
                    "Wind Onshore": 50.0 + 5.0 * signal,
                    "Solar": max(0.0, 30.0 * np.sin(idx / 12.0)),
                    "total export": 5.0 + signal,
                    "total import": 6.0 + signal,
                }
            )
    return pd.DataFrame(rows)


class NegativePriceRunnerTest(unittest.TestCase):
    def test_smoke_run_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            config = ExperimentConfig(
                name="SMOKE",
                data_path=data_path,
                countries=("AT", "BE"),
                feature_group="public",
                window_hours=24,
                horizon_hours=1,
                models=("Majority", "LogisticRegression"),
                split_strategy="unit",
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
            )
            folds = (
                WalkForwardFold(
                    "F1",
                    TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-04 00:00:00")),
                    TimeRange(utc_ts("2024-01-04 00:00:00"), utc_ts("2024-01-06 00:00:00")),
                ),
                WalkForwardFold(
                    "F2",
                    TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-06 00:00:00")),
                    TimeRange(utc_ts("2024-01-06 00:00:00"), utc_ts("2024-01-08 00:00:00")),
                ),
            )

            artifacts = run_experiment(
                config,
                output_dir=out_dir,
                folds=folds,
                final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
            )

            self.assertTrue(all(path.exists() for path in artifacts.values()))
            metrics = pd.read_csv(artifacts["metrics_summary"])
            predictions = pd.read_csv(artifacts["predictions"])
            self.assertIn("model", metrics.columns)
            self.assertIn("y_prob", predictions.columns)
            self.assertIn("Majority", set(metrics["model"]))
            self.assertIn("LogisticRegression", set(metrics["model"]))

    def test_skip_unavailable_models_keeps_tabular_run_alive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            config = ExperimentConfig(
                name="SKIP_DEPS",
                data_path=data_path,
                countries=("AT", "BE"),
                feature_group="public",
                window_hours=24,
                horizon_hours=1,
                models=("Majority", "XGBoost", "GRU"),
                split_strategy="unit",
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
            )
            folds = (
                WalkForwardFold(
                    "F1",
                    TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-04 00:00:00")),
                    TimeRange(utc_ts("2024-01-04 00:00:00"), utc_ts("2024-01-06 00:00:00")),
                ),
            )

            artifacts = run_experiment(
                config,
                output_dir=out_dir,
                folds=folds,
                final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-05 00:00:00")),
                final_test_range=TimeRange(utc_ts("2024-01-05 00:00:00"), utc_ts("2024-01-06 00:00:00")),
                skip_unavailable_models=True,
            )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            expected_models = {"Majority"}
            if HAS_XGBOOST:
                expected_models.add("XGBoost")
            if HAS_TORCH:
                expected_models.add("GRU")
            self.assertEqual(set(metrics["model"]), expected_models)
