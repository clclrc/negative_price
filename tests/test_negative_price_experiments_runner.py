from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from negative_price_experiments.config import AdaptBudget, ExperimentConfig, TimeRange, TransferConfig, WalkForwardFold, utc_ts
from negative_price_experiments.models import HAS_TORCH, HAS_XGBOOST, TorchTrainingOutcome
from negative_price_experiments.pipeline import (
    _build_meta_prediction_frame,
    _collapse_seed_averaged_predictions,
    _merge_member_prediction_frames,
    _meta_metric_score,
    _normalize_member_weights,
    _run_best_member_late_fusion_experiment,
    _run_late_fusion_experiment,
    _run_repeated_seed_experiment,
    _run_stacking_experiment,
    _weighted_member_probability,
    run_experiment,
    run_transfer_experiment,
)


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


def build_transfer_frame() -> pd.DataFrame:
    rows = []
    times = pd.date_range("2024-01-01 00:00:00", periods=144, freq="1h", tz="UTC")
    for country, offset in [("AT", 0.0), ("BE", 1.0), ("BG", 2.0)]:
        for idx, ts in enumerate(times):
            signal = np.sin((idx + offset) / 6.0)
            rows.append(
                {
                    "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "country": country,
                    "time_zone": "UTC",
                    "is_weekend_local": float(ts.weekday() >= 5),
                    "is_holiday_local": 0.0,
                    "price": -3.0 if signal > 0.65 else 30.0 + signal,
                    "load": 100.0 + 5.0 * np.cos(idx / 5.0),
                    "temp_2m_c": 10.0 + signal,
                    "wind_speed_10m_ms": 4.0 + signal,
                    "shortwave_radiation_wm2": max(0.0, 150.0 * np.sin(idx / 10.0)),
                    "cloud_cover_pct": 40.0 + 10.0 * signal,
                    "precipitation_mm": max(0.0, signal),
                    "pressure_msl_hpa": 1000.0 + signal,
                    "Wind Onshore": 50.0 + 5.0 * signal,
                    "Solar": max(0.0, 20.0 * np.sin(idx / 10.0)),
                    "total export": 5.0 + signal,
                    "total import": 6.0 + signal,
                }
            )
    return pd.DataFrame(rows)


def write_fake_artifacts(
    output_dir: Path,
    *,
    experiment: str,
    model: str,
    val_prob: float,
    test_prob: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = pd.DataFrame(
        {
            "country": ["AT", "AT"],
            "anchor_time": ["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"],
            "target_time": ["2024-01-01 01:00:00+00:00", "2024-01-01 02:00:00+00:00"],
            "y_true": [0, 1],
        }
    )
    predictions = pd.DataFrame(
        {
            "country": ["AT", "AT", "AT", "AT"],
            "anchor_time": [
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-01 00:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
            ],
            "target_time": [
                "2024-01-01 01:00:00+00:00",
                "2024-01-01 02:00:00+00:00",
                "2024-01-01 01:00:00+00:00",
                "2024-01-01 02:00:00+00:00",
            ],
            "y_true": [0, 1, 0, 1],
            "y_prob": [val_prob, 1.0 - val_prob, test_prob, 1.0 - test_prob],
            "split": ["val", "val", "test", "test"],
            "experiment": [experiment] * 4,
            "model": [model] * 4,
            "candidate": ["default"] * 4,
            "threshold": [0.5] * 4,
            "fold": ["F1", "F1", pd.NA, pd.NA],
            "protocol": [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    metrics = pd.DataFrame(
        {
            "experiment": [experiment, experiment],
            "model": [model, model],
            "split": ["val", "test"],
            "candidate": ["default", "default"],
            "pr_auc": [0.6, 0.55],
            "roc_auc": [0.7, 0.68],
            "precision": [0.5, 0.48],
            "recall": [0.5, 0.5],
            "f1": [0.5, 0.49],
            "accuracy": [0.5, 0.5],
            "balanced_accuracy": [0.5, 0.5],
            "positive_rate": [0.5, 0.5],
            "sample_count": [2, 2],
            "positive_count": [1, 1],
            "negative_count": [1, 1],
            "threshold": [0.5, 0.5],
        }
    )
    sample_manifest.to_csv(output_dir / "sample_manifest.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "progress.log").write_text(f"[{experiment}] reused\n", encoding="utf-8")
    return {
        "sample_manifest": output_dir / "sample_manifest.csv",
        "metrics_summary": output_dir / "metrics_summary.csv",
        "predictions": output_dir / "predictions.csv",
        "progress_log": output_dir / "progress.log",
    }


class NegativePriceRunnerTest(unittest.TestCase):
    def test_meta_helper_functions_build_weighted_predictions(self) -> None:
        member_a = pd.DataFrame(
            {
                "country": ["AT", "BE"],
                "anchor_time": ["2024-01-01 00:00:00+00:00", "2024-01-01 01:00:00+00:00"],
                "target_time": ["2024-01-01 01:00:00+00:00", "2024-01-01 02:00:00+00:00"],
                "y_true": [0, 1],
                "split": ["val", "val"],
                "fold": ["F1", "F1"],
                "y_prob": [0.2, 0.7],
            }
        )
        member_b = member_a.copy()
        member_b["y_prob"] = [0.4, 0.5]

        merged = _merge_member_prediction_frames({"E30": member_a, "E31": member_b})
        weights = _normalize_member_weights({"E30": 0.6, "E31": 0.4})
        y_prob = _weighted_member_probability(merged, weights)

        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(y_prob, np.array([0.28, 0.62], dtype=np.float32), atol=1e-6)

        frame = _build_meta_prediction_frame(
            merged,
            y_prob,
            experiment="E35",
            model="LateFusion",
            split="val",
            threshold=0.5,
            candidate="E30+E31",
            extra_columns={"fusion_strategy": "val_pr_auc_weighted"},
        )
        self.assertIn("fold", frame.columns)
        self.assertIn("fusion_strategy", frame.columns)
        self.assertEqual(frame["model"].iloc[0], "LateFusion")

    def test_collapse_seed_averaged_predictions_ignores_threshold_metadata(self) -> None:
        repeated_seed_predictions = pd.DataFrame(
            {
                "country": ["AT", "AT", "BE", "BE"],
                "anchor_time": [
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                ],
                "target_time": [
                    "2024-01-01 01:00:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                    "2024-01-01 02:00:00+00:00",
                    "2024-01-01 02:00:00+00:00",
                ],
                "y_true": [0, 0, 1, 1],
                "split": ["val", "val", "val", "val"],
                "fold": ["F1", "F1", "F1", "F1"],
                "experiment": ["E49", "E49", "E49", "E49"],
                "model": ["GRUMultiMarket"] * 4,
                "candidate": ["default"] * 4,
                "threshold": [0.2, 0.8, 0.3, 0.7],
                "seed": [42, 52, 42, 52],
                "y_prob": [0.2, 0.4, 0.6, 0.8],
            }
        )

        collapsed = _collapse_seed_averaged_predictions(repeated_seed_predictions)

        self.assertEqual(len(collapsed), 2)
        self.assertNotIn("threshold", collapsed.columns)
        self.assertNotIn("seed", collapsed.columns)
        np.testing.assert_allclose(collapsed["y_prob"].to_numpy(dtype=np.float32), np.array([0.3, 0.7], dtype=np.float32))

    def test_meta_metric_score_uses_raw_rows_when_seed_aggregation_present(self) -> None:
        metrics = pd.DataFrame(
            {
                "split": ["val", "val", "val", "test"],
                "pr_auc": [0.31, 0.29, 0.30, 0.28],
                "seed_aggregation": ["raw", "raw", "mean", "raw"],
            }
        )

        score = _meta_metric_score(metrics, split="val", metric="pr_auc")
        self.assertAlmostEqual(score, 0.30)

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

            stdout = io.StringIO()
            with redirect_stdout(stdout):
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
            progress_log = Path(artifacts["progress_log"])
            self.assertIn("model", metrics.columns)
            self.assertIn("y_prob", predictions.columns)
            self.assertIn("Majority", set(metrics["model"]))
            self.assertIn("LogisticRegression", set(metrics["model"]))
            self.assertTrue(progress_log.exists())
            output = stdout.getvalue()
            log_output = progress_log.read_text(encoding="utf-8")
            self.assertIn("[SMOKE] data prep started", output)
            self.assertIn("[SMOKE][Majority][F1] fold 1/2 completed", output)
            self.assertIn("[SMOKE][LogisticRegression] final stage 2/2 completed", output)
            self.assertIn("[SMOKE] experiment started", log_output)
            self.assertIn("[SMOKE][Majority][F1] fold 1/2 completed", log_output)
            self.assertIn("[SMOKE] experiment completed", log_output)

    def test_meta_experiments_smoke_run_with_patched_member_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e30 = ExperimentConfig(name="E30", models=("Majority",), **common_kwargs)
            e31 = ExperimentConfig(name="E31", models=("LogisticRegression",), **common_kwargs)
            e35 = ExperimentConfig(
                name="E35",
                models=(),
                meta_kind="late_fusion",
                meta_members=("E30", "E31"),
                **common_kwargs,
            )
            e36 = ExperimentConfig(
                name="E36",
                models=(),
                meta_kind="calibration",
                meta_members=("E30", "E31", "E35"),
                meta_calibration_method="sigmoid",
                **common_kwargs,
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
            config_map = {"E30": e30, "E31": e31, "E35": e35, "E36": e36}

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map):
                e35_artifacts = run_experiment(
                    e35,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
                )
                e36_artifacts = run_experiment(
                    e36,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
                )

            e35_metrics = pd.read_csv(e35_artifacts["metrics_summary"])
            e36_metrics = pd.read_csv(e36_artifacts["metrics_summary"])

            self.assertTrue(Path(e35_artifacts["member_weights"]).exists())
            self.assertTrue(Path(e36_artifacts["candidate_scores"]).exists())
            self.assertIn("LateFusion", set(e35_metrics["model"]))
            self.assertTrue(any(str(model).startswith("Calibrated") for model in set(e36_metrics["model"])))

    def test_late_fusion_smoke_run_supports_repeated_seed_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e49 = ExperimentConfig(
                name="E49",
                models=("Majority",),
                repeat_random_seeds=(42, 52),
                **common_kwargs,
            )
            e44 = ExperimentConfig(name="E44", models=("LogisticRegression",), **common_kwargs)
            e56 = ExperimentConfig(
                name="E56",
                models=(),
                meta_kind="late_fusion",
                meta_members=("E49", "E44"),
                **common_kwargs,
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
            config_map = {"E49": e49, "E44": e44, "E56": e56}

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map):
                e56_artifacts = run_experiment(
                    e56,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
                )

            metrics = pd.read_csv(e56_artifacts["metrics_summary"])
            self.assertIn("LateFusion", set(metrics["model"]))
            self.assertTrue(Path(e56_artifacts["member_weights"]).exists())

    def test_late_fusion_reuses_existing_member_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e30 = ExperimentConfig(name="E30", models=("Majority",), **common_kwargs)
            e31 = ExperimentConfig(name="E31", models=("LogisticRegression",), **common_kwargs)
            e35 = ExperimentConfig(
                name="E35",
                models=(),
                meta_kind="late_fusion",
                meta_members=("E30", "E31"),
                **common_kwargs,
            )

            write_fake_artifacts(out_dir / "cached" / "E30", experiment="E30", model="Majority", val_prob=0.2, test_prob=0.3)
            write_fake_artifacts(
                out_dir / "cached" / "E31",
                experiment="E31",
                model="LogisticRegression",
                val_prob=0.4,
                test_prob=0.6,
            )
            config_map = {"E30": e30, "E31": e31, "E35": e35}

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map), patch(
                "negative_price_experiments.pipeline.run_experiment",
                side_effect=AssertionError("member run should have been reused"),
            ):
                artifacts = _run_late_fusion_experiment(
                    e35,
                    output_path=out_dir / "E35",
                    folds=(),
                    final_train_range=None,
                    final_test_range=None,
                    skip_unavailable_models=False,
                    reporter=None,
                )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            self.assertIn("LateFusion", set(metrics["model"]))
            self.assertTrue(Path(artifacts["member_weights"]).exists())

    def test_repeated_seed_reuses_existing_seed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            config = ExperimentConfig(
                name="E60",
                data_path=data_path,
                countries=("AT", "BE"),
                feature_group="public",
                window_hours=24,
                horizon_hours=1,
                models=("Majority",),
                split_strategy="unit",
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
                repeat_random_seeds=(42, 52),
            )

            write_fake_artifacts(out_dir / "cached" / "E60_SEED_42", experiment="E60_SEED_42", model="Majority", val_prob=0.2, test_prob=0.3)
            write_fake_artifacts(out_dir / "cached" / "E60_SEED_52", experiment="E60_SEED_52", model="Majority", val_prob=0.4, test_prob=0.6)

            with patch(
                "negative_price_experiments.pipeline.run_experiment",
                side_effect=AssertionError("seed run should have been reused"),
            ):
                artifacts = _run_repeated_seed_experiment(
                    config,
                    output_path=out_dir / "E60",
                    folds=(),
                    final_train_range=None,
                    final_test_range=None,
                    skip_unavailable_models=False,
                    reporter=None,
                )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            self.assertIn("raw", set(metrics["seed_aggregation"].dropna()))
            self.assertTrue({42, 52}.issubset(set(metrics["seed"].dropna().astype(int))))

    def test_grouped_stacking_reuses_existing_member_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e30 = ExperimentConfig(name="E30", models=("Majority",), **common_kwargs)
            e31 = ExperimentConfig(name="E31", models=("LogisticRegression",), **common_kwargs)
            e69 = ExperimentConfig(
                name="E69",
                models=(),
                meta_kind="stacking",
                meta_members=("E30", "E31"),
                meta_group_strategy="country",
                **common_kwargs,
            )

            write_fake_artifacts(out_dir / "cached" / "E30", experiment="E30", model="Majority", val_prob=0.2, test_prob=0.3)
            write_fake_artifacts(
                out_dir / "cached" / "E31",
                experiment="E31",
                model="LogisticRegression",
                val_prob=0.4,
                test_prob=0.6,
            )
            config_map = {"E30": e30, "E31": e31, "E69": e69}

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map), patch(
                "negative_price_experiments.pipeline.run_experiment",
                side_effect=AssertionError("member run should have been reused"),
            ):
                artifacts = _run_stacking_experiment(
                    e69,
                    output_path=out_dir / "E69",
                    folds=(),
                    final_train_range=None,
                    final_test_range=None,
                    skip_unavailable_models=False,
                    reporter=None,
                )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            self.assertIn("StackingLogisticRegression", set(metrics["model"]))
            coefficients = pd.read_csv(artifacts["stacking_coefficients"])
            self.assertIn("stacker_group", coefficients.columns)

    def test_best_member_late_fusion_reuses_existing_member_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e75 = ExperimentConfig(name="E75", models=("Majority",), **common_kwargs)
            e77 = ExperimentConfig(name="E77", models=("LogisticRegression",), **common_kwargs)
            e78 = ExperimentConfig(name="E78", models=("LogisticRegression",), **common_kwargs)
            e79 = ExperimentConfig(name="E79", models=("LogisticRegression",), **common_kwargs)
            e80 = ExperimentConfig(
                name="E80",
                models=(),
                meta_kind="best_member_late_fusion",
                meta_members=("E75", "E77", "E78", "E79"),
                **common_kwargs,
            )

            write_fake_artifacts(out_dir / "cached" / "E75", experiment="E75", model="Majority", val_prob=0.2, test_prob=0.3)
            write_fake_artifacts(
                out_dir / "cached" / "E77",
                experiment="E77",
                model="LogisticRegression",
                val_prob=0.45,
                test_prob=0.55,
            )
            write_fake_artifacts(
                out_dir / "cached" / "E78",
                experiment="E78",
                model="LogisticRegression",
                val_prob=0.6,
                test_prob=0.7,
            )
            write_fake_artifacts(
                out_dir / "cached" / "E79",
                experiment="E79",
                model="LogisticRegression",
                val_prob=0.35,
                test_prob=0.4,
            )
            for experiment_name, score in (("E77", 0.58), ("E78", 0.72), ("E79", 0.51)):
                metrics_path = out_dir / "cached" / experiment_name / "metrics_summary.csv"
                metrics = pd.read_csv(metrics_path)
                metrics.loc[metrics["split"] == "val", "pr_auc"] = score
                metrics.to_csv(metrics_path, index=False)
            config_map = {"E75": e75, "E77": e77, "E78": e78, "E79": e79, "E80": e80}

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map), patch(
                "negative_price_experiments.pipeline.run_experiment",
                side_effect=AssertionError("member run should have been reused"),
            ):
                artifacts = _run_best_member_late_fusion_experiment(
                    e80,
                    output_path=out_dir / "E80",
                    folds=(),
                    final_train_range=None,
                    final_test_range=None,
                    skip_unavailable_models=False,
                    reporter=None,
                )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            candidate_scores = pd.read_csv(artifacts["candidate_scores"])
            self.assertIn("LateFusion", set(metrics["model"]))
            self.assertEqual(candidate_scores.iloc[0]["candidate_experiment"], "E78")

    def test_stacking_and_cross_seed_meta_experiments_smoke_run_with_patched_member_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            common_kwargs = {
                "data_path": data_path,
                "countries": ("AT", "BE"),
                "feature_group": "public",
                "window_hours": 24,
                "horizon_hours": 1,
                "split_strategy": "unit",
                "ffill_limit": 3,
                "primary_metric": "pr_auc",
                "random_seed": 42,
            }
            e30 = ExperimentConfig(name="E30", models=("Majority",), **common_kwargs)
            e31 = ExperimentConfig(name="E31", models=("LogisticRegression",), **common_kwargs)
            e33 = ExperimentConfig(name="E33", models=("Majority",), repeat_random_seeds=(42, 52), **common_kwargs)
            e34 = ExperimentConfig(name="E34", models=("LogisticRegression",), repeat_random_seeds=(42, 52), **common_kwargs)
            e39 = ExperimentConfig(
                name="E39",
                models=(),
                meta_kind="stacking",
                meta_members=("E30", "E31"),
                **common_kwargs,
            )
            e40 = ExperimentConfig(
                name="E40",
                models=(),
                meta_kind="cross_seed_ensemble",
                meta_members=("E33", "E34"),
                **common_kwargs,
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
            config_map = {
                "E30": e30,
                "E31": e31,
                "E33": e33,
                "E34": e34,
                "E39": e39,
                "E40": e40,
            }

            with patch("negative_price_experiments.pipeline.build_default_experiment_configs", return_value=config_map):
                e39_artifacts = run_experiment(
                    e39,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
                )
                e40_artifacts = run_experiment(
                    e40,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-07 00:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-07 00:00:00"), utc_ts("2024-01-08 12:00:00")),
                )

            e39_output = Path(e39_artifacts["metrics_summary"]).resolve().parent
            e40_output = Path(e40_artifacts["metrics_summary"]).resolve().parent
            self.assertTrue((e39_output / "stacking_coefficients.csv").exists())
            self.assertTrue((e40_output / "member_weights.csv").exists())
            self.assertIn("StackingLogisticRegression", set(pd.read_csv(e39_artifacts["metrics_summary"])["model"]))
            self.assertIn("CrossSeedEnsemble", set(pd.read_csv(e40_artifacts["metrics_summary"])["model"]))

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
            self.assertTrue(Path(artifacts["progress_log"]).exists())

    @unittest.skipUnless(HAS_TORCH, "torch is required for sequence progress tests")
    def test_sequence_run_prints_epoch_level_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().loc[lambda df: df["country"] == "AT"].to_csv(data_path, index=False)

            config = ExperimentConfig(
                name="SEQ",
                data_path=data_path,
                countries=("AT",),
                feature_group="public",
                window_hours=12,
                horizon_hours=1,
                models=("GRU",),
                split_strategy="unit",
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
            )
            folds = (
                WalkForwardFold(
                    "F1",
                    TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-03 12:00:00")),
                    TimeRange(utc_ts("2024-01-03 12:00:00"), utc_ts("2024-01-04 12:00:00")),
                ),
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                artifacts = run_experiment(
                    config,
                    output_dir=out_dir,
                    folds=folds,
                    final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-04 12:00:00")),
                    final_test_range=TimeRange(utc_ts("2024-01-04 12:00:00"), utc_ts("2024-01-05 12:00:00")),
                )

            output = stdout.getvalue()
            log_output = Path(artifacts["progress_log"]).read_text(encoding="utf-8")
            self.assertIn("[SEQ][GRU][F1] sequence training start", output)
            self.assertIn("[SEQ][GRU][F1] epoch 1/", output)
            self.assertIn("[SEQ][GRU][Final] final training start", output)
            self.assertIn("[SEQ][GRU][Final] epoch 1/", output)
            self.assertIn("[SEQ][GRU][F1] sequence training start", log_output)
            self.assertIn("[SEQ][GRU][Final] final training completed", log_output)

    @unittest.skipUnless(HAS_TORCH, "torch is required for graph-temporal smoke tests")
    def test_graph_temporal_hybrid_smoke_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy.csv"
            out_dir = tmp_path / "out"
            build_runner_frame().to_csv(data_path, index=False)

            config = ExperimentConfig(
                name="GRAPHSEQ",
                data_path=data_path,
                countries=("AT", "BE"),
                feature_group="public",
                window_hours=12,
                horizon_hours=1,
                models=("GraphTemporalHybrid",),
                split_strategy="unit",
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
                sequence_max_epochs=1,
                sequence_patience=1,
                use_mechanism_features=True,
            )
            folds = (
                WalkForwardFold(
                    "F1",
                    TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-03 12:00:00")),
                    TimeRange(utc_ts("2024-01-03 12:00:00"), utc_ts("2024-01-04 12:00:00")),
                ),
            )

            artifacts = run_experiment(
                config,
                output_dir=out_dir,
                folds=folds,
                final_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-04 12:00:00")),
                final_test_range=TimeRange(utc_ts("2024-01-04 12:00:00"), utc_ts("2024-01-05 12:00:00")),
            )

            metrics = pd.read_csv(artifacts["metrics_summary"])
            self.assertIn("GraphTemporalHybrid", set(metrics["model"]))
            self.assertTrue(Path(artifacts["progress_log"]).exists())

    def test_transfer_run_prints_country_budget_and_protocol_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "toy_transfer.csv"
            out_dir = tmp_path / "out"
            build_transfer_frame().to_csv(data_path, index=False)

            config = TransferConfig(
                name="E6TEST",
                data_path=data_path,
                source_countries=("AT", "BE"),
                target_countries=("BG",),
                adapt_budget=(
                    AdaptBudget(
                        name="B24",
                        train_range=TimeRange(utc_ts("2024-01-03 00:00:00"), utc_ts("2024-01-04 12:00:00")),
                        val_range=TimeRange(utc_ts("2024-01-04 12:00:00"), utc_ts("2024-01-05 00:00:00")),
                    ),
                ),
                pretrain_train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-01-03 00:00:00")),
                pretrain_val_range=TimeRange(utc_ts("2024-01-03 00:00:00"), utc_ts("2024-01-04 00:00:00")),
                target_test_range=TimeRange(utc_ts("2024-01-05 00:00:00"), utc_ts("2024-01-06 00:00:00")),
                window_hours=12,
                horizon_hours=1,
                ffill_limit=3,
                primary_metric="pr_auc",
                random_seed=42,
            )

            def fake_predict_sequence_model(_model, dataset) -> np.ndarray:
                return np.full(len(dataset), 0.4, dtype=np.float32)

            stdout = io.StringIO()
            with redirect_stdout(stdout), patch(
                "negative_price_experiments.pipeline.train_sequence_model",
                return_value=TorchTrainingOutcome(state_dict={}, best_epoch=2, best_score=0.5),
            ), patch(
                "negative_price_experiments.pipeline.load_sequence_model",
                return_value=object(),
            ), patch(
                "negative_price_experiments.pipeline.predict_sequence_model",
                side_effect=fake_predict_sequence_model,
            ), patch(
                "negative_price_experiments.pipeline.require_torch",
                return_value=None,
            ):
                artifacts = run_transfer_experiment(config, output_dir=out_dir)

            self.assertTrue(all(path.exists() for path in artifacts.values()))
            output = stdout.getvalue()
            log_output = Path(artifacts["progress_log"]).read_text(encoding="utf-8")
            self.assertIn("[E6TEST][SourcePretrain] source pretraining started", output)
            self.assertIn("[E6TEST][BG][ZeroShot] evaluation completed", output)
            self.assertIn("[E6TEST][BG][B24][TargetOnly] evaluation started", output)
            self.assertIn("[E6TEST][BG][B24][TransferFineTune] evaluation completed", output)
            self.assertIn("[E6TEST][BG] country 1/1 completed", output)
            self.assertIn("[E6TEST] experiment started", log_output)
            self.assertIn("[E6TEST][SourcePretrain] source pretraining started", log_output)
            self.assertIn("[E6TEST] experiment completed", log_output)
