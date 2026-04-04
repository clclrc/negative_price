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
    _merge_member_prediction_frames,
    _meta_metric_score,
    _normalize_member_weights,
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
