from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from negative_price_experiments.config import ExperimentConfig, TimeRange, utc_ts
from negative_price_experiments.data import NumericScaler, prepare_experiment_data


def build_toy_frame(*, periods: int = 16, country: str = "AT") -> pd.DataFrame:
    times = pd.date_range("2024-01-01 00:00:00", periods=periods, freq="1h", tz="UTC")
    rows = []
    for idx, ts in enumerate(times):
        rows.append(
            {
                "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "country": country,
                "time_zone": "UTC",
                "is_weekend_local": float(ts.weekday() >= 5),
                "is_holiday_local": 0.0,
                "price": -1.0 if idx in {7, 8, 9} else 20.0 + idx,
                "load": 100.0 + idx,
                "temp_2m_c": 10.0 + idx,
                "wind_speed_10m_ms": 4.0 + idx,
                "shortwave_radiation_wm2": float(idx),
                "cloud_cover_pct": 30.0 + idx,
                "precipitation_mm": 0.0,
                "pressure_msl_hpa": 1000.0 + idx,
                "Wind Onshore": 50.0 + idx,
                "Solar": float(idx),
                "total export": 5.0,
                "total import": 6.0,
            }
        )
    return pd.DataFrame(rows)


class NegativePriceDataPrepTest(unittest.TestCase):
    def write_frame(self, df: pd.DataFrame) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "toy.csv"
        df.to_csv(path, index=False)
        return path

    def test_select_samples_uses_target_time_not_anchor_time(self) -> None:
        df = build_toy_frame(periods=12)
        path = self.write_frame(df)
        config = ExperimentConfig(
            name="TEST",
            data_path=path,
            countries=("AT",),
            feature_group="public",
            window_hours=3,
            horizon_hours=2,
            models=("Majority",),
            split_strategy="unit",
            ffill_limit=3,
            primary_metric="pr_auc",
            random_seed=42,
        )

        prepared = prepare_experiment_data(config)
        selected = prepared.select_samples(TimeRange(utc_ts("2024-01-01 04:00:00"), utc_ts("2024-01-01 05:00:00")))

        self.assertEqual(len(selected), 1)
        self.assertEqual(str(selected.iloc[0]["anchor_time"]), "2024-01-01 02:00:00+00:00")
        self.assertEqual(str(selected.iloc[0]["target_time"]), "2024-01-01 04:00:00+00:00")

    def test_ffill_limit_and_missing_target_drop_samples(self) -> None:
        df = build_toy_frame(periods=14)
        df.loc[4:7, "load"] = pd.NA
        df.loc[10, "price"] = pd.NA
        path = self.write_frame(df)
        config = ExperimentConfig(
            name="TEST",
            data_path=path,
            countries=("AT",),
            feature_group="public",
            window_hours=4,
            horizon_hours=1,
            models=("Majority",),
            split_strategy="unit",
            ffill_limit=3,
            primary_metric="pr_auc",
            random_seed=42,
        )

        prepared = prepare_experiment_data(config)
        target_times = pd.to_datetime(prepared.sample_manifest["target_time"], utc=True)

        self.assertNotIn(utc_ts("2024-01-01 10:00:00"), set(target_times))
        self.assertFalse(((target_times >= utc_ts("2024-01-01 08:00:00")) & (target_times <= utc_ts("2024-01-01 09:00:00"))).any())

    def test_numeric_scaler_ignores_nan_history_and_outputs_finite_values(self) -> None:
        values = np.asarray(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 30.0],
            ],
            dtype=np.float32,
        )
        scaler = NumericScaler.fit(values, ("a", "b"))

        transformed = scaler.transform(np.asarray([[2.0, 20.0], [np.nan, 20.0]], dtype=np.float32))

        np.testing.assert_allclose(scaler.mean_, np.asarray([2.0, 20.0], dtype=np.float32))
        np.testing.assert_allclose(scaler.scale_, np.asarray([0.8164966, 10.0], dtype=np.float32), rtol=1e-5)
        self.assertTrue(np.isfinite(transformed).all())

    def test_parallel_tabular_bundle_matches_serial_output(self) -> None:
        df = pd.concat([build_toy_frame(periods=24, country="AT"), build_toy_frame(periods=24, country="BE")], ignore_index=True)
        path = self.write_frame(df)
        config = ExperimentConfig(
            name="TEST",
            data_path=path,
            countries=("AT", "BE"),
            feature_group="public",
            window_hours=6,
            horizon_hours=1,
            models=("Majority",),
            split_strategy="unit",
            ffill_limit=3,
            primary_metric="pr_auc",
            random_seed=42,
        )

        prepared = prepare_experiment_data(config)
        samples = prepared.sample_manifest.head(12).copy()

        with patch("negative_price_experiments.data.get_parallel_worker_count", return_value=1):
            serial_bundle = prepared.build_tabular_bundle(samples, include_country=True)
        with patch("negative_price_experiments.data.get_parallel_worker_count", return_value=2):
            parallel_bundle = prepared.build_tabular_bundle(samples, include_country=True)

        np.testing.assert_allclose(serial_bundle.X, parallel_bundle.X)
        np.testing.assert_array_equal(serial_bundle.y, parallel_bundle.y)
