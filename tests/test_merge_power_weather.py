from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from merge_power_weather import JOIN_KEYS, WEATHER_COLUMNS, merge_power_and_weather


class MergePowerWeatherTest(unittest.TestCase):
    def write_csv(self, directory: Path, name: str, df: pd.DataFrame) -> Path:
        path = directory / name
        df.to_csv(path, index=False)
        return path

    def test_merge_preserves_left_order_and_appends_weather_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            power_df = pd.DataFrame(
                [
                    {"time": "2024-01-01 01:00:00", "country": "AT", "time_zone": "UTC", "price": 12.0},
                    {"time": "2024-01-01 00:00:00", "country": "AT", "time_zone": "UTC", "price": 10.0},
                ]
            )
            weather_df = pd.DataFrame(
                [
                    {
                        "time": "2024-01-01 00:00:00",
                        "country": "AT",
                        "time_zone": "UTC",
                        "weather_source": "era5",
                        "temp_2m_c": 1.2,
                        "wind_speed_10m_ms": 2.1,
                        "shortwave_radiation_wm2": 0.0,
                        "cloud_cover_pct": 82.0,
                        "precipitation_mm": 0.1,
                        "pressure_msl_hpa": 1012.0,
                        "point_count": 5,
                        "has_missing": False,
                        "has_interpolated": False,
                    },
                    {
                        "time": "2024-01-01 01:00:00",
                        "country": "AT",
                        "time_zone": "UTC",
                        "weather_source": "era5",
                        "temp_2m_c": 1.3,
                        "wind_speed_10m_ms": 2.2,
                        "shortwave_radiation_wm2": 0.0,
                        "cloud_cover_pct": 83.0,
                        "precipitation_mm": 0.0,
                        "pressure_msl_hpa": 1011.8,
                        "point_count": 5,
                        "has_missing": False,
                        "has_interpolated": False,
                    },
                ]
            )

            power_path = self.write_csv(tmp_path, "power.csv", power_df)
            weather_path = self.write_csv(tmp_path, "weather.csv", weather_df)
            output_path = tmp_path / "merged.csv"

            merged = merge_power_and_weather(power_path, weather_path, output_path)

            self.assertEqual(
                merged.columns.tolist(),
                power_df.columns.tolist() + WEATHER_COLUMNS,
            )
            self.assertEqual(merged["time"].tolist(), power_df["time"].tolist())
            self.assertEqual(merged["price"].tolist(), [12.0, 10.0])
            self.assertEqual(merged["temp_2m_c"].tolist(), [1.3, 1.2])
            self.assertTrue(output_path.exists())

    def test_merge_raises_for_duplicate_power_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            power_df = pd.DataFrame(
                [
                    {"time": "2024-01-01 00:00:00", "country": "AT", "time_zone": "UTC", "price": 10.0},
                    {"time": "2024-01-01 00:00:00", "country": "AT", "time_zone": "UTC", "price": 11.0},
                ]
            )
            weather_df = pd.DataFrame(
                [
                    {
                        "time": "2024-01-01 00:00:00",
                        "country": "AT",
                        "time_zone": "UTC",
                        "weather_source": "era5",
                        "temp_2m_c": 1.2,
                        "wind_speed_10m_ms": 2.1,
                        "shortwave_radiation_wm2": 0.0,
                        "cloud_cover_pct": 82.0,
                        "precipitation_mm": 0.1,
                        "pressure_msl_hpa": 1012.0,
                        "point_count": 5,
                        "has_missing": False,
                        "has_interpolated": False,
                    }
                ]
            )

            power_path = self.write_csv(tmp_path, "power.csv", power_df)
            weather_path = self.write_csv(tmp_path, "weather.csv", weather_df)

            with self.assertRaisesRegex(RuntimeError, "duplicate join keys"):
                merge_power_and_weather(power_path, weather_path, tmp_path / "merged.csv")

    def test_merge_raises_for_duplicate_weather_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            power_df = pd.DataFrame(
                [
                    {"time": "2024-01-01 00:00:00", "country": "AT", "time_zone": "UTC", "price": 10.0}
                ]
            )
            weather_row = {
                "time": "2024-01-01 00:00:00",
                "country": "AT",
                "time_zone": "UTC",
                "weather_source": "era5",
                "temp_2m_c": 1.2,
                "wind_speed_10m_ms": 2.1,
                "shortwave_radiation_wm2": 0.0,
                "cloud_cover_pct": 82.0,
                "precipitation_mm": 0.1,
                "pressure_msl_hpa": 1012.0,
                "point_count": 5,
                "has_missing": False,
                "has_interpolated": False,
            }
            weather_df = pd.DataFrame([weather_row, weather_row])

            power_path = self.write_csv(tmp_path, "power.csv", power_df)
            weather_path = self.write_csv(tmp_path, "weather.csv", weather_df)

            with self.assertRaisesRegex(RuntimeError, "duplicate join keys"):
                merge_power_and_weather(power_path, weather_path, tmp_path / "merged.csv")

    def test_join_key_contract_is_explicit(self) -> None:
        self.assertEqual(JOIN_KEYS, ["time", "country", "time_zone"])


if __name__ == "__main__":
    unittest.main()
