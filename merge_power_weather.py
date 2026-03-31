#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


JOIN_KEYS = ["time", "country", "time_zone"]
WEATHER_COLUMNS = [
    "weather_source",
    "temp_2m_c",
    "wind_speed_10m_ms",
    "shortwave_radiation_wm2",
    "cloud_cover_pct",
    "precipitation_mm",
    "pressure_msl_hpa",
    "point_count",
    "has_missing",
    "has_interpolated",
]
DEFAULT_POWER_FILE = "ALL_COUNTRIES_2024_2025.csv"
DEFAULT_WEATHER_FILE = "WEATHER_ERA5_2024_2025.csv"
DEFAULT_OUTPUT_FILE = "ALL_COUNTRIES_2024_2025_WITH_ERA5.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the power panel with ERA5 weather data.")
    parser.add_argument("--power-file", default=DEFAULT_POWER_FILE)
    parser.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def require_columns(columns: pd.Index, required: list[str], frame_name: str) -> None:
    missing = [column for column in required if column not in columns]
    if missing:
        raise RuntimeError(f"{frame_name} is missing required columns: {missing}")


def validate_unique_keys(df: pd.DataFrame, frame_name: str) -> None:
    duplicated = df.duplicated(JOIN_KEYS)
    if duplicated.any():
        duplicate_rows = df.loc[duplicated, JOIN_KEYS].head(5).to_dict(orient="records")
        raise RuntimeError(f"{frame_name} has duplicate join keys. Sample duplicates: {duplicate_rows}")


def prepare_weather_frame(power_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(weather_df.columns, JOIN_KEYS + WEATHER_COLUMNS, "weather file")

    overlapping = sorted(set(power_df.columns).intersection(WEATHER_COLUMNS))
    if overlapping:
        raise RuntimeError(f"power file already contains weather columns: {overlapping}")

    return weather_df[JOIN_KEYS + WEATHER_COLUMNS].copy()


def validate_merged_output(merged: pd.DataFrame, power_df: pd.DataFrame) -> None:
    if len(merged) != len(power_df):
        raise RuntimeError(f"row count mismatch after merge: {len(merged)} != {len(power_df)}")

    if merged.duplicated(JOIN_KEYS).any():
        duplicate_rows = merged.loc[merged.duplicated(JOIN_KEYS), JOIN_KEYS].head(5).to_dict(orient="records")
        raise RuntimeError(f"merged output has duplicate join keys. Sample duplicates: {duplicate_rows}")

    missing_counts = merged[WEATHER_COLUMNS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise RuntimeError(f"merged output has missing weather values: {missing_counts.to_dict()}")


def merge_power_and_weather(power_path: Path, weather_path: Path, output_path: Path) -> pd.DataFrame:
    power_df = pd.read_csv(power_path, low_memory=False)
    weather_df = pd.read_csv(weather_path, low_memory=False)

    require_columns(power_df.columns, JOIN_KEYS, "power file")
    require_columns(weather_df.columns, JOIN_KEYS, "weather file")
    validate_unique_keys(power_df, "power file")
    validate_unique_keys(weather_df, "weather file")

    weather_selected = prepare_weather_frame(power_df, weather_df)
    merged = power_df.merge(weather_selected, on=JOIN_KEYS, how="left", sort=False)
    merged = merged[power_df.columns.tolist() + WEATHER_COLUMNS]
    validate_merged_output(merged, power_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    return merged


def main() -> int:
    args = parse_args()
    merge_power_and_weather(
        power_path=Path(args.power_file).resolve(),
        weather_path=Path(args.weather_file).resolve(),
        output_path=Path(args.output).resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
