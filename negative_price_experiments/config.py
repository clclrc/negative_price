from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PUBLIC_NUMERIC_FEATURES = (
    "price",
    "load",
    "temp_2m_c",
    "wind_speed_10m_ms",
    "shortwave_radiation_wm2",
    "cloud_cover_pct",
    "precipitation_mm",
    "pressure_msl_hpa",
)
RENEWABLE_FEATURES = ("Wind Onshore", "Solar")
FLOW_FEATURES = ("total export", "total import")
CALENDAR_CONTEXT_FEATURES = (
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    "is_weekend_local",
    "is_holiday_local",
)
ALL_REQUIRED_BASE_COLUMNS = (
    "time",
    "country",
    "time_zone",
    "is_weekend_local",
    "is_holiday_local",
    *PUBLIC_NUMERIC_FEATURES,
    *RENEWABLE_FEATURES,
    *FLOW_FEATURES,
)

MAIN_COUNTRIES = (
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE_LU",
    "DK_1",
    "DK_2",
    "EE",
    "FI",
    "LT",
    "LV",
    "NL",
    "NO_1",
    "NO_2",
    "RO",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
)
RENEWABLE_COUNTRIES = (
    "AT",
    "BE",
    "BG",
    "CH",
    "DE_LU",
    "DK_1",
    "DK_2",
    "EE",
    "FI",
    "LT",
    "LV",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
)
FLOW_COUNTRIES = ("AT", "BE", "BG", "CH", "CZ", "DE_LU", "RO")
TRANSFER_SOURCE_COUNTRIES = (
    "AT",
    "BE",
    "CH",
    "CZ",
    "DE_LU",
    "DK_1",
    "DK_2",
    "FI",
    "NL",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
)
TRANSFER_TARGET_COUNTRIES = ("BG", "EE", "LT", "LV", "RO")

CORE_TABULAR_MODELS = ("Majority", "LogisticRegression", "XGBoost")
ADVANCED_TABULAR_MODELS = ("LightGBM", "CatBoost", "XGBoostWeightedCalibrated")
TABULAR_MODELS = CORE_TABULAR_MODELS + ADVANCED_TABULAR_MODELS
CORE_SEQUENCE_MODELS = ("GRU", "TCN")
ADVANCED_SEQUENCE_MODELS = ("PatchTST",)
SEQUENCE_MODELS = CORE_SEQUENCE_MODELS + ADVANCED_SEQUENCE_MODELS
DEFAULT_MODELS = CORE_TABULAR_MODELS + CORE_SEQUENCE_MODELS


def utc_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class TimeRange:
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("TimeRange end must be greater than start.")

    def contains(self, values: pd.Series) -> pd.Series:
        normalized = pd.to_datetime(values, utc=True)
        return (normalized >= self.start) & (normalized < self.end)

    def last_anchor_time(self, horizon_hours: int) -> pd.Timestamp:
        return self.end - pd.Timedelta(hours=horizon_hours)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_range: TimeRange
    val_range: TimeRange


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_path: Path
    countries: tuple[str, ...]
    feature_group: str
    window_hours: int
    horizon_hours: int
    models: tuple[str, ...]
    split_strategy: str
    ffill_limit: int
    primary_metric: str
    random_seed: int
    use_country_features: bool = True

    @property
    def numeric_features(self) -> tuple[str, ...]:
        if self.feature_group == "public":
            return PUBLIC_NUMERIC_FEATURES
        if self.feature_group == "renewables":
            return PUBLIC_NUMERIC_FEATURES + RENEWABLE_FEATURES
        if self.feature_group == "flows":
            return PUBLIC_NUMERIC_FEATURES + FLOW_FEATURES
        raise ValueError(f"Unsupported feature_group: {self.feature_group}")


@dataclass(frozen=True)
class AdaptBudget:
    name: str
    train_range: TimeRange
    val_range: TimeRange


@dataclass(frozen=True)
class TransferConfig:
    name: str
    data_path: Path
    source_countries: tuple[str, ...]
    target_countries: tuple[str, ...]
    adapt_budget: tuple[AdaptBudget, ...]
    pretrain_train_range: TimeRange
    pretrain_val_range: TimeRange
    target_test_range: TimeRange
    window_hours: int
    horizon_hours: int
    ffill_limit: int
    primary_metric: str
    random_seed: int

    @property
    def all_countries(self) -> tuple[str, ...]:
        return self.source_countries + self.target_countries


WALK_FORWARD_FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        name="F1",
        train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-07-01 00:00:00")),
        val_range=TimeRange(utc_ts("2024-07-01 00:00:00"), utc_ts("2024-10-01 00:00:00")),
    ),
    WalkForwardFold(
        name="F2",
        train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2024-10-01 00:00:00")),
        val_range=TimeRange(utc_ts("2024-10-01 00:00:00"), utc_ts("2025-01-01 00:00:00")),
    ),
    WalkForwardFold(
        name="F3",
        train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2025-01-01 00:00:00")),
        val_range=TimeRange(utc_ts("2025-01-01 00:00:00"), utc_ts("2025-04-01 00:00:00")),
    ),
    WalkForwardFold(
        name="F4",
        train_range=TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2025-04-01 00:00:00")),
        val_range=TimeRange(utc_ts("2025-04-01 00:00:00"), utc_ts("2025-07-01 00:00:00")),
    ),
)

FINAL_TRAIN_RANGE = TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2025-07-01 00:00:00"))
FINAL_TEST_RANGE = TimeRange(utc_ts("2025-07-01 00:00:00"), utc_ts("2026-01-01 00:00:00"))

TRANSFER_PRETRAIN_TRAIN_RANGE = TimeRange(utc_ts("2024-01-01 00:00:00"), utc_ts("2025-04-01 00:00:00"))
TRANSFER_PRETRAIN_VAL_RANGE = TimeRange(utc_ts("2025-04-01 00:00:00"), utc_ts("2025-07-01 00:00:00"))
TRANSFER_TEST_RANGE = TimeRange(utc_ts("2025-07-01 00:00:00"), utc_ts("2026-01-01 00:00:00"))
TRANSFER_BUDGETS = (
    AdaptBudget(
        name="B90",
        train_range=TimeRange(utc_ts("2025-04-01 00:00:00"), utc_ts("2025-06-01 00:00:00")),
        val_range=TimeRange(utc_ts("2025-06-01 00:00:00"), utc_ts("2025-07-01 00:00:00")),
    ),
    AdaptBudget(
        name="B180",
        train_range=TimeRange(utc_ts("2025-01-01 00:00:00"), utc_ts("2025-05-01 00:00:00")),
        val_range=TimeRange(utc_ts("2025-05-01 00:00:00"), utc_ts("2025-07-01 00:00:00")),
    ),
)


def build_default_experiment_configs(data_path: str | Path) -> dict[str, ExperimentConfig]:
    resolved = Path(data_path).resolve()
    common = {
        "data_path": resolved,
        "window_hours": 72,
        "models": DEFAULT_MODELS,
        "split_strategy": "expanding_walk_forward_v1",
        "ffill_limit": 3,
        "primary_metric": "pr_auc",
        "random_seed": 42,
        "use_country_features": True,
    }
    return {
        "E1": ExperimentConfig(
            name="E1",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=6,
            **common,
        ),
        "E2": ExperimentConfig(
            name="E2",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=24,
            **common,
        ),
        "E3": ExperimentConfig(
            name="E3",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            horizon_hours=6,
            **common,
        ),
        "E4": ExperimentConfig(
            name="E4",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            horizon_hours=24,
            **common,
        ),
        "E5": ExperimentConfig(
            name="E5",
            countries=FLOW_COUNTRIES,
            feature_group="flows",
            horizon_hours=6,
            **common,
        ),
        "E7": ExperimentConfig(
            name="E7",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=6,
            models=("LightGBM",),
            **{key: value for key, value in common.items() if key != "models"},
        ),
        "E8": ExperimentConfig(
            name="E8",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=6,
            models=("CatBoost",),
            **{key: value for key, value in common.items() if key != "models"},
        ),
        "E9": ExperimentConfig(
            name="E9",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=6,
            models=("XGBoostWeightedCalibrated",),
            **{key: value for key, value in common.items() if key != "models"},
        ),
        "E10": ExperimentConfig(
            name="E10",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            horizon_hours=6,
            models=("PatchTST",),
            **{key: value for key, value in common.items() if key != "models"},
        ),
        "E11": ExperimentConfig(
            name="E11",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=120,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E12": ExperimentConfig(
            name="E12",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E13": ExperimentConfig(
            name="E13",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("TCN",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
    }


def build_default_transfer_config(data_path: str | Path) -> TransferConfig:
    return TransferConfig(
        name="E6",
        data_path=Path(data_path).resolve(),
        source_countries=TRANSFER_SOURCE_COUNTRIES,
        target_countries=TRANSFER_TARGET_COUNTRIES,
        adapt_budget=TRANSFER_BUDGETS,
        pretrain_train_range=TRANSFER_PRETRAIN_TRAIN_RANGE,
        pretrain_val_range=TRANSFER_PRETRAIN_VAL_RANGE,
        target_test_range=TRANSFER_TEST_RANGE,
        window_hours=72,
        horizon_hours=6,
        ffill_limit=3,
        primary_metric="pr_auc",
        random_seed=42,
    )
