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
ADVANCED_SEQUENCE_MODELS = (
    "PatchTST",
    "GRUHybrid",
    "GRUHybridAttn",
    "GRUHybridGated",
    "GRUHybridGatedMultiTask",
    "GRUMultiMarket",
    "GRUMultiMarketHybrid",
    "GRUMultiMarketTargetAttn",
    "GRUMultiMarketTemporalAttn",
    "GraphTemporal",
    "GraphTemporalHybrid",
)
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
    sequence_learning_rate: float = 1e-3
    sequence_max_epochs: int = 30
    sequence_patience: int = 5
    sequence_loss: str = "bce"
    focal_gamma: float = 2.0
    sequence_aux_target: str | None = None
    sequence_aux_weight: float = 0.2
    sample_filter_feature_group: str | None = None
    allow_window_missing: bool = False
    repeat_random_seeds: tuple[int, ...] = ()
    use_mechanism_features: bool = False
    use_mechanism_sequence_features: bool = False
    meta_kind: str | None = None
    meta_members: tuple[str, ...] = ()
    meta_calibration_method: str = "sigmoid"
    meta_stacker_model: str = "logistic_regression"
    meta_stacker_num_leaves: int | None = None
    meta_stacker_learning_rate: float | None = None
    meta_stacker_n_estimators: int | None = None
    meta_group_strategy: str | None = None
    meta_context_features: tuple[str, ...] = ()

    @property
    def numeric_features(self) -> tuple[str, ...]:
        if self.feature_group == "public":
            return PUBLIC_NUMERIC_FEATURES
        if self.feature_group == "renewables":
            return PUBLIC_NUMERIC_FEATURES + RENEWABLE_FEATURES
        if self.feature_group == "flows":
            return PUBLIC_NUMERIC_FEATURES + FLOW_FEATURES
        raise ValueError(f"Unsupported feature_group: {self.feature_group}")

    @property
    def sample_filter_numeric_features(self) -> tuple[str, ...]:
        feature_group = self.sample_filter_feature_group or self.feature_group
        if feature_group == "public":
            return PUBLIC_NUMERIC_FEATURES
        if feature_group == "renewables":
            return PUBLIC_NUMERIC_FEATURES + RENEWABLE_FEATURES
        if feature_group == "flows":
            return PUBLIC_NUMERIC_FEATURES + FLOW_FEATURES
        raise ValueError(f"Unsupported sample_filter_feature_group: {feature_group}")


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
        "E14": ExperimentConfig(
            name="E14",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            sequence_loss="focal",
            focal_gamma=2.0,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E15A": ExperimentConfig(
            name="E15A",
            countries=RENEWABLE_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E15B": ExperimentConfig(
            name="E15B",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E16A": ExperimentConfig(
            name="E16A",
            countries=FLOW_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E16B": ExperimentConfig(
            name="E16B",
            countries=FLOW_COUNTRIES,
            feature_group="flows",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E17A": ExperimentConfig(
            name="E17A",
            countries=RENEWABLE_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E17B": ExperimentConfig(
            name="E17B",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E18": ExperimentConfig(
            name="E18",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            random_seed=42,
            sample_filter_feature_group="renewables",
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E19": ExperimentConfig(
            name="E19",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E20": ExperimentConfig(
            name="E20",
            countries=MAIN_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRU",),
            allow_window_missing=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E21": ExperimentConfig(
            name="E21",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            random_seed=42,
            sample_filter_feature_group="renewables",
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E22A": ExperimentConfig(
            name="E22A",
            countries=RENEWABLE_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E22B": ExperimentConfig(
            name="E22B",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E23": ExperimentConfig(
            name="E23",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E24": ExperimentConfig(
            name="E24",
            countries=MAIN_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            allow_window_missing=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E25": ExperimentConfig(
            name="E25",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            random_seed=42,
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E26": ExperimentConfig(
            name="E26",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            sequence_loss="focal",
            focal_gamma=2.0,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E27": ExperimentConfig(
            name="E27",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            sequence_max_epochs=45,
            sequence_patience=8,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E28": ExperimentConfig(
            name="E28",
            countries=MAIN_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybrid",),
            allow_window_missing=True,
            sequence_max_epochs=45,
            sequence_patience=8,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E29": ExperimentConfig(
            name="E29",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridAttn",),
            sequence_max_epochs=60,
            sequence_patience=10,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E30": ExperimentConfig(
            name="E30",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_max_epochs=60,
            sequence_patience=10,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E31": ExperimentConfig(
            name="E31",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_max_epochs=60,
            sequence_patience=10,
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E32": ExperimentConfig(
            name="E32",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGatedMultiTask",),
            sequence_max_epochs=60,
            sequence_patience=10,
            sequence_aux_target="target_price",
            sequence_aux_weight=0.2,
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E33": ExperimentConfig(
            name="E33",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_max_epochs=60,
            sequence_patience=10,
            random_seed=42,
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E34": ExperimentConfig(
            name="E34",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_max_epochs=60,
            sequence_patience=10,
            use_mechanism_features=True,
            random_seed=42,
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E35": ExperimentConfig(
            name="E35",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E30", "E31"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E36": ExperimentConfig(
            name="E36",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="calibration",
            meta_members=("E30", "E31", "E35"),
            meta_calibration_method="sigmoid",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E37": ExperimentConfig(
            name="E37",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E38": ExperimentConfig(
            name="E38",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUHybridGated",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E39": ExperimentConfig(
            name="E39",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E30", "E31"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E40": ExperimentConfig(
            name="E40",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="cross_seed_ensemble",
            meta_members=("E33", "E34"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E41": ExperimentConfig(
            name="E41",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("LogisticRegression",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E42": ExperimentConfig(
            name="E42",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("XGBoost",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E43": ExperimentConfig(
            name="E43",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("CatBoost",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E44": ExperimentConfig(
            name="E44",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("LightGBM",),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E45": ExperimentConfig(
            name="E45",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E46": ExperimentConfig(
            name="E46",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GraphTemporal",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E47": ExperimentConfig(
            name="E47",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GraphTemporalHybrid",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E48": ExperimentConfig(
            name="E48",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GraphTemporalHybrid",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=80,
            sequence_patience=12,
            use_mechanism_features=True,
            random_seed=42,
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E49": ExperimentConfig(
            name="E49",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            random_seed=42,
            repeat_random_seeds=(42, 52, 62),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours", "random_seed"}},
        ),
        "E50": ExperimentConfig(
            name="E50",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarketHybrid",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=100,
            sequence_patience=15,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E51": ExperimentConfig(
            name="E51",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarketHybrid",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=100,
            sequence_patience=15,
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E52": ExperimentConfig(
            name="E52",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E45", "E35"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E53": ExperimentConfig(
            name="E53",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarketTargetAttn",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=100,
            sequence_patience=15,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E54": ExperimentConfig(
            name="E54",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarketTemporalAttn",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=100,
            sequence_patience=15,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E55": ExperimentConfig(
            name="E55",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            sequence_learning_rate=5e-4,
            sequence_max_epochs=100,
            sequence_patience=15,
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E56": ExperimentConfig(
            name="E56",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E44"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E57": ExperimentConfig(
            name="E57",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E42"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E58": ExperimentConfig(
            name="E58",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E42", "E44"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E59": ExperimentConfig(
            name="E59",
            countries=RENEWABLE_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E60": ExperimentConfig(
            name="E60",
            countries=RENEWABLE_COUNTRIES,
            feature_group="renewables",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            repeat_random_seeds=(42, 52, 62),
            sample_filter_feature_group="renewables",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E61": ExperimentConfig(
            name="E61",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E43"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E62": ExperimentConfig(
            name="E62",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E43", "E44"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E63": ExperimentConfig(
            name="E63",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E42", "E43", "E44"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E64": ExperimentConfig(
            name="E64",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E65": ExperimentConfig(
            name="E65",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E66": ExperimentConfig(
            name="E66",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_group_strategy="country",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E67": ExperimentConfig(
            name="E67",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_group_strategy="season_weekpart",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E68": ExperimentConfig(
            name="E68",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            meta_context_features=("country", "month", "hour", "is_weekend"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E69": ExperimentConfig(
            name="E69",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            meta_group_strategy="country",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E70": ExperimentConfig(
            name="E70",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            meta_group_strategy="season_weekpart",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E71": ExperimentConfig(
            name="E71",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E44"),
            meta_stacker_model="lightgbm",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E72": ExperimentConfig(
            name="E72",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E43", "E44"),
            meta_stacker_model="lightgbm",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E73": ExperimentConfig(
            name="E73",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43"),
            meta_stacker_model="lightgbm",
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E74": ExperimentConfig(
            name="E74",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            meta_stacker_num_leaves=7,
            meta_stacker_learning_rate=0.03,
            meta_stacker_n_estimators=400,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E75": ExperimentConfig(
            name="E75",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E43", "E44"),
            meta_stacker_model="lightgbm",
            meta_stacker_num_leaves=31,
            meta_stacker_learning_rate=0.03,
            meta_stacker_n_estimators=500,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E76": ExperimentConfig(
            name="E76",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="stacking",
            meta_members=("E49", "E42", "E44"),
            meta_stacker_model="lightgbm",
            meta_stacker_num_leaves=7,
            meta_stacker_learning_rate=0.03,
            meta_stacker_n_estimators=400,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E77": ExperimentConfig(
            name="E77",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("LightGBM",),
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E78": ExperimentConfig(
            name="E78",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("CatBoost",),
            use_mechanism_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E79": ExperimentConfig(
            name="E79",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=("GRUMultiMarket",),
            use_mechanism_sequence_features=True,
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E80": ExperimentConfig(
            name="E80",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="best_member_late_fusion",
            meta_members=("E75", "E77", "E78", "E79"),
            **{key: value for key, value in common.items() if key not in {"models", "window_hours"}},
        ),
        "E81": ExperimentConfig(
            name="E81",
            countries=MAIN_COUNTRIES,
            feature_group="public",
            window_hours=168,
            horizon_hours=6,
            models=(),
            meta_kind="late_fusion",
            meta_members=("E75", "E78"),
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
