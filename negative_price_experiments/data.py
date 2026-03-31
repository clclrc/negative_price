from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ALL_REQUIRED_BASE_COLUMNS, CALENDAR_CONTEXT_FEATURES, ExperimentConfig


MASK_SUFFIX = "__missing"
TABULAR_LAG_STEPS = (0, 1, 6, 24)
TABULAR_WINDOW_STEPS = (6, 24, 72)


@dataclass(frozen=True)
class NumericScaler:
    feature_names: tuple[str, ...]
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, feature_names: tuple[str, ...]) -> "NumericScaler":
        mean_ = values.mean(axis=0)
        scale_ = values.std(axis=0)
        scale_[scale_ == 0] = 1.0
        return cls(feature_names=feature_names, mean_=mean_.astype(np.float32), scale_=scale_.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean_) / self.scale_).astype(np.float32)


@dataclass(frozen=True)
class TabularBundle:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    continuous_indices: tuple[int, ...]


class TabularScaler:
    def __init__(self, continuous_indices: tuple[int, ...]) -> None:
        self.continuous_indices = continuous_indices
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "TabularScaler":
        if not self.continuous_indices:
            self.mean_ = np.empty(0, dtype=np.float32)
            self.scale_ = np.empty(0, dtype=np.float32)
            return self
        subset = X[:, self.continuous_indices]
        self.mean_ = subset.mean(axis=0).astype(np.float32)
        self.scale_ = subset.std(axis=0).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.astype(np.float32, copy=True)
        if self.mean_ is None or self.scale_ is None or not self.continuous_indices:
            return out
        out[:, self.continuous_indices] = (out[:, self.continuous_indices] - self.mean_) / self.scale_
        return out


@dataclass(frozen=True)
class CountryPanel:
    country: str
    times: np.ndarray
    numeric_values: np.ndarray
    original_price: np.ndarray
    missing_masks: np.ndarray
    context_values: np.ndarray


class SequenceDataset:
    def __init__(
        self,
        prepared: "PreparedExperimentData",
        samples: pd.DataFrame,
        scaler: NumericScaler,
        *,
        include_country: bool,
    ) -> None:
        self.prepared = prepared
        self.samples = samples.reset_index(drop=True).copy()
        self.scaler = scaler
        self.include_country = include_country

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.float32 | np.int64]:
        row = self.samples.iloc[index]
        panel = self.prepared.country_panels[row["country"]]
        anchor_idx = int(row["anchor_index"])
        window_slice = slice(anchor_idx - self.prepared.config.window_hours + 1, anchor_idx + 1)
        numeric = panel.numeric_values[window_slice]
        x_numeric = self.scaler.transform(numeric)
        x_mask = panel.missing_masks[window_slice].astype(np.float32)
        x_context = panel.context_values[window_slice].astype(np.float32)
        x = np.concatenate([x_numeric, x_mask, x_context], axis=1).astype(np.float32)
        country_idx = self.prepared.country_to_index.get(row["country"], -1) if self.include_country else -1
        return {
            "x": x,
            "country_idx": np.int64(country_idx),
            "y": np.float32(row["y_true"]),
        }

    @property
    def metadata(self) -> pd.DataFrame:
        return self.samples.copy()


class PreparedExperimentData:
    def __init__(self, config: ExperimentConfig, country_panels: dict[str, CountryPanel], sample_manifest: pd.DataFrame) -> None:
        self.config = config
        self.country_panels = country_panels
        self.sample_manifest = sample_manifest.reset_index(drop=True)
        self.country_to_index = {country: idx for idx, country in enumerate(config.countries)}
        self.numeric_features = config.numeric_features
        self.mask_feature_names = tuple(f"{feature}{MASK_SUFFIX}" for feature in self.numeric_features)
        self.context_feature_names = CALENDAR_CONTEXT_FEATURES
        self.sequence_feature_names = self.numeric_features + self.mask_feature_names + self.context_feature_names

    def select_samples(self, time_range, *, countries: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
        mask = time_range.contains(self.sample_manifest["target_time"])
        if countries is not None:
            mask &= self.sample_manifest["country"].isin(list(countries))
        return self.sample_manifest.loc[mask].copy().reset_index(drop=True)

    def fit_sequence_scaler(self, train_samples: pd.DataFrame) -> NumericScaler:
        if train_samples.empty:
            raise ValueError("Cannot fit sequence scaler on empty sample set.")
        arrays: list[np.ndarray] = []
        for country, group in train_samples.groupby("country", sort=False):
            panel = self.country_panels[country]
            max_anchor = int(group["anchor_index"].max())
            arrays.append(panel.numeric_values[: max_anchor + 1])
        values = np.concatenate(arrays, axis=0)
        return NumericScaler.fit(values, self.numeric_features)

    def build_sequence_dataset(
        self,
        samples: pd.DataFrame,
        scaler: NumericScaler,
        *,
        include_country: bool,
    ) -> SequenceDataset:
        return SequenceDataset(self, samples, scaler, include_country=include_country)

    def build_tabular_bundle(self, samples: pd.DataFrame, *, include_country: bool) -> TabularBundle:
        feature_names, continuous_indices = self._tabular_feature_spec(include_country=include_country)
        X = np.zeros((len(samples), len(feature_names)), dtype=np.float32)
        y = samples["y_true"].to_numpy(dtype=np.int8)

        for row_idx, row in enumerate(samples.itertuples(index=False)):
            panel = self.country_panels[row.country]
            anchor_idx = int(row.anchor_index)
            window_slice = slice(anchor_idx - self.config.window_hours + 1, anchor_idx + 1)
            numeric_window = panel.numeric_values[window_slice]
            mask_window = panel.missing_masks[window_slice]
            context_last = panel.context_values[anchor_idx]

            values: list[float] = []
            for feature_idx in range(len(self.numeric_features)):
                series = numeric_window[:, feature_idx]
                for lag in TABULAR_LAG_STEPS:
                    lag_index = max(len(series) - 1 - lag, 0)
                    values.append(float(series[lag_index]))
                for window in TABULAR_WINDOW_STEPS:
                    start = max(len(series) - window, 0)
                    slice_values = series[start:]
                    values.extend(
                        [
                            float(slice_values.mean()),
                            float(slice_values.std(ddof=0)),
                            float(slice_values.min()),
                            float(slice_values.max()),
                        ]
                    )

            for feature_idx in range(len(self.numeric_features)):
                series = mask_window[:, feature_idx]
                for window in TABULAR_WINDOW_STEPS:
                    start = max(len(series) - window, 0)
                    values.append(float(series[start:].mean()))

            values.extend(float(item) for item in context_last.tolist())
            if include_country:
                for country in self.config.countries:
                    values.append(1.0 if country == row.country else 0.0)
            X[row_idx] = np.asarray(values, dtype=np.float32)

        return TabularBundle(
            X=X,
            y=y,
            metadata=samples.reset_index(drop=True).copy(),
            feature_names=feature_names,
            continuous_indices=continuous_indices,
        )

    def _tabular_feature_spec(self, *, include_country: bool) -> tuple[tuple[str, ...], tuple[int, ...]]:
        feature_names: list[str] = []
        continuous_indices: list[int] = []

        for feature in self.numeric_features:
            for lag in TABULAR_LAG_STEPS:
                continuous_indices.append(len(feature_names))
                feature_names.append(f"{feature}__lag_{lag}")
            for window in TABULAR_WINDOW_STEPS:
                for stat in ("mean", "std", "min", "max"):
                    continuous_indices.append(len(feature_names))
                    feature_names.append(f"{feature}__{stat}_{window}")

        for feature in self.numeric_features:
            mask_name = f"{feature}{MASK_SUFFIX}"
            for window in TABULAR_WINDOW_STEPS:
                feature_names.append(f"{mask_name}__mean_{window}")

        feature_names.extend(self.context_feature_names)
        if include_country:
            feature_names.extend(f"country__{country}" for country in self.config.countries)
        return tuple(feature_names), tuple(continuous_indices)


def prepare_experiment_data(config: ExperimentConfig) -> PreparedExperimentData:
    selected_columns = ["time", "country", "time_zone", *config.numeric_features, "is_weekend_local", "is_holiday_local"]
    df = pd.read_csv(config.data_path, usecols=lambda column: column in set(ALL_REQUIRED_BASE_COLUMNS).union(selected_columns))
    missing_required = sorted(set(selected_columns) - set(df.columns))
    if missing_required:
        raise KeyError(f"Dataset is missing required columns: {missing_required}")

    df = df[df["country"].isin(config.countries)].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering selected countries.")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.sort_values(["country", "time"], inplace=True)
    duplicated = df.duplicated(["country", "time"])
    if duplicated.any():
        example = df.loc[duplicated, ["country", "time"]].head(5).to_dict(orient="records")
        raise RuntimeError(f"Duplicate country/time rows found. Sample: {example}")

    _validate_hourly_panel(df)
    _add_time_context(df)

    df["_original_price"] = df["price"].astype(float)
    for feature in config.numeric_features:
        mask_col = f"{feature}{MASK_SUFFIX}"
        df[mask_col] = df[feature].isna().astype(np.float32)
    df[list(config.numeric_features)] = (
        df.groupby("country", sort=False)[list(config.numeric_features)].transform(lambda group: group.ffill(limit=config.ffill_limit))
    )

    country_panels: dict[str, CountryPanel] = {}
    manifest_frames: list[pd.DataFrame] = []
    for country, group in df.groupby("country", sort=False):
        group = group.reset_index(drop=True)
        panel = CountryPanel(
            country=country,
            times=group["time"].to_numpy(),
            numeric_values=group[list(config.numeric_features)].to_numpy(dtype=np.float32),
            original_price=group["_original_price"].to_numpy(dtype=np.float32),
            missing_masks=group[[f"{feature}{MASK_SUFFIX}" for feature in config.numeric_features]].to_numpy(dtype=np.float32),
            context_values=group[list(CALENDAR_CONTEXT_FEATURES)].to_numpy(dtype=np.float32),
        )
        country_panels[country] = panel
        manifest_frames.append(_build_sample_manifest_for_country(config, panel))

    sample_manifest = pd.concat(manifest_frames, ignore_index=True)
    sample_manifest.sort_values(["country", "target_time"], inplace=True)
    sample_manifest.reset_index(drop=True, inplace=True)
    return PreparedExperimentData(config, country_panels, sample_manifest)


def _validate_hourly_panel(df: pd.DataFrame) -> None:
    for country, group in df.groupby("country", sort=False):
        diffs = group["time"].diff().dropna()
        if not diffs.eq(pd.Timedelta(hours=1)).all():
            raise RuntimeError(f"Country {country} does not form an hourly panel.")


def _add_time_context(df: pd.DataFrame) -> None:
    hour = df["time"].dt.hour.to_numpy(dtype=np.float32)
    month = df["time"].dt.month.to_numpy(dtype=np.float32)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12).astype(np.float32)
    df["is_weekend_local"] = df["is_weekend_local"].astype(np.float32)
    df["is_holiday_local"] = df["is_holiday_local"].astype(np.float32)


def _build_sample_manifest_for_country(config: ExperimentConfig, panel: CountryPanel) -> pd.DataFrame:
    row_has_missing = np.isnan(panel.numeric_values).any(axis=1).astype(np.int16)
    rolling_missing = np.cumsum(row_has_missing)
    counts = rolling_missing.copy()
    window = config.window_hours
    if window < len(counts):
        counts[window:] = rolling_missing[window:] - rolling_missing[:-window]
    counts[: window - 1] = 1

    total_rows = len(panel.times)
    anchors = np.arange(window - 1, total_rows - config.horizon_hours, dtype=np.int32)
    target_indices = anchors + config.horizon_hours
    valid_window = counts[anchors] == 0
    valid_target = ~np.isnan(panel.original_price[target_indices])
    keep = valid_window & valid_target
    anchors = anchors[keep]
    target_indices = target_indices[keep]
    labels = (panel.original_price[target_indices] < 0).astype(np.int8)

    return pd.DataFrame(
        {
            "country": panel.country,
            "anchor_index": anchors,
            "target_index": target_indices,
            "anchor_time": panel.times[anchors],
            "target_time": panel.times[target_indices],
            "y_true": labels,
            "horizon_hours": config.horizon_hours,
        }
    )
