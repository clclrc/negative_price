#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Point, shape
from shapely.ops import unary_union

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback only
    cKDTree = None


TZ = "UTC"
TIME_FMT = "%Y-%m-%d %H:%M:%S"
OUTPUT_OPENMETEO = "WEATHER_OPENMETEO_2024_2025.csv"
OUTPUT_ERA5 = "WEATHER_ERA5_2024_2025.csv"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ENTSOE_GEO_BASE = "https://raw.githubusercontent.com/EnergieID/entsoe-py/V0.6.18/entsoe/geo/geojson"
ELECTRICITYMAPS_WORLD_URL = (
    "https://raw.githubusercontent.com/electricitymaps/electricitymaps-contrib/"
    "refs/tags/v1.238.0/web/geo/world.geojson"
)
ERA5_REANALYSIS_URL = "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2"
ERA5_FORECAST_URL = "gs://gcp-public-data-arco-era5/co/single-level-forecast.zarr-v2/"
VARIABLE_COLUMNS = [
    "temp_2m_c",
    "wind_speed_10m_ms",
    "shortwave_radiation_wm2",
    "cloud_cover_pct",
    "precipitation_mm",
    "pressure_msl_hpa",
]
FINAL_COLUMNS = [
    "time",
    "country",
    "time_zone",
    "weather_source",
    *VARIABLE_COLUMNS,
    "point_count",
    "has_missing",
    "has_interpolated",
]
OPENMETEO_HOURLY = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "pressure_msl",
]
ZONE_ALIASES = {
    "IE_SEM": ["CL_SEM"],
}


@dataclass(frozen=True)
class ZonePoint:
    country: str
    point_id: str
    point_rank: int
    latitude: float
    longitude: float
    weight: float
    geometry_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Open-Meteo and ERA5 weather CSVs for 2024-2025.")
    parser.add_argument("--power-file", required=True)
    parser.add_argument("--source", choices=["openmeteo", "era5", "all"], default="all")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cache-dir", default="weather_cache")
    parser.add_argument(
        "--zones",
        default="",
        help="Optional comma-separated subset of market zones for smoke runs.",
    )
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def normalize_zone_code(value: str) -> str:
    code = str(value).strip().upper().replace("-", "_").replace(" ", "")
    match = re.fullmatch(r"([A-Z]+)(\d)", code)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return code


def request_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict[str, object]] = None,
    timeout: int = 120,
    max_retries: int = 4,
) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    sleep_seconds = max(float(retry_after), 20.0) if retry_after else 20.0 * (attempt + 1)
                except ValueError:
                    sleep_seconds = 20.0 * (attempt + 1)
                time.sleep(sleep_seconds)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def load_power_keys(power_file: Path, zone_filter: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_df = pd.read_csv(power_file, usecols=["time", "country", "time_zone"])
    key_df["country"] = key_df["country"].map(normalize_zone_code)
    if zone_filter:
        key_df = key_df[key_df["country"].isin(zone_filter)].copy()
    if key_df.empty:
        raise RuntimeError("No rows left after applying zone filter.")

    key_df["_row_id"] = np.arange(len(key_df))
    key_df["time"] = pd.to_datetime(key_df["time"])
    if key_df["time_zone"].nunique() != 1 or key_df["time_zone"].iloc[0] != TZ:
        raise RuntimeError(f"Expected a single time_zone '{TZ}', found {sorted(key_df['time_zone'].unique())}")

    canonical = (
        key_df[["time", "country", "time_zone"]]
        .drop_duplicates()
        .sort_values(["country", "time"])
        .reset_index(drop=True)
    )
    return key_df.reset_index(drop=True), canonical


def expected_hour_index(canonical: pd.DataFrame) -> pd.DatetimeIndex:
    start = canonical["time"].min()
    end = canonical["time"].max()
    expected = pd.date_range(start=start, end=end, freq="1h")
    if len(expected) != canonical["time"].nunique() // canonical["country"].nunique():
        # The data should be a complete hour-by-hour panel.
        pass
    return expected


def load_zone_geometry(
    zone: str,
    session: requests.Session,
    geo_cache_dir: Path,
    max_retries: int,
) -> tuple[object, str]:
    aliases = [zone, *ZONE_ALIASES.get(zone, [])]
    entsoe_dir = geo_cache_dir / "entsoepy"
    entsoe_dir.mkdir(parents=True, exist_ok=True)

    for alias in aliases:
        entsoe_path = entsoe_dir / f"{alias}.geojson"
        if not entsoe_path.exists():
            url = f"{ENTSOE_GEO_BASE}/{alias}.geojson"
            response = session.get(url, timeout=60)
            if response.ok:
                entsoe_path.write_text(response.text, encoding="utf-8")

        if entsoe_path.exists():
            data = json.loads(entsoe_path.read_text(encoding="utf-8"))
            geometry = unary_union([shape(feature["geometry"]) for feature in data.get("features", [])])
            if geometry.is_empty:
                raise RuntimeError(f"Empty geometry in {entsoe_path}")
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            return geometry, "bidding_zones_entsoepy"

    world_path = geo_cache_dir / "electricitymaps_world.geojson"
    if not world_path.exists():
        response = request_with_retry(
            session,
            ELECTRICITYMAPS_WORLD_URL,
            timeout=120,
            max_retries=max_retries,
        )
        world_path.write_text(response.text, encoding="utf-8")

    data = json.loads(world_path.read_text(encoding="utf-8"))
    matches = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        candidates = {
            normalize_zone_code(props.get(key, ""))
            for key in ("zoneName", "zoneId", "countryCode", "name", "id")
            if isinstance(props.get(key), str)
        }
        if any(alias in candidates for alias in aliases):
            matches.append(shape(feature["geometry"]))

    if not matches:
        raise RuntimeError(f"Could not find geometry for zone {zone} in entsoepy or electricitymaps fallback.")

    geometry = unary_union(matches)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry, "bidding_zones_electricitymaps"


def generate_candidate_points(geometry: object, target_count: int) -> list[tuple[float, float]]:
    representative = geometry.representative_point()
    minx, miny, maxx, maxy = geometry.bounds
    candidates: list[tuple[float, float]] = [(representative.x, representative.y)]
    seen = {(round(representative.x, 6), round(representative.y, 6))}

    for step in (0.5, 0.25, 0.1):
        xs = np.arange(math.floor(minx / step) * step, math.ceil(maxx / step) * step + step * 0.5, step)
        ys = np.arange(math.floor(miny / step) * step, math.ceil(maxy / step) * step + step * 0.5, step)
        for y in ys:
            for x in xs:
                key = (round(float(x), 6), round(float(y), 6))
                if key in seen:
                    continue
                point = Point(float(x), float(y))
                if geometry.covers(point):
                    candidates.append((float(x), float(y)))
                    seen.add(key)
        if len(candidates) >= max(target_count * 4, target_count + 4):
            break
    return candidates


def select_representative_points(geometry: object) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = geometry.bounds
    width = maxx - minx
    height = maxy - miny
    short_side = max(min(width, height), 1e-6)
    aspect_ratio = max(width, height) / short_side
    target_count = 5 if geometry.area > 12 or aspect_ratio > 1.8 else 3

    candidates = generate_candidate_points(geometry, target_count)
    coords = np.asarray(candidates, dtype=float)
    if len(coords) <= target_count:
        return [(float(x), float(y)) for x, y in coords]

    selected = [0]
    while len(selected) < target_count:
        remaining = [idx for idx in range(len(coords)) if idx not in selected]
        selected_coords = coords[np.asarray(selected)]
        remaining_coords = coords[np.asarray(remaining)]
        diff = remaining_coords[:, None, :] - selected_coords[None, :, :]
        min_dist_sq = np.min(np.sum(diff * diff, axis=2), axis=1)
        next_idx = remaining[int(np.argmax(min_dist_sq))]
        selected.append(next_idx)

    return [(float(coords[idx, 0]), float(coords[idx, 1])) for idx in selected]


def build_zone_points(
    zones: list[str],
    session: requests.Session,
    cache_dir: Path,
    max_retries: int,
) -> pd.DataFrame:
    zone_points_path = cache_dir / "zone_points.csv"
    if zone_points_path.exists():
        cached = pd.read_csv(zone_points_path)
        cached["country"] = cached["country"].map(normalize_zone_code)
        cached = cached[cached["country"].isin(zones)].copy()
        if set(cached["country"].unique()) == set(zones):
            return cached.sort_values(["country", "point_rank"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    geo_cache_dir = cache_dir / "geo"
    for zone in zones:
        geometry, geometry_source = load_zone_geometry(zone, session, geo_cache_dir, max_retries)
        points = select_representative_points(geometry)
        weight = 1.0 / len(points)
        for rank, (lon, lat) in enumerate(points, start=1):
            rows.append(
                {
                    "country": zone,
                    "point_id": f"{zone}_P{rank}",
                    "point_rank": rank,
                    "latitude": lat,
                    "longitude": lon,
                    "weight": weight,
                    "geometry_source": geometry_source,
                }
            )

    zone_points = pd.DataFrame(rows).sort_values(["country", "point_rank"]).reset_index(drop=True)
    zone_points_path.parent.mkdir(parents=True, exist_ok=True)
    zone_points.to_csv(zone_points_path, index=False)
    return zone_points


def new_zone_frames(canonical: pd.DataFrame) -> dict[str, pd.DataFrame]:
    zone_frames: dict[str, pd.DataFrame] = {}
    for country, group in canonical.groupby("country", sort=True):
        frame = group[["time", "country", "time_zone"]].copy().set_index("time")
        for column in VARIABLE_COLUMNS:
            frame[column] = np.nan
            frame[f"__sum_{column}"] = 0.0
            frame[f"__weight_{column}"] = 0.0
        zone_frames[country] = frame
    return zone_frames


def load_cached_point_frame(cache_file: Path, expected_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    if not cache_file.exists():
        return None
    df = pd.read_csv(cache_file)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"])
    if len(df) != len(expected_index):
        return None
    if not df["time"].equals(pd.Series(expected_index, name="time")):
        return None
    return df


def save_point_frame(cache_file: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out["time"] = out["time"].dt.strftime(TIME_FMT)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_file, index=False)


def clip_weather_ranges(df: pd.DataFrame) -> pd.DataFrame:
    clipped = df.copy()
    if "wind_speed_10m_ms" in clipped:
        clipped["wind_speed_10m_ms"] = clipped["wind_speed_10m_ms"].clip(lower=0)
    if "shortwave_radiation_wm2" in clipped:
        clipped["shortwave_radiation_wm2"] = clipped["shortwave_radiation_wm2"].clip(lower=0)
    if "cloud_cover_pct" in clipped:
        clipped["cloud_cover_pct"] = clipped["cloud_cover_pct"].clip(lower=0, upper=100)
    if "precipitation_mm" in clipped:
        clipped["precipitation_mm"] = clipped["precipitation_mm"].clip(lower=0)
    return clipped


def fetch_openmeteo_point(
    point: ZonePoint,
    session: requests.Session,
    cache_dir: Path,
    expected_index: pd.DatetimeIndex,
    max_retries: int,
    *,
    model: Optional[str] = None,
) -> pd.DataFrame:
    cache_file = cache_dir / f"{point.point_id}.csv"
    cached = load_cached_point_frame(cache_file, expected_index)
    if cached is not None:
        return cached

    params = {
        "latitude": f"{point.latitude:.5f}",
        "longitude": f"{point.longitude:.5f}",
        "start_date": expected_index[0].strftime("%Y-%m-%d"),
        "end_date": expected_index[-1].strftime("%Y-%m-%d"),
        "hourly": ",".join(OPENMETEO_HOURLY),
        "timezone": TZ,
    }
    if model:
        params["models"] = model
    response = request_with_retry(
        session,
        OPENMETEO_URL,
        params=params,
        timeout=120,
        max_retries=max_retries,
    )
    payload = response.json()
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError(f"Malformed Open-Meteo payload for point {point.point_id}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "temperature_2m": "temp_2m_c",
            "wind_speed_10m": "wind_speed_10m_kmh",
            "shortwave_radiation": "shortwave_radiation_wm2",
            "cloud_cover": "cloud_cover_pct",
            "precipitation": "precipitation_mm",
            "pressure_msl": "pressure_msl_hpa",
        }
    )
    df["wind_speed_10m_ms"] = df["wind_speed_10m_kmh"] / 3.6
    df = df[["time", "temp_2m_c", "wind_speed_10m_ms", "shortwave_radiation_wm2", "cloud_cover_pct", "precipitation_mm", "pressure_msl_hpa"]]
    df = df.set_index("time").reindex(expected_index).reset_index(names="time")
    df = clip_weather_ranges(df)
    save_point_frame(cache_file, df)
    time.sleep(3.0 if model == "era5" else 1.0)
    return df


def aggregate_point_into_zone(zone_frame: pd.DataFrame, point_df: pd.DataFrame, weight: float) -> None:
    aligned = point_df.set_index("time").reindex(zone_frame.index)
    for column in VARIABLE_COLUMNS:
        values = aligned[column].to_numpy(dtype=float, na_value=np.nan)
        mask = np.isfinite(values)
        zone_frame[f"__sum_{column}"] += np.where(mask, values * weight, 0.0)
        zone_frame[f"__weight_{column}"] += np.where(mask, weight, 0.0)


def finalize_zone_frame(zone_frame: pd.DataFrame, point_count: int, weather_source: str) -> pd.DataFrame:
    out = zone_frame.copy()
    for column in VARIABLE_COLUMNS:
        weight_col = f"__weight_{column}"
        sum_col = f"__sum_{column}"
        with np.errstate(divide="ignore", invalid="ignore"):
            out[column] = out[sum_col] / out[weight_col]
        out.loc[out[weight_col] == 0, column] = np.nan
        out = out.drop(columns=[sum_col, weight_col])

    out[VARIABLE_COLUMNS] = clip_weather_ranges(out[VARIABLE_COLUMNS])
    before = out[VARIABLE_COLUMNS].copy()
    interpolated = out[VARIABLE_COLUMNS].interpolate(
        method="linear",
        limit=3,
        limit_direction="both",
        limit_area="inside",
    )
    filled_mask = before.isna() & interpolated.notna()
    out[VARIABLE_COLUMNS] = clip_weather_ranges(interpolated)
    out["point_count"] = int(point_count)
    out["has_interpolated"] = filled_mask.any(axis=1)
    out["has_missing"] = out[VARIABLE_COLUMNS].isna().any(axis=1)
    out["weather_source"] = weather_source
    return out.reset_index(names="time")


def assemble_final_output(key_df: pd.DataFrame, zone_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weather_df = pd.concat(zone_frames.values(), ignore_index=True)
    weather_df = weather_df[FINAL_COLUMNS]
    merged = key_df.merge(weather_df, on=["time", "country", "time_zone"], how="left", sort=False)
    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
    return merged


def validate_final_frame(name: str, final_df: pd.DataFrame, key_df: pd.DataFrame) -> None:
    if len(final_df) != len(key_df):
        raise RuntimeError(f"{name}: row count mismatch {len(final_df)} != {len(key_df)}")
    if final_df.duplicated(subset=["time", "country", "time_zone"]).any() and not key_df.duplicated(
        subset=["time", "country", "time_zone"]
    ).any():
        raise RuntimeError(f"{name}: unexpected duplicate weather keys.")

    for column in VARIABLE_COLUMNS:
        if column not in final_df.columns:
            raise RuntimeError(f"{name}: missing expected column {column}")

    if (final_df["cloud_cover_pct"].dropna().between(0, 100) == False).any():
        raise RuntimeError(f"{name}: cloud_cover_pct outside 0-100.")
    for column in ("precipitation_mm", "shortwave_radiation_wm2", "wind_speed_10m_ms"):
        if (final_df[column].dropna() < 0).any():
            raise RuntimeError(f"{name}: negative values in {column}.")


def build_openmeteo_family_output(
    key_df: pd.DataFrame,
    canonical: pd.DataFrame,
    zone_points_df: pd.DataFrame,
    session: requests.Session,
    cache_dir: Path,
    max_retries: int,
    *,
    weather_source: str,
    model: Optional[str],
) -> pd.DataFrame:
    expected_index = expected_hour_index(canonical)
    zone_frames = new_zone_frames(canonical)
    point_cache_dir = cache_dir / weather_source

    point_count_map = zone_points_df.groupby("country")["point_id"].count().to_dict()
    zone_points = [
        ZonePoint(**row)
        for row in zone_points_df[
            ["country", "point_id", "point_rank", "latitude", "longitude", "weight", "geometry_source"]
        ].to_dict(orient="records")
    ]

    for idx, point in enumerate(zone_points, start=1):
        print(f"[{weather_source}] {idx}/{len(zone_points)} {point.point_id}", flush=True)
        point_df = fetch_openmeteo_point(
            point,
            session,
            point_cache_dir,
            expected_index,
            max_retries,
            model=model,
        )
        aggregate_point_into_zone(zone_frames[point.country], point_df, point.weight)

    final_zone_frames: dict[str, pd.DataFrame] = {}
    for country, zone_frame in zone_frames.items():
        final_zone_frames[country] = finalize_zone_frame(zone_frame, point_count_map[country], weather_source)

    final_df = assemble_final_output(key_df, final_zone_frames)
    validate_final_frame(weather_source, final_df, key_df)
    return final_df


def build_openmeteo_output(
    key_df: pd.DataFrame,
    canonical: pd.DataFrame,
    zone_points_df: pd.DataFrame,
    session: requests.Session,
    cache_dir: Path,
    max_retries: int,
) -> pd.DataFrame:
    return build_openmeteo_family_output(
        key_df,
        canonical,
        zone_points_df,
        session,
        cache_dir,
        max_retries,
        weather_source="openmeteo",
        model="best_match",
    )


def latlon_to_unit(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    cos_lat = np.cos(lat_rad)
    return np.column_stack(
        [
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )


def find_nearest_indices(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    point_lat: np.ndarray,
    point_lon: np.ndarray,
) -> np.ndarray:
    point_lon_aligned = point_lon.copy()
    if np.nanmin(grid_lon) >= 0 and np.any(point_lon_aligned < 0):
        point_lon_aligned = np.mod(point_lon_aligned, 360.0)

    grid_xyz = latlon_to_unit(grid_lat.astype(float), grid_lon.astype(float))
    point_xyz = latlon_to_unit(point_lat.astype(float), point_lon_aligned.astype(float))

    if cKDTree is not None:
        tree = cKDTree(grid_xyz)
        _, indices = tree.query(point_xyz, k=1)
        return np.asarray(indices, dtype=int)

    distances = ((point_xyz[:, None, :] - grid_xyz[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(int)


def build_era5_point_cache(
    zone_points_df: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    cache_dir: Path,
) -> None:
    point_cache_dir = cache_dir / "era5"
    missing_points = []
    for row in zone_points_df.itertuples(index=False):
        cache_file = point_cache_dir / f"{row.point_id}.csv"
        if load_cached_point_frame(cache_file, expected_index) is None:
            missing_points.append(row)

    if not missing_points:
        return

    print(f"[era5] building {len(missing_points)} missing point caches", flush=True)

    reanalysis = xr.open_zarr(ERA5_REANALYSIS_URL, chunks=None, storage_options={"token": "anon"})
    forecast = xr.open_zarr(ERA5_FORECAST_URL, chunks=None, storage_options={"token": "anon"})

    point_lats = np.asarray([row.latitude for row in missing_points], dtype=float)
    point_lons = np.asarray([row.longitude for row in missing_points], dtype=float)

    re_idx = find_nearest_indices(
        reanalysis["latitude"].values,
        reanalysis["longitude"].values,
        point_lats,
        point_lons,
    )
    fc_idx = find_nearest_indices(
        forecast["latitude"].values,
        forecast["longitude"].values,
        point_lats,
        point_lons,
    )

    point_dim = xr.DataArray(np.arange(len(missing_points)), dims="point")
    re_indexer = xr.DataArray(re_idx, dims="point")
    fc_indexer = xr.DataArray(fc_idx, dims="point")

    re_subset = (
        reanalysis[["t2m", "u10", "v10", "msl", "tcc"]]
        .sel(time=slice(expected_index[0], expected_index[-1]))
        .isel(values=re_indexer)
        .load()
    )

    forecast_start = expected_index[0] - pd.Timedelta(hours=18)
    fc_subset = (
        forecast[["tp", "ssrd"]]
        .sel(time=slice(forecast_start, expected_index[-1]))
        .isel(step=slice(0, 13), values=fc_indexer)
        .load()
    )

    re_times = pd.to_datetime(re_subset["time"].values)
    if len(re_times) != len(expected_index):
        raise RuntimeError(
            f"ERA5 reanalysis time coverage mismatch: {len(re_times)} rows vs expected {len(expected_index)}"
        )

    tp_hourly = np.diff(fc_subset["tp"].values, axis=1).reshape(-1, len(missing_points))
    ssrd_hourly = np.diff(fc_subset["ssrd"].values, axis=1).reshape(-1, len(missing_points))
    fc_valid_time = pd.to_datetime(fc_subset["valid_time"].values[:, 1:].reshape(-1))
    fc_mask = (fc_valid_time >= expected_index[0]) & (fc_valid_time <= expected_index[-1])
    fc_times = fc_valid_time[fc_mask]
    tp_hourly = tp_hourly[fc_mask]
    ssrd_hourly = ssrd_hourly[fc_mask]

    if len(fc_times) != len(expected_index):
        unique_fc = pd.Index(fc_times)
        if not unique_fc.equals(pd.Index(expected_index)):
            raise RuntimeError(
                f"ERA5 forecast hourly coverage mismatch: {len(fc_times)} rows vs expected {len(expected_index)}"
            )

    re_t2m = re_subset["t2m"].values
    re_u10 = re_subset["u10"].values
    re_v10 = re_subset["v10"].values
    re_msl = re_subset["msl"].values
    re_tcc = re_subset["tcc"].values

    expected_frame = pd.DataFrame({"time": expected_index})
    for point_pos, row in enumerate(missing_points):
        print(f"[era5] cache {point_pos + 1}/{len(missing_points)} {row.point_id}", flush=True)
        point_df = pd.DataFrame(
            {
                "time": re_times,
                "temp_2m_c": re_t2m[:, point_pos] - 273.15,
                "wind_speed_10m_ms": np.sqrt(re_u10[:, point_pos] ** 2 + re_v10[:, point_pos] ** 2),
                "pressure_msl_hpa": re_msl[:, point_pos] / 100.0,
                "cloud_cover_pct": re_tcc[:, point_pos] * 100.0,
            }
        )
        forecast_df = pd.DataFrame(
            {
                "time": fc_times,
                "precipitation_mm": tp_hourly[:, point_pos] * 1000.0,
                "shortwave_radiation_wm2": ssrd_hourly[:, point_pos] / 3600.0,
            }
        )
        point_df = expected_frame.merge(point_df, on="time", how="left")
        point_df = point_df.merge(forecast_df, on="time", how="left")
        point_df = point_df[
            [
                "time",
                "temp_2m_c",
                "wind_speed_10m_ms",
                "shortwave_radiation_wm2",
                "cloud_cover_pct",
                "precipitation_mm",
                "pressure_msl_hpa",
            ]
        ]
        point_df = clip_weather_ranges(point_df)
        save_point_frame(point_cache_dir / f"{row.point_id}.csv", point_df)


def build_era5_output(
    key_df: pd.DataFrame,
    canonical: pd.DataFrame,
    zone_points_df: pd.DataFrame,
    session: requests.Session,
    cache_dir: Path,
    max_retries: int,
) -> pd.DataFrame:
    return build_openmeteo_family_output(
        key_df,
        canonical,
        zone_points_df,
        session,
        cache_dir,
        max_retries,
        weather_source="era5",
        model="era5",
    )


def write_final_csv(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out["time"] = out["time"].dt.strftime(TIME_FMT)
    out.to_csv(path, index=False)


def sanity_checks(output_dir: Path, run_openmeteo: bool, run_era5: bool) -> None:
    if not (run_openmeteo and run_era5):
        return

    openmeteo_path = output_dir / OUTPUT_OPENMETEO
    era5_path = output_dir / OUTPUT_ERA5
    om = pd.read_csv(openmeteo_path, parse_dates=["time"])
    er = pd.read_csv(era5_path, parse_dates=["time"])

    joined = om.merge(
        er,
        on=["time", "country", "time_zone"],
        suffixes=("_om", "_era5"),
        how="inner",
    )
    sample_vars = VARIABLE_COLUMNS
    for column in sample_vars:
        if joined[f"{column}_om"].equals(joined[f"{column}_era5"]):
            raise RuntimeError(f"Sanity check failed: {column} is identical between Open-Meteo and ERA5.")

    monthly = om.assign(month=om["time"].dt.to_period("M").astype(str))
    available_zones = set(monthly["country"].unique())
    for a, b in [("DK_1", "DK_2"), ("NO_1", "NO_5"), ("IT_NORD", "IT_SICI")]:
        if a not in available_zones or b not in available_zones:
            continue
        left = (
            monthly[monthly["country"] == a]
            .groupby("month")[VARIABLE_COLUMNS]
            .mean(numeric_only=True)
            .reset_index()
        )
        right = (
            monthly[monthly["country"] == b]
            .groupby("month")[VARIABLE_COLUMNS]
            .mean(numeric_only=True)
            .reset_index()
        )
        pair = left.merge(right, on="month", suffixes=("_a", "_b"))
        if pair.empty:
            raise RuntimeError(f"Sanity check failed: no monthly overlap for {a} and {b}.")
        if all(pair[f"{column}_a"].equals(pair[f"{column}_b"]) for column in VARIABLE_COLUMNS):
            raise RuntimeError(f"Sanity check failed: monthly weather is identical for {a} and {b}.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    power_file = Path(args.power_file).resolve()

    zone_filter = {normalize_zone_code(item) for item in args.zones.split(",") if item.strip()}
    key_df, canonical = load_power_keys(power_file, zone_filter)
    zones = canonical["country"].drop_duplicates().sort_values().tolist()

    print(
        f"Loaded {len(key_df):,} power rows across {len(zones)} zones "
        f"from {canonical['time'].min().strftime(TIME_FMT)} to {canonical['time'].max().strftime(TIME_FMT)} UTC.",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "6002-weather-builder/1.0"})

    zone_points_df = build_zone_points(zones, session, cache_dir, args.max_retries)
    print(f"Prepared {len(zone_points_df)} representative points.", flush=True)

    run_openmeteo = args.source in {"openmeteo", "all"}
    run_era5 = args.source in {"era5", "all"}

    if run_openmeteo:
        openmeteo_df = build_openmeteo_output(key_df, canonical, zone_points_df, session, cache_dir, args.max_retries)
        write_final_csv(output_dir / OUTPUT_OPENMETEO, openmeteo_df)
        print(f"Wrote {output_dir / OUTPUT_OPENMETEO}", flush=True)

    if run_era5:
        era5_df = build_era5_output(key_df, canonical, zone_points_df, session, cache_dir, args.max_retries)
        write_final_csv(output_dir / OUTPUT_ERA5, era5_df)
        print(f"Wrote {output_dir / OUTPUT_ERA5}", flush=True)

    sanity_checks(output_dir, run_openmeteo, run_era5)
    print("Weather build completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
