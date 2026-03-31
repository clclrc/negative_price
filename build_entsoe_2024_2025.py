#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
from entsoe import EntsoeRawClient
from entsoe.mappings import NEIGHBOURS, PSRTYPE_MAPPINGS
from entsoe.exceptions import (
    InvalidBusinessParameterError,
    InvalidPSRTypeError,
    NoMatchingDataError,
    PaginationError,
)

from calendar_features import (
    CALENDAR_FLAG_COLUMNS,
    HolidaySource,
    add_calendar_features,
    compute_calendar_flags_for_zone,
    ensure_holidays_dependency,
    get_holiday_dates_for_sources,
    get_local_dates_for_zone,
    validate_supported_zones,
)


TZ = "UTC"
DEFAULT_START = pd.Timestamp("2024-01-01 00:00:00", tz=TZ)
DEFAULT_END = pd.Timestamp("2026-01-01 00:00:00", tz=TZ)
DEFAULT_START_ARG = "2024-01-01 00:00:00"
DEFAULT_END_ARG = "2026-01-01 00:00:00"
DEFAULT_COUNTRY_SOURCE = "ALL_COUNTRIES(1).csv"
DEFAULT_OUTPUT = "ALL_COUNTRIES_2024_2025.csv"
DEFAULT_VALIDATION_OUTPUT = "ALL_COUNTRIES_2024_2025_validation.csv"
DEFAULT_CACHE_DIR = "entsoe_2024_2025_cache"
PSR_ORDER = [f"B{i:02d}" for i in range(1, 26)]
PSR_NAME_ORDER = [PSRTYPE_MAPPINGS.get(code, code) for code in PSR_ORDER]
RESOLUTION_PRIORITY = {"15min": 3, "30min": 2, "1h": 1}


@dataclass
class FailureRecord:
    country: str
    dataset: str
    month: str
    source: str
    error: str
    fatal: bool = False


def sanitize_error_text(value: object) -> str:
    text = str(value)
    return re.sub(r"(securityToken=)[^&\\s]+", r"\1REDACTED", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2024-2025 ENTSO-E dataset with raw XML caching."
    )
    parser.add_argument("--start", default=DEFAULT_START_ARG)
    parser.add_argument("--end", default=DEFAULT_END_ARG)
    parser.add_argument("--country-source", default=DEFAULT_COUNTRY_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--countries",
        default="",
        help="Comma-separated subset of countries. Defaults to all countries in the source CSV.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=4.0)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--request-chunk-months", type=int, default=12)
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-crossborder", action="store_true")
    parser.add_argument(
        "--crossborder-cache-only",
        action="store_true",
        help="Read cached crossborder monthly files without issuing network requests.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.getenv("ENTSOE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY environment variable is required.")
    return api_key


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(TZ)
    return ts.tz_convert(TZ)


def load_country_list(country_source: Path, requested: str) -> list[str]:
    if requested.strip():
        countries = [value.strip() for value in requested.split(",") if value.strip()]
    else:
        df = pd.read_csv(country_source, usecols=["country"])
        countries = sorted(df["country"].dropna().astype(str).unique().tolist())

    missing = [country for country in countries if country not in NEIGHBOURS]
    if missing:
        raise ValueError(f"Countries missing in entsoe mappings: {missing}")
    return countries


def is_binary_flag_series(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.empty:
        return True
    return clean.isin([0, 1]).all()


def de_lu_union_holiday_check(country_df: pd.DataFrame) -> bool:
    if country_df.empty or "is_holiday_local" not in country_df.columns:
        return False

    local_dates = get_local_dates_for_zone(country_df["time"], "DE_LU")
    covered_dates = set(local_dates.dropna().tolist())
    if not covered_dates:
        return False

    years = sorted({item.year for item in covered_dates})
    lu_only_dates = get_holiday_dates_for_sources((HolidaySource("LU"),), years) - get_holiday_dates_for_sources(
        (HolidaySource("DE"),),
        years,
    )
    candidate_dates = sorted(covered_dates.intersection(lu_only_dates))
    if not candidate_dates:
        return True

    matching = country_df[local_dates.isin(candidate_dates)]
    if matching.empty:
        return False
    return matching["is_holiday_local"].eq(1).all()


def month_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    current = start
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while current < end:
        next_month = current + pd.offsets.MonthBegin(1)
        windows.append((current, min(next_month, end)))
        current = next_month
    return windows


def normalize_resolution(resolution: object) -> Optional[str]:
    if pd.isna(resolution):
        return None

    value = str(resolution).strip()
    mapping = {
        "PT15M": "15min",
        "15min": "15min",
        "15T": "15min",
        "PT30M": "30min",
        "30min": "30min",
        "30T": "30min",
        "PT60M": "1h",
        "PT1H": "1h",
        "60min": "1h",
        "1h": "1h",
        "H": "1h",
    }
    return mapping.get(value, value)


def resolution_to_freq(resolution: str) -> str:
    normalized = normalize_resolution(resolution)
    mapping = {"15min": "15min", "30min": "30min", "1h": "1h"}
    if normalized not in mapping:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return mapping[normalized]


def select_best_resolution(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    available = [value for value in df["resolution"].dropna().unique().tolist() if value in RESOLUTION_PRIORITY]
    if not available:
        raise ValueError("No supported resolution found in raw data.")
    best = max(available, key=lambda item: RESOLUTION_PRIORITY[item])
    return df[df["resolution"] == best].copy(), best


def get_namespace(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0][1:]
    return ""


def qname(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}" if ns else tag


def parse_xml_generic(xml_text: str, value_tag: str, value_col_name: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    ns = get_namespace(root)
    rows: list[dict[str, object]] = []

    for ts in root.findall(".//" + qname(ns, "TimeSeries")):
        business_type = ts.findtext(qname(ns, "businessType"), default="")
        in_domain = ts.findtext(qname(ns, "in_Domain.mRID"), default="")
        out_domain = ts.findtext(qname(ns, "out_Domain.mRID"), default="")

        psr_type = ""
        mkt_psr = ts.find(qname(ns, "MktPSRType"))
        if mkt_psr is not None:
            psr_elem = mkt_psr.find(qname(ns, "psrType"))
            if psr_elem is not None and psr_elem.text is not None:
                psr_type = psr_elem.text

        currency_unit = ts.findtext(qname(ns, "currency_Unit.name"), default="")
        price_measure_unit = ts.findtext(qname(ns, "price_Measure_Unit.name"), default="")

        for period in ts.findall(qname(ns, "Period")):
            interval = period.find(qname(ns, "timeInterval"))
            if interval is None:
                continue

            start_elem = interval.find(qname(ns, "start"))
            end_elem = interval.find(qname(ns, "end"))
            res_elem = period.find(qname(ns, "resolution"))

            if start_elem is None or end_elem is None or res_elem is None:
                continue

            resolution = normalize_resolution(res_elem.text)
            if resolution not in RESOLUTION_PRIORITY:
                continue

            idx = pd.date_range(
                start=start_elem.text,
                end=end_elem.text,
                freq=resolution_to_freq(resolution),
                inclusive="left",
                tz=TZ,
            )

            for point in period.findall(qname(ns, "Point")):
                pos_elem = point.find(qname(ns, "position"))
                val_elem = point.find(qname(ns, value_tag))
                if pos_elem is None or val_elem is None or val_elem.text in (None, ""):
                    continue

                position = int(pos_elem.text)
                if position < 1 or position > len(idx):
                    continue

                rows.append(
                    {
                        "datetime": idx[position - 1],
                        value_col_name: float(val_elem.text),
                        "position": position,
                        "resolution": resolution,
                        "business_type": business_type,
                        "psr_type": psr_type,
                        "in_domain": in_domain,
                        "out_domain": out_domain,
                        "currency_unit": currency_unit,
                        "price_measure_unit": price_measure_unit,
                    }
                )

    return pd.DataFrame(rows)


def full_hour_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {"datetime": pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz=TZ)}
    )


def empty_series_frame(start: pd.Timestamp, end: pd.Timestamp, value_col: str) -> pd.DataFrame:
    out = full_hour_frame(start, end)
    out[value_col] = pd.NA
    return out


def empty_generation_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = full_hour_frame(start, end)
    for col in PSR_NAME_ORDER:
        out[col] = pd.NA
    return out


def convert_series_raw_to_hourly(
    raw_df: pd.DataFrame,
    value_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if raw_df.empty:
        return empty_series_frame(start, end, value_col)

    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()
    if df.empty:
        return empty_series_frame(start, end, value_col)

    df["resolution"] = df["resolution"].apply(normalize_resolution)
    df, best_resolution = select_best_resolution(df)
    df = df[["datetime", value_col]].copy()
    df = df.groupby("datetime", as_index=False)[value_col].mean()
    df = df.sort_values("datetime").reset_index(drop=True)

    if best_resolution == "1h":
        hourly = df
    else:
        hourly = (
            df.set_index("datetime")
            .resample("1h", label="left", closed="left")
            .mean()
            .reset_index()
        )

    out = full_hour_frame(start, end).merge(hourly, on="datetime", how="left")
    return out


def convert_generation_raw_to_hourly(
    raw_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if raw_df.empty:
        return empty_generation_frame(start, end)

    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()
    if df.empty:
        return empty_generation_frame(start, end)

    df["resolution"] = df["resolution"].apply(normalize_resolution)
    df, best_resolution = select_best_resolution(df)

    if "psr_type" not in df.columns:
        return empty_generation_frame(start, end)

    df["psr_type"] = df["psr_type"].fillna("").astype(str).str.strip()
    df = df[df["psr_type"].isin(PSR_ORDER)].copy()
    if df.empty:
        return empty_generation_frame(start, end)

    df["psr_name"] = df["psr_type"].map(PSRTYPE_MAPPINGS).fillna("N/A")
    wide = (
        df.groupby(["datetime", "psr_name"], as_index=False)["quantity"]
        .mean()
        .pivot(index="datetime", columns="psr_name", values="quantity")
        .reset_index()
    )

    for col in PSR_NAME_ORDER:
        if col not in wide.columns:
            wide[col] = pd.NA

    wide = wide[["datetime"] + PSR_NAME_ORDER]
    wide = wide.sort_values("datetime").reset_index(drop=True)

    if best_resolution == "1h":
        hourly = wide
    else:
        hourly = (
            wide.set_index("datetime")
            .resample("1h", label="left", closed="left")
            .mean()
            .reset_index()
        )

    out = full_hour_frame(start, end).merge(hourly, on="datetime", how="left")
    return out[["datetime"] + PSR_NAME_ORDER]


def read_cached_hourly(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    return df


def write_hourly_cache(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(csv_path, index=False, encoding="utf-8")


def read_cached_hourly_or_empty(
    *,
    cache_dir: Path,
    dataset: str,
    key: str,
    month_start: pd.Timestamp,
    cache_validator: Optional[Callable[[pd.DataFrame], bool]],
    empty_builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    hourly_path = cache_dir / "monthly_hourly" / dataset / key / f"{month_label(month_start)}.csv"
    if hourly_path.exists():
        try:
            cached = read_cached_hourly(hourly_path)
            if cache_validator is None or cache_validator(cached):
                return cached
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] skipping unreadable cached frame {hourly_path}: {sanitize_error_text(exc)}", flush=True)
    return empty_builder()


def read_xml_cache(xml_path: Path) -> Optional[str]:
    if not xml_path.exists():
        return None
    return xml_path.read_text(encoding="utf-8")


def write_xml_cache(xml_text: str, xml_path: Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text, encoding="utf-8")


def bad_window_marker_path(cache_dir: Path, dataset: str, key: str, window_label: str) -> Path:
    return cache_dir / "bad_request_windows" / dataset / key / f"{window_label}.txt"


def mark_bad_request_window(cache_dir: Path, dataset: str, key: str, window_label: str, reason: str) -> None:
    marker = bad_window_marker_path(cache_dir, dataset, key, window_label)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(sanitize_error_text(reason), encoding="utf-8")


def make_request_with_retries(
    request_fn: Callable[[], str],
    label: str,
    max_retries: int,
    retry_sleep: float,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return request_fn()
        except (NoMatchingDataError, InvalidBusinessParameterError, InvalidPSRTypeError, PaginationError):
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(
                f"[retry {attempt}/{max_retries}] {label} failed: {sanitize_error_text(exc)}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"{label} failed after {max_retries} attempts: {sanitize_error_text(last_error)}")


def month_label(start: pd.Timestamp) -> str:
    return start.strftime("%Y-%m")


def window_cache_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.strftime('%Y%m%dT%H%M')}_{end.strftime('%Y%m%dT%H%M')}"


def aligned_request_window(
    month_start: pd.Timestamp,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
    chunk_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if chunk_months <= 1:
        return month_start, min(month_start + pd.DateOffset(months=1), overall_end)

    month_offset = (month_start.year - overall_start.year) * 12 + (month_start.month - overall_start.month)
    aligned_offset = (month_offset // chunk_months) * chunk_months
    request_start = overall_start + pd.DateOffset(months=aligned_offset)
    request_end = min(request_start + pd.DateOffset(months=chunk_months), overall_end)
    return request_start, request_end


def request_window_candidates(
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
    request_chunk_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    candidates = [aligned_request_window(month_start, overall_start, overall_end, request_chunk_months)]
    candidates.append((month_start, month_end))

    unique: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen: set[tuple[str, str]] = set()
    for start, end in candidates:
        key = (str(start), str(end))
        if key not in seen:
            seen.add(key)
            unique.append((start, end))
    return unique


def populate_monthly_caches_from_raw(
    *,
    raw_df: pd.DataFrame,
    cache_dir: Path,
    dataset: str,
    key: str,
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    monthly_converter: Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp], pd.DataFrame],
) -> None:
    for sub_start, sub_end in month_windows(request_start, request_end):
        hourly = monthly_converter(raw_df, sub_start, sub_end)
        hourly_path = cache_dir / "monthly_hourly" / dataset / key / f"{month_label(sub_start)}.csv"
        write_hourly_cache(hourly, hourly_path)


def get_monthly_hourly_frame(
    *,
    cache_dir: Path,
    dataset: str,
    key: str,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
    request_chunk_months: int,
    request_factory: Callable[[pd.Timestamp, pd.Timestamp], Callable[[], str]],
    raw_parser: Callable[[str], pd.DataFrame],
    monthly_converter: Callable[[pd.DataFrame, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    cache_validator: Optional[Callable[[pd.DataFrame], bool]],
    empty_builder: Callable[[], pd.DataFrame],
    failures: list[FailureRecord],
    failure_country: str,
    failure_source: str,
    fatal: bool,
    max_retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    month = month_label(month_start)
    monthly_window_label = window_cache_label(month_start, month_end)
    hourly_path = cache_dir / "monthly_hourly" / dataset / key / f"{month}.csv"
    if hourly_path.exists():
        try:
            cached = read_cached_hourly(hourly_path)
            if cache_validator is None or cache_validator(cached):
                return cached
            print(f"[cache] invalid monthly cache detected for {hourly_path}; rebuilding", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] rebuilding corrupt hourly cache {hourly_path}: {exc}", flush=True)

    last_error: Optional[Exception] = None
    for request_start, request_end in request_window_candidates(
        month_start,
        month_end,
        overall_start,
        overall_end,
        request_chunk_months,
    ):
        window_label = window_cache_label(request_start, request_end)
        marker_path = bad_window_marker_path(cache_dir, dataset, key, window_label)
        if marker_path.exists():
            if (request_start, request_end) != (month_start, month_end):
                continue
            if not fatal:
                return empty_builder()

        xml_path = cache_dir / "raw_xml_request" / dataset / key / f"{window_label}.xml"
        xml_text = read_xml_cache(xml_path)

        if xml_text is None:
            label = f"{dataset}:{key}:{window_label}"
            try:
                xml_text = make_request_with_retries(
                    request_factory(request_start, request_end),
                    label,
                    max_retries,
                    retry_sleep,
                )
                write_xml_cache(xml_text, xml_path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if (request_start, request_end) != (month_start, month_end):
                    mark_bad_request_window(cache_dir, dataset, key, window_label, str(exc))
                    print(
                        f"[fallback] {dataset}:{key}:{month} falling back to a smaller request window after: {exc}",
                        flush=True,
                    )
                    continue
                failures.append(
                    FailureRecord(
                        country=failure_country,
                        dataset=dataset,
                        month=month,
                        source=failure_source,
                        error=sanitize_error_text(exc),
                        fatal=fatal,
                    )
                )
                if fatal:
                    raise
                mark_bad_request_window(cache_dir, dataset, key, window_label, sanitize_error_text(exc))
                return empty_builder()

        try:
            raw_df = raw_parser(xml_text)
            populate_monthly_caches_from_raw(
                raw_df=raw_df,
                cache_dir=cache_dir,
                dataset=dataset,
                key=key,
                request_start=request_start,
                request_end=request_end,
                monthly_converter=monthly_converter,
            )
            hourly = read_cached_hourly(hourly_path)
            if (
                cache_validator is not None
                and not cache_validator(hourly)
                and (request_start, request_end) != (month_start, month_end)
            ):
                mark_bad_request_window(
                    cache_dir,
                    dataset,
                    key,
                    window_label,
                    f"Target month {month} was not fully covered",
                )
                print(
                    f"[fallback] {dataset}:{key}:{month} request window {window_label} did not fully cover the target month; retrying monthly",
                    flush=True,
                )
                continue
            return hourly
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] reparsing failed for {xml_path}: {sanitize_error_text(exc)}", flush=True)
            last_error = exc
            try:
                xml_text = make_request_with_retries(
                    request_factory(request_start, request_end),
                    f"{dataset}:{key}:{window_label}",
                    max_retries,
                    retry_sleep,
                )
                write_xml_cache(xml_text, xml_path)
                raw_df = raw_parser(xml_text)
                populate_monthly_caches_from_raw(
                    raw_df=raw_df,
                    cache_dir=cache_dir,
                    dataset=dataset,
                    key=key,
                    request_start=request_start,
                    request_end=request_end,
                    monthly_converter=monthly_converter,
                )
                hourly = read_cached_hourly(hourly_path)
                if (
                    cache_validator is not None
                    and not cache_validator(hourly)
                    and (request_start, request_end) != (month_start, month_end)
                ):
                    mark_bad_request_window(
                        cache_dir,
                        dataset,
                        key,
                        window_label,
                        f"Target month {month} was not fully covered after refetch",
                    )
                    print(
                        f"[fallback] {dataset}:{key}:{month} refetched window {window_label} still missed the target month; retrying monthly",
                        flush=True,
                    )
                    continue
                return hourly
            except Exception as refetch_exc:  # noqa: BLE001
                last_error = refetch_exc
                if (request_start, request_end) != (month_start, month_end):
                    mark_bad_request_window(
                        cache_dir,
                        dataset,
                        key,
                        window_label,
                        sanitize_error_text(refetch_exc),
                    )
                    print(
                        f"[fallback] {dataset}:{key}:{month} switching to monthly request after parse/refetch failure: {sanitize_error_text(refetch_exc)}",
                        flush=True,
                    )
                    continue

    failures.append(
        FailureRecord(
            country=failure_country,
            dataset=dataset,
            month=month,
            source=failure_source,
            error=sanitize_error_text(last_error),
            fatal=fatal,
        )
    )
    if fatal:
        raise RuntimeError(f"{dataset}:{key}:{month} failed: {sanitize_error_text(last_error)}")
    mark_bad_request_window(cache_dir, dataset, key, monthly_window_label, sanitize_error_text(last_error))
    return empty_builder()


def concat_monthly_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(list(frames), axis=0, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["datetime"])
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def sum_series_with_min_count(series_list: list[pd.Series]) -> pd.Series:
    if not series_list:
        return pd.Series(pd.NA, index=pd.RangeIndex(0))
    return pd.concat(series_list, axis=1).sum(axis=1, min_count=1)


def build_country_dataset(
    *,
    client: EntsoeRawClient,
    cache_dir: Path,
    country: str,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    failures: list[FailureRecord],
    max_retries: int,
    retry_sleep: float,
    request_chunk_months: int,
    skip_load: bool,
    skip_generation: bool,
    skip_crossborder: bool,
    crossborder_cache_only: bool,
) -> pd.DataFrame:
    print(f"\n=== Building {country} ===", flush=True)
    overall_start = windows[0][0]
    overall_end = windows[-1][1]
    aux_max_retries = 1
    aux_retry_sleep = 0.0
    full_hours = full_hour_frame(windows[0][0], windows[-1][1])
    price_months: list[pd.DataFrame] = []
    load_months: list[pd.DataFrame] = []
    generation_months: list[pd.DataFrame] = []

    for month_start, month_end in windows:
        price_month = get_monthly_hourly_frame(
                cache_dir=cache_dir,
                dataset="price",
                key=country,
                month_start=month_start,
                month_end=month_end,
                overall_start=overall_start,
                overall_end=overall_end,
                request_chunk_months=request_chunk_months,
                request_factory=lambda request_start, request_end, c=country: (
                    lambda: client.query_day_ahead_prices(c, request_start, request_end)
                ),
                raw_parser=lambda text: parse_xml_generic(text, "price.amount", "price"),
                monthly_converter=lambda raw_df, s, e: convert_series_raw_to_hourly(
                    raw_df,
                    "price",
                    s,
                    e,
                ),
                cache_validator=lambda df: df["price"].notna().any(),
                empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "price"),
                failures=failures,
                failure_country=country,
                failure_source=country,
                fatal=True,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
        if price_month["price"].isna().all():
            raise RuntimeError(f"price data is empty for {country} in {month_label(month_start)}")
        price_months.append(price_month)
        if skip_load:
            load_months.append(
                read_cached_hourly_or_empty(
                    cache_dir=cache_dir,
                    dataset="load",
                    key=country,
                    month_start=month_start,
                    cache_validator=lambda df: df["quantity"].notna().any(),
                    empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                )
            )
        else:
            load_months.append(
                get_monthly_hourly_frame(
                    cache_dir=cache_dir,
                    dataset="load",
                    key=country,
                    month_start=month_start,
                    month_end=month_end,
                    overall_start=overall_start,
                    overall_end=overall_end,
                    request_chunk_months=request_chunk_months,
                    request_factory=lambda request_start, request_end, c=country: (
                        lambda: client.query_load(c, request_start, request_end)
                    ),
                    raw_parser=lambda text: parse_xml_generic(text, "quantity", "quantity"),
                    monthly_converter=lambda raw_df, s, e: convert_series_raw_to_hourly(
                        raw_df,
                        "quantity",
                        s,
                        e,
                    ),
                    cache_validator=lambda df: df["quantity"].notna().any(),
                    empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                    failures=failures,
                    failure_country=country,
                    failure_source=country,
                    fatal=False,
                    max_retries=aux_max_retries,
                    retry_sleep=aux_retry_sleep,
                )
            )
        if skip_generation:
            generation_months.append(
                read_cached_hourly_or_empty(
                    cache_dir=cache_dir,
                    dataset="generation",
                    key=country,
                    month_start=month_start,
                    cache_validator=lambda df: df[PSR_NAME_ORDER].notna().any().any(),
                    empty_builder=lambda s=month_start, e=month_end: empty_generation_frame(s, e),
                )
            )
        else:
            generation_months.append(
                get_monthly_hourly_frame(
                    cache_dir=cache_dir,
                    dataset="generation",
                    key=country,
                    month_start=month_start,
                    month_end=month_end,
                    overall_start=overall_start,
                    overall_end=overall_end,
                    request_chunk_months=request_chunk_months,
                    request_factory=lambda request_start, request_end, c=country: (
                        lambda: client.query_generation(c, request_start, request_end)
                    ),
                    raw_parser=lambda text: parse_xml_generic(text, "quantity", "quantity"),
                    monthly_converter=lambda raw_df, s, e: convert_generation_raw_to_hourly(
                        raw_df,
                        s,
                        e,
                    ),
                    cache_validator=lambda df: df[PSR_NAME_ORDER].notna().any().any(),
                    empty_builder=lambda s=month_start, e=month_end: empty_generation_frame(s, e),
                    failures=failures,
                    failure_country=country,
                    failure_source=country,
                    fatal=False,
                    max_retries=aux_max_retries,
                    retry_sleep=aux_retry_sleep,
                )
            )

    price_df = concat_monthly_frames(price_months).rename(columns={"datetime": "time"})
    load_df = concat_monthly_frames(load_months).rename(columns={"datetime": "time", "quantity": "load"})
    gen_df = concat_monthly_frames(generation_months).rename(columns={"datetime": "time"})

    export_series_list: list[pd.Series] = []
    import_series_list: list[pd.Series] = []

    if not skip_crossborder:
        for neighbour in NEIGHBOURS[country]:
            pair_key = f"{country}__{neighbour}"
            reverse_key = f"{neighbour}__{country}"
            export_months: list[pd.DataFrame] = []
            import_months: list[pd.DataFrame] = []

            for month_start, month_end in windows:
                if crossborder_cache_only:
                    export_months.append(
                        read_cached_hourly_or_empty(
                            cache_dir=cache_dir,
                            dataset="crossborder",
                            key=pair_key,
                            month_start=month_start,
                            cache_validator=lambda df: df["quantity"].notna().any(),
                            empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                        )
                    )
                    import_months.append(
                        read_cached_hourly_or_empty(
                            cache_dir=cache_dir,
                            dataset="crossborder",
                            key=reverse_key,
                            month_start=month_start,
                            cache_validator=lambda df: df["quantity"].notna().any(),
                            empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                        )
                    )
                else:
                    export_months.append(
                        get_monthly_hourly_frame(
                            cache_dir=cache_dir,
                            dataset="crossborder",
                            key=pair_key,
                            month_start=month_start,
                            month_end=month_end,
                            overall_start=overall_start,
                            overall_end=overall_end,
                            request_chunk_months=request_chunk_months,
                            request_factory=lambda request_start, request_end, c=country, n=neighbour: (
                                lambda: client.query_crossborder_flows(c, n, request_start, request_end)
                            ),
                            raw_parser=lambda text: parse_xml_generic(text, "quantity", "quantity"),
                            monthly_converter=lambda raw_df, s, e: convert_series_raw_to_hourly(
                                raw_df,
                                "quantity",
                                s,
                                e,
                            ),
                            cache_validator=lambda df: df["quantity"].notna().any(),
                            empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                            failures=failures,
                            failure_country=country,
                            failure_source=f"{country}->{neighbour}",
                            fatal=False,
                            max_retries=aux_max_retries,
                            retry_sleep=aux_retry_sleep,
                        )
                    )
                    import_months.append(
                        get_monthly_hourly_frame(
                            cache_dir=cache_dir,
                            dataset="crossborder",
                            key=reverse_key,
                            month_start=month_start,
                            month_end=month_end,
                            overall_start=overall_start,
                            overall_end=overall_end,
                            request_chunk_months=request_chunk_months,
                            request_factory=lambda request_start, request_end, c=country, n=neighbour: (
                                lambda: client.query_crossborder_flows(n, c, request_start, request_end)
                            ),
                            raw_parser=lambda text: parse_xml_generic(text, "quantity", "quantity"),
                            monthly_converter=lambda raw_df, s, e: convert_series_raw_to_hourly(
                                raw_df,
                                "quantity",
                                s,
                                e,
                            ),
                            cache_validator=lambda df: df["quantity"].notna().any(),
                            empty_builder=lambda s=month_start, e=month_end: empty_series_frame(s, e, "quantity"),
                            failures=failures,
                            failure_country=country,
                            failure_source=f"{neighbour}->{country}",
                            fatal=False,
                            max_retries=aux_max_retries,
                            retry_sleep=aux_retry_sleep,
                        )
                    )

            export_df = concat_monthly_frames(export_months)
            import_df = concat_monthly_frames(import_months)
            export_series_list.append(export_df["quantity"].reset_index(drop=True))
            import_series_list.append(import_df["quantity"].reset_index(drop=True))

    df = full_hours.rename(columns={"datetime": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["price"] = price_df["price"].reset_index(drop=True)
    df = df.merge(load_df[["time", "load"]], on="time", how="left")
    df = df.merge(gen_df, on="time", how="left")

    if export_series_list:
        df["total export"] = pd.concat(export_series_list, axis=1).sum(axis=1, min_count=1)
    else:
        df["total export"] = pd.NA

    if import_series_list:
        df["total import"] = pd.concat(import_series_list, axis=1).sum(axis=1, min_count=1)
    else:
        df["total import"] = pd.NA

    df.insert(1, "country", country)
    df.insert(2, "time_zone", TZ)
    df = add_calendar_features(df)

    final_cols = ["time", "country", "time_zone", *CALENDAR_FLAG_COLUMNS, "price", "load"] + PSR_NAME_ORDER + [
        "total export",
        "total import",
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[final_cols]


def clean_time_column(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dt.tz_localize(None)
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_dataset(
    df: pd.DataFrame,
    countries: list[str],
    failures: list[FailureRecord],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    expected_hours = int((end - start) / pd.Timedelta(hours=1))
    expected_index = pd.date_range(start=start, end=end, freq="1h", inclusive="left", tz=TZ)
    failure_df = pd.DataFrame([record.__dict__ for record in failures])
    rows: list[dict[str, object]] = []

    for country in countries:
        country_df = df[df["country"] == country].copy()
        times = pd.to_datetime(country_df["time"], errors="coerce", utc=True)
        dedup_times = times.dropna()
        unique_times = dedup_times.drop_duplicates()
        missing_hours = len(expected_index.difference(unique_times))
        duplicate_hours = int(dedup_times.duplicated().sum())
        weekend_missing_values = (
            int(country_df["is_weekend_local"].isna().sum())
            if "is_weekend_local" in country_df
            else expected_hours
        )
        holiday_missing_values = (
            int(country_df["is_holiday_local"].isna().sum())
            if "is_holiday_local" in country_df
            else expected_hours
        )
        weekend_invalid_values = (
            0 if "is_weekend_local" in country_df and is_binary_flag_series(country_df["is_weekend_local"]) else len(country_df)
        )
        holiday_invalid_values = (
            0 if "is_holiday_local" in country_df and is_binary_flag_series(country_df["is_holiday_local"]) else len(country_df)
        )
        calendar_mismatch_values = len(country_df)
        de_lu_union_ok = country != "DE_LU"
        if (
            "is_weekend_local" in country_df
            and "is_holiday_local" in country_df
            and weekend_missing_values == 0
            and holiday_missing_values == 0
            and weekend_invalid_values == 0
            and holiday_invalid_values == 0
        ):
            expected_flags = compute_calendar_flags_for_zone(country_df["time"], country)
            actual_flags = country_df[list(CALENDAR_FLAG_COLUMNS)].reset_index(drop=True).astype("int8")
            calendar_mismatch_values = int((actual_flags != expected_flags.reset_index(drop=True)).any(axis=1).sum())
            if country == "DE_LU":
                de_lu_union_ok = de_lu_union_holiday_check(country_df)

        country_failures = (
            failure_df[failure_df["country"] == country] if not failure_df.empty else pd.DataFrame()
        )
        failure_count = int(len(country_failures))
        failure_details = ""
        if failure_count:
            detail_rows = [
                f"{row.dataset}:{row.month}:{row.source}:{row.error}"
                for row in country_failures.itertuples(index=False)
            ]
            failure_details = " | ".join(detail_rows[:20])

        row = {
            "country": country,
            "row_count": int(len(country_df)),
            "min_time": clean_time_column(pd.Series([times.min()])).iloc[0] if not country_df.empty else "",
            "max_time": clean_time_column(pd.Series([times.max()])).iloc[0] if not country_df.empty else "",
            "duplicate_hours": duplicate_hours,
            "missing_hours": missing_hours,
            "price_missing_values": int(country_df["price"].isna().sum()) if "price" in country_df else expected_hours,
            "load_missing_values": int(country_df["load"].isna().sum()) if "load" in country_df else expected_hours,
            "generation_all_null_hours": int(country_df[PSR_NAME_ORDER].isna().all(axis=1).sum()),
            "total_export_missing_values": int(country_df["total export"].isna().sum())
            if "total export" in country_df
            else expected_hours,
            "total_import_missing_values": int(country_df["total import"].isna().sum())
            if "total import" in country_df
            else expected_hours,
            "is_weekend_local_missing_values": weekend_missing_values,
            "is_holiday_local_missing_values": holiday_missing_values,
            "is_weekend_local_invalid_values": weekend_invalid_values,
            "is_holiday_local_invalid_values": holiday_invalid_values,
            "calendar_mismatch_values": calendar_mismatch_values,
            "de_lu_union_holiday_check": de_lu_union_ok,
            "failure_count": failure_count,
            "failure_details": failure_details,
            "passed": bool(
                len(country_df) == expected_hours
                and duplicate_hours == 0
                and missing_hours == 0
                and (country_df["country"].nunique() == 1 if not country_df.empty else False)
                and weekend_missing_values == 0
                and holiday_missing_values == 0
                and weekend_invalid_values == 0
                and holiday_invalid_values == 0
                and calendar_mismatch_values == 0
                and de_lu_union_ok
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compare_january_subset(new_df: pd.DataFrame, old_csv: Path) -> dict[str, object]:
    old_df = pd.read_csv(old_csv)
    old_jan = old_df.copy()
    new_jan = new_df[new_df["time"] < "2024-02-01 00:00:00"].copy()

    return {
        "same_country_count": old_jan["country"].nunique() == new_jan["country"].nunique(),
        "same_country_set": set(old_jan["country"].dropna().unique()) == set(new_jan["country"].dropna().unique()),
        "same_columns": old_jan.columns.tolist() == new_jan.columns.tolist(),
        "time_format_ok": bool(new_jan["time"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()),
    }


def main() -> int:
    args = parse_args()
    start = parse_utc_timestamp(args.start)
    end = parse_utc_timestamp(args.end)
    if end <= start:
        raise ValueError("end must be greater than start")

    root = Path.cwd()
    country_source = root / args.country_source
    output_path = root / args.output
    validation_path = root / args.validation_output
    cache_dir = root / args.cache_dir

    countries = load_country_list(country_source, args.countries)
    validate_supported_zones(countries)
    ensure_holidays_dependency()
    api_key = ensure_api_key()
    windows = month_windows(start, end)
    client = EntsoeRawClient(
        api_key=api_key,
        retry_count=1,
        retry_delay=max(int(args.retry_sleep), 1),
        timeout=args.request_timeout,
    )
    failures: list[FailureRecord] = []
    country_frames: list[pd.DataFrame] = []

    print(f"Countries: {len(countries)}", flush=True)
    print(f"Window: {start} -> {end}", flush=True)
    print(f"Monthly chunks: {len(windows)}", flush=True)

    for country in countries:
        country_frames.append(
            build_country_dataset(
                client=client,
                cache_dir=cache_dir,
                country=country,
                windows=windows,
                failures=failures,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                request_chunk_months=args.request_chunk_months,
                skip_load=args.skip_load,
                skip_generation=args.skip_generation,
                skip_crossborder=args.skip_crossborder,
                crossborder_cache_only=args.crossborder_cache_only,
            )
        )

    merged = pd.concat(country_frames, axis=0, ignore_index=True, sort=False)
    merged["time"] = clean_time_column(merged["time"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    validation_df = validate_dataset(merged, countries, failures, start, end)
    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")

    jan_check = compare_january_subset(merged, country_source)
    expected_hours = int((end - start) / pd.Timedelta(hours=1))
    expected_rows = expected_hours * len(countries)
    passed = bool(
        len(merged) == expected_rows
        and validation_df["passed"].all()
        and validation_df["row_count"].eq(expected_hours).all()
        and validation_df["min_time"].eq(start.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")).all()
        and validation_df["max_time"].eq(
            (end - pd.Timedelta(hours=1)).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
        ).all()
        and merged["country"].nunique() == len(countries)
    )

    print("\n=== Build Summary ===", flush=True)
    print(f"Output file: {output_path}", flush=True)
    print(f"Validation file: {validation_path}", flush=True)
    print(f"Rows: {len(merged)} (expected {expected_rows})", flush=True)
    print(f"Country count: {merged['country'].nunique()} (expected {len(countries)})", flush=True)
    print(f"Time min: {merged['time'].min()}", flush=True)
    print(f"Time max: {merged['time'].max()}", flush=True)
    print(f"January compatibility: {jan_check}", flush=True)
    print(f"Failure count: {len(failures)}", flush=True)
    print(f"Validation passed: {passed}", flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {sanitize_error_text(exc)}", file=sys.stderr)
        raise SystemExit(1)
