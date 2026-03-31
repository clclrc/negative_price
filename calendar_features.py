#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Iterable

import pandas as pd

try:
    import holidays as holidays_lib
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in build-time guard
    holidays_lib = None
    HOLIDAYS_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    HOLIDAYS_IMPORT_ERROR = None


CALENDAR_FLAG_COLUMNS = ("is_weekend_local", "is_holiday_local")


@dataclass(frozen=True)
class HolidaySource:
    country: str
    subdiv: str | None = None


@dataclass(frozen=True)
class ZoneCalendarConfig:
    timezone: str
    holiday_sources: tuple[HolidaySource, ...]


ZONE_CALENDAR_CONFIGS: dict[str, ZoneCalendarConfig] = {
    "AT": ZoneCalendarConfig("Europe/Vienna", (HolidaySource("AT"),)),
    "BE": ZoneCalendarConfig("Europe/Brussels", (HolidaySource("BE"),)),
    "BG": ZoneCalendarConfig("Europe/Sofia", (HolidaySource("BG"),)),
    "CH": ZoneCalendarConfig("Europe/Zurich", (HolidaySource("CH"),)),
    "CZ": ZoneCalendarConfig("Europe/Prague", (HolidaySource("CZ"),)),
    "DE_LU": ZoneCalendarConfig("Europe/Berlin", (HolidaySource("DE"), HolidaySource("LU"))),
    "DK_1": ZoneCalendarConfig("Europe/Copenhagen", (HolidaySource("DK"),)),
    "DK_2": ZoneCalendarConfig("Europe/Copenhagen", (HolidaySource("DK"),)),
    "EE": ZoneCalendarConfig("Europe/Tallinn", (HolidaySource("EE"),)),
    "ES": ZoneCalendarConfig("Europe/Madrid", (HolidaySource("ES"),)),
    "FI": ZoneCalendarConfig("Europe/Helsinki", (HolidaySource("FI"),)),
    "FR": ZoneCalendarConfig("Europe/Paris", (HolidaySource("FR"),)),
    "GR": ZoneCalendarConfig("Europe/Athens", (HolidaySource("GR"),)),
    "HR": ZoneCalendarConfig("Europe/Zagreb", (HolidaySource("HR"),)),
    "HU": ZoneCalendarConfig("Europe/Budapest", (HolidaySource("HU"),)),
    "IE_SEM": ZoneCalendarConfig(
        "Europe/Dublin",
        (HolidaySource("IE"), HolidaySource("GB", subdiv="NIR")),
    ),
    "IT_CALA": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_CNOR": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_CSUD": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_NORD": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_SARD": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_SICI": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "IT_SUD": ZoneCalendarConfig("Europe/Rome", (HolidaySource("IT"),)),
    "LT": ZoneCalendarConfig("Europe/Vilnius", (HolidaySource("LT"),)),
    "LV": ZoneCalendarConfig("Europe/Riga", (HolidaySource("LV"),)),
    "ME": ZoneCalendarConfig("Europe/Podgorica", (HolidaySource("ME"),)),
    "MK": ZoneCalendarConfig("Europe/Skopje", (HolidaySource("MK"),)),
    "NL": ZoneCalendarConfig("Europe/Amsterdam", (HolidaySource("NL"),)),
    "NO_1": ZoneCalendarConfig("Europe/Oslo", (HolidaySource("NO"),)),
    "NO_2": ZoneCalendarConfig("Europe/Oslo", (HolidaySource("NO"),)),
    "NO_3": ZoneCalendarConfig("Europe/Oslo", (HolidaySource("NO"),)),
    "NO_4": ZoneCalendarConfig("Europe/Oslo", (HolidaySource("NO"),)),
    "NO_5": ZoneCalendarConfig("Europe/Oslo", (HolidaySource("NO"),)),
    "PL": ZoneCalendarConfig("Europe/Warsaw", (HolidaySource("PL"),)),
    "PT": ZoneCalendarConfig("Europe/Lisbon", (HolidaySource("PT"),)),
    "RO": ZoneCalendarConfig("Europe/Bucharest", (HolidaySource("RO"),)),
    "RS": ZoneCalendarConfig("Europe/Belgrade", (HolidaySource("RS"),)),
    "SE_1": ZoneCalendarConfig("Europe/Stockholm", (HolidaySource("SE"),)),
    "SE_2": ZoneCalendarConfig("Europe/Stockholm", (HolidaySource("SE"),)),
    "SE_3": ZoneCalendarConfig("Europe/Stockholm", (HolidaySource("SE"),)),
    "SE_4": ZoneCalendarConfig("Europe/Stockholm", (HolidaySource("SE"),)),
    "SI": ZoneCalendarConfig("Europe/Ljubljana", (HolidaySource("SI"),)),
    "SK": ZoneCalendarConfig("Europe/Bratislava", (HolidaySource("SK"),)),
}


def ensure_holidays_dependency() -> None:
    if holidays_lib is None:
        raise RuntimeError(
            "python-holidays is required to build local holiday flags. "
            "Install it with `python -m pip install holidays`."
        ) from HOLIDAYS_IMPORT_ERROR


def validate_supported_zones(zones: Iterable[object]) -> None:
    unknown = sorted(
        {
            str(zone).strip()
            for zone in zones
            if zone is not None and str(zone).strip() and str(zone).strip() not in ZONE_CALENDAR_CONFIGS
        }
    )
    if unknown:
        raise ValueError(
            "Calendar configuration is missing for zone(s): "
            + ", ".join(unknown)
        )


def get_zone_config(zone: str) -> ZoneCalendarConfig:
    validate_supported_zones([zone])
    return ZONE_CALENDAR_CONFIGS[zone]


@lru_cache(maxsize=None)
def _holiday_dates_for_sources(
    sources: tuple[HolidaySource, ...],
    years: tuple[int, ...],
) -> frozenset[date]:
    ensure_holidays_dependency()
    dates: set[date] = set()
    for source in sources:
        kwargs: dict[str, object] = {"years": list(years)}
        if source.subdiv is not None:
            kwargs["subdiv"] = source.subdiv
        calendar = holidays_lib.country_holidays(source.country, **kwargs)
        dates.update(calendar.keys())
    return frozenset(dates)


def get_holiday_dates_for_zone(zone: str, years: Iterable[int]) -> frozenset[date]:
    normalized_years = tuple(sorted({int(year) for year in years}))
    if not normalized_years:
        return frozenset()
    return _holiday_dates_for_sources(get_zone_config(zone).holiday_sources, normalized_years)


def get_holiday_dates_for_sources(
    sources: Iterable[HolidaySource],
    years: Iterable[int],
) -> frozenset[date]:
    normalized_years = tuple(sorted({int(year) for year in years}))
    if not normalized_years:
        return frozenset()
    return _holiday_dates_for_sources(tuple(sources), normalized_years)


def get_local_dates_for_zone(times: pd.Series, zone: str) -> pd.Series:
    local_times = pd.to_datetime(times, errors="coerce", utc=True)
    if local_times.isna().any():
        raise ValueError(f"Invalid UTC timestamps found while building calendar features for zone {zone}")
    return local_times.dt.tz_convert(get_zone_config(zone).timezone).dt.date


def compute_calendar_flags_for_zone(times: pd.Series, zone: str) -> pd.DataFrame:
    local_times = pd.to_datetime(times, errors="coerce", utc=True)
    if local_times.isna().any():
        raise ValueError(f"Invalid UTC timestamps found while building calendar features for zone {zone}")

    config = get_zone_config(zone)
    local_times = local_times.dt.tz_convert(config.timezone)
    local_dates = local_times.dt.date
    holiday_dates = get_holiday_dates_for_zone(zone, (item.year for item in local_dates))

    return pd.DataFrame(
        {
            "is_weekend_local": local_times.dt.weekday.isin([5, 6]).astype("int8"),
            "is_holiday_local": local_dates.isin(holiday_dates).astype("int8"),
        },
        index=times.index,
    )


def add_calendar_features(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    zone_col: str = "country",
) -> pd.DataFrame:
    if time_col not in df.columns:
        raise KeyError(f"Missing required time column: {time_col}")
    if zone_col not in df.columns:
        raise KeyError(f"Missing required zone column: {zone_col}")

    out = df.copy()
    validate_supported_zones(out[zone_col].dropna().unique().tolist())

    weekend = pd.Series(0, index=out.index, dtype="int8")
    holiday = pd.Series(0, index=out.index, dtype="int8")

    for zone, zone_df in out.groupby(zone_col, sort=False):
        flags = compute_calendar_flags_for_zone(zone_df[time_col], zone)
        weekend.loc[flags.index] = flags["is_weekend_local"]
        holiday.loc[flags.index] = flags["is_holiday_local"]

    out["is_weekend_local"] = weekend.astype("int8")
    out["is_holiday_local"] = holiday.astype("int8")
    return out
