from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from calendar_features import (
    HolidaySource,
    add_calendar_features,
    compute_calendar_flags_for_zone,
    get_holiday_dates_for_sources,
)


class CalendarFeaturesTest(unittest.TestCase):
    def test_weekend_uses_local_date_not_utc_date(self) -> None:
        flags = compute_calendar_flags_for_zone(pd.Series(["2024-03-29 23:00:00"]), "AT")

        self.assertEqual(int(flags.iloc[0]["is_weekend_local"]), 1)
        self.assertEqual(int(flags.iloc[0]["is_holiday_local"]), 0)

    def test_de_lu_uses_union_of_germany_and_luxembourg_holidays(self) -> None:
        lu_only_date = date(2025, 5, 9)
        lu_dates = get_holiday_dates_for_sources((HolidaySource("LU"),), [2025])
        de_dates = get_holiday_dates_for_sources((HolidaySource("DE"),), [2025])
        self.assertIn(lu_only_date, lu_dates)
        self.assertNotIn(lu_only_date, de_dates)

        flags = compute_calendar_flags_for_zone(pd.Series(["2025-05-09 12:00:00"]), "DE_LU")
        self.assertEqual(int(flags.iloc[0]["is_holiday_local"]), 1)

    def test_danish_bidding_zones_share_same_calendar_flags(self) -> None:
        df = pd.DataFrame(
            {
                "time": ["2024-12-25 12:00:00", "2024-12-25 12:00:00"],
                "country": ["DK_1", "DK_2"],
            }
        )

        out = add_calendar_features(df)

        self.assertEqual(int(out.loc[0, "is_weekend_local"]), int(out.loc[1, "is_weekend_local"]))
        self.assertEqual(int(out.loc[0, "is_holiday_local"]), int(out.loc[1, "is_holiday_local"]))
        self.assertEqual(int(out.loc[0, "is_holiday_local"]), 1)

    def test_regular_weekday_non_holiday_is_zero_zero(self) -> None:
        flags = compute_calendar_flags_for_zone(pd.Series(["2024-02-14 12:00:00"]), "AT")

        self.assertEqual(int(flags.iloc[0]["is_weekend_local"]), 0)
        self.assertEqual(int(flags.iloc[0]["is_holiday_local"]), 0)


if __name__ == "__main__":
    unittest.main()
