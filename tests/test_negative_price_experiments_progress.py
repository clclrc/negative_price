from __future__ import annotations

import unittest

from negative_price_experiments.progress import ProgressReporter, format_duration, format_metric


class NegativePriceProgressTest(unittest.TestCase):
    def test_format_helpers_cover_short_long_and_nan_values(self) -> None:
        self.assertEqual(format_duration(12.3), "12.3s")
        self.assertEqual(format_duration(65.0), "00:01:05")
        self.assertEqual(format_duration(None), "n/a")
        self.assertEqual(format_metric(float("nan")), "nan")
        self.assertEqual(format_metric(0.12345), "0.1235")

    def test_log_step_uses_deterministic_clock_for_elapsed_and_eta(self) -> None:
        lines: list[str] = []

        def fake_clock() -> float:
            return 10.0

        reporter = ProgressReporter(time_fn=fake_clock, print_fn=lines.append)
        reporter.log_step(
            ("E1", "GRU", "F1"),
            label="fold",
            index=2,
            total=4,
            loop_started_at=0.0,
            step_started_at=8.0,
            extra="pr_auc=0.5000",
        )

        self.assertEqual(
            lines,
            [
                "[E1][GRU][F1] fold 2/4 completed | step=2.0s | elapsed=10.0s | eta=10.0s | pr_auc=0.5000",
            ],
        )
