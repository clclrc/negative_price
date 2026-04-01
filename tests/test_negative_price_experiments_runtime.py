from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from negative_price_experiments.runtime import get_cpu_worker_count, get_parallel_worker_count


class NegativePriceRuntimeTest(unittest.TestCase):
    def test_cpu_worker_count_reserves_one_core_by_default(self) -> None:
        with patch("os.cpu_count", return_value=8):
            self.assertEqual(get_cpu_worker_count(), 7)
            self.assertEqual(get_cpu_worker_count(max_workers=4), 4)

    def test_cpu_worker_count_honors_env_override(self) -> None:
        with patch.dict(os.environ, {"NEGPRICE_CPU_WORKERS": "3"}, clear=False):
            with patch("os.cpu_count", return_value=16):
                self.assertEqual(get_cpu_worker_count(), 3)
                self.assertEqual(get_cpu_worker_count(max_workers=2), 2)

    def test_parallel_worker_count_avoids_tiny_workloads(self) -> None:
        with patch("os.cpu_count", return_value=12):
            self.assertEqual(get_parallel_worker_count(100, max_workers=8, min_items_per_worker=256), 1)
            self.assertEqual(get_parallel_worker_count(2048, max_workers=8, min_items_per_worker=256), 8)
