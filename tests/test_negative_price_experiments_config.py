from __future__ import annotations

import unittest
from pathlib import Path

from negative_price_experiments.config import build_default_experiment_configs


class NegativePriceConfigTest(unittest.TestCase):
    def test_default_configs_include_advanced_post_e6_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        self.assertIn("E7", configs)
        self.assertIn("E8", configs)
        self.assertIn("E9", configs)
        self.assertIn("E10", configs)
        self.assertEqual(configs["E7"].models, ("LightGBM",))
        self.assertEqual(configs["E8"].models, ("CatBoost",))
        self.assertEqual(configs["E9"].models, ("XGBoostWeightedCalibrated",))
        self.assertEqual(configs["E10"].models, ("PatchTST",))
        self.assertEqual(configs["E7"].horizon_hours, 6)
        self.assertEqual(configs["E10"].feature_group, "public")
