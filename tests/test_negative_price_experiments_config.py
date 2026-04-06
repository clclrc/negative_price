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

    def test_default_configs_include_next_phase_deep_learning_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        self.assertIn("E11", configs)
        self.assertIn("E12", configs)
        self.assertIn("E13", configs)
        self.assertIn("E14", configs)
        self.assertEqual(configs["E11"].models, ("GRU",))
        self.assertEqual(configs["E12"].models, ("GRU",))
        self.assertEqual(configs["E13"].models, ("TCN",))
        self.assertEqual(configs["E14"].models, ("GRU",))
        self.assertEqual(configs["E11"].window_hours, 120)
        self.assertEqual(configs["E12"].window_hours, 168)
        self.assertEqual(configs["E13"].window_hours, 168)
        self.assertEqual(configs["E14"].window_hours, 168)
        self.assertEqual(configs["E11"].horizon_hours, 6)
        self.assertEqual(configs["E12"].feature_group, "public")
        self.assertEqual(configs["E13"].countries, configs["E1"].countries)
        self.assertEqual(configs["E14"].sequence_loss, "focal")
        self.assertEqual(configs["E14"].focal_gamma, 2.0)

    def test_default_configs_include_richer_feature_gru_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        self.assertIn("E15A", configs)
        self.assertIn("E15B", configs)
        self.assertIn("E16A", configs)
        self.assertIn("E16B", configs)

        self.assertEqual(configs["E15A"].models, ("GRU",))
        self.assertEqual(configs["E15B"].models, ("GRU",))
        self.assertEqual(configs["E16A"].models, ("GRU",))
        self.assertEqual(configs["E16B"].models, ("GRU",))

        self.assertEqual(configs["E15A"].window_hours, 168)
        self.assertEqual(configs["E15B"].window_hours, 168)
        self.assertEqual(configs["E16A"].window_hours, 168)
        self.assertEqual(configs["E16B"].window_hours, 168)

        self.assertEqual(configs["E15A"].feature_group, "public")
        self.assertEqual(configs["E15B"].feature_group, "renewables")
        self.assertEqual(configs["E16A"].feature_group, "public")
        self.assertEqual(configs["E16B"].feature_group, "flows")

        self.assertEqual(configs["E15A"].countries, configs["E3"].countries)
        self.assertEqual(configs["E15B"].countries, configs["E3"].countries)
        self.assertEqual(configs["E16A"].countries, configs["E5"].countries)
        self.assertEqual(configs["E16B"].countries, configs["E5"].countries)

    def test_default_configs_include_next_iteration_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E17A", "E17B", "E18", "E19", "E20"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E17A"].models, ("GRU",))
        self.assertEqual(configs["E17B"].models, ("GRU",))
        self.assertEqual(configs["E18"].models, ("GRU",))
        self.assertEqual(configs["E19"].models, ("GRUHybrid",))
        self.assertEqual(configs["E20"].models, ("GRU",))

        self.assertEqual(configs["E17A"].sample_filter_feature_group, "renewables")
        self.assertEqual(configs["E17B"].sample_filter_feature_group, "renewables")
        self.assertEqual(configs["E18"].sample_filter_feature_group, "renewables")
        self.assertEqual(configs["E19"].sample_filter_feature_group, "renewables")
        self.assertEqual(configs["E20"].sample_filter_feature_group, None)

        self.assertEqual(configs["E18"].repeat_random_seeds, (42, 52, 62))
        self.assertTrue(configs["E20"].allow_window_missing)
        self.assertFalse(configs["E17A"].allow_window_missing)
        self.assertEqual(configs["E17A"].countries, configs["E15A"].countries)
        self.assertEqual(configs["E17B"].countries, configs["E15B"].countries)
        self.assertEqual(configs["E20"].countries, configs["E12"].countries)

    def test_default_configs_include_hybrid_follow_up_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E21", "E22A", "E22B", "E23", "E24"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E21"].models, ("GRUHybrid",))
        self.assertEqual(configs["E22A"].models, ("GRUHybrid",))
        self.assertEqual(configs["E22B"].models, ("GRUHybrid",))
        self.assertEqual(configs["E23"].models, ("GRUHybrid",))
        self.assertEqual(configs["E24"].models, ("GRUHybrid",))

        self.assertEqual(configs["E21"].repeat_random_seeds, (42, 52, 62))
        self.assertEqual(configs["E22A"].sample_filter_feature_group, "renewables")
        self.assertEqual(configs["E22B"].sample_filter_feature_group, "renewables")
        self.assertIsNone(configs["E23"].sample_filter_feature_group)
        self.assertIsNone(configs["E24"].sample_filter_feature_group)

        self.assertTrue(configs["E24"].allow_window_missing)
        self.assertFalse(configs["E23"].allow_window_missing)

        self.assertEqual(configs["E22A"].feature_group, "public")
        self.assertEqual(configs["E22B"].feature_group, "renewables")
        self.assertEqual(configs["E23"].feature_group, "public")
        self.assertEqual(configs["E24"].feature_group, "renewables")

        self.assertEqual(configs["E21"].countries, configs["E17B"].countries)
        self.assertEqual(configs["E22A"].countries, configs["E17A"].countries)
        self.assertEqual(configs["E22B"].countries, configs["E17B"].countries)
        self.assertEqual(configs["E23"].countries, configs["E12"].countries)
        self.assertEqual(configs["E24"].countries, configs["E20"].countries)

    def test_default_configs_include_e23_centered_follow_up_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E25", "E26", "E27", "E28"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E25"].models, ("GRUHybrid",))
        self.assertEqual(configs["E26"].models, ("GRUHybrid",))
        self.assertEqual(configs["E27"].models, ("GRUHybrid",))
        self.assertEqual(configs["E28"].models, ("GRUHybrid",))

        self.assertEqual(configs["E25"].repeat_random_seeds, (42, 52, 62))
        self.assertEqual(configs["E26"].sequence_loss, "focal")
        self.assertEqual(configs["E26"].focal_gamma, 2.0)
        self.assertEqual(configs["E27"].sequence_max_epochs, 45)
        self.assertEqual(configs["E27"].sequence_patience, 8)
        self.assertEqual(configs["E28"].sequence_max_epochs, 45)
        self.assertEqual(configs["E28"].sequence_patience, 8)
        self.assertTrue(configs["E28"].allow_window_missing)

        self.assertEqual(configs["E25"].feature_group, "public")
        self.assertEqual(configs["E26"].feature_group, "public")
        self.assertEqual(configs["E27"].feature_group, "public")
        self.assertEqual(configs["E28"].feature_group, "renewables")

        self.assertEqual(configs["E25"].countries, configs["E23"].countries)
        self.assertEqual(configs["E26"].countries, configs["E23"].countries)
        self.assertEqual(configs["E27"].countries, configs["E23"].countries)
        self.assertEqual(configs["E28"].countries, configs["E24"].countries)

    def test_default_configs_include_next_generation_hybrid_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E29", "E30", "E31", "E32"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E29"].models, ("GRUHybridAttn",))
        self.assertEqual(configs["E30"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E31"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E32"].models, ("GRUHybridGatedMultiTask",))

        for name in ("E29", "E30", "E31", "E32"):
            self.assertEqual(configs[name].sequence_max_epochs, 60)
            self.assertEqual(configs[name].sequence_patience, 10)
            self.assertEqual(configs[name].countries, configs["E23"].countries)
            self.assertEqual(configs[name].feature_group, "public")

        self.assertFalse(configs["E29"].use_mechanism_features)
        self.assertFalse(configs["E30"].use_mechanism_features)
        self.assertTrue(configs["E31"].use_mechanism_features)
        self.assertTrue(configs["E32"].use_mechanism_features)
        self.assertIsNone(configs["E31"].sequence_aux_target)
        self.assertEqual(configs["E32"].sequence_aux_target, "target_price")
        self.assertEqual(configs["E32"].sequence_aux_weight, 0.2)

    def test_default_configs_include_e30_e31_follow_up_meta_experiments(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E33", "E34", "E35", "E36"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E33"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E34"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E33"].repeat_random_seeds, (42, 52, 62))
        self.assertEqual(configs["E34"].repeat_random_seeds, (42, 52, 62))
        self.assertFalse(configs["E33"].use_mechanism_features)
        self.assertTrue(configs["E34"].use_mechanism_features)

        self.assertEqual(configs["E35"].meta_kind, "late_fusion")
        self.assertEqual(configs["E35"].meta_members, ("E30", "E31"))
        self.assertEqual(configs["E35"].models, ())

        self.assertEqual(configs["E36"].meta_kind, "calibration")
        self.assertEqual(configs["E36"].meta_members, ("E30", "E31", "E35"))
        self.assertEqual(configs["E36"].meta_calibration_method, "sigmoid")
        self.assertEqual(configs["E36"].models, ())

    def test_default_configs_include_stability_and_ensemble_follow_ups(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E37", "E38", "E39", "E40"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E37"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E38"].models, ("GRUHybridGated",))
        self.assertEqual(configs["E37"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E38"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E37"].sequence_max_epochs, 80)
        self.assertEqual(configs["E38"].sequence_max_epochs, 80)
        self.assertEqual(configs["E37"].sequence_patience, 12)
        self.assertEqual(configs["E38"].sequence_patience, 12)
        self.assertFalse(configs["E37"].use_mechanism_features)
        self.assertTrue(configs["E38"].use_mechanism_features)

        self.assertEqual(configs["E39"].meta_kind, "stacking")
        self.assertEqual(configs["E39"].meta_members, ("E30", "E31"))
        self.assertEqual(configs["E39"].models, ())

        self.assertEqual(configs["E40"].meta_kind, "cross_seed_ensemble")
        self.assertEqual(configs["E40"].meta_members, ("E33", "E34"))
        self.assertEqual(configs["E40"].models, ())

    def test_default_configs_include_current_task_classical_baselines(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E41", "E42", "E43", "E44"):
            self.assertIn(name, configs)
            self.assertEqual(configs[name].countries, configs["E30"].countries)
            self.assertEqual(configs[name].feature_group, "public")
            self.assertEqual(configs[name].window_hours, 168)
            self.assertEqual(configs[name].horizon_hours, 6)

        self.assertEqual(configs["E41"].models, ("LogisticRegression",))
        self.assertEqual(configs["E42"].models, ("XGBoost",))
        self.assertEqual(configs["E43"].models, ("CatBoost",))
        self.assertEqual(configs["E44"].models, ("LightGBM",))

    def test_default_configs_include_graph_temporal_follow_ups(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E45", "E46", "E47", "E48"):
            self.assertIn(name, configs)
            self.assertEqual(configs[name].countries, configs["E30"].countries)
            self.assertEqual(configs[name].feature_group, "public")
            self.assertEqual(configs[name].window_hours, 168)
            self.assertEqual(configs[name].horizon_hours, 6)
            self.assertEqual(configs[name].sequence_learning_rate, 5e-4)
            self.assertEqual(configs[name].sequence_max_epochs, 80)
            self.assertEqual(configs[name].sequence_patience, 12)

        self.assertEqual(configs["E45"].models, ("GRUMultiMarket",))
        self.assertEqual(configs["E46"].models, ("GraphTemporal",))
        self.assertEqual(configs["E47"].models, ("GraphTemporalHybrid",))
        self.assertEqual(configs["E48"].models, ("GraphTemporalHybrid",))
        self.assertFalse(configs["E45"].use_mechanism_features)
        self.assertFalse(configs["E46"].use_mechanism_features)
        self.assertTrue(configs["E47"].use_mechanism_features)
        self.assertTrue(configs["E48"].use_mechanism_features)
        self.assertEqual(configs["E48"].repeat_random_seeds, (42, 52, 62))

    def test_default_configs_include_e45_centered_follow_ups(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E49", "E50", "E51", "E52"):
            self.assertIn(name, configs)
            self.assertEqual(configs[name].countries, configs["E45"].countries)
            self.assertEqual(configs[name].feature_group, "public")
            self.assertEqual(configs[name].window_hours, 168)
            self.assertEqual(configs[name].horizon_hours, 6)

        self.assertEqual(configs["E49"].models, ("GRUMultiMarket",))
        self.assertEqual(configs["E49"].repeat_random_seeds, (42, 52, 62))
        self.assertEqual(configs["E49"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E49"].sequence_max_epochs, 100)
        self.assertEqual(configs["E49"].sequence_patience, 15)

        self.assertEqual(configs["E50"].models, ("GRUMultiMarketHybrid",))
        self.assertFalse(configs["E50"].use_mechanism_features)
        self.assertEqual(configs["E50"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E50"].sequence_max_epochs, 100)
        self.assertEqual(configs["E50"].sequence_patience, 15)

        self.assertEqual(configs["E51"].models, ("GRUMultiMarketHybrid",))
        self.assertTrue(configs["E51"].use_mechanism_features)
        self.assertEqual(configs["E51"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E51"].sequence_max_epochs, 100)
        self.assertEqual(configs["E51"].sequence_patience, 15)

        self.assertEqual(configs["E52"].meta_kind, "late_fusion")
        self.assertEqual(configs["E52"].meta_members, ("E45", "E35"))
        self.assertEqual(configs["E52"].models, ())

    def test_default_configs_include_e49_centered_follow_ups(self) -> None:
        configs = build_default_experiment_configs(Path("toy.csv"))

        for name in ("E53", "E54", "E55", "E56"):
            self.assertIn(name, configs)

        self.assertEqual(configs["E53"].models, ("GRUMultiMarketTargetAttn",))
        self.assertEqual(configs["E54"].models, ("GRUMultiMarketTemporalAttn",))
        self.assertEqual(configs["E55"].models, ("GRUMultiMarket",))
        self.assertEqual(configs["E56"].models, ())

        self.assertEqual(configs["E53"].countries, configs["E49"].countries)
        self.assertEqual(configs["E54"].countries, configs["E49"].countries)
        self.assertEqual(configs["E56"].countries, configs["E49"].countries)
        self.assertEqual(configs["E55"].countries, configs["E17B"].countries)

        self.assertEqual(configs["E53"].feature_group, "public")
        self.assertEqual(configs["E54"].feature_group, "public")
        self.assertEqual(configs["E55"].feature_group, "renewables")
        self.assertEqual(configs["E56"].feature_group, "public")

        self.assertEqual(configs["E53"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E53"].sequence_max_epochs, 100)
        self.assertEqual(configs["E53"].sequence_patience, 15)
        self.assertEqual(configs["E54"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E54"].sequence_max_epochs, 100)
        self.assertEqual(configs["E54"].sequence_patience, 15)
        self.assertEqual(configs["E55"].sequence_learning_rate, 5e-4)
        self.assertEqual(configs["E55"].sequence_max_epochs, 100)
        self.assertEqual(configs["E55"].sequence_patience, 15)
        self.assertEqual(configs["E55"].sample_filter_feature_group, "renewables")

        self.assertEqual(configs["E56"].meta_kind, "late_fusion")
        self.assertEqual(configs["E56"].meta_members, ("E49", "E44"))
