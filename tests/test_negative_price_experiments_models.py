from __future__ import annotations

import unittest
from unittest.mock import patch

from negative_price_experiments.models import HAS_TORCH, build_sequence_model, get_preferred_torch_device


@unittest.skipUnless(HAS_TORCH, "torch is required for device-selection tests")
class NegativePriceModelsDeviceTest(unittest.TestCase):
    def test_prefers_cuda_over_mps(self) -> None:
        with patch("negative_price_experiments.models.has_cuda", return_value=True), patch(
            "negative_price_experiments.models.has_mps", return_value=True
        ):
            self.assertEqual(get_preferred_torch_device().type, "cuda")

    def test_uses_mps_when_cuda_is_unavailable(self) -> None:
        with patch("negative_price_experiments.models.has_cuda", return_value=False), patch(
            "negative_price_experiments.models.has_mps", return_value=True
        ):
            self.assertEqual(get_preferred_torch_device().type, "mps")

    def test_falls_back_to_cpu(self) -> None:
        with patch("negative_price_experiments.models.has_cuda", return_value=False), patch(
            "negative_price_experiments.models.has_mps", return_value=False
        ):
            self.assertEqual(get_preferred_torch_device().type, "cpu")

    def test_builds_patchtst_sequence_model(self) -> None:
        model = build_sequence_model(
            "PatchTST",
            input_dim=4,
            use_country_embedding=False,
            num_countries=0,
        )
        self.assertEqual(model.__class__.__name__, "PatchTSTClassifier")
