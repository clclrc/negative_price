from __future__ import annotations

import unittest
from unittest.mock import patch

from negative_price_experiments.models import HAS_TORCH, build_sequence_model, get_preferred_torch_device

if HAS_TORCH:
    import torch

    from negative_price_experiments.models import BinaryFocalLossWithLogits


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

    def test_builds_gru_hybrid_sequence_model(self) -> None:
        model = build_sequence_model(
            "GRUHybrid",
            input_dim=4,
            use_country_embedding=True,
            num_countries=3,
            tabular_dim=6,
        )
        self.assertEqual(model.__class__.__name__, "GRUHybridClassifier")

    def test_builds_next_generation_hybrid_sequence_models(self) -> None:
        attn = build_sequence_model(
            "GRUHybridAttn",
            input_dim=4,
            use_country_embedding=True,
            num_countries=3,
            tabular_dim=6,
        )
        gated = build_sequence_model(
            "GRUHybridGated",
            input_dim=4,
            use_country_embedding=True,
            num_countries=3,
            tabular_dim=6,
        )
        multitask = build_sequence_model(
            "GRUHybridGatedMultiTask",
            input_dim=4,
            use_country_embedding=True,
            num_countries=3,
            tabular_dim=6,
        )
        self.assertEqual(attn.__class__.__name__, "GRUHybridAttnClassifier")
        self.assertEqual(gated.__class__.__name__, "GRUHybridGatedClassifier")
        self.assertEqual(multitask.__class__.__name__, "GRUHybridGatedMultiTaskClassifier")

    def test_multitask_hybrid_forward_returns_logits_and_auxiliary_output(self) -> None:
        model = build_sequence_model(
            "GRUHybridGatedMultiTask",
            input_dim=4,
            use_country_embedding=True,
            num_countries=3,
            tabular_dim=6,
        )
        x = torch.randn(2, 8, 4)
        country_idx = torch.tensor([0, 1], dtype=torch.long)
        tabular_x = torch.randn(2, 6)

        logits, aux = model(x, country_idx, tabular_x)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(aux.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(aux).all())


@unittest.skipUnless(HAS_TORCH, "torch is required for focal loss tests")
class NegativePriceModelLossTest(unittest.TestCase):
    def test_binary_focal_loss_with_logits_returns_finite_scalar(self) -> None:
        logits = torch.tensor([0.2, -1.5, 2.1, -0.7], dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        pos_weight = torch.tensor([3.0], dtype=torch.float32)
        loss_fn = BinaryFocalLossWithLogits(gamma=2.0, pos_weight=pos_weight)

        loss = loss_fn(logits, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.ndim, 0)
