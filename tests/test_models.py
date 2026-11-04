"""tests/test_models.py — Unit tests for model architectures and losses."""
import pytest
import torch
import numpy as np
from src.models.unet import UNetHeightEstimator
from src.models.losses import CombinedHeightLoss, SmoothL1Loss, SSIMLoss, EdgeAwareLoss


class TestUNet:
    def setup_method(self):
        self.model = UNetHeightEstimator(in_channels=4, encoder_weights=None)
        self.model.eval()

    def test_output_shape(self):
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {out.shape}"

    def test_output_nonnegative(self):
        """Height predictions should be ≥ 0."""
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        assert (out >= 0).all(), "Negative height predictions found"

    def test_different_batch_sizes(self):
        for bs in [1, 4]:
            x = torch.randn(bs, 4, 256, 256)
            with torch.no_grad():
                out = self.model(x)
            assert out.shape[0] == bs


class TestLosses:
    def test_smooth_l1(self):
        loss_fn = SmoothL1Loss()
        pred = torch.rand(2, 1, 64, 64) * 50
        target = torch.rand(2, 1, 64, 64) * 50
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_ssim_range(self):
        loss_fn = SSIMLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 2

    def test_edge_loss(self):
        loss_fn = EdgeAwareLoss()
        pred = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_combined_loss(self):
        loss_fn = CombinedHeightLoss()
        pred = torch.rand(2, 1, 64, 64) * 30
        target = torch.rand(2, 1, 64, 64) * 30
        total, parts = loss_fn(pred, target)
        assert total.item() >= 0
        assert "smooth_l1" in parts
        assert "ssim" in parts
        assert "edge" in parts

    def test_perfect_prediction_low_loss(self):
        loss_fn = CombinedHeightLoss()
        pred = torch.ones(2, 1, 64, 64) * 20.0
        total, _ = loss_fn(pred, pred)
        assert total.item() < 0.1
