"""tests/test_metrics.py — Unit tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics


def test_perfect_prediction():
    arr = np.random.rand(100, 100) * 30 + 1
    m = compute_metrics(arr, arr)
    assert m["rmse"] < 0.001
    assert m["mae"]  < 0.001
    assert m["r2"]   > 0.999


def test_rmse_calculation():
    pred   = np.array([[[5.0, 10.0, 15.0]]])
    target = np.array([[[4.0,  9.0, 14.0]]])
    m = compute_metrics(pred, target)
    assert abs(m["rmse"] - 1.0) < 0.01


def test_empty_prediction():
    """All-zero predictions → handled gracefully."""
    pred   = np.zeros((10, 10))
    target = np.zeros((10, 10))
    m = compute_metrics(pred, target)
    # With all zeros, falls into "no valid pixels" path
    assert "rmse" in m


def test_delta_accuracy():
    pred   = np.ones((50, 50)) * 10.0
    target = np.ones((50, 50)) * 11.0   # 9% off
    m = compute_metrics(pred, target)
    assert m["delta1"] == pytest.approx(1.0, abs=0.01)  # Within 25%


def test_metrics_keys():
    pred   = np.random.rand(20, 20) * 30 + 1
    target = np.random.rand(20, 20) * 30 + 1
    m = compute_metrics(pred, target)
    for key in ["rmse", "mae", "r2", "delta1", "delta2", "mape"]:
        assert key in m
