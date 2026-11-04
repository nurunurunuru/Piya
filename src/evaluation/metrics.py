"""
src/evaluation/metrics.py
Evaluation metrics for building height analysis.

Supports:
  - Regression:     RMSE, MAE, R², δ1/δ2, MAPE
  - Classification: Accuracy, mIoU, per-class IoU, F1, Confusion Matrix
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Class definitions (must match src/models/unet.py and dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

HEIGHT_THRESHOLDS = [3.0, 15.0, 30.0]   # bin edges in metres
NUM_HEIGHT_CLASSES = 4
CLASS_NAMES = ["Background", "Low", "Medium", "High"]


def height_to_class(height: np.ndarray) -> np.ndarray:
    """
    Convert continuous height array (metres) → class index array.

    Classes:
        0 = Background  (height < 3m)
        1 = Low         (3m – 15m)
        2 = Medium      (15m – 30m)
        3 = High        (> 30m)
    """
    classes = np.zeros_like(height, dtype=np.int64)
    classes[height >= HEIGHT_THRESHOLDS[0]] = 1   # Low
    classes[height >= HEIGHT_THRESHOLDS[1]] = 2   # Medium
    classes[height >= HEIGHT_THRESHOLDS[2]] = 3   # High
    return classes


# ─────────────────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:   np.ndarray,
    targets: np.ndarray,
    task:    str = "classification",
    mask:    Optional[np.ndarray] = None,
    height_threshold: float = 1.0,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for building height prediction.

    For classification mode:
        preds   = predicted class indices [N, H, W] or logits [N, C, H, W]
        targets = ground-truth class indices [N, H, W] OR continuous heights [N, 1, H, W]

    For regression mode:
        preds   = predicted heights [N, 1, H, W]
        targets = ground-truth heights [N, 1, H, W]
    """
    if task == "classification":
        return _compute_classification_metrics(preds, targets)
    else:
        return _compute_regression_metrics(preds, targets, mask, height_threshold)


def _compute_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics: Accuracy, mIoU, per-class IoU & F1.

    preds:   [N, C, H, W] logits  OR  [N, H, W] class indices
    targets: [N, H, W] class indices  OR  [N, 1, H, W] continuous heights
    """
    # Convert logits → class indices if needed
    if preds.ndim == 4 and preds.shape[1] > 1:
        pred_classes = preds.argmax(axis=1)            # [N, H, W]
    else:
        pred_classes = preds.squeeze().astype(np.int64)

    # Convert continuous heights → class indices if needed
    if targets.dtype in [np.float32, np.float64]:
        tgt_classes = height_to_class(targets.squeeze())
    else:
        tgt_classes = targets.squeeze().astype(np.int64)

    p = pred_classes.ravel()
    t = tgt_classes.ravel()

    n_classes = NUM_HEIGHT_CLASSES
    metrics = {}

    # Overall Accuracy
    accuracy = float((p == t).mean())
    metrics["accuracy"] = accuracy

    # Per-class IoU & F1
    iou_per_class = []
    f1_per_class  = []
    for c in range(n_classes):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())

        iou = tp / (tp + fp + fn + 1e-8)
        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        iou_per_class.append(iou)
        f1_per_class.append(f1)
        metrics[f"iou_{CLASS_NAMES[c].lower()}"] = float(iou)
        metrics[f"f1_{CLASS_NAMES[c].lower()}"]  = float(f1)

    metrics["miou"]    = float(np.mean(iou_per_class))
    metrics["macro_f1"] = float(np.mean(f1_per_class))

    # Building-only mIoU (classes 1-3, excluding background)
    metrics["building_miou"] = float(np.mean(iou_per_class[1:]))

    return metrics


def _compute_regression_metrics(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    Optional[np.ndarray] = None,
    height_threshold: float = 1.0,
) -> Dict[str, float]:
    """Compute regression metrics: RMSE, MAE, R², δ1, δ2, MAPE."""
    try:
        from sklearn.metrics import r2_score
    except ImportError:
        r2_score = None

    preds   = preds.squeeze().ravel().astype(np.float32)
    targets = targets.squeeze().ravel().astype(np.float32)

    if mask is not None:
        valid = mask.ravel().astype(bool)
    else:
        valid = targets > height_threshold

    if valid.sum() == 0:
        logger.warning("No valid pixels found for evaluation!")
        return {"rmse": 999.0, "mae": 999.0, "r2": 0.0, "delta1": 0.0}

    p = preds[valid]
    t = targets[valid]

    rmse   = float(np.sqrt(np.mean((p - t) ** 2)))
    mae    = float(np.mean(np.abs(p - t)))
    r2     = float(r2_score(t, p)) if r2_score else 0.0
    rel    = np.maximum(p / (t + 1e-8), t / (p + 1e-8))
    delta1 = float((rel < 1.25).mean())
    delta2 = float((rel < 1.25 ** 2).mean())
    mape   = float(np.mean(np.abs(p - t) / (t + 1e-8)) * 100)

    return {
        "rmse":     rmse,
        "mae":      mae,
        "r2":       r2,
        "mape":     mape,
        "delta1":   delta1,
        "delta2":   delta2,
        "n_pixels": int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty-print metrics table to console (uses rich if available)."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value",  style="green", justify="right")

        for name, value in metrics.items():
            if isinstance(value, float):
                table.add_row(name.upper(), f"{value:.4f}")
            else:
                table.add_row(name.upper(), str(value))

        console.print(table)

    except ImportError:
        # Fallback plain-text table
        print(f"\n{'─'*40}")
        print(f"  {title}")
        print(f"{'─'*40}")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {name.upper():<20} {value:.4f}")
            else:
                print(f"  {name.upper():<20} {value}")
        print(f"{'─'*40}\n")
