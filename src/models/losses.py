"""
src/models/losses.py
Loss functions for building height analysis.

Supports:
  - Regression:     CombinedHeightLoss (SmoothL1 + SSIM + EdgeAware)
  - Classification: HeightClassificationLoss (CrossEntropyLoss with class weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Classification Loss
# ─────────────────────────────────────────────────────────────────────────────

class HeightClassificationLoss(nn.Module):
    """
    Weighted Cross-Entropy loss for Low / Medium / High building classification.

    Class weights account for class imbalance:
        Background and Low buildings are abundant in urban satellite imagery,
        while High buildings are rare → assign higher weight to rare classes.

    Classes:
        0 = Background  (height < 3m)
        1 = Low         (3m – 15m)
        2 = Medium      (15m – 30m)
        3 = High        (> 30m)
    """

    # Default class weights (inverse-frequency heuristic for Dhaka)
    DEFAULT_WEIGHTS = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)

    def __init__(
        self,
        class_weights: torch.Tensor = None,
        ignore_index:  int = -1,
        dice_weight:   float = 0.3,
        ce_weight:     float = 0.7,
    ):
        super().__init__()
        weights = class_weights if class_weights is not None else self.DEFAULT_WEIGHTS
        self.register_buffer("class_weights", weights)
        self.ignore_index = ignore_index
        self.dice_weight  = dice_weight
        self.ce_weight    = ce_weight

    def forward(
        self,
        logits:  torch.Tensor,   # [B, C, H, W]
        targets: torch.Tensor,   # [B, H, W] — class indices (int64)
    ):
        """
        Args:
            logits:  Raw model output [B, num_classes, H, W]
            targets: Ground-truth class index map [B, H, W] dtype=torch.long

        Returns:
            (total_loss, parts_dict)
        """
        num_classes = logits.shape[1]

        # ── Cross-Entropy ────────────────────────────────────────────────────
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.device),
            ignore_index=self.ignore_index,
        )

        # ── Dice Loss ────────────────────────────────────────────────────────
        probs = F.softmax(logits, dim=1)                   # [B, C, H, W]
        # One-hot encode targets
        tgt_oh = F.one_hot(targets.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(num_classes):
            p = probs[:, c]
            t = tgt_oh[:, c]
            intersection = (p * t).sum()
            dice_c = 1.0 - (2 * intersection + 1) / (p.sum() + t.sum() + 1)
            dice_per_class.append(dice_c)
        dice = torch.stack(dice_per_class).mean()

        total = self.ce_weight * ce + self.dice_weight * dice
        return total, {"ce": ce.item(), "dice": dice.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Regression Losses (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class SmoothL1Loss(nn.Module):
    """Huber loss — less sensitive to outlier heights than MSE."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = (target > 0).float()
        return self.loss_fn(pred * mask, target * mask)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss — preserves spatial structure."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _create_window(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        return window.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2
        w = self.window.to(x.device)
        mu_x = F.conv2d(x, w, padding=self.window_size // 2, groups=1)
        mu_y = F.conv2d(y, w, padding=self.window_size // 2, groups=1)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sigma_x2 = F.conv2d(x**2, w, padding=self.window_size // 2) - mu_x2
        sigma_y2 = F.conv2d(y**2, w, padding=self.window_size // 2) - mu_y2
        sigma_xy = F.conv2d(x * y, w, padding=self.window_size // 2) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        return (num / (den + 1e-8)).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        max_h = target.max().clamp(min=1.0)
        return 1.0 - self._ssim(pred / max_h, target / max_h)


class EdgeAwareLoss(nn.Module):
    """Penalizes blurry building edges using Sobel-based gradient difference."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._gradient(pred), self._gradient(target))


class CombinedHeightLoss(nn.Module):
    """
    Weighted combination for regression:
        L = w1 * SmoothL1 + w2 * SSIM + w3 * EdgeAware
    """

    def __init__(
        self,
        smooth_l1_weight: float = 0.85,
        ssim_weight:      float = 0.10,
        edge_weight:      float = 0.05,
    ):
        super().__init__()
        self.smooth_l1 = SmoothL1Loss()
        self.ssim      = SSIMLoss()
        self.edge      = EdgeAwareLoss()
        self.w1 = smooth_l1_weight
        self.w2 = ssim_weight
        self.w3 = edge_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1   = self.smooth_l1(pred, target)
        ssim = self.ssim(pred, target)
        edge = self.edge(pred, target)
        total = self.w1 * l1 + self.w2 * ssim + self.w3 * edge
        return total, {"smooth_l1": l1.item(), "ssim": ssim.item(), "edge": edge.item()}
