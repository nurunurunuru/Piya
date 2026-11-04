"""
src/models/ensemble.py
Weighted ensemble of UNet and Swin Transformer predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .unet import UNetHeightEstimator
from .transformer import SwinTransformerHeightEstimator

logger = logging.getLogger(__name__)


class EnsembleHeightEstimator(nn.Module):
    """
    Weighted average ensemble of multiple height estimator models.
    Weights are learned or set manually.
    """

    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        weights: Optional[List[float]] = None,
        in_channels: int = 12,
    ):
        super().__init__()

        if models is None:
            models = [
                UNetHeightEstimator(in_channels=in_channels),
                SwinTransformerHeightEstimator(in_channels=in_channels),
            ]

        self.models  = nn.ModuleList(models)
        n = len(models)
        w = weights if weights else [1.0 / n] * n
        self.register_buffer("weights", torch.tensor(w, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([m(x) for m in self.models], dim=0)   # [N, B, 1, H, W]
        w = self.weights.view(-1, 1, 1, 1, 1)
        return (preds * w).sum(dim=0)                              # [B, 1, H, W]

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: Dict[str, str],
        weights: List[float],
        in_channels: int = 12,
        device: str = "cpu",
    ) -> "EnsembleHeightEstimator":
        """Load individual models from checkpoints and build ensemble."""
        models = []
        for name, ckpt_path in checkpoint_paths.items():
            if name == "unet":
                m = UNetHeightEstimator(in_channels=in_channels)
            elif name == "swin":
                m = SwinTransformerHeightEstimator(in_channels=in_channels)
            else:
                raise ValueError(f"Unknown model: {name}")

            state = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(state.get("model_state_dict", state))
            m.eval()
            models.append(m)
            logger.info(f"Loaded {name} from {ckpt_path}")

        return cls(models=models, weights=weights, in_channels=in_channels)
