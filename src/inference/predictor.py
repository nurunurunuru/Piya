"""
src/inference/predictor.py
Sliding-window inference over full Dhaka city extent.
Stitches patches back into a city-wide height raster.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.preprocessor import PatchGenerator, save_raster
from ..data.dataset import InferencePatchDataset

logger = logging.getLogger(__name__)


class CityPredictor:
    """
    Runs inference on a full city GeoTIFF using sliding-window approach.
    Averages overlapping predictions for smooth boundaries.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 4,
        mean: Optional[np.ndarray] = None,
        std:  Optional[np.ndarray] = None,
    ):
        self.model      = model.eval().to(device)
        self.device     = device
        self.patch_gen  = PatchGenerator(patch_size=patch_size, stride=stride,
                                          min_building_pct=0.0)
        self.patch_size = patch_size
        self.stride     = stride
        self.batch_size = batch_size
        self.mean       = mean
        self.std        = std

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        mean: np.ndarray,
        std:  np.ndarray,
        device: str = "cpu",
    ) -> "CityPredictor":
        """Load model weights from checkpoint."""
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info(f"Model loaded from {checkpoint_path}")
        return cls(model=model, device=device, mean=mean, std=std)

    @torch.no_grad()
    def predict(self, image_path: str, output_path: str) -> str:
        """
        Run full-city height prediction.

        Args:
            image_path:  Path to city-wide Sentinel-2 GeoTIFF.
            output_path: Where to save the predicted height GeoTIFF.

        Returns:
            Path to saved prediction raster.
        """
        logger.info(f"Loading image: {image_path}")
        patches, positions, meta = self.patch_gen.generate_inference_patches(image_path)

        # Set up normalization
        if self.mean is None:
            img_data = np.load(image_path) if image_path.endswith(".npy") else None
            self.mean = np.zeros(meta["count"], dtype=np.float32)
            self.std  = np.ones(meta["count"],  dtype=np.float32)

        # Build inference dataset
        dataset = InferencePatchDataset(patches, positions, self.mean, self.std)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Prepare canvas for prediction averaging
        H, W = meta["height"], meta["width"]
        pred_sum   = np.zeros((H, W), dtype=np.float64)
        pred_count = np.zeros((H, W), dtype=np.float64)

        logger.info(f"Running inference on {len(patches)} patches...")

        for batch in tqdm(loader, desc="Predicting"):
            images    = batch["image"].to(self.device)
            positions_batch = batch["position"]

            preds = self.model(images).squeeze(1).cpu().numpy()  # [B, H, W]

            for i, (row, col) in enumerate(zip(*positions_batch)):
                row, col = int(row), int(col)
                pred_patch = preds[i]

                r_end = min(row + self.patch_size, H)
                c_end = min(col + self.patch_size, W)
                ph    = r_end - row
                pw    = c_end - col

                pred_sum[row:r_end, col:c_end]   += pred_patch[:ph, :pw]
                pred_count[row:r_end, col:c_end] += 1.0

        # Average overlapping predictions
        pred_count = np.maximum(pred_count, 1.0)
        height_map = (pred_sum / pred_count).astype(np.float32)
        height_map = np.clip(height_map, 0.0, 250.0)

        # Save as GeoTIFF
        out_meta = meta.copy()
        out_meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_raster(height_map[np.newaxis], out_meta, output_path)

        logger.info(f"Prediction saved → {output_path}")
        logger.info(f"  Height stats: min={height_map.min():.1f}m, "
                    f"max={height_map.max():.1f}m, "
                    f"mean={height_map.mean():.1f}m")
        return output_path


class PostProcessor:
    """Smoothing and cleanup of raw prediction rasters."""

    def smooth(self, height_array: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing while preserving zero regions."""
        from scipy.ndimage import gaussian_filter
        mask = height_array > 0
        smoothed = gaussian_filter(height_array, sigma=sigma)
        smoothed[~mask] = 0.0
        return smoothed

    def remove_isolated_pixels(
        self, height_array: np.ndarray, min_area: int = 5
    ) -> np.ndarray:
        """Remove small isolated height predictions (noise)."""
        from scipy import ndimage
        binary = height_array > 1.0
        labeled, _ = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(1, labeled.max() + 1))
        mask  = np.zeros_like(binary)
        for i, size in enumerate(sizes):
            if size >= min_area:
                mask[labeled == i + 1] = True
        return height_array * mask

    def clip_to_buildings(
        self, height_array: np.ndarray, building_mask: np.ndarray
    ) -> np.ndarray:
        """Zero out predictions outside known building footprints."""
        return height_array * (building_mask > 0)
