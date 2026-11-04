"""
src/data/dataset.py
PyTorch Dataset classes for building height regression.

Datasets supported:
  1. HeightPatchDataset      — pre-extracted .npy patch pairs (images + labels)
  2. SyntheticDhakaDataset   — synthetic training data (no downloads needed)
  3. DFC2023Dataset          — IEEE Data Fusion Contest 2023 (building heights)
  4. OSMHeightDataset        — OSM-derived building heights + Sentinel-2 tiles
  5. DatasetFactory          — unified factory that selects / merges datasets
"""

import logging
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .augmentation import apply_transforms, get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HeightPatchDataset  — pre-extracted .npy patches
# ═══════════════════════════════════════════════════════════════════════════════

class HeightPatchDataset(Dataset):
    """
    Loads pre-extracted .npy patch pairs produced by PatchGenerator.

    Expected directory layout:
        root/
          {split}/
            images/patch_RRRRR_CCCCC.npy   → float32 [C, H, W]
            labels/patch_RRRRR_CCCCC.npy   → float32 [1, H, W]
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        normalize: bool = True,
        augment: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        max_height: float = 150.0,
    ):
        self.root = Path(root) / split
        self.split = split
        self.normalize = normalize
        self.augment = augment and split == "train"
        self.max_height = max_height

        img_dir = self.root / "images"
        lbl_dir = self.root / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                f"Run pipeline/stage_02_preprocess.py first, or use SyntheticDhakaDataset."
            )

        self.image_paths = sorted(img_dir.glob("*.npy"))
        self.label_paths = sorted(lbl_dir.glob("*.npy"))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .npy patches found in {img_dir}")

        assert len(self.image_paths) == len(self.label_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
        )

        self.mean = mean
        self.std = std
        if normalize and self.mean is None:
            self._compute_stats()

        self.transform = get_train_transforms() if self.augment else get_val_transforms()
        logger.info(f"[HeightPatchDataset/{split}] {len(self)} patches")

    def _compute_stats(self):
        """Compute per-channel mean/std over random subset of patches."""
        logger.info("Computing channel statistics from training patches...")
        n_sample = min(200, len(self.image_paths))
        indices = np.random.choice(len(self.image_paths), n_sample, replace=False)
        all_pixels = []
        for i in indices:
            img = np.load(self.image_paths[i])
            all_pixels.append(img.reshape(img.shape[0], -1))
        pixels = np.concatenate(all_pixels, axis=1)
        self.mean = pixels.mean(axis=1).astype(np.float32)
        self.std = pixels.std(axis=1).clip(min=1e-6).astype(np.float32)
        logger.info(f"Stats ready — {pixels.shape[0]} channels, {n_sample} sample patches.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = np.load(self.image_paths[idx]).astype(np.float32)   # [C, H, W]
        label = np.load(self.label_paths[idx]).astype(np.float32)   # [1, H, W]

        # Clamp label to valid height range
        label = np.clip(label, 0.0, self.max_height)

        if self.augment:
            image, label = apply_transforms(image, label, self.transform)

        if self.normalize and self.mean is not None:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return {
            "image":    torch.from_numpy(image),
            "label":    torch.from_numpy(label),
            "filename": self.image_paths[idx].name,
            "source":   "npy_patch",
        }

    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.mean, self.std


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SyntheticDhakaDataset — works WITHOUT any downloaded data
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticDhakaDataset(Dataset):
    """
    Generates realistic synthetic Sentinel-2 + height patches for Dhaka City.

    Uses physically-motivated spectral signatures for urban/vegetation/water
    and height distributions calibrated to real Dhaka zone statistics.
    This dataset lets you train and test the full pipeline with ZERO downloads.

    Sentinel-2 channel order (12 bands + 5 indices = 17 channels):
        0:B02  1:B03  2:B04  3:B05  4:B06  5:B07
        6:B08  7:B8A  8:B09  9:B11  10:B12
        11:NDVI 12:NDBI 13:MNDWI 14:Shadow 15:Texture 16:Elevation-proxy
    """

    # Sentinel-2 reflectance signature per land-cover class (normalized 0-1)
    SPECTRAL_SIGNATURES = {
        #           B02   B03   B04   B05   B06   B07   B08   B8A   B09   B11   B12
        "building": [0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.27, 0.27, 0.22, 0.30, 0.28],
        "road":     [0.16, 0.18, 0.20, 0.21, 0.22, 0.23, 0.24, 0.24, 0.20, 0.28, 0.26],
        "veg":      [0.04, 0.08, 0.05, 0.12, 0.28, 0.38, 0.42, 0.44, 0.38, 0.12, 0.07],
        "water":    [0.07, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02],
        "shadow":   [0.03, 0.04, 0.03, 0.04, 0.05, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03],
        "bare":     [0.15, 0.17, 0.19, 0.20, 0.21, 0.22, 0.22, 0.23, 0.20, 0.25, 0.23],
    }

    # Height distributions per Dhaka zone (mean, std, max)
    ZONE_HEIGHT_PARAMS = {
        "Gulshan":   {"mean": 28.5, "std": 12.0, "max": 95},
        "Banani":    {"mean": 22.1, "std": 10.0, "max": 78},
        "Dhanmondi": {"mean": 14.3, "std": 6.5,  "max": 48},
        "Motijheel": {"mean": 32.7, "std": 15.0, "max": 120},
        "OldDhaka":  {"mean": 7.2,  "std": 3.0,  "max": 22},
        "Mirpur":    {"mean": 11.4, "std": 5.0,  "max": 42},
        "Uttara":    {"mean": 16.8, "std": 7.0,  "max": 55},
        "Mohakhali": {"mean": 19.6, "std": 8.5,  "max": 68},
    }

    IN_CHANNELS = 17

    # Height class thresholds in metres (must match unet.py and metrics.py)
    HEIGHT_THRESHOLDS = [3.0, 15.0, 30.0]   # → 4 classes: BG, Low, Med, High

    def __init__(
        self,
        num_samples: int = 2000,
        patch_size: int = 256,
        split: str = "train",
        augment: bool = True,
        seed: int = 42,
        task: str = "classification",   # "regression" | "classification"
    ):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.split = split
        self.augment = augment and split == "train"
        self.seed = seed
        self.task = task

        # Train/val/test deterministic split via seed offset
        split_offset = {"train": 0, "val": 100_000, "test": 200_000}
        self.seed_base = seed + split_offset.get(split, 0)

        self.transform = get_train_transforms() if self.augment else get_val_transforms()
        self.zone_names = list(self.ZONE_HEIGHT_PARAMS.keys())

        logger.info(f"[SyntheticDhakaDataset/{split}] {num_samples} samples (seed={seed}, task={task})")

    def __len__(self) -> int:
        return self.num_samples

    def _make_building_mask(
        self, rng: np.random.Generator, H: int, W: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a realistic building footprint mask and height map.
        Simulates block-like urban patterns with varied building sizes.
        """
        mask = np.zeros((H, W), dtype=np.float32)
        heights = np.zeros((H, W), dtype=np.float32)

        zone = rng.choice(self.zone_names)
        z = self.ZONE_HEIGHT_PARAMS[zone]
        n_buildings = int(rng.integers(20, 80))

        for _ in range(n_buildings):
            bw = int(rng.integers(8, 50))
            bh = int(rng.integers(8, 50))
            x  = int(rng.integers(0, max(1, W - bw)))
            y  = int(rng.integers(0, max(1, H - bh)))

            # Building height from zone distribution
            h = float(np.clip(rng.normal(z["mean"], z["std"]), 3, z["max"]))
            mask[y:y+bh, x:x+bw]   = 1.0
            heights[y:y+bh, x:x+bw] = h

        return mask, heights

    def _make_spectral_image(
        self,
        rng: np.random.Generator,
        building_mask: np.ndarray,
        height_map: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Generate a synthetic Sentinel-2 image from building mask.
        Each pixel's spectrum is a mixture of land-cover signatures.
        """
        n_bands = 11
        image = np.zeros((n_bands, H, W), dtype=np.float32)

        # Base: bare soil + vegetation background
        veg_frac = rng.uniform(0.1, 0.4)
        base = np.array(
            np.array(self.SPECTRAL_SIGNATURES["bare"]) * (1 - veg_frac)
            + np.array(self.SPECTRAL_SIGNATURES["veg"]) * veg_frac
        )
        for b in range(n_bands):
            image[b] = base[b] + rng.normal(0, 0.01, (H, W)).astype(np.float32)

        # Add buildings
        bld_sig = np.array(self.SPECTRAL_SIGNATURES["building"])
        for b in range(n_bands):
            image[b] = np.where(
                building_mask > 0,
                bld_sig[b] + rng.normal(0, 0.008, (H, W)),
                image[b],
            )

        # Add shadows (south side of buildings, simplified)
        shadow_mask = np.roll(building_mask, shift=3, axis=0) * (1 - building_mask)
        shd_sig = np.array(self.SPECTRAL_SIGNATURES["shadow"])
        for b in range(n_bands):
            image[b] = np.where(
                shadow_mask > 0,
                shd_sig[b] + rng.normal(0, 0.005, (H, W)),
                image[b],
            )

        # Water body (random chance)
        if rng.random() < 0.3:
            wx = int(rng.integers(0, W // 2))
            wy = int(rng.integers(0, H // 2))
            ww = int(rng.integers(20, 80))
            wh = int(rng.integers(15, 60))
            wat_sig = np.array(self.SPECTRAL_SIGNATURES["water"])
            for b in range(n_bands):
                image[b, wy:wy+wh, wx:wx+ww] = (
                    wat_sig[b] + rng.normal(0, 0.003, (wh, min(ww, W - wx)))
                )

        # Roads
        if rng.random() < 0.7:
            road_y = int(rng.integers(H // 4, 3 * H // 4))
            road_w = int(rng.integers(3, 10))
            road_sig = np.array(self.SPECTRAL_SIGNATURES["road"])
            for b in range(n_bands):
                image[b, road_y:road_y+road_w, :] = (
                    road_sig[b] + rng.normal(0, 0.006, (road_w, W))
                )

        image = np.clip(image, 0.0, 1.0)
        return image

    def _compute_indices(self, image: np.ndarray) -> np.ndarray:
        """Compute spectral indices and stack as extra channels."""
        nir  = image[6]   # B08
        red  = image[2]   # B04
        swir = image[9]   # B11
        grn  = image[1]   # B03
        blu  = image[0]   # B02

        ndvi   = (nir - red)  / (nir + red + 1e-8)
        ndbi   = (swir - nir) / (swir + nir + 1e-8)
        mndwi  = (grn - swir) / (grn + swir + 1e-8)
        shadow = 1.0 - (blu + grn + nir) / 3.0

        # Simple texture: local variance of NIR
        from scipy.ndimage import uniform_filter
        nir_mean = uniform_filter(nir, size=5)
        nir_sq   = uniform_filter(nir**2, size=5)
        texture  = np.sqrt(np.maximum(nir_sq - nir_mean**2, 0))

        # Elevation proxy (height smoothed + noise)
        return np.stack([ndvi, ndbi, mndwi, shadow, texture], axis=0)

    def _height_to_class(self, height_map: np.ndarray) -> np.ndarray:
        """Bin continuous heights [H, W] → class indices [H, W] (int64)."""
        cls = np.zeros_like(height_map, dtype=np.int64)
        cls[height_map >= self.HEIGHT_THRESHOLDS[0]] = 1  # Low
        cls[height_map >= self.HEIGHT_THRESHOLDS[1]] = 2  # Medium
        cls[height_map >= self.HEIGHT_THRESHOLDS[2]] = 3  # High
        return cls

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed_base + idx)
        H = W = self.patch_size

        # 1. Generate building mask and ground-truth height map
        building_mask, height_map = self._make_building_mask(rng, H, W)

        # 2. Generate synthetic Sentinel-2 image
        image_11band = self._make_spectral_image(rng, building_mask, height_map, H, W)

        # 3. Compute spectral indices (5 channels)
        indices = self._compute_indices(image_11band)

        # 4. Elevation proxy channel
        elevation_proxy = (height_map / 100.0 + rng.normal(0, 0.02, (H, W))).astype(np.float32)

        # 5. Stack into full feature tensor [17, H, W]
        image = np.concatenate(
            [image_11band, indices, elevation_proxy[np.newaxis]], axis=0
        ).astype(np.float32)

        # 6. Prepare label based on task
        if self.task == "classification":
            label = self._height_to_class(height_map)    # [H, W] int64
        else:
            label = height_map[np.newaxis].astype(np.float32)  # [1, H, W] float32

        # 7. Augmentation (only for float labels; skip for int class maps)
        if self.augment and self.task == "regression":
            image, label = apply_transforms(image, label, self.transform)

        if self.task == "classification":
            return {
                "image":    torch.from_numpy(image),
                "label":    torch.from_numpy(label),          # [H, W] int64
                "filename": f"synthetic_{idx:06d}.npy",
                "source":   "synthetic",
            }
        return {
            "image":    torch.from_numpy(image),
            "label":    torch.from_numpy(label),              # [1, H, W] float32
            "filename": f"synthetic_{idx:06d}.npy",
            "source":   "synthetic",
        }

    @property
    def mean(self) -> np.ndarray:
        """Pre-computed approximate channel means for Sentinel-2 urban imagery."""
        return np.array([
            0.09, 0.11, 0.12, 0.13, 0.16, 0.19,  # B02–B07
            0.22, 0.22, 0.19, 0.18, 0.15,          # B08–B12
            0.20, 0.02, 0.05, 0.55, 0.08, 0.10,    # indices + elev
        ], dtype=np.float32)

    @property
    def std(self) -> np.ndarray:
        """Pre-computed approximate channel stds."""
        return np.array([
            0.06, 0.06, 0.07, 0.07, 0.08, 0.10,
            0.11, 0.11, 0.09, 0.08, 0.07,
            0.25, 0.18, 0.20, 0.20, 0.06, 0.12,
        ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DFC2023Dataset — IEEE Data Fusion Contest 2023
# ═══════════════════════════════════════════════════════════════════════════════

class DFC2023Dataset(Dataset):
    """
    IEEE GRSS Data Fusion Contest 2023 — Track 1: Building Height Estimation.
    Uses Sentinel-1 SAR + Sentinel-2 optical imagery with building height labels.

    Dataset structure (after download from IEEE DataPort):
        root/
          train/
            sar/    <tile_id>_S1.tif
            opt/    <tile_id>_S2.tif
            label/  <tile_id>_DSM.tif
          val/  ...
          test/ ...

    Download:
        https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest

    Reference:
        Schmitt et al., "SEN12MS — A curated dataset of georeferenced
        multi-spectral Sentinel-1/2 imagery," ISPRS 2019.
    """

    N_OPT_BANDS = 11     # Sentinel-2 bands
    N_SAR_BANDS = 2      # Sentinel-1: VV + VH
    IN_CHANNELS  = N_OPT_BANDS + N_SAR_BANDS  # = 13

    def __init__(
        self,
        root: str,
        split: str = "train",
        use_sar: bool = True,
        patch_size: int = 256,
        augment: bool = True,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.use_sar = use_sar
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.normalize = normalize

        split_dir = self.root / split
        self._validate_structure(split_dir)

        self.opt_paths   = sorted((split_dir / "opt").glob("*_S2.tif"))
        self.label_paths = sorted((split_dir / "label").glob("*_DSM.tif"))
        if use_sar:
            self.sar_paths = sorted((split_dir / "sar").glob("*_S1.tif"))
        else:
            self.sar_paths = []

        if len(self.opt_paths) == 0:
            raise FileNotFoundError(
                f"No Sentinel-2 tiles found in {split_dir / 'opt'}\n"
                f"Please download DFC2023 from: https://ieee-dataport.org/competitions/"
                f"2023-ieee-grss-data-fusion-contest"
            )

        # Default normalization stats for DFC2023
        self.mean = mean if mean is not None else self._default_mean()
        self.std  = std  if std  is not None else self._default_std()

        self.transform = get_train_transforms() if self.augment else get_val_transforms()
        logger.info(f"[DFC2023Dataset/{split}] {len(self)} tiles found.")

    @staticmethod
    def _validate_structure(split_dir: Path):
        required = ["opt", "label"]
        for d in required:
            if not (split_dir / d).exists():
                raise FileNotFoundError(
                    f"DFC2023 directory missing: {split_dir / d}\n"
                    f"Expected structure: root/{split_dir.name}/opt/, label/, sar/"
                )

    def _default_mean(self) -> np.ndarray:
        # DFC2023 approximate means (Sentinel-2 surface reflectance / 10000)
        s2_mean = np.array([0.103, 0.125, 0.131, 0.150, 0.185, 0.212,
                             0.232, 0.240, 0.215, 0.190, 0.141], dtype=np.float32)
        if self.use_sar:
            s1_mean = np.array([-12.5, -19.0], dtype=np.float32)  # dB
            return np.concatenate([s2_mean, s1_mean])
        return s2_mean

    def _default_std(self) -> np.ndarray:
        s2_std = np.array([0.045, 0.050, 0.058, 0.062, 0.072, 0.082,
                            0.090, 0.092, 0.080, 0.075, 0.060], dtype=np.float32)
        if self.use_sar:
            s1_std = np.array([4.5, 4.0], dtype=np.float32)
            return np.concatenate([s2_std, s1_std])
        return s2_std

    def __len__(self) -> int:
        return len(self.opt_paths)

    def _load_tile(self, path: Path) -> np.ndarray:
        """Load a GeoTIFF tile as a float32 numpy array [C, H, W]."""
        import rasterio
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)

    def _random_crop(
        self, *arrays: np.ndarray, size: int
    ) -> Tuple[np.ndarray, ...]:
        """Random-crop all arrays to (size × size) at the same position."""
        _, H, W = arrays[0].shape
        if H <= size and W <= size:
            return arrays
        row = random.randint(0, max(0, H - size))
        col = random.randint(0, max(0, W - size))
        return tuple(a[:, row:row+size, col:col+size] for a in arrays)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load optical (Sentinel-2)
        opt = self._load_tile(self.opt_paths[idx])
        opt = opt / 10_000.0   # Scale from DN to reflectance

        # Load label (DSM / height)
        lbl = self._load_tile(self.label_paths[idx])
        lbl = np.clip(lbl, 0.0, 250.0)

        # Optionally load SAR (Sentinel-1)
        if self.use_sar and len(self.sar_paths) > idx:
            sar = self._load_tile(self.sar_paths[idx])
            image = np.concatenate([opt, sar], axis=0)
        else:
            image = opt

        # Ensure label is [1, H, W]
        if lbl.ndim == 2:
            lbl = lbl[np.newaxis]

        # Random crop to patch_size
        if self.augment:
            image, lbl = self._random_crop(image, lbl, size=self.patch_size)
        else:
            # Center crop
            _, H, W = image.shape
            row = max(0, (H - self.patch_size) // 2)
            col = max(0, (W - self.patch_size) // 2)
            image = image[:, row:row+self.patch_size, col:col+self.patch_size]
            lbl   = lbl[:,   row:row+self.patch_size, col:col+self.patch_size]

        # Augmentation
        if self.augment:
            image, lbl = apply_transforms(image, lbl, self.transform)

        # Normalization
        if self.normalize:
            mean = self.mean[:image.shape[0], None, None]
            std  = self.std[:image.shape[0], None, None]
            image = (image - mean) / (std + 1e-8)

        return {
            "image":    torch.from_numpy(image.astype(np.float32)),
            "label":    torch.from_numpy(lbl.astype(np.float32)),
            "filename": self.opt_paths[idx].name,
            "source":   "dfc2023",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OSMHeightDataset — OSM-derived heights with Sentinel-2 tiles
# ═══════════════════════════════════════════════════════════════════════════════

class OSMHeightDataset(Dataset):
    """
    Uses OpenStreetMap building:height / building:levels as ground truth labels
    rasterized over Sentinel-2 tiles. Suitable for Dhaka where many buildings
    have OSM height attributes.

    Directory structure:
        root/
          {split}/
            s2/      <tile_id>_S2.tif          (Sentinel-2, 12 bands)
            osm_h/   <tile_id>_osm_height.tif  (Rasterized OSM heights)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        min_coverage: float = 0.05,      # Minimum fraction of labeled pixels
    ):
        self.root = Path(root) / split
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.normalize = normalize
        self.min_coverage = min_coverage

        s2_dir  = self.root / "s2"
        osm_dir = self.root / "osm_h"

        if not s2_dir.exists():
            raise FileNotFoundError(
                f"Sentinel-2 tiles not found: {s2_dir}\n"
                f"Run src/data/osm_rasterizer.py to generate OSM height rasters."
            )

        self.s2_paths  = sorted(s2_dir.glob("*_S2.tif"))
        self.osm_paths = sorted(osm_dir.glob("*_osm_height.tif"))

        # Filter pairs by label coverage
        self.valid_pairs = self._filter_by_coverage()

        # Normalization
        self.mean = mean
        self.std  = std

        self.transform = get_train_transforms() if self.augment else get_val_transforms()
        logger.info(
            f"[OSMHeightDataset/{split}] {len(self.valid_pairs)} valid tile pairs "
            f"(min_coverage={min_coverage:.0%})"
        )

    def _filter_by_coverage(self) -> List[Tuple[Path, Path]]:
        """Keep only tile pairs where OSM labels cover >= min_coverage area."""
        import rasterio
        valid = []
        for s2, osm in zip(self.s2_paths, self.osm_paths):
            try:
                with rasterio.open(osm) as src:
                    lbl = src.read(1)
                if (lbl > 0).mean() >= self.min_coverage:
                    valid.append((s2, osm))
            except Exception:
                pass
        return valid

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import rasterio
        s2_path, osm_path = self.valid_pairs[idx]

        with rasterio.open(s2_path) as src:
            image = src.read().astype(np.float32) / 10_000.0

        with rasterio.open(osm_path) as src:
            label = src.read(1).astype(np.float32)

        label = np.clip(label, 0.0, 200.0)[np.newaxis]

        # Random crop
        _, H, W = image.shape
        if H > self.patch_size and W > self.patch_size:
            row = random.randint(0, H - self.patch_size)
            col = random.randint(0, W - self.patch_size)
            image = image[:, row:row+self.patch_size, col:col+self.patch_size]
            label = label[:, row:row+self.patch_size, col:col+self.patch_size]

        if self.augment:
            image, label = apply_transforms(image, label, self.transform)

        if self.normalize and self.mean is not None:
            image = (image - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        return {
            "image":    torch.from_numpy(image),
            "label":    torch.from_numpy(label),
            "filename": s2_path.name,
            "source":   "osm_height",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. InferencePatchDataset — for city-wide prediction (no labels)
# ═══════════════════════════════════════════════════════════════════════════════

class InferencePatchDataset(Dataset):
    """Dataset for running inference on a list of patches without labels."""

    def __init__(
        self,
        patches:   List[np.ndarray],
        positions: List[Tuple[int, int]],
        mean: np.ndarray,
        std:  np.ndarray,
    ):
        self.patches   = patches
        self.positions = positions
        self.mean = mean
        self.std  = std

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict:
        image = self.patches[idx].astype(np.float32)
        image = (image - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        return {
            "image":    torch.from_numpy(image),
            "position": torch.tensor(self.positions[idx]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DatasetFactory — unified dataset loader
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetFactory:
    """
    Builds the right dataset based on what data is available.

    Priority order:
        1. DFC2023Dataset     — if dfc2023_root is found
        2. HeightPatchDataset — if pre-extracted .npy patches exist
        3. OSMHeightDataset   — if Sentinel-2 + OSM rasters exist
        4. SyntheticDhakaDataset — always available as fallback

    Datasets can also be concatenated (combine real + synthetic for more data).
    """

    @staticmethod
    def build(
        config: Dict,
        split: str = "train",
        combine_synthetic: bool = False,
    ) -> Dataset:
        """
        Args:
            config:             Project config dict (loads data paths from it).
            split:              'train', 'val', or 'test'
            combine_synthetic:  Append synthetic data to the real dataset.

        Returns:
            A ready-to-use PyTorch Dataset.
        """
        datasets = []

        patch_dir = config.get("patch_dir", "data/processed/patches")
        dfc_root  = config.get("dfc2023_root", "")
        osm_root  = config.get("osm_height_root", "")
        syn_size  = config.get("synthetic_samples", {"train": 2000, "val": 400, "test": 400})
        use_syn   = config.get("use_synthetic", True)

        # ── DFC2023 ──────────────────────────────────────────────────────────
        if dfc_root and Path(dfc_root).exists():
            try:
                ds = DFC2023Dataset(root=dfc_root, split=split)
                datasets.append(ds)
                logger.info(f"[Factory] Added DFC2023Dataset ({len(ds)} samples)")
            except Exception as e:
                logger.warning(f"[Factory] DFC2023Dataset failed: {e}")

        # ── HeightPatchDataset ────────────────────────────────────────────────
        patch_split_dir = Path(patch_dir) / split / "images"
        if patch_split_dir.exists() and any(patch_split_dir.glob("*.npy")):
            try:
                ds = HeightPatchDataset(root=patch_dir, split=split)
                datasets.append(ds)
                logger.info(f"[Factory] Added HeightPatchDataset ({len(ds)} samples)")
            except Exception as e:
                logger.warning(f"[Factory] HeightPatchDataset failed: {e}")

        # ── OSMHeightDataset ──────────────────────────────────────────────────
        if osm_root and Path(osm_root).exists():
            try:
                ds = OSMHeightDataset(root=osm_root, split=split)
                datasets.append(ds)
                logger.info(f"[Factory] Added OSMHeightDataset ({len(ds)} samples)")
            except Exception as e:
                logger.warning(f"[Factory] OSMHeightDataset failed: {e}")

        # ── Synthetic (fallback or supplement) ───────────────────────────────
        n_syn = syn_size.get(split, 400) if isinstance(syn_size, dict) else syn_size
        if use_syn and (len(datasets) == 0 or combine_synthetic):
            ds_syn = SyntheticDhakaDataset(num_samples=n_syn, split=split)
            datasets.append(ds_syn)
            logger.info(f"[Factory] Added SyntheticDhakaDataset ({n_syn} samples)")

        if not datasets:
            raise RuntimeError(
                "No datasets could be loaded. Check your data paths in configs/data_config.yaml"
            )

        if len(datasets) == 1:
            return datasets[0]

        combined = ConcatDataset(datasets)
        logger.info(f"[Factory] Combined dataset: {len(combined)} total samples")
        return combined

    @staticmethod
    def get_dataloaders(
        config: Dict,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        combine_synthetic: bool = False,
    ) -> Dict[str, DataLoader]:
        """
        Build DataLoaders for train / val / test splits.
        Returns dict with 'train', 'val', 'test' keys.
        """
        loaders = {}
        for split in ["train", "val", "test"]:
            ds = DatasetFactory.build(config, split=split,
                                       combine_synthetic=combine_synthetic)
            shuffle = (split == "train")
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(split == "train"),
            )
            logger.info(f"[DataLoader/{split}] {len(ds)} samples | "
                        f"batch={batch_size} | {len(loaders[split])} batches")
        return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = "data/processed/patches",
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Backward-compatible wrapper.
    Tries HeightPatchDataset first; falls back to SyntheticDhakaDataset.
    """
    config = {
        "patch_dir":          data_dir,
        "dfc2023_root":       "",
        "osm_height_root":    "",
        "use_synthetic":      True,
        "synthetic_samples":  {"train": 2000, "val": 400, "test": 400},
    }
    return DatasetFactory.get_dataloaders(
        config, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
