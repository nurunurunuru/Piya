"""
src/data/preprocessor.py
Preprocessing pipeline: cloud masking, patch generation, DSM computation,
feature extraction (NDVI, NDBI, shadow index, texture).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None
    from_bounds = None
    reproject = None
    Resampling = None
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raster Utilities
# ---------------------------------------------------------------------------

def load_raster(path: str) -> Tuple[np.ndarray, Dict]:
    """Load a GeoTIFF and return (array[C,H,W], meta)."""
    with rasterio.open(path) as src:
        arr  = src.read().astype(np.float32)
        meta = src.meta.copy()
    return arr, meta


def save_raster(array: np.ndarray, meta: Dict, output_path: str):
    """Save array [C,H,W] as GeoTIFF."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    meta.update({
        "count": array.shape[0],
        "dtype": "float32",
        "compress": "lzw",
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array)


def resample_raster(source_path: str, target_path: str, target_res: float = 10.0):
    """Resample a raster to match target spatial resolution."""
    with rasterio.open(source_path) as src:
        scale = src.res[0] / target_res
        new_h = int(src.height * scale)
        new_w = int(src.width  * scale)
        new_transform = src.transform * src.transform.scale(
            src.width / new_w, src.height / new_h
        )
        data = src.read(
            out_shape=(src.count, new_h, new_w),
            resampling=Resampling.bilinear,
        )
        meta = src.meta.copy()
        meta.update({
            "height": new_h,
            "width":  new_w,
            "transform": new_transform,
        })
    save_raster(data, meta, target_path)
    logger.info(f"Resampled {source_path} → {target_path} at {target_res}m")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Computes spectral indices and texture features from Sentinel-2 imagery."""

    # Band indices in Sentinel-2 12-band stack (0-indexed)
    BAND_IDX = {
        "B02": 0, "B03": 1, "B04": 2,  # Blue, Green, Red
        "B05": 3, "B06": 4, "B07": 5,  # Red Edges
        "B08": 6, "B8A": 7, "B09": 8,  # NIR, Narrow NIR, Water Vapour
        "B11": 9, "B12": 10,           # SWIR 1, SWIR 2
        "NDVI": 11,                     # Pre-computed NDVI
    }

    def compute_ndvi(self, img: np.ndarray) -> np.ndarray:
        """Normalized Difference Vegetation Index."""
        nir = img[self.BAND_IDX["B08"]]
        red = img[self.BAND_IDX["B04"]]
        denom = nir + red + 1e-8
        return ((nir - red) / denom).astype(np.float32)

    def compute_ndbi(self, img: np.ndarray) -> np.ndarray:
        """Normalized Difference Built-up Index."""
        swir = img[self.BAND_IDX["B11"]]
        nir  = img[self.BAND_IDX["B08"]]
        denom = swir + nir + 1e-8
        return ((swir - nir) / denom).astype(np.float32)

    def compute_mndwi(self, img: np.ndarray) -> np.ndarray:
        """Modified Normalized Difference Water Index."""
        green = img[self.BAND_IDX["B03"]]
        swir  = img[self.BAND_IDX["B11"]]
        denom = green + swir + 1e-8
        return ((green - swir) / denom).astype(np.float32)

    def compute_shadow_index(self, img: np.ndarray) -> np.ndarray:
        """Building shadow index using blue band."""
        blue  = img[self.BAND_IDX["B02"]]
        green = img[self.BAND_IDX["B03"]]
        nir   = img[self.BAND_IDX["B08"]]
        # Shadow appears dark in visible + NIR
        return (1.0 - (blue + green + nir) / 3.0).astype(np.float32)

    def compute_texture(self, img: np.ndarray, band_idx: int = 6) -> np.ndarray:
        """Local standard deviation as a texture feature."""
        band = (img[band_idx] * 255).clip(0, 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.float32) / 25
        mean   = cv2.filter2D(band.astype(np.float32), -1, kernel)
        sq_mean = cv2.filter2D((band.astype(np.float32))**2, -1, kernel)
        std    = np.sqrt(np.maximum(sq_mean - mean**2, 0))
        return (std / 255.0).astype(np.float32)

    def extract_all(self, img: np.ndarray) -> np.ndarray:
        """Return stacked feature array [C+4, H, W]."""
        ndvi    = self.compute_ndvi(img)[np.newaxis]
        ndbi    = self.compute_ndbi(img)[np.newaxis]
        mndwi   = self.compute_mndwi(img)[np.newaxis]
        shadow  = self.compute_shadow_index(img)[np.newaxis]
        texture = self.compute_texture(img)[np.newaxis]
        return np.concatenate([img, ndvi, ndbi, mndwi, shadow, texture], axis=0)


# ---------------------------------------------------------------------------
# nDSM Generator (Ground Truth)
# ---------------------------------------------------------------------------

class NDSMGenerator:
    """
    Generates Normalized Digital Surface Model (nDSM = DSM - DEM).
    nDSM represents building heights above ground.
    """

    def generate(self, dsm_path: str, dem_path: str, output_path: str):
        """
        Compute nDSM from DSM and DEM rasters.

        Args:
            dsm_path:    Path to Digital Surface Model.
            dem_path:    Path to Digital Elevation Model.
            output_path: Output nDSM GeoTIFF path.
        """
        dsm, dsm_meta = load_raster(dsm_path)
        dem, _        = load_raster(dem_path)

        # Resample DEM to match DSM if needed
        if dsm.shape[1:] != dem.shape[1:]:
            logger.info("Resampling DEM to match DSM resolution...")
            dem_resampled = np.zeros_like(dsm)
            reproject(
                source=dem,
                destination=dem_resampled,
                src_transform=None,
                dst_transform=dsm_meta["transform"],
                resampling=Resampling.bilinear,
            )
            dem = dem_resampled

        ndsm = dsm - dem
        ndsm = np.clip(ndsm, 0, 250)   # Clamp to [0, 250m]
        ndsm[ndsm < 0.5] = 0.0         # Ground level noise removal

        save_raster(ndsm, dsm_meta, output_path)
        logger.info(f"nDSM saved → {output_path}")
        logger.info(f"  Height range: {ndsm.min():.1f}m – {ndsm.max():.1f}m")


# ---------------------------------------------------------------------------
# Patch Generator
# ---------------------------------------------------------------------------

class PatchGenerator:
    """
    Splits large rasters into overlapping square patches for model input/training.
    """

    def __init__(self, patch_size: int = 256, stride: int = 128,
                 min_building_pct: float = 0.05):
        self.patch_size = patch_size
        self.stride     = stride
        self.min_building_pct = min_building_pct

    def generate_patches(
        self,
        image_path: str,
        label_path: str,
        output_dir: str,
        split: str = "train",
    ) -> int:
        """
        Extract and save image/label patch pairs.

        Returns:
            Number of patches saved.
        """
        img, img_meta = load_raster(image_path)
        lbl, _        = load_raster(label_path)

        C, H, W = img.shape
        patch_dir = Path(output_dir) / split
        (patch_dir / "images").mkdir(parents=True, exist_ok=True)
        (patch_dir / "labels").mkdir(parents=True, exist_ok=True)

        count = 0
        for row in range(0, H - self.patch_size + 1, self.stride):
            for col in range(0, W - self.patch_size + 1, self.stride):
                img_patch = img[:, row:row+self.patch_size, col:col+self.patch_size]
                lbl_patch = lbl[:, row:row+self.patch_size, col:col+self.patch_size]

                # Skip patches with no building pixels
                building_pct = (lbl_patch > 1.0).mean()
                if building_pct < self.min_building_pct:
                    continue

                # Skip patches with too many NaN values
                if np.isnan(img_patch).mean() > 0.1:
                    continue

                patch_id = f"patch_{row:05d}_{col:05d}"
                np.save(patch_dir / "images" / f"{patch_id}.npy", img_patch)
                np.save(patch_dir / "labels" / f"{patch_id}.npy", lbl_patch)
                count += 1

        logger.info(f"Generated {count} patches → {patch_dir}")
        return count

    def generate_inference_patches(
        self, image_path: str
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Dict]:
        """
        Generate patches for inference (no label required).
        Returns patches, their (row, col) positions, and image metadata.
        """
        img, meta = load_raster(image_path)
        _, H, W = img.shape

        patches   = []
        positions = []

        for row in range(0, H, self.stride):
            for col in range(0, W, self.stride):
                r_end = min(row + self.patch_size, H)
                c_end = min(col + self.patch_size, W)
                patch = np.zeros(
                    (img.shape[0], self.patch_size, self.patch_size), dtype=np.float32
                )
                patch[:, :r_end-row, :c_end-col] = img[:, row:r_end, col:c_end]
                patches.append(patch)
                positions.append((row, col))

        return patches, positions, meta
