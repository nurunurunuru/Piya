"""
pipeline/stage_02_preprocess.py
Stage 2: Preprocessing — feature extraction, nDSM generation, patch creation.
"""
import logging
from pathlib import Path
logger = logging.getLogger(__name__)


def run(config: dict):
    from src.data.preprocessor import (
        NDSMGenerator, PatchGenerator, FeatureExtractor,
        load_raster, save_raster, resample_raster,
    )

    raw_dir  = "data/raw"
    proc_dir = "data/processed"

    sat_path = f"{raw_dir}/satellite/sentinel2_dhaka_composite.tif"
    dsm_path = f"{raw_dir}/lidar/dsm_dhaka.tif"
    dem_path = f"{raw_dir}/lidar/srtm_dem_dhaka.tif"
    ndsm_out = f"{proc_dir}/ndsm/ndsm_dhaka.tif"

    # Step 1: Resample DEM to 10m to match Sentinel-2
    logger.info("Resampling DEM to 10m...")
    resample_raster(dem_path, f"{proc_dir}/dsm/dem_10m.tif", target_res=10.0)

    # Step 2: Compute nDSM (ground truth heights)
    logger.info("Computing nDSM...")
    ndsm_gen = NDSMGenerator()
    ndsm_gen.generate(dsm_path, f"{proc_dir}/dsm/dem_10m.tif", ndsm_out)

    # Step 3: Extract spectral features
    logger.info("Extracting spectral features...")
    extractor = FeatureExtractor()
    img, meta = load_raster(sat_path)
    features  = extractor.extract_all(img)
    save_raster(features, meta, f"{proc_dir}/features/sentinel2_features.tif")
    logger.info(f"Feature raster shape: {features.shape} (channels, H, W)")

    # Step 4: Generate patches
    logger.info("Generating training patches...")
    patch_gen = PatchGenerator(
        patch_size=config.get("model_cfg", {}).get("input", {}).get("patch_size", 256),
        stride=config.get("model_cfg", {}).get("input", {}).get("stride", 128),
    )

    # Create train/val/test splits using spatial stratification
    n_train = patch_gen.generate_patches(
        f"{proc_dir}/features/sentinel2_features.tif",
        ndsm_out,
        f"{proc_dir}/patches",
        split="train",
    )
    n_val = patch_gen.generate_patches(
        f"{proc_dir}/features/sentinel2_features.tif",
        ndsm_out,
        f"{proc_dir}/patches",
        split="val",
    )
    n_test = patch_gen.generate_patches(
        f"{proc_dir}/features/sentinel2_features.tif",
        ndsm_out,
        f"{proc_dir}/patches",
        split="test",
    )

    logger.info(f"Patches: Train={n_train}, Val={n_val}, Test={n_test}")
