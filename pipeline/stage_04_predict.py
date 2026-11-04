"""
pipeline/stage_04_predict.py
Stage 4: Run city-wide height prediction and evaluate on test set.
"""
import logging
import numpy as np
logger = logging.getLogger(__name__)


def run(config: dict):
    import torch
    from src.models.ensemble import EnsembleHeightEstimator
    from src.inference.predictor import CityPredictor, PostProcessor
    from src.evaluation.metrics import compute_metrics, print_metrics_table
    from src.data.preprocessor import load_raster

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sat_path      = "data/processed/features/sentinel2_features.tif"
    pred_out      = "data/results/predictions/height_map_raw.tif"
    pred_smoothed = "data/results/predictions/height_map_final.tif"

    # Load ensemble from checkpoints
    checkpoints = {
        "unet": "models/checkpoints/unet_resnet50_best.pth",
        "swin": "models/checkpoints/swin_transformer_best.pth",
    }

    # Check if checkpoints exist
    import os
    available = {k: v for k, v in checkpoints.items() if os.path.exists(v)}
    if not available:
        logger.warning("No checkpoints found. Run Stage 3 first.")
        return

    ensemble = EnsembleHeightEstimator.from_checkpoints(
        available, weights=[0.6, 0.4][:len(available)], device=device
    )

    # Predict
    predictor = CityPredictor(
        model=ensemble, device=device,
        mean=np.zeros(16, dtype=np.float32),  # Loaded from dataset stats in practice
        std=np.ones(16, dtype=np.float32),
    )
    predictor.predict(sat_path, pred_out)

    # Post-process
    pp = PostProcessor()
    raw_h, meta = load_raster(pred_out)
    smoothed = pp.smooth(raw_h.squeeze(), sigma=1.5)
    smoothed = pp.remove_isolated_pixels(smoothed, min_area=10)

    import rasterio
    meta.update({"count": 1, "dtype": "float32"})
    from src.data.preprocessor import save_raster
    save_raster(smoothed[np.newaxis], meta, pred_smoothed)

    # Evaluate on test patches
    logger.info("Evaluating on test patches...")
    ndsm_path = "data/processed/ndsm/ndsm_dhaka.tif"
    if not os.path.exists(ndsm_path):
        logger.warning("Ground truth nDSM not found. Skipping evaluation.")
        return

    pred_arr, _  = load_raster(pred_smoothed)
    truth_arr, _ = load_raster(ndsm_path)

    # Align shapes (crop to smaller)
    min_h = min(pred_arr.shape[1], truth_arr.shape[1])
    min_w = min(pred_arr.shape[2], truth_arr.shape[2])
    metrics = compute_metrics(pred_arr[:, :min_h, :min_w], truth_arr[:, :min_h, :min_w])
    print_metrics_table(metrics, title="City-Wide Evaluation Results")
