"""
pipeline/stage_03_train.py
Stage 3: Train deep learning models (UNet + Swin Transformer).
"""
import logging
import torch
logger = logging.getLogger(__name__)


def run(config: dict):
    from src.data.dataset import get_dataloaders
    from src.models.unet import UNetHeightEstimator
    from src.models.transformer import SwinTransformerHeightEstimator
    from src.models.losses import CombinedHeightLoss
    from src.training.trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    proc_dir    = "data/processed/patches"
    model_cfg   = config.get("model_cfg", {})
    in_channels = model_cfg.get("input", {}).get("channels", 12)

    # Data loaders
    loaders = get_dataloaders(
        data_dir   = proc_dir,
        batch_size = config["training"]["batch_size"],
        num_workers= config["training"]["num_workers"],
        pin_memory = config["training"].get("pin_memory", True),
    )

    loss_fn = CombinedHeightLoss()

    # --- Train UNet ---
    logger.info("=" * 50)
    logger.info("Training UNet-ResNet50...")
    unet = UNetHeightEstimator(in_channels=in_channels)
    trainer_unet = Trainer(unet, loss_fn, config, device=device)
    trainer_unet.fit(loaders["train"], loaders["val"])

    # --- Train Swin Transformer ---
    logger.info("=" * 50)
    logger.info("Training Swin Transformer...")
    swin = SwinTransformerHeightEstimator(in_channels=in_channels)
    trainer_swin = Trainer(swin, loss_fn, config, device=device)
    trainer_swin.fit(loaders["train"], loaders["val"])

    logger.info("Both models trained. Checkpoints saved to models/checkpoints/")
