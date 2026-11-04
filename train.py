"""
train.py
═══════════════════════════════════════════════════════════════════════════
Standalone training script for Building Height Analysis — Dhaka City.
Runs with ZERO downloads using SyntheticDhakaDataset.
Automatically upgrades to real data (DFC2023 / .npy patches) if available.

Usage:
    # Quick start — synthetic data (no downloads):
    python train.py --mode synthetic --epochs 30 --batch 8

    # Train on DFC2023 dataset:
    python train.py --mode dfc2023 --data-root ./data/dfc2023 --epochs 100

    # Train on pre-extracted .npy patches:
    python train.py --mode patches --data-root ./data/processed/patches

    # Combine synthetic + real data:
    python train.py --mode patches --combine-synthetic --epochs 50

    # Specify model:
    python train.py --model unet --epochs 50
    python train.py --model swin --epochs 100
    python train.py --model ensemble --epochs 80
═══════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")

# ─────────────────────────────────────────────────────────────────────────────
# Imports (lazy to keep startup fast)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import (
    DatasetFactory,
    SyntheticDhakaDataset,
    HeightPatchDataset,
    DFC2023Dataset,
)
from src.models.unet import UNetHeightEstimator, NUM_HEIGHT_CLASSES
from src.models.transformer import SwinTransformerHeightEstimator
from src.models.ensemble import EnsembleHeightEstimator
from src.models.losses import CombinedHeightLoss, HeightClassificationLoss
from src.evaluation.metrics import compute_metrics, print_metrics_table


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dhaka Building Height — Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Task
    p.add_argument("--task", choices=["classification", "regression"],
                   default="classification",
                   help=("'classification' → predict Low/Medium/High class (default). "
                         "'regression' → predict continuous height in metres."))

    # Dataset
    p.add_argument("--mode", choices=["synthetic", "dfc2023", "patches", "osm"],
                   default="synthetic",
                   help="Dataset mode. 'synthetic' works with no downloads.")
    p.add_argument("--data-root", default="data/processed/patches",
                   help="Root directory for real dataset (ignored in synthetic mode)")
    p.add_argument("--combine-synthetic", action="store_true",
                   help="Add synthetic data on top of real data")
    p.add_argument("--synthetic-train", type=int, default=2000,
                   help="Number of synthetic training samples")
    p.add_argument("--synthetic-val",   type=int, default=400)
    p.add_argument("--synthetic-test",  type=int, default=400)

    # Model
    p.add_argument("--model", choices=["unet", "unetpp", "yolov8", "yolov9", "swin", "ensemble"],
                   default="unet",
                   help="Model architecture to train")
    p.add_argument("--in-channels", type=int, default=17,
                   help="Number of input channels (17 for synthetic, 12 for Sentinel-2 only)")
    p.add_argument("--encoder", default="resnet50",
                   help="UNet encoder (resnet50, efficientnet-b4, mobilenet_v2, ...)")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="Use ImageNet-pretrained encoder weights")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Training
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=8)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patch-size",   type=int, default=256)
    p.add_argument("--num-workers",  type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable Automatic Mixed Precision")
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--patience",     type=int,   default=15,
                   help="Early stopping patience (epochs)")
    p.add_argument("--seed",         type=int,   default=42)

    # Checkpointing
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--resume",   default="",
                   help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=10,
                   help="Save checkpoint every N epochs (in addition to best)")

    # Misc
    p.add_argument("--device", default="",
                   help="'cuda', 'cpu', or '' (auto-detect)")
    p.add_argument("--use-mlflow", action="store_true",
                   help="Enable MLflow experiment tracking")
    p.add_argument("--experiment", default="dhaka-height-analysis",
                   help="MLflow experiment name")
    p.add_argument("--config", default="configs/training_config.yaml",
                   help="Optional YAML config (CLI args override YAML)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args) -> nn.Module:
    """Instantiate the requested model architecture."""
    enc_weights = "imagenet" if args.pretrained else None

    if args.model == "unet":
        from src.models.unet import UNetHeightEstimator
        model = UNetHeightEstimator(
            in_channels=args.in_channels,
            encoder_name=args.encoder,
            encoder_weights=enc_weights,
            task=args.task,
            num_classes=NUM_HEIGHT_CLASSES,
        )
        logger.info(f"Model: UNet-{args.encoder} | in_channels={args.in_channels} | task={args.task}")

    elif args.model == "unetpp":
        from src.models.unet import UNetPlusPlusHeightEstimator
        model = UNetPlusPlusHeightEstimator(
            in_channels=args.in_channels,
            encoder_name=args.encoder,
            task=args.task,
        )
        logger.info(f"Model: UNet++-{args.encoder} | in_channels={args.in_channels} | task={args.task}")

    elif args.model == "yolov8":
        from src.models.yolo import YOLOv8HeightEstimator
        model = YOLOv8HeightEstimator(
            in_channels=args.in_channels,
            num_classes=NUM_HEIGHT_CLASSES,
            task=args.task,
        )
        logger.info(f"Model: YOLOv8 (C2f + SPPF Architecture) | in_channels={args.in_channels} | task={args.task}")

    elif args.model == "yolov9":
        from src.models.yolo import YOLOv9HeightEstimator
        model = YOLOv9HeightEstimator(
            in_channels=args.in_channels,
            num_classes=NUM_HEIGHT_CLASSES,
            task=args.task,
        )
        logger.info(f"Model: YOLOv9 (GELAN Architecture) | in_channels={args.in_channels} | task={args.task}")

    elif args.model == "swin":
        from src.models.transformer import SwinTransformerHeightEstimator
        model = SwinTransformerHeightEstimator(
            in_channels=args.in_channels,
            img_size=args.patch_size,
            pretrained=args.pretrained,
        )
        logger.info(f"Model: SwinTransformer | in_channels={args.in_channels}")

    elif args.model == "ensemble":
        from src.models.unet import UNetHeightEstimator
        from src.models.transformer import SwinTransformerHeightEstimator
        from src.models.ensemble import EnsembleHeightEstimator
        unet = UNetHeightEstimator(in_channels=args.in_channels,
                                    encoder_weights=enc_weights)
        swin = SwinTransformerHeightEstimator(in_channels=args.in_channels,
                                               pretrained=args.pretrained)
        model = EnsembleHeightEstimator(models=[unet, swin], weights=[0.6, 0.4])
        logger.info(f"Model: Ensemble (UNet + Swin) | in_channels={args.in_channels}")

    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")
    return model


def build_datasets(args) -> tuple:
    """Build train / val / test datasets based on --mode."""

    syn_sizes = {
        "train": args.synthetic_train,
        "val":   args.synthetic_val,
        "test":  args.synthetic_test,
    }

    if args.mode == "synthetic":
        logger.info("Dataset: SyntheticDhakaDataset (no download required)")
        train_ds = SyntheticDhakaDataset(
            num_samples=syn_sizes["train"], split="train", seed=args.seed, task=args.task)
        val_ds   = SyntheticDhakaDataset(
            num_samples=syn_sizes["val"],   split="val",   seed=args.seed, task=args.task)
        test_ds  = SyntheticDhakaDataset(
            num_samples=syn_sizes["test"],  split="test",  seed=args.seed, task=args.task)

    elif args.mode == "dfc2023":
        logger.info(f"Dataset: DFC2023 from {args.data_root}")
        train_ds = DFC2023Dataset(root=args.data_root, split="train")
        val_ds   = DFC2023Dataset(root=args.data_root, split="val")
        test_ds  = DFC2023Dataset(root=args.data_root, split="test")

    elif args.mode == "patches":
        logger.info(f"Dataset: HeightPatchDataset from {args.data_root}")
        train_ds = HeightPatchDataset(root=args.data_root, split="train", augment=True)
        val_ds   = HeightPatchDataset(root=args.data_root, split="val",   augment=False,
                                       mean=train_ds.mean, std=train_ds.std)
        test_ds  = HeightPatchDataset(root=args.data_root, split="test",  augment=False,
                                       mean=train_ds.mean, std=train_ds.std)

    else:  # osm or auto
        config = {
            "patch_dir":         args.data_root,
            "dfc2023_root":      args.data_root if args.mode == "dfc2023" else "",
            "osm_height_root":   args.data_root if args.mode == "osm"    else "",
            "use_synthetic":     True,
            "synthetic_samples": syn_sizes,
        }
        from torch.utils.data import ConcatDataset
        train_ds = DatasetFactory.build(config, split="train",
                                         combine_synthetic=args.combine_synthetic)
        val_ds   = DatasetFactory.build(config, split="val")
        test_ds  = DatasetFactory.build(config, split="test")

    # Optionally add synthetic on top of real data
    if args.combine_synthetic and args.mode != "synthetic":
        from torch.utils.data import ConcatDataset
        syn_tr = SyntheticDhakaDataset(syn_sizes["train"], split="train", seed=args.seed)
        syn_va = SyntheticDhakaDataset(syn_sizes["val"],   split="val",   seed=args.seed)
        train_ds = ConcatDataset([train_ds, syn_tr])
        val_ds   = ConcatDataset([val_ds,   syn_va])
        logger.info("Synthetic data combined with real dataset.")

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training / Evaluation Loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, grad_clip, task="classification"
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for batch in pbar:
        imgs    = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if targets.dtype == torch.float32 and targets.ndim == 4:
            targets = targets.squeeze(1)   # [B,1,H,W] → [B,H,W] for CE loss

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            preds = model(imgs)
            loss, parts = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu().float().numpy())
        all_targets.append(targets.detach().cpu().float().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_targets), task=task
    )
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, scaler, task="classification") -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False, ncols=90):
        imgs    = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if targets.dtype == torch.float32 and targets.ndim == 4:
            targets = targets.squeeze(1)   # [B,1,H,W] → [B,H,W]
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            preds = model(imgs)
            loss, _ = loss_fn(preds, targets)
        total_loss += loss.item()
        all_preds.append(preds.cpu().float().numpy())
        all_targets.append(targets.cpu().float().numpy())

    metrics = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_targets), task=task
    )
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, scheduler, epoch, metrics, args):
    torch.save({
        "epoch":                epoch,
        "model_name":           args.model,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics":              metrics,
        "args":                 vars(args),
    }, path)
    logger.info(f"  [+] Checkpoint -> {path}")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    epoch = state.get("epoch", 0)
    logger.info(f"Resumed from {path} (epoch {epoch})")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# History Logger
# ─────────────────────────────────────────────────────────────────────────────

class HistoryLogger:
    """Saves per-epoch metrics to JSON and CSV."""

    def __init__(self, log_dir: str = "logs"):
        self.path_json = Path(log_dir) / "training_history.json"
        self.path_csv  = Path(log_dir) / "training_history.csv"
        self.history   = []

    def append(self, epoch: int, train_m: dict, val_m: dict, lr: float):
        row = {"epoch": epoch, "lr": lr}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}":   v for k, v in val_m.items()})
        self.history.append(row)

        # JSON
        with open(self.path_json, "w") as f:
            json.dump(self.history, f, indent=2)

        # CSV
        import csv
        with open(self.path_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

    def print_last(self):
        r = self.history[-1]
        logger.info(
            f"  Epoch {r['epoch']:3d} | lr={r['lr']:.2e} | "
            f"Train: loss={r.get('train_loss',0):.4f} RMSE={r.get('train_rmse',0):.2f}m "
            f"R²={r.get('train_r2',0):.3f} | "
            f"Val: loss={r.get('val_loss',0):.4f} RMSE={r.get('val_rmse',0):.2f}m "
            f"R²={r.get('val_r2',0):.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU. Training will be slower.")

    logger.info("=" * 60)
    logger.info("  Dhaka Building Height Analysis — Training")
    logger.info("=" * 60)
    logger.info(f"  Mode:       {args.mode}")
    logger.info(f"  Task:       {args.task}  (classification=Low/Med/High | regression=metres)")
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  LR:         {args.lr}")
    logger.info("=" * 60)

    # ── Datasets & DataLoaders ─────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_datasets(args)

    # Adjust in_channels automatically for synthetic dataset
    if isinstance(train_ds, SyntheticDhakaDataset):
        args.in_channels = SyntheticDhakaDataset.IN_CHANNELS

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin,
                               drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    if args.task == "classification":
        from src.models.losses import HeightClassificationLoss
        loss_fn = HeightClassificationLoss()
        logger.info("Loss: HeightClassificationLoss (CrossEntropy + Dice | 4 classes)")
    else:
        loss_fn = CombinedHeightLoss(
            smooth_l1_weight=0.85,
            ssim_weight=0.10,
            edge_weight=0.05,
        )
        logger.info("Loss: CombinedHeightLoss (SmoothL1 + SSIM + EdgeAware)")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ──────────────────────────────────────────────────────────
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 10,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    # ── AMP Scaler ─────────────────────────────────────────────────────────
    use_amp = not args.no_amp and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        logger.info("Mixed precision (AMP) enabled.")

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_run = None
    if args.use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(args.experiment)
            mlflow_run = mlflow.start_run(
                run_name=f"{args.model}_{args.mode}_{int(time.time())}"
            )
            mlflow.log_params(vars(args))
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        ) + 1

    # ── History & Early Stopping ────────────────────────────────────────────
    history = HistoryLogger()
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"{args.model}_best.pth"

    # ── Training Loop ───────────────────────────────────────────────────────
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")

        train_m = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, args.grad_clip, task=args.task
        )
        val_m = evaluate(model, val_loader, loss_fn, device, scaler, task=args.task)

        # Step the scheduler once per epoch if not per-batch
        # (OneCycleLR steps per batch already inside train_one_epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(epoch, train_m, val_m, current_lr)
        history.print_last()

        # MLflow logging
        if mlflow_run:
            import mlflow
            mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}":   v for k, v in val_m.items()},   step=epoch)

        # Best model checkpoint — use mIoU for classification, RMSE for regression
        monitor_key  = "miou"   if args.task == "classification" else "rmse"
        monitor_sign = -1       if args.task == "classification" else 1     # higher is better for mIoU
        current_score = val_m.get(monitor_key, 0.0) * monitor_sign

        if current_score < best_val_rmse * monitor_sign or best_val_rmse == float("inf"):
            best_val_rmse = val_m.get(monitor_key, float("inf"))
            epochs_no_improve = 0
            save_checkpoint(str(best_ckpt), model, optimizer, scheduler, epoch,
                             val_m, args)
            if args.task == "classification":
                logger.info(f"  [*] New best Val mIoU: {best_val_rmse:.4f}")
            else:
                logger.info(f"  [*] New best Val RMSE: {best_val_rmse:.3f}m")
        else:
            epochs_no_improve += 1
            logger.info(
                f"  No improvement ({epochs_no_improve}/{args.patience}) "
                f"| Best so far: {best_val_rmse:.3f}m"
            )

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            periodic = ckpt_dir / f"{args.model}_epoch{epoch:03d}.pth"
            save_checkpoint(str(periodic), model, optimizer, scheduler,
                             epoch, val_m, args)

        # Early stopping
        if epochs_no_improve >= args.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # ── Final Evaluation on Test Set ────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("Final evaluation on test set...")

    # Load best checkpoint
    if best_ckpt.exists():
        load_checkpoint(str(best_ckpt), model, device=device)

    test_m = evaluate(model, test_loader, loss_fn, device, scaler, task=args.task)
    print_metrics_table(test_m, title="🧪 Test Set Results")

    if mlflow_run:
        import mlflow
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.end_run()

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training complete in {elapsed/60:.1f} minutes.")
    if args.task == "classification":
        logger.info(f"Best Val mIoU : {best_val_rmse:.4f}")
        logger.info(f"Test mIoU     : {test_m.get('miou', 'N/A')}")
        logger.info(f"Test Accuracy : {test_m.get('accuracy', 'N/A')}")
        logger.info(f"Building mIoU : {test_m.get('building_miou', 'N/A')}")
    else:
        logger.info(f"Best Val RMSE : {best_val_rmse:.3f} m")
        logger.info(f"Test RMSE     : {test_m.get('rmse', 'N/A')}")
        logger.info(f"Test R²       : {test_m.get('r2',   'N/A')}")
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"History saved : logs/training_history.json")


if __name__ == "__main__":
    main()
