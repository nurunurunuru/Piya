"""
src/training/trainer.py
Full training loop with validation, checkpointing, and MLflow logging.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for building height estimation models.
    Supports mixed precision, gradient clipping, early stopping, and MLflow tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict,
        device: Optional[str] = None,
    ):
        self.model   = model
        self.loss_fn = loss_fn
        self.config  = config
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self._setup_optimizer()
        self._setup_logging()

        self.best_val_rmse = float("inf")
        self.epochs_no_improve = 0
        self.scaler = GradScaler(enabled=config["training"].get("mixed_precision", True))

    def _setup_optimizer(self):
        cfg = self.config["training"]
        opt_cfg = self.config["optimizer"]

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=opt_cfg["betas"],
        )

    def _setup_logging(self):
        mlflow.set_tracking_uri(self.config.get("mlflow_uri", "http://localhost:5000"))
        mlflow.set_experiment(
            self.config.get("experiment_name", "dhaka-height-analysis")
        )

    def setup_scheduler(self, train_loader: DataLoader):
        cfg = self.config["scheduler"]
        steps_per_epoch = len(train_loader)
        total_steps = self.config["training"]["num_epochs"] * steps_per_epoch

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=cfg["max_lr"],
            total_steps=total_steps,
            pct_start=cfg["pct_start"],
            div_factor=cfg["div_factor"],
            final_div_factor=cfg["final_div_factor"],
        )

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        pbar = tqdm(loader, desc="  Train", leave=False)
        for batch in pbar:
            images  = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.scaler.is_enabled()):
                preds = self.model(images)
                loss, loss_parts = self.loss_fn(preds, targets)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"].get("gradient_clip", 1.0),
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        preds_all   = np.concatenate(all_preds,   axis=0)
        targets_all = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(preds_all, targets_all)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in tqdm(loader, desc="  Val  ", leave=False):
            images  = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)

            with autocast(enabled=self.scaler.is_enabled()):
                preds = self.model(images)
                loss, _ = self.loss_fn(preds, targets)

            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        preds_all   = np.concatenate(all_preds,   axis=0)
        targets_all = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(preds_all, targets_all)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, tag: str = "best"):
        ckpt_dir = Path(self.config["checkpoint"]["save_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{self.model.name}_{tag}.pth"
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics":            metrics,
            "config":             self.config,
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   Optional[int] = None,
    ):
        """Full training loop."""
        num_epochs = num_epochs or self.config["training"]["num_epochs"]
        patience   = self.config["training"].get("early_stopping_patience", 15)
        self.setup_scheduler(train_loader)

        with mlflow.start_run(run_name=f"{self.model.name}_{int(time.time())}"):
            mlflow.log_params({
                "model":      self.model.name,
                "epochs":     num_epochs,
                "batch_size": self.config["training"]["batch_size"],
                "lr":         self.config["optimizer"]["lr"],
            })

            for epoch in range(1, num_epochs + 1):
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch}/{num_epochs}")

                train_metrics = self.train_epoch(train_loader)
                val_metrics   = self.val_epoch(val_loader)

                # Log metrics
                mlflow.log_metrics({
                    f"train_{k}": v for k, v in train_metrics.items()
                }, step=epoch)
                mlflow.log_metrics({
                    f"val_{k}": v for k, v in val_metrics.items()
                }, step=epoch)

                logger.info(
                    f"  Train → RMSE: {train_metrics['rmse']:.3f}m | "
                    f"MAE: {train_metrics['mae']:.3f}m | R²: {train_metrics['r2']:.4f}"
                )
                logger.info(
                    f"  Val   → RMSE: {val_metrics['rmse']:.3f}m | "
                    f"MAE: {val_metrics['mae']:.3f}m | R²: {val_metrics['r2']:.4f}"
                )

                # Save best model
                if val_metrics["rmse"] < self.best_val_rmse:
                    self.best_val_rmse = val_metrics["rmse"]
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, val_metrics, tag="best")
                    mlflow.log_metric("best_val_rmse", self.best_val_rmse, step=epoch)
                else:
                    self.epochs_no_improve += 1

                # Save periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, val_metrics, tag=f"epoch_{epoch}")

                # Early stopping
                if self.epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

            logger.info(f"\nTraining complete. Best Val RMSE: {self.best_val_rmse:.3f}m")
