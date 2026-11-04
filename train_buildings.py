"""
train_buildings.py
══════════════════════════════════════════════════════════════════════════
YOLOv8 / YOLOv11 Building Detection Training Script
Dataset: Dhaka City drone imagery (YOLO polygon label format)

Usage:
    # Default — YOLOv8n-seg, 50 epochs (fast run):
    python train_buildings.py

    # Larger model, more epochs:
    python train_buildings.py --model yolov8m-seg --epochs 100

    # Use instance segmentation (polygon masks):
    python train_buildings.py --task segment --model yolov8n-seg --epochs 50

    # Object detection with OBB (Oriented Bounding Boxes):
    python train_buildings.py --task obb --model yolov8n-obb --epochs 50

    # Resume from checkpoint:
    python train_buildings.py --resume runs/segment/dhaka_buildings/weights/last.pt
══════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train_buildings.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_buildings")

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dhaka Building Detection — YOLOv8 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task & Model
    p.add_argument("--task", choices=["segment", "obb", "detect"],
                   default="segment",
                   help=(
                       "segment = polygon instance segmentation (recommended)\n"
                       "obb     = oriented bounding boxes\n"
                       "detect  = standard bounding box detection"
                   ))
    p.add_argument("--model",
                   default="yolov8n-seg.pt",
                   help=(
                       "YOLOv8/YOLOv11 model variant. Examples:\n"
                       "  yolov8n-seg.pt  (fastest, small)\n"
                       "  yolov8s-seg.pt  (fast, good)\n"
                       "  yolov8m-seg.pt  (balanced)\n"
                       "  yolov8l-seg.pt  (accurate, slow)\n"
                       "  yolov8n-obb.pt  (for --task obb)\n"
                       "  yolov8n.pt      (for --task detect)"
                   ))

    # Dataset
    p.add_argument("--data",
                   default="Dataset/data.yaml",
                   help="Path to data.yaml file")

    # Training hyperparameters
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch",     type=int,   default=8,
                   help="Batch size (-1 for auto-batch)")
    p.add_argument("--imgsz",     type=int,   default=640,
                   help="Training image size (pixels)")
    p.add_argument("--lr0",       type=float, default=0.01,
                   help="Initial learning rate")
    p.add_argument("--lrf",       type=float, default=0.01,
                   help="Final lr = lr0 * lrf")
    p.add_argument("--momentum",  type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--patience",  type=int,   default=20,
                   help="Early stopping patience (epochs)")

    # Augmentation
    p.add_argument("--hsv-h",     type=float, default=0.015)
    p.add_argument("--hsv-s",     type=float, default=0.7)
    p.add_argument("--hsv-v",     type=float, default=0.4)
    p.add_argument("--degrees",   type=float, default=10.0,
                   help="Rotation augmentation degrees")
    p.add_argument("--translate",  type=float, default=0.1)
    p.add_argument("--scale",     type=float, default=0.5)
    p.add_argument("--fliplr",    type=float, default=0.5)
    p.add_argument("--flipud",    type=float, default=0.3)
    p.add_argument("--mosaic",    type=float, default=1.0)

    # Output
    p.add_argument("--project",   default="runs",
                   help="Output directory root")
    p.add_argument("--name",      default="dhaka_buildings",
                   help="Run name (sub-directory under --project/--task)")
    p.add_argument("--workers",   type=int,   default=0,
                   help="DataLoader workers (0 = main thread, safe on Windows)")
    p.add_argument("--device",    default="",
                   help="Device: '' auto, '0' GPU 0, 'cpu' force CPU")

    # Resume
    p.add_argument("--resume",    default="",
                   help="Path to last.pt to resume training from")

    # Misc
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--exist-ok",  action="store_true",
                   help="Allow overwriting existing run directory")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model name resolver
# ─────────────────────────────────────────────────────────────────────────────

def resolve_model_name(args) -> str:
    """
    Auto-adjust the model suffix based on the chosen task so the user
    does not have to remember the naming conventions.
    """
    name = args.model

    # Strip existing suffix to get the base size (n/s/m/l/x)
    for suffix in ["-seg.pt", "-obb.pt", "-cls.pt", "-pose.pt", ".pt"]:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            break
    else:
        base = name  # no known suffix

    # Map task -> correct suffix
    if args.task == "segment":
        resolved = f"{base}-seg.pt"
    elif args.task == "obb":
        resolved = f"{base}-obb.pt"
    else:  # detect
        resolved = f"{base}.pt"

    if resolved != args.model:
        logger.info(f"  Model name adjusted: {args.model} -> {resolved}")

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────

def preflight_checks(args):
    """Validate paths and dataset before starting training."""
    import yaml

    errors = []

    # data.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        errors.append(f"data.yaml not found: {data_path.resolve()}")
    else:
        logger.info(f"  data.yaml    : {data_path.resolve()}")

    # Dataset directories
    if data_path.exists():
        with open(data_path) as f:
            cfg = yaml.safe_load(f)
        path_val = cfg.get("path", ".")
        root = Path(path_val)
        if not root.is_absolute():
            root = (data_path.parent / root).resolve()
        for split in ["train", "val"]:
            img_dir = root / cfg.get(split, split)
            if not img_dir.exists():
                errors.append(f"Missing {split} images dir: {img_dir}")
            else:
                n_imgs = len(list(img_dir.glob("*.jpg")) +
                             list(img_dir.glob("*.jpeg")) +
                             list(img_dir.glob("*.png")))
                logger.info(f"  {split:10s} images: {n_imgs}")

    if errors:
        for e in errors:
            logger.error(f"  [!] {e}")
        sys.exit(1)


def ensure_valid_weights(model_name: str) -> None:
    """Check if model weights exist and are valid PyTorch checkpoint; download if missing/corrupt."""
    path = Path(model_name)
    if not model_name.endswith(".pt"):
        return

    import torch
    import ultralytics.nn.tasks
    import urllib.request
    import gc

    # Register safe globals for PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.SegmentationModel,
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.nn.tasks.OBBModel,
        ])
    except Exception:
        pass

    if path.exists():
        try:
            with open(path, "rb") as f:
                torch.load(f, map_location="cpu", weights_only=False)
            logger.info(f"  Weights verified: {path.resolve()}")
            return
        except Exception as e:
            logger.warning(f"  Existing weights {path} corrupted ({e}). Removing...")
            gc.collect()
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{path.name}"
    logger.info(f"  Downloading weights from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
        with open(path, "rb") as f:
            torch.load(f, map_location="cpu", weights_only=False)
        logger.info(f"  Downloaded and verified weights: {path.resolve()}")
    except Exception as e:
        logger.error(f"  Failed to download/verify weights: {e}")
        gc.collect()
        path.unlink(missing_ok=True)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logger.info("=" * 62)
    logger.info("  Dhaka Building Height Classification -- YOLOv8 Training")
    logger.info("  Classes: Low (1-3 floors) | Medium (4-8 floors) | High (9+ floors)")
    logger.info("=" * 62)
    logger.info(f"  Task       : {args.task}")
    logger.info(f"  Model      : {args.model}")
    logger.info(f"  Dataset    : {args.data}")
    logger.info(f"  Epochs     : {args.epochs}")
    logger.info(f"  Batch size : {args.batch}")
    logger.info(f"  Image size : {args.imgsz}")
    logger.info("=" * 62)

    # -- Import ultralytics --------------------------------------------------
    try:
        from ultralytics import YOLO
        import ultralytics
        logger.info(f"  Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        logger.error(
            "\n[!] ultralytics is not installed.\n"
            "    Install it with:\n"
            "        .\\venv\\Scripts\\python.exe -m pip install ultralytics\n"
        )
        sys.exit(1)

    # -- Pre-flight ----------------------------------------------------------
    preflight_checks(args)

    # -- Model name ----------------------------------------------------------
    model_name = resolve_model_name(args)

    # -- Ensure weights are downloaded & verified cleanly --------------------
    ensure_valid_weights(model_name)

    # -- Load / init model ---------------------------------------------------
    if args.resume and Path(args.resume).exists():
        logger.info(f"  Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        logger.info(f"  Loading model: {model_name}")
        model = YOLO(model_name)

    # -- Training ------------------------------------------------------------
    t_start = time.time()
    logger.info("\n  Starting training...\n")

    results = model.train(
        data=str(Path(args.data).resolve()),
        task=args.task,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        # Optimizer
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        optimizer="auto",               # AdamW for seg/obb

        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,           # rotation useful for buildings
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,

        # Training control
        patience=args.patience,
        workers=args.workers,
        device=args.device if args.device else None,
        verbose=args.verbose,

        # Output — use absolute path so output always lands inside this project
        project=str(Path(__file__).parent / args.project),
        name=f"{args.task}/{args.name}",
        exist_ok=args.exist_ok,

        # Misc
        save=True,
        save_period=10,                 # also save every 10 epochs
        plots=True,                     # save training plots
        val=True,
        cache=False,                    # disable caching (saves RAM)
        # 3 classes: low / medium / high  (single_cls=False)
    )

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 62)
    logger.info(f"  Training complete in {elapsed / 60:.1f} minutes")

    # -- Validation on test set ----------------------------------------------
    logger.info("\n  Running final evaluation on test split...")
    run_dir = Path(__file__).parent / args.project / args.task / args.name
    best_weights = run_dir / "weights" / "best.pt"

    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        test_results = best_model.val(
            data=str(Path(args.data).resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            verbose=True,
        )
        logger.info("\n  Test Results:")
        try:
            if args.task == "segment":
                logger.info(f"    Box mAP50   : {test_results.box.map50:.4f}")
                logger.info(f"    Seg mAP50   : {test_results.seg.map50:.4f}")
                logger.info(f"    Seg mAP50-95: {test_results.seg.map:.4f}")
                logger.info(f"    Precision   : {test_results.seg.mp:.4f}")
                logger.info(f"    Recall      : {test_results.seg.mr:.4f}")
            else:
                logger.info(f"    mAP50       : {test_results.box.map50:.4f}")
                logger.info(f"    mAP50-95    : {test_results.box.map:.4f}")
                logger.info(f"    Precision   : {test_results.box.mp:.4f}")
                logger.info(f"    Recall      : {test_results.box.mr:.4f}")
        except Exception:
            logger.info("    (see detailed results in the val run directory)")
    else:
        logger.warning(f"  Best weights not found at {best_weights}")

    # -- Summary -------------------------------------------------------------
    logger.info("\n" + "=" * 62)
    logger.info("  OUTPUT SUMMARY")
    logger.info("=" * 62)
    logger.info(f"  Best model  : {run_dir}/weights/best.pt")
    logger.info(f"  Last model  : {run_dir}/weights/last.pt")
    logger.info(f"  Training CSV: {run_dir}/results.csv")
    logger.info(f"  Plots       : {run_dir}/*.png")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
