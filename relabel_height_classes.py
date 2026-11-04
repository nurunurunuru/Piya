"""
relabel_height_classes.py
══════════════════════════════════════════════════════════════════════════
Relabels YOLO polygon labels from single class (0 = building) to three
building height categories based on polygon area:

  Class 0 → low    (small footprint  → 1-3 floors)
  Class 1 → medium (medium footprint → 4-8 floors)
  Class 2 → high   (large footprint  → 9+ floors)

Strategy: Compute the normalized polygon area for every building in
the training split, then assign percentile-based thresholds:
  - Bottom 50% by area → low
  - 50th–85th percentile → medium
  - Top 15%            → high

You can override the thresholds with --low-pct / --high-pct.

Usage:
    # Preview thresholds (dry run, no files changed):
    python relabel_height_classes.py --dry-run

    # Apply relabeling to all splits:
    python relabel_height_classes.py

    # Custom thresholds (40% low, 30% medium, 30% high):
    python relabel_height_classes.py --low-pct 40 --high-pct 70
══════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("relabel")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def polygon_area(coords: list[float]) -> float:
    """
    Computes area for bounding box [x_center, y_center, width, height]
    or polygon [x1, y1, x2, y2, ...].
    Returns normalized area (0–1 range because coords are 0–1 normalized).
    """
    if len(coords) == 4:
        return coords[2] * coords[3]  # width * height
    n = len(coords) // 2
    if n < 3:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0



def parse_label_file(path: Path) -> list[dict]:
    """
    Parse a YOLO polygon label file.
    Each line: class_id x1 y1 x2 y2 ... xn yn
    Returns list of dicts with 'cls' and 'coords'.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        rows.append({"cls": cls, "coords": coords})
    return rows


def write_label_file(path: Path, rows: list[dict]) -> None:
    """Write label rows back to file in YOLO format."""
    lines = []
    for r in rows:
        coord_str = " ".join(f"{v:.10f}" for v in r["coords"])
        lines.append(f"{r['cls']} {coord_str}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def collect_areas(label_dir: Path) -> list[float]:
    """Collect all polygon areas from a label directory."""
    areas = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        rows = parse_label_file(lbl_path)
        for r in rows:
            areas.append(polygon_area(r["coords"]))
    return areas


def assign_class(area: float, low_thresh: float, high_thresh: float) -> int:
    """
    Assign height class based on area thresholds.
      area < low_thresh  → 0 (low)
      low_thresh <= area < high_thresh → 1 (medium)
      area >= high_thresh → 2 (high)
    """
    if area < low_thresh:
        return 0
    elif area < high_thresh:
        return 1
    else:
        return 2


def relabel_split(
    label_dir: Path,
    low_thresh: float,
    high_thresh: float,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    """
    Relabel all .txt files in label_dir.
    Returns a summary dict with class counts.
    """
    counts = {0: 0, 1: 0, 2: 0}
    label_files = sorted(label_dir.glob("*.txt"))

    if not label_files:
        logger.warning(f"  No label files found in {label_dir}")
        return counts

    # Backup original labels once
    if backup and not dry_run:
        backup_dir = label_dir.parent / (label_dir.name + "_original_backup")
        if not backup_dir.exists():
            shutil.copytree(label_dir, backup_dir)
            logger.info(f"  Backup saved -> {backup_dir}")
        else:
            logger.info(f"  Backup already exists at {backup_dir}")

    for lbl_path in label_files:
        rows = parse_label_file(lbl_path)
        updated = []
        for r in rows:
            area = polygon_area(r["coords"])
            new_cls = assign_class(area, low_thresh, high_thresh)
            counts[new_cls] += 1
            updated.append({"cls": new_cls, "coords": r["coords"]})

        if not dry_run:
            write_label_file(lbl_path, updated)

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Relabel YOLO building polygons -> Low / Medium / High classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-root", default="Dataset",
                   help="Root of the dataset (parent of train/valid/test)")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   help="Which splits to relabel")
    p.add_argument("--low-pct",  type=float, default=50.0,
                   help="Percentile below which buildings are 'low' (default 50)")
    p.add_argument("--high-pct", type=float, default=85.0,
                   help="Percentile above which buildings are 'high' (default 85)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview class distribution without writing files")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip backing up original label files")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_root)

    logger.info("=" * 62)
    logger.info("  Building Height Class Relabeling")
    logger.info("=" * 62)
    logger.info(f"  Dataset root : {root.resolve()}")
    logger.info(f"  Low pct      : {args.low_pct}%")
    logger.info(f"  High pct     : {args.high_pct}%")
    logger.info(f"  Splits       : {args.splits}")
    logger.info(f"  Dry run      : {args.dry_run}")
    logger.info("=" * 62)

    # ── Step 1: Compute thresholds from training set ─────────────────────
    train_label_dir = root / "train" / "labels"
    if not train_label_dir.exists():
        logger.error(f"Training labels not found: {train_label_dir}")
        sys.exit(1)

    logger.info(f"\n  Collecting polygon areas from {train_label_dir} ...")
    all_areas = collect_areas(train_label_dir)

    if not all_areas:
        logger.error("No polygons found in training labels!")
        sys.exit(1)

    arr = np.array(all_areas)
    low_thresh  = float(np.percentile(arr, args.low_pct))
    high_thresh = float(np.percentile(arr, args.high_pct))

    logger.info(f"\n  Area statistics from {len(arr):,} building polygons:")
    logger.info(f"    Min area   : {arr.min():.6f}")
    logger.info(f"    Max area   : {arr.max():.6f}")
    logger.info(f"    Mean area  : {arr.mean():.6f}")
    logger.info(f"    Median area: {np.median(arr):.6f}")
    logger.info(f"\n  Thresholds:")
    logger.info(f"    LOW  threshold (<{args.low_pct:.0f}th pct) : area < {low_thresh:.6f}")
    logger.info(f"    HIGH threshold (>{args.high_pct:.0f}th pct) : area > {high_thresh:.6f}")
    logger.info(f"    -> Class 0 LOW    : area < {low_thresh:.6f}")
    logger.info(f"    -> Class 1 MEDIUM : {low_thresh:.6f} <= area < {high_thresh:.6f}")
    logger.info(f"    -> Class 2 HIGH   : area >= {high_thresh:.6f}")

    # ── Step 2: Relabel each split ─────────────────────────────────────────
    total = {0: 0, 1: 0, 2: 0}
    for split in args.splits:
        label_dir = root / split / "labels"
        if not label_dir.exists():
            logger.warning(f"  Skipping missing split: {label_dir}")
            continue

        logger.info(f"\n  Processing split: {split} ({label_dir})")
        counts = relabel_split(
            label_dir,
            low_thresh=low_thresh,
            high_thresh=high_thresh,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_split = sum(counts.values())
        logger.info(f"  Results ({split}):")
        logger.info(f"    Low    (class 0): {counts[0]:5d}  ({100*counts[0]/max(total_split,1):.1f}%)")
        logger.info(f"    Medium (class 1): {counts[1]:5d}  ({100*counts[1]/max(total_split,1):.1f}%)")
        logger.info(f"    High   (class 2): {counts[2]:5d}  ({100*counts[2]/max(total_split,1):.1f}%)")

        for k in total:
            total[k] += counts[k]

    # ── Summary ────────────────────────────────────────────────────────────
    grand_total = sum(total.values())
    logger.info("\n" + "=" * 62)
    logger.info("  OVERALL SUMMARY (all splits combined)")
    logger.info("=" * 62)
    logger.info(f"  Low    (class 0): {total[0]:6d}  ({100*total[0]/max(grand_total,1):.1f}%)")
    logger.info(f"  Medium (class 1): {total[1]:6d}  ({100*total[1]/max(grand_total,1):.1f}%)")
    logger.info(f"  High   (class 2): {total[2]:6d}  ({100*total[2]/max(grand_total,1):.1f}%)")
    logger.info(f"  Total buildings : {grand_total:6d}")
    logger.info("=" * 62)

    if args.dry_run:
        logger.info("\n  [DRY RUN] No files were modified.")
        logger.info("  Run without --dry-run to apply changes.")
    else:
        logger.info("\n  Relabeling complete! Original labels backed up as *_original_backup/")
        logger.info("  You can now train with: python train_buildings.py")


if __name__ == "__main__":
    main()
