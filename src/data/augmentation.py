"""
src/data/augmentation.py
Albumentations-based augmentation pipeline for satellite image patches.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """Strong augmentations for training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                        min_holes=1, fill_value=0, p=0.2),
    ], additional_targets={"label": "image"})


def get_val_transforms() -> A.Compose:
    """Minimal transforms for validation/test."""
    return A.Compose([], additional_targets={"label": "image"})


def apply_transforms(
    image: np.ndarray,
    label: np.ndarray,
    transform: A.Compose,
) -> tuple:
    """
    Apply albumentations transforms to image [C,H,W] and label [1,H,W].
    Albumentations expects [H,W,C] format.
    """
    # Convert to HWC
    img_hwc = image.transpose(1, 2, 0)   # [H, W, C]
    lbl_hwc = label.transpose(1, 2, 0)   # [H, W, 1]

    augmented = transform(image=img_hwc, label=lbl_hwc)

    # Back to CHW
    aug_img = augmented["image"].transpose(2, 0, 1)
    aug_lbl = augmented["label"].transpose(2, 0, 1)

    return aug_img, aug_lbl
