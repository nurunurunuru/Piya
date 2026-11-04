"""
src/models/unet.py
U-Net with ResNet-50 encoder for building height estimation.
Supports both regression (height in meters) and classification (Low/Medium/High).
Uses segmentation_models_pytorch as backbone.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# Height class thresholds (meters)
# Class 0: Background (no building, height < 3m)
# Class 1: Low       (3m – 15m,  ~1–4 stories)
# Class 2: Medium    (15m – 30m, ~5–9 stories)
# Class 3: High      (> 30m,     ~10+ stories)
HEIGHT_THRESHOLDS = [3.0, 15.0, 30.0]   # bin edges
NUM_HEIGHT_CLASSES = 4
CLASS_NAMES = ["Background", "Low (3–15m)", "Medium (15–30m)", "High (>30m)"]
CLASS_COLORS = ["#2d2d2d", "#4CAF50", "#FFC107", "#F44336"]  # dark, green, amber, red


class UNetHeightEstimator(nn.Module):
    """
    U-Net with ResNet-50 encoder for building height estimation.

    Two modes:
        regression      → Output: [B, 1, H, W] — predicted height in metres
        classification  → Output: [B, 4, H, W] — raw logits per height class

    Input:  [B, in_channels, H, W]
    """

    def __init__(
        self,
        in_channels:       int  = 12,
        encoder_name:      str  = "resnet50",
        encoder_weights:   str  = "imagenet",
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        dropout:          float = 0.2,
        task:              str  = "classification",   # "regression" | "classification"
        num_classes:       int  = NUM_HEIGHT_CLASSES,
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes if task == "classification" else 1

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=self.num_classes,
            activation=None,                    # raw logits / raw regression values
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )

        if task == "regression":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.ReLU(),                      # Heights ≥ 0
            )
        else:
            # Classification head: dropout → 1×1 conv for per-class logits
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, padding=1),
                # No activation — CrossEntropyLoss expects raw logits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.unet(x)
        return self.head(feat)

    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax class indices [B, H, W] for classification mode."""
        logits = self.forward(x)
        return logits.argmax(dim=1)

    @property
    def name(self) -> str:
        return f"unet_resnet50_{self.task}"


class UNetPlusPlusHeightEstimator(nn.Module):
    """
    UNet++ (nested skip connections) variant.
    Supports regression and classification modes.
    """

    def __init__(
        self,
        in_channels:  int = 12,
        encoder_name: str = "resnet50",
        task:         str = "classification",
        num_classes:  int = NUM_HEIGHT_CLASSES,
    ):
        super().__init__()
        self.task = task
        out_ch = num_classes if task == "classification" else 1

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_ch,
            activation=None,
        )
        self.final = nn.ReLU() if task == "regression" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.model(x))

    @property
    def name(self) -> str:
        return f"unetplusplus_resnet50_{self.task}"
