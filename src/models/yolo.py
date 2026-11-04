"""
src/models/yolo.py
YOLOv9 (GELAN - Generalized Efficient Layer Aggregation Network) Architecture 
adapted for dense building height estimation & satellite segmentation.

Features:
  - RepConv / ConvBNSiLU building blocks
  - GELAN feature extraction blocks (Programmable Gradient Information compatible)
  - SPPELAN (Spatial Pyramid Pooling - ELAN) multi-scale context module
  - Feature Pyramid Network (FPN) decoder for dense pixel-level prediction
  - Supports arbitrary input channels (e.g., 17 Sentinel-2 + spectral index channels)
  - Supports both classification (Low/Med/High) and regression tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """Re-parameterizable Convolutional Block used in YOLOv9 / GELAN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ConvBNSiLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvBNSiLU(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv2(x)


class GELANBlock(nn.Module):
    """
    Generalized Efficient Layer Aggregation Network (GELAN) Block.
    Core architectural building block of YOLOv9.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_blocks: int = 2):
        super().__init__()
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.cv2 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            RepConv(hidden_channels, hidden_channels, kernel_size=3)
            for _ in range(num_blocks)
        ])

        # Concatenates cv1 + cv2 + output of all intermediate blocks
        concat_channels = hidden_channels * (2 + num_blocks)
        self.cv3 = ConvBNSiLU(concat_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        feats = [y1, y2]
        
        current = y2
        for b in self.blocks:
            current = b(current)
            feats.append(current)

        return self.cv3(torch.cat(feats, dim=1))


class SPPELAN(nn.Module):
    """
    Spatial Pyramid Pooling GELAN (SPPELAN) Module.
    Aggregates multi-scale contextual features without losing spatial resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1)
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = ConvBNSiLU(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.m1(y1)
        y3 = self.m2(y2)
        y4 = self.m3(y3)
        return self.cv2(torch.cat([y1, y2, y3, y4], dim=1))


class YOLOv9HeightEstimator(nn.Module):
    """
    YOLOv9-based Dense Segmentation & Height Estimation Network.

    Backbone: GELAN-C multi-stage feature extractor.
    Neck: SPPELAN + FPN (Feature Pyramid Network) upsampling decoder.
    Head: Dense pixel-level classification (Low/Med/High) or regression (height in metres).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 4,
        task: str = "classification",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes if task == "classification" else 1

        # ── Stage 1: Stem ──────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, 32, kernel_size=3, stride=1),
            ConvBNSiLU(32, 64, kernel_size=3, stride=2),   # 1/2 resolution
        )

        # ── Stage 2: GELAN Stage 1 ──────────────────────────────────────────
        self.gelan1 = GELANBlock(64, 128, hidden_channels=64, num_blocks=2)
        self.down1 = ConvBNSiLU(128, 128, kernel_size=3, stride=2)  # 1/4 resolution

        # ── Stage 3: GELAN Stage 2 ──────────────────────────────────────────
        self.gelan2 = GELANBlock(128, 256, hidden_channels=128, num_blocks=3)
        self.down2 = ConvBNSiLU(256, 256, kernel_size=3, stride=2)  # 1/8 resolution

        # ── Stage 4: GELAN Stage 3 ──────────────────────────────────────────
        self.gelan3 = GELANBlock(256, 512, hidden_channels=256, num_blocks=3)
        self.down3 = ConvBNSiLU(512, 512, kernel_size=3, stride=2)  # 1/16 resolution

        # ── Stage 5: SPPELAN Bottleneck ─────────────────────────────────────
        self.spp = SPPELAN(512, 512, hidden_channels=256)

        # ── Decoder / FPN Neck ─────────────────────────────────────────────
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBNSiLU(512 + 256, 256, kernel_size=3)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBNSiLU(256 + 256, 128, kernel_size=3)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBNSiLU(128 + 64, 64, kernel_size=3)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # ── Prediction Head ────────────────────────────────────────────────
        out_ch = self.num_classes
        if task == "classification":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                ConvBNSiLU(64, 32, kernel_size=3),
                nn.Conv2d(32, out_ch, kernel_size=1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                ConvBNSiLU(64, 32, kernel_size=3),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.ReLU(),  # Non-negative height
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (GELAN Backbone)
        f1 = self.stem(x)          # [B, 64, H/2, W/2]
        
        f2 = self.gelan1(f1)       # [B, 128, H/2, W/2]
        d1 = self.down1(f2)        # [B, 128, H/4, W/4]

        f3 = self.gelan2(d1)       # [B, 256, H/4, W/4]
        d2 = self.down2(f3)        # [B, 256, H/8, W/8]

        f4 = self.gelan3(d2)       # [B, 512, H/8, W/8]
        d3 = self.down3(f4)        # [B, 512, H/16, W/16]

        b = self.spp(d3)           # [B, 512, H/16, W/16]

        # Decoder (FPN Skip Connections)
        u3 = self.up3(b)                                      # [B, 512, H/8, W/8]
        p3 = self.dec3(torch.cat([u3, d2], dim=1))            # [B, 256, H/8, W/8]

        u2 = self.up2(p3)                                     # [B, 256, H/4, W/4]
        p2 = self.dec2(torch.cat([u2, f3], dim=1))            # [B, 128, H/4, W/4]

        u1 = self.up1(p2)                                     # [B, 128, H/2, W/2]
        p1 = self.dec1(torch.cat([u1, f1], dim=1))            # [B, 64, H/2, W/2]

        feat = self.final_up(p1)                              # [B, 64, H, W]
        return self.head(feat)

    @property
    def name(self) -> str:
        return f"yolov9_gelan_{self.task}"


# ═══════════════════════════════════════════════════════════════════════════════
# YOLOv8 Architecture (C2f + SPPF)
# ═══════════════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """Standard Bottleneck block used in YOLOv8 C2f."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, c_, 3, 1)
        self.cv2 = ConvBNSiLU(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (C2f) - Core building block of YOLOv8.
    Provides faster and richer gradient flow for satellite feature extraction.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv8."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOv8HeightEstimator(nn.Module):
    """
    YOLOv8-based Dense Segmentation & Height Estimation Network.

    Backbone: Darknet-v8 with C2f feature extraction blocks.
    Neck: SPPF + FPN (Feature Pyramid Network) decoder.
    Head: Pixel-level height classification or continuous regression.
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_classes: int = 4,
        task: str = "classification",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.task = task
        self.num_classes = num_classes if task == "classification" else 1

        # ── Backbone ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, 32, kernel_size=3, stride=1),
            ConvBNSiLU(32, 64, kernel_size=3, stride=2),          # 1/2 resolution
        )

        self.c2f_1 = C2f(64, 128, n=1)
        self.down1 = ConvBNSiLU(128, 128, kernel_size=3, stride=2) # 1/4 resolution

        self.c2f_2 = C2f(128, 256, n=2)
        self.down2 = ConvBNSiLU(256, 256, kernel_size=3, stride=2) # 1/8 resolution

        self.c2f_3 = C2f(256, 512, n=2)
        self.down3 = ConvBNSiLU(512, 512, kernel_size=3, stride=2) # 1/16 resolution

        self.sppf = SPPF(512, 512)

        # ── Decoder / FPN Neck ─────────────────────────────────────────────
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = C2f(512 + 256, 256, n=1)

        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = C2f(256 + 256, 128, n=1)

        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = C2f(128 + 64, 64, n=1)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # ── Head ───────────────────────────────────────────────────────────
        out_ch = self.num_classes
        if task == "classification":
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                ConvBNSiLU(64, 32, kernel_size=3),
                nn.Conv2d(32, out_ch, kernel_size=1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout2d(p=dropout),
                ConvBNSiLU(64, 32, kernel_size=3),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stem(x)          # [B, 64, H/2, W/2]
        
        f2 = self.c2f_1(f1)        # [B, 128, H/2, W/2]
        d1 = self.down1(f2)        # [B, 128, H/4, W/4]

        f3 = self.c2f_2(d1)        # [B, 256, H/4, W/4]
        d2 = self.down2(f3)        # [B, 256, H/8, W/8]

        f4 = self.c2f_3(d2)        # [B, 512, H/8, W/8]
        d3 = self.down3(f4)        # [B, 512, H/16, W/16]

        b = self.sppf(d3)          # [B, 512, H/16, W/16]

        # Decoder
        u3 = self.up3(b)
        p3 = self.dec3(torch.cat([u3, d2], dim=1))            # [B, 256, H/8, W/8]

        u2 = self.up2(p3)
        p2 = self.dec2(torch.cat([u2, f3], dim=1))            # [B, 128, H/4, W/4]

        u1 = self.up1(p2)
        p1 = self.dec1(torch.cat([u1, f1], dim=1))            # [B, 64, H/2, W/2]

        feat = self.final_up(p1)                              # [B, 64, H, W]
        return self.head(feat)

    @property
    def name(self) -> str:
        return f"yolov8_c2f_{self.task}"

