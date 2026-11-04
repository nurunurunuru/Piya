"""
src/models/transformer.py
Swin Transformer-based height estimator using timm backbone.
"""

import torch
import torch.nn as nn
import timm


class SwinTransformerHeightEstimator(nn.Module):
    """
    Swin Transformer encoder + lightweight CNN decoder for height regression.
    Captures long-range spatial dependencies across large urban areas.

    Input:  [B, in_channels, H, W]  (H=W=256)
    Output: [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 12,
        img_size: int = 256,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        embed_dim: int = 128,
    ):
        super().__init__()

        # Input projection: adapt N-channel input to 3 channels for pretrained model
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # Swin Transformer encoder (feature extractor only)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
        )

        # Get encoder output channels
        encoder_channels = self.encoder.feature_info.channels()

        # Upsampling decoder
        self.decoder = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for out_ch in [512, 256, 128, 64, 32]:
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(),   # Heights >= 0
        )

        # Ensure output resolution matches input
        self.upsample_final = nn.Upsample(size=(img_size, img_size), mode="bilinear",
                                           align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapt input channels
        x_rgb = self.input_proj(x)

        # Encode
        features = self.encoder(x_rgb)   # List of feature maps at different scales
        feat = features[-1]              # Use deepest features

        # Rearrange if needed (Swin outputs [B, H, W, C])
        if feat.dim() == 4 and feat.shape[-1] != feat.shape[-2]:
            feat = feat.permute(0, 3, 1, 2).contiguous()

        # Decode
        for layer in self.decoder:
            feat = layer(feat)

        out = self.head(feat)
        return self.upsample_final(out)

    @property
    def name(self) -> str:
        return "swin_transformer"
