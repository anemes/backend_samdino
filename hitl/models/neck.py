"""Neck adapter: projects single-resolution ViT features to multi-scale for UperNet.

DINOv3 ViT outputs features at a single spatial resolution from all layers
(all are H_p x W_p, where H_p = H // patch_size). UperNet's FPN expects a
multi-scale pyramid like a CNN produces (1/4, 1/8, 1/16, 1/32).

This adapter:
1. Projects each layer's features to different channel widths via 1x1 convs
2. Upsamples/downsamples to simulate scale hierarchy
3. Produces 4 feature maps at different scales for FPN

This is the standard ViT→dense-prediction adapter pattern (mmseg MLN).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeckAdapter(nn.Module):
    """Multi-Level Neck: project + rescale ViT features for FPN.

    Args:
        in_channels: Channel count of backbone features (1024 for ViT-L).
        out_channels: Per-level output channels, e.g. (256, 512, 1024, 1024).
        scale_factors: Per-level spatial scale relative to input, e.g. (4.0, 2.0, 1.0, 0.5).
            >1 = upsample (finer), <1 = downsample (coarser).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: Tuple[int, ...] = (256, 512, 1024, 1024),
        scale_factors: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
    ):
        super().__init__()
        assert len(out_channels) == len(scale_factors)

        self.projections = nn.ModuleList()
        for out_c in out_channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_c, 1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            )

        self.scale_factors = scale_factors
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project and rescale features.

        Args:
            features: List of (B, in_channels, H_p, W_p) from backbone.

        Returns:
            List of (B, out_channels[i], H_i, W_i) at different scales.
        """
        assert len(features) == len(self.projections)

        scaled = []
        for feat, proj, scale in zip(features, self.projections, self.scale_factors):
            out = proj(feat)
            if scale != 1.0:
                out = F.interpolate(
                    out,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            scaled.append(out)

        return scaled
