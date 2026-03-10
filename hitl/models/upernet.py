"""UperNet segmentation head: Pyramid Pooling Module + Feature Pyramid Network.

UperNet (Xiao et al., 2018) combines:
- PPM on the deepest feature map for global context
- FPN for multi-scale feature fusion
- Final fusion + classifier for per-pixel prediction

This is the standard segmentation head used by DINOv2/v3 for evaluation,
and is the natural pairing for ViT backbones in dense prediction tasks.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module (Zhao et al., PSPNet).

    Pools features at multiple scales and concatenates with the input
    to capture global context at different granularities.
    """

    def __init__(
        self,
        in_channels: int,
        pool_channels: int = 256,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, pool_channels, 1, bias=False),
                    nn.BatchNorm2d(pool_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.out_channels = in_channels + pool_channels * len(pool_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PPM and concatenate all pooled features with input.

        Args:
            x: (B, C, H, W)

        Returns:
            (B, C + pool_channels * num_scales, H, W)
        """
        H, W = x.shape[2:]
        outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(pooled)
        return torch.cat(outs, dim=1)


class FPN(nn.Module):
    """Feature Pyramid Network with top-down pathway.

    Takes multi-scale features and produces refined features at each scale
    via lateral connections and top-down merging.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply FPN with top-down merging.

        Args:
            features: List of (B, C_i, H_i, W_i) from coarsest to finest,
                      or any ordering — laterals handle channel projection.

        Returns:
            List of (B, out_channels, H_i, W_i) refined features.
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: from deepest (smallest spatial) to shallowest
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]


class UperNetHead(nn.Module):
    """UperNet segmentation head.

    Architecture:
    1. PPM on the deepest feature map → bottleneck
    2. Replace deepest FPN input with PPM output
    3. FPN top-down fusion across all scales
    4. Upsample all FPN outputs to largest scale, concatenate
    5. Fusion conv → classifier → per-pixel logits

    Args:
        in_channels_list: Channel counts for each input scale level.
        num_classes: Number of output segmentation classes.
        fpn_channels: Internal FPN channel width (default 256).
        pool_scales: PPM pooling scales.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        fpn_channels: int = 256,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)

        # PPM on deepest feature
        self.ppm = PPM(in_channels_list[-1], pool_channels=fpn_channels, pool_scales=pool_scales)

        # Bottleneck to reduce PPM output to fpn_channels
        self.ppm_bottleneck = nn.Sequential(
            nn.Conv2d(self.ppm.out_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # FPN: replace deepest input channels with fpn_channels (PPM output)
        fpn_in_channels = list(in_channels_list[:-1]) + [fpn_channels]
        self.fpn = FPN(fpn_in_channels, fpn_channels)

        # Fusion: concatenate all FPN levels → single feature map
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * self.num_levels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute segmentation logits.

        Args:
            features: List of multi-scale feature maps from NeckAdapter.
                      Ordered from finest (largest spatial) to coarsest.

        Returns:
            (B, num_classes, H_finest, W_finest) logits at the finest scale.
        """
        # PPM on deepest (coarsest) feature
        ppm_out = self.ppm(features[-1])
        ppm_out = self.ppm_bottleneck(ppm_out)

        # FPN with PPM output replacing deepest input
        fpn_in = list(features[:-1]) + [ppm_out]
        fpn_outs = self.fpn(fpn_in)

        # Upsample all to finest scale and concatenate
        target_size = fpn_outs[0].shape[2:]
        resized = []
        for f in fpn_outs:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            resized.append(f)

        fused = self.fusion(torch.cat(resized, dim=1))
        return self.classifier(fused)
