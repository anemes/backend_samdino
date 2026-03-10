"""Full segmentation model: DINOv3 backbone + NeckAdapter + UperNet head.

Composes the three components into a single nn.Module with convenience
methods for training (returns loss) and inference (returns predictions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone
from .neck import NeckAdapter
from .upernet import UperNetHead


class Segmentor(nn.Module):
    """Composed segmentation model.

    Forward pass:
        input image → backbone (frozen) → neck (trainable) → head (trainable) → logits
        logits are bilinearly upsampled to input resolution.
    """

    def __init__(
        self,
        backbone: DINOv3Backbone,
        neck: NeckAdapter,
        head: UperNetHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @classmethod
    def build(
        cls,
        model_path: str,
        num_classes: int,
        extract_layers: tuple[int, ...] = (6, 12, 18, 24),
        patch_size: int = 16,
        num_register_tokens: int = 4,
        freeze_backbone: bool = True,
        fpn_channels: int = 256,
        neck_out_channels: tuple[int, ...] = (256, 512, 1024, 1024),
        neck_scale_factors: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
    ) -> "Segmentor":
        """Factory method to build a Segmentor from config parameters."""
        backbone = DINOv3Backbone(
            model_path=model_path,
            extract_layers=extract_layers,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            freeze=freeze_backbone,
        )
        neck = NeckAdapter(
            in_channels=backbone.hidden_size,
            out_channels=neck_out_channels,
            scale_factors=neck_scale_factors,
        )
        head = UperNetHead(
            in_channels_list=list(neck_out_channels),
            num_classes=num_classes,
            fpn_channels=fpn_channels,
        )
        return cls(backbone=backbone, neck=neck, head=head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: (B, 3, H, W) normalized input images.

        Returns:
            (B, num_classes, H, W) logits upsampled to input resolution.
        """
        input_size = x.shape[2:]

        # Backbone (frozen by default, no grad)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Neck: project + rescale
        multi_scale = self.neck(features)

        # Head: FPN + PPM + classifier
        logits = self.head(multi_scale)

        # Upsample to input resolution
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only trainable parameters (neck + head, optionally unfrozen backbone layers)."""
        for param in self.neck.parameters():
            if param.requires_grad:
                yield param
        for param in self.head.parameters():
            if param.requires_grad:
                yield param
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    def save_checkpoint(self, path: str | Path, metadata: Optional[dict] = None) -> None:
        """Save trainable state (neck + head + any unfrozen backbone layers)."""
        state = {
            "neck": self.neck.state_dict(),
            "head": self.head.state_dict(),
        }
        # Save backbone state only if it has trainable params
        if any(p.requires_grad for p in self.backbone.parameters()):
            state["backbone_trainable"] = {
                k: v
                for k, v in self.backbone.model.state_dict().items()
                if any(
                    v.data_ptr() == p.data_ptr()
                    for p in self.backbone.parameters()
                    if p.requires_grad
                )
            }
        if metadata:
            state["metadata"] = metadata
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load trainable state. Returns metadata dict if present."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.neck.load_state_dict(state["neck"])
        self.head.load_state_dict(state["head"])
        if "backbone_trainable" in state:
            self.backbone.model.load_state_dict(state["backbone_trainable"], strict=False)
        return state.get("metadata", {})
