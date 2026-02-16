"""Factory function for building a Segmentor from AppConfig."""

from __future__ import annotations

from hitl.models.segmentor import Segmentor


def build_segmentor(config, num_classes: int) -> Segmentor:
    """Build a Segmentor from AppConfig."""
    dinov3 = config.models.dinov3
    train = config.training

    return Segmentor.build(
        model_path=dinov3.path,
        num_classes=num_classes,
        extract_layers=tuple(dinov3.extract_layers),
        patch_size=dinov3.patch_size,
        num_register_tokens=dinov3.num_register_tokens,
        freeze_backbone=train.freeze_backbone,
        fpn_channels=train.fpn_channels,
        neck_out_channels=tuple(train.neck_out_channels),
        neck_scale_factors=tuple(train.neck_scale_factors),
    )
