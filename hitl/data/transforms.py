"""Data transforms: augmentations and normalization for training/inference.

Uses albumentations for spatial augmentations (applied to both image and mask)
and torchvision-style normalization for DINOv3 input preprocessing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def build_train_transforms(
    tile_size: int = 512,
    norm_mean: Tuple[float, ...] = (0.430, 0.411, 0.296),
    norm_std: Tuple[float, ...] = (0.213, 0.156, 0.143),
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    rotate90: bool = True,
    color_jitter: bool = True,
    random_scale: Optional[List[float]] = None,
) -> A.Compose:
    """Build training augmentation pipeline.

    Applied to both image and mask simultaneously to keep alignment.
    """
    transforms = []

    # Random scaling (resize + crop back to tile_size)
    if random_scale and len(random_scale) == 2:
        transforms.append(
            A.RandomScale(scale_limit=(random_scale[0] - 1.0, random_scale[1] - 1.0), p=0.5)
        )
        transforms.append(A.PadIfNeeded(min_height=tile_size, min_width=tile_size, border_mode=0))
        transforms.append(A.RandomCrop(height=tile_size, width=tile_size))

    if horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    if rotate90:
        transforms.append(A.RandomRotate90(p=0.5))
    if color_jitter:
        transforms.append(
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
        )

    # Normalize and convert to tensor
    transforms.extend(
        [
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def build_val_transforms(
    norm_mean: Tuple[float, ...] = (0.430, 0.411, 0.296),
    norm_std: Tuple[float, ...] = (0.213, 0.156, 0.143),
) -> A.Compose:
    """Build validation/inference transform pipeline (normalize only)."""
    return A.Compose(
        [
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )


def normalize_for_inference(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.430, 0.411, 0.296),
    std: Tuple[float, ...] = (0.213, 0.156, 0.143),
) -> torch.Tensor:
    """Normalize a single image for inference.

    Args:
        image: (H, W, 3) uint8 numpy array.

    Returns:
        (1, 3, H, W) float32 tensor, normalized.
    """
    transform = build_val_transforms(norm_mean=mean, norm_std=std)
    result = transform(image=image)
    return result["image"].unsqueeze(0)
