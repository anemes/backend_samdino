"""PyTorch Dataset for tiled image + mask pairs.

Reads tiles produced by DatasetBuilder:
    {split}/images/{tile_id}.tif  (3-band uint8 GeoTIFF)
    {split}/masks/{tile_id}.tif   (1-band uint8 GeoTIFF, class IDs)

Applies augmentations (training) or just normalization (validation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .transforms import build_train_transforms, build_val_transforms

logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """Dataset of tiled image + mask pairs for segmentation training.

    Args:
        root: Directory containing images/ and masks/ subdirectories.
        split: "train", "val", or "test".
        tile_size: Expected tile size in pixels.
        transform: albumentations Compose pipeline. If None, uses defaults.
        ignore_index: Class ID to ignore in loss (default 0).
        min_labeled_fraction: Skip tiles where labeled pixels < this fraction.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 512,
        norm_mean: Tuple[float, ...] = (0.430, 0.411, 0.296),
        norm_std: Tuple[float, ...] = (0.213, 0.156, 0.143),
        augment: bool = True,
        ignore_index: int = 0,
        min_labeled_fraction: float = 0.05,
        aug_config: Optional[dict] = None,
    ):
        self.root = Path(root) / split
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.tile_size = tile_size
        self.ignore_index = ignore_index

        # Build transform
        if augment and split == "train":
            cfg = aug_config or {}
            self.transform = build_train_transforms(
                tile_size=tile_size, norm_mean=norm_mean, norm_std=norm_std, **cfg
            )
        else:
            self.transform = build_val_transforms(norm_mean=norm_mean, norm_std=norm_std)

        # Discover tiles
        self.tile_ids = self._discover_tiles(min_labeled_fraction)
        logger.info("TileDataset %s: %d tiles (split=%s)", root, len(self.tile_ids), split)

    def _discover_tiles(self, min_labeled_fraction: float) -> List[str]:
        """Find valid tiles (sufficient labeled pixels)."""
        if not self.images_dir.exists():
            return []

        tile_ids = []
        for img_path in sorted(self.images_dir.glob("*.tif")):
            tile_id = img_path.stem
            mask_path = self.masks_dir / f"{tile_id}.tif"
            if not mask_path.exists():
                continue

            # Check labeled fraction
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            labeled_fraction = np.mean(mask != self.ignore_index)
            if labeled_fraction >= min_labeled_fraction:
                tile_ids.append(tile_id)

        return tile_ids

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_id = self.tile_ids[idx]

        # Read image
        img_path = self.images_dir / f"{tile_id}.tif"
        with rasterio.open(img_path) as src:
            image = src.read()  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # (H, W, C) for albumentations

        # Read mask
        mask_path = self.masks_dir / f"{tile_id}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image_t = transformed["image"]  # (C, H, W) float32, normalized
        mask_t = transformed["mask"]  # (H, W) int

        return {
            "image": image_t,
            "mask": mask_t.long(),
            "tile_id": tile_id,
        }

    def get_class_distribution(self) -> Dict[int, int]:
        """Count pixels per class across all tiles."""
        counts: Dict[int, int] = {}
        for tile_id in self.tile_ids:
            mask_path = self.masks_dir / f"{tile_id}.tif"
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            unique, cts = np.unique(mask, return_counts=True)
            for cls_id, count in zip(unique, cts):
                counts[int(cls_id)] = counts.get(int(cls_id), 0) + int(count)
        return counts
