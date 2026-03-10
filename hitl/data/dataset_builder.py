"""Dataset builder: converts vector labels + raster imagery into tiled training data.

Pipeline:
1. Read annotation regions and annotation polygons from LabelStore
2. For each region, fetch the underlying raster imagery
3. Rasterize annotations:
   - Inside region + has class polygon → class_id
   - Inside region + no polygon → background (class_id=1)
   - Outside all regions → ignore_index (0)
4. Tile imagery + mask into (tile_size x tile_size) patches
5. Spatial train/val/test split by geographic blocks
6. Write to disk as GeoTIFF tiles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

from .label_store import LabelStore
from .raster_source import GeoTIFFSource, RasterSource

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics from dataset building."""

    num_regions: int = 0
    num_annotations: int = 0
    num_tiles_total: int = 0
    num_tiles_train: int = 0
    num_tiles_val: int = 0
    num_tiles_test: int = 0
    num_skipped_low_label: int = 0
    class_pixel_counts: Dict[int, int] = field(default_factory=dict)
    target_crs: str = ""


class DatasetBuilder:
    """Builds a tiled training dataset from labels + imagery.

    Args:
        label_store: LabelStore with classes, annotations, regions.
        raster_sources: Dict mapping source_id → RasterSource.
        tile_size: Tile size in pixels.
        tile_overlap: Overlap in pixels between adjacent tiles.
        ignore_index: Class ID for unlabeled (outside regions) pixels.
        background_class_id: Class ID for background (inside region, no label).
        min_labeled_fraction: Skip tiles with less than this fraction of non-ignore pixels.
        val_fraction: Fraction of spatial blocks for validation.
        test_fraction: Fraction of spatial blocks for test.
        split_block_size: Spatial block size in CRS units for train/val/test split.
    """

    def __init__(
        self,
        label_store: LabelStore,
        tile_size: int = 512,
        tile_overlap: int = 64,
        ignore_index: int = 0,
        background_class_id: int = 1,
        min_labeled_fraction: float = 0.05,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        split_block_size: float = 500.0,
    ):
        self.label_store = label_store
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.ignore_index = ignore_index
        self.background_class_id = background_class_id
        self.min_labeled_fraction = min_labeled_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_block_size = split_block_size

    def build(
        self,
        raster_source: RasterSource,
        output_dir: str | Path,
        target_crs: Optional[str] = None,
    ) -> DatasetStats:
        """Build the complete tiled dataset.

        Args:
            raster_source: Source of imagery.
            output_dir: Output directory for tiles.
            target_crs: CRS for the dataset. Defaults to raster source CRS.

        Returns:
            DatasetStats with counts and class distribution.
        """
        output_dir = Path(output_dir)
        if target_crs is None:
            target_crs = raster_source.get_crs()

        stats = DatasetStats(target_crs=target_crs)

        # Read only approved regions and annotations for training
        regions = self.label_store.get_regions(crs=target_crs, status="active")
        annotations = self.label_store.get_annotations(crs=target_crs, status="approved")
        stats.num_regions = len(regions)
        stats.num_annotations = len(annotations)

        if len(regions) == 0:
            logger.warning("No annotation regions found. Cannot build dataset.")
            return stats

        # For each region: extract imagery, rasterize labels, tile
        all_tiles = []  # (tile_id, center_x, center_y, image_array, mask_array, transform)

        for _, region_row in regions.iterrows():
            region_geom = region_row.geometry
            region_id = region_row["region_id"]
            region_bounds = region_geom.bounds  # (minx, miny, maxx, maxy)

            # Get annotations within this region
            region_annots = annotations[annotations["region_id"] == region_id]

            # Compute pixel dimensions from bounds and raster resolution
            x_res, y_res = raster_source.get_resolution()
            minx, miny, maxx, maxy = region_bounds
            width = max(1, int((maxx - minx) / x_res))
            height = max(1, int((maxy - miny) / y_res))

            # Fetch imagery for this region
            chip = raster_source.get_chip(region_bounds, width, height)
            image = chip.data  # (C, H, W)

            # Build mask
            mask = self._rasterize_region(
                region_geom=region_geom,
                annotations=region_annots,
                bounds=region_bounds,
                width=width,
                height=height,
            )

            # Tile this region
            region_tiles = self._tile_region(
                image=image,
                mask=mask,
                bounds=region_bounds,
                region_id=region_id,
                target_crs=target_crs,
            )
            all_tiles.extend(region_tiles)

        if not all_tiles:
            logger.warning("No valid tiles produced.")
            return stats

        # Spatial train/val/test split
        splits = self._spatial_split(all_tiles)

        # Write tiles to disk
        for split_name, split_tiles in splits.items():
            split_dir = output_dir / split_name
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "masks").mkdir(parents=True, exist_ok=True)

            for tile_info in split_tiles:
                tile_id = tile_info["tile_id"]
                self._write_tile(
                    split_dir / "images" / f"{tile_id}.tif",
                    tile_info["image"],
                    tile_info["transform"],
                    target_crs,
                )
                self._write_mask(
                    split_dir / "masks" / f"{tile_id}.tif",
                    tile_info["mask"],
                    tile_info["transform"],
                    target_crs,
                )

        stats.num_tiles_train = len(splits.get("train", []))
        stats.num_tiles_val = len(splits.get("val", []))
        stats.num_tiles_test = len(splits.get("test", []))
        stats.num_tiles_total = sum(len(v) for v in splits.values())
        stats.num_skipped_low_label = len(all_tiles) - stats.num_tiles_total

        # Class distribution
        for split_tiles in splits.values():
            for t in split_tiles:
                unique, counts = np.unique(t["mask"], return_counts=True)
                for cls_id, count in zip(unique, counts):
                    stats.class_pixel_counts[int(cls_id)] = (
                        stats.class_pixel_counts.get(int(cls_id), 0) + int(count)
                    )

        logger.info(
            "Dataset built: %d train, %d val, %d test tiles",
            stats.num_tiles_train,
            stats.num_tiles_val,
            stats.num_tiles_test,
        )
        return stats

    def _rasterize_region(
        self,
        region_geom,
        annotations: gpd.GeoDataFrame,
        bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Rasterize labels within an annotation region.

        Returns:
            (H, W) uint8 array with class IDs.
            - Inside region, no label → background_class_id (1)
            - Inside region, has label → class_id (2+)
            - Outside region → ignore_index (0)
        """
        transform = from_bounds(*bounds, width, height)

        # Start with ignore everywhere
        mask = np.full((height, width), self.ignore_index, dtype=np.uint8)

        # Fill region area with background
        region_shapes = [(region_geom, self.background_class_id)]
        rasterize(region_shapes, out=mask, transform=transform, dtype=np.uint8)

        # Rasterize annotation polygons on top
        if len(annotations) > 0:
            ann_shapes = [
                (row.geometry, int(row["class_id"]))
                for _, row in annotations.iterrows()
                if row.geometry is not None and not row.geometry.is_empty
            ]
            if ann_shapes:
                rasterize(ann_shapes, out=mask, transform=transform, dtype=np.uint8)

        return mask

    def _tile_region(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bounds: Tuple[float, float, float, float],
        region_id: int,
        target_crs: str,
    ) -> list:
        """Split a region's image + mask into overlapping tiles."""
        C, H, W = image.shape
        step = self.tile_size - self.tile_overlap
        tiles = []
        tile_idx = 0

        minx, miny, maxx, maxy = bounds
        x_res = (maxx - minx) / W
        y_res = (maxy - miny) / H

        for y_start in range(0, H, step):
            for x_start in range(0, W, step):
                y_end = min(y_start + self.tile_size, H)
                x_end = min(x_start + self.tile_size, W)

                # Skip incomplete tiles at edges
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                if tile_h < self.tile_size // 2 or tile_w < self.tile_size // 2:
                    continue

                img_tile = image[:, y_start:y_end, x_start:x_end]
                mask_tile = mask[y_start:y_end, x_start:x_end]

                # Pad if needed
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    pad_img = np.zeros(
                        (C, self.tile_size, self.tile_size), dtype=img_tile.dtype
                    )
                    pad_mask = np.full(
                        (self.tile_size, self.tile_size),
                        self.ignore_index,
                        dtype=mask_tile.dtype,
                    )
                    pad_img[:, :tile_h, :tile_w] = img_tile
                    pad_mask[:tile_h, :tile_w] = mask_tile
                    img_tile = pad_img
                    mask_tile = pad_mask

                # Check labeled fraction
                labeled_fraction = np.mean(mask_tile != self.ignore_index)
                if labeled_fraction < self.min_labeled_fraction:
                    continue

                # Compute geo transform for this tile
                tile_minx = minx + x_start * x_res
                tile_maxy = maxy - y_start * y_res
                tile_transform = rasterio.transform.from_origin(
                    tile_minx, tile_maxy, x_res, y_res
                )

                tile_id = f"r{region_id}_t{tile_idx:04d}"
                center_x = tile_minx + (self.tile_size * x_res) / 2
                center_y = tile_maxy - (self.tile_size * y_res) / 2

                tiles.append(
                    {
                        "tile_id": tile_id,
                        "image": img_tile,
                        "mask": mask_tile,
                        "transform": tile_transform,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )
                tile_idx += 1

        return tiles

    def _spatial_split(self, tiles: list) -> Dict[str, list]:
        """Split tiles into train/val/test by spatial blocks.

        Assigns each tile to a block based on its center coordinate,
        then assigns blocks to splits. This prevents spatial leakage.
        """
        if not tiles:
            return {"train": [], "val": [], "test": []}

        # Assign each tile to a spatial block
        block_assignments = {}
        for tile in tiles:
            bx = int(tile["center_x"] / self.split_block_size)
            by = int(tile["center_y"] / self.split_block_size)
            block_key = (bx, by)
            if block_key not in block_assignments:
                block_assignments[block_key] = []
            block_assignments[block_key].append(tile)

        # Shuffle and assign blocks to splits
        block_keys = list(block_assignments.keys())
        rng = np.random.RandomState(42)  # deterministic
        rng.shuffle(block_keys)

        n_blocks = len(block_keys)
        n_test = max(1, int(n_blocks * self.test_fraction))
        n_val = max(1, int(n_blocks * self.val_fraction))

        test_blocks = set(map(tuple, block_keys[:n_test]))
        val_blocks = set(map(tuple, block_keys[n_test : n_test + n_val]))
        train_blocks = set(map(tuple, block_keys[n_test + n_val :]))

        # If only 1-2 blocks, put everything in train
        if n_blocks <= 2:
            return {"train": tiles, "val": [], "test": []}

        splits = {"train": [], "val": [], "test": []}
        for block_key, block_tiles in block_assignments.items():
            if block_key in test_blocks:
                splits["test"].extend(block_tiles)
            elif block_key in val_blocks:
                splits["val"].extend(block_tiles)
            else:
                splits["train"].extend(block_tiles)

        return splits

    @staticmethod
    def _write_tile(path: Path, data: np.ndarray, transform, crs: str) -> None:
        """Write an image tile as GeoTIFF."""
        C, H, W = data.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=C,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

    @staticmethod
    def _write_mask(path: Path, data: np.ndarray, transform, crs: str) -> None:
        """Write a mask tile as single-band GeoTIFF."""
        H, W = data.shape
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
