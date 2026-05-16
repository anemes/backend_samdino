"""Tiler: splits an AOI into overlapping geo-referenced patches for inference.

Fetches the AOI image from the raster source, then slices chips
directly with numpy at exact pixel boundaries. 

A margin is used to handle edge preditions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..data.raster_source import RasterSource

logger = logging.getLogger(__name__)


@dataclass
class InferenceTile:
    """A single tile for inference with its geo metadata."""

    tile_idx: int
    image: np.ndarray  # (C, H, W) uint8
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    pixel_offset: Tuple[int, int]  # (x_off, y_off) in the full output raster


class Tiler:
    """Generate overlapping tiles from a raster source over an AOI.

    Fetches the full AOI (plus optional margin) as a single image, then
    extracts overlapping patches via numpy slicing.

    Args:
        patch_size: Tile size in pixels.
        overlap: Overlap in pixels between adjacent tiles.
    """

    def __init__(self, patch_size: int = 512, overlap: int = 128):
        self.patch_size = patch_size
        self.overlap = overlap

    def tile(
        self,
        raster_source: RasterSource,
        aoi_bounds: Tuple[float, float, float, float],
        margin: int = 0,
    ) -> Tuple[List[InferenceTile], Tuple[int, int], Tuple[int, int, int, int]]:
        """Generate tiles covering the AOI.

        Args:
            raster_source: Source of imagery.
            aoi_bounds: (minx, miny, maxx, maxy) in source CRS.
            margin: Extra pixels to fetch beyond AOI in each direction.
                    Ensures edge tiles have real imagery context instead
                    of zero-padding, allowing proper predictions at the edge.

        Returns:
            (tiles, output_shape, crop_box) where:
            - output_shape is (height, width) of the full stitched raster
              (including margin).
            - crop_box is (top, left, height, width) to trim the stitched
              output back to the original AOI.
        """
        x_res, y_res = raster_source.get_resolution()
        minx, miny, maxx, maxy = aoi_bounds

        # Extend bounds by margin pixels for real edge context
        ext_minx = minx - margin * x_res
        ext_miny = miny - margin * y_res
        ext_maxx = maxx + margin * x_res
        ext_maxy = maxy + margin * y_res

        # Output grid dimensions for the extended area
        full_width = round((ext_maxx - ext_minx) / x_res)
        full_height = round((ext_maxy - ext_miny) / y_res)
        x_res = (ext_maxx - ext_minx) / full_width
        y_res = (ext_maxy - ext_miny) / full_height

        # Crop box to trim back to the original AOI after stitching
        crop_top = margin
        crop_left = margin
        crop_h = round((maxy - miny) / y_res)
        crop_w = round((maxx - minx) / x_res)

        # Fetch the extended AOI as a single image at the target resolution.
        logger.info(
            "Fetching AOI image: %d×%d px (%.1f×%.1f m, margin=%d px)",
            full_width, full_height,
            ext_maxx - ext_minx, ext_maxy - ext_miny, margin,
        )
        full_chip = raster_source.get_chip(
            bounds=(ext_minx, ext_miny, ext_maxx, ext_maxy),
            width=full_width, height=full_height,
        )
        full_image = full_chip.data  # (C, full_height, full_width)

        # Slice into overlapping patches using exact pixel indices
        step = self.patch_size - self.overlap
        tiles = []
        tile_idx = 0

        for y_off in range(0, full_height, step):
            for x_off in range(0, full_width, step):
                # Actual slice extent (may be smaller at edges)
                y_end = min(y_off + self.patch_size, full_height)
                x_end = min(x_off + self.patch_size, full_width)
                h = y_end - y_off
                w = x_end - x_off

                # Skip very small edge slivers
                if w < self.patch_size // 4 or h < self.patch_size // 4:
                    continue

                # Direct numpy slice — zero alignment error
                chip = full_image[:, y_off:y_end, x_off:x_end]

                # Zero-pad edge chips to patch_size × patch_size
                if h < self.patch_size or w < self.patch_size:
                    padded = np.zeros(
                        (chip.shape[0], self.patch_size, self.patch_size),
                        dtype=np.uint8,
                    )
                    padded[:, :h, :w] = chip
                    chip = padded

                # Geo bounds for this tile (used for metadata only)
                tile_minx = ext_minx + x_off * x_res
                tile_maxy = ext_maxy - y_off * y_res
                tile_maxx = tile_minx + self.patch_size * x_res
                tile_miny = tile_maxy - self.patch_size * y_res

                tiles.append(
                    InferenceTile(
                        tile_idx=tile_idx,
                        image=chip,
                        bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                        pixel_offset=(x_off, y_off),
                    )
                )
                tile_idx += 1

        logger.info("Tiled AOI into %d patches (%d×%d, overlap=%d, margin=%d)",
                    len(tiles), self.patch_size, self.patch_size, self.overlap, margin)
        return tiles, (full_height, full_width), (crop_top, crop_left, crop_h, crop_w)
