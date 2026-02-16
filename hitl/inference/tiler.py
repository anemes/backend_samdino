"""Tiler: splits an AOI into overlapping geo-referenced patches for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..data.raster_source import RasterSource


@dataclass
class InferenceTile:
    """A single tile for inference with its geo metadata."""

    tile_idx: int
    image: np.ndarray  # (C, H, W) uint8
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)
    pixel_offset: Tuple[int, int]  # (x_off, y_off) in the full output raster


class Tiler:
    """Generate overlapping tiles from a raster source over an AOI.

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
    ) -> Tuple[List[InferenceTile], Tuple[int, int]]:
        """Generate tiles covering the AOI.

        Args:
            raster_source: Source of imagery.
            aoi_bounds: (minx, miny, maxx, maxy) in source CRS.

        Returns:
            (tiles, output_shape) where output_shape is (height, width) of the
            full stitched output raster.
        """
        x_res, y_res = raster_source.get_resolution()
        minx, miny, maxx, maxy = aoi_bounds

        # Full output dimensions
        full_width = int((maxx - minx) / x_res)
        full_height = int((maxy - miny) / y_res)

        step = self.patch_size - self.overlap
        tiles = []
        tile_idx = 0

        for y_off in range(0, full_height, step):
            for x_off in range(0, full_width, step):
                # Compute geo bounds for this tile
                tile_minx = minx + x_off * x_res
                tile_maxy = maxy - y_off * y_res
                tile_maxx = tile_minx + self.patch_size * x_res
                tile_miny = tile_maxy - self.patch_size * y_res

                # Clamp to AOI
                tile_maxx = min(tile_maxx, maxx)
                tile_miny = max(tile_miny, miny)

                tile_w = int((tile_maxx - tile_minx) / x_res)
                tile_h = int((tile_maxy - tile_miny) / y_res)

                if tile_w < self.patch_size // 4 or tile_h < self.patch_size // 4:
                    continue

                # Fetch imagery
                chip = raster_source.get_chip(
                    bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                    width=self.patch_size,
                    height=self.patch_size,
                )

                tiles.append(
                    InferenceTile(
                        tile_idx=tile_idx,
                        image=chip.data,
                        bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                        pixel_offset=(x_off, y_off),
                    )
                )
                tile_idx += 1

        return tiles, (full_height, full_width)
