"""Geospatial utilities: CRS transforms, bounds math, affine helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def reproject_bounds(bounds: BBox, src_crs: str, dst_crs: str) -> BBox:
    """Reproject a bounding box between CRS."""
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geom = box(*bounds)
    reprojected = shapely_transform(transformer.transform, geom)
    return reprojected.bounds


def make_transform(bounds: BBox, width: int, height: int) -> rasterio.Affine:
    """Create a rasterio Affine transform from bounds and pixel dimensions."""
    return from_bounds(*bounds, width, height)


def pixel_to_geo(
    row: int, col: int, transform: rasterio.Affine
) -> Tuple[float, float]:
    """Convert pixel (row, col) to geographic coordinates (x, y)."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def geo_to_pixel(
    x: float, y: float, transform: rasterio.Affine
) -> Tuple[int, int]:
    """Convert geographic (x, y) to pixel (row, col)."""
    col, row = ~transform * (x, y)
    return int(row), int(col)


def compute_resolution(bounds: BBox, width: int, height: int) -> Tuple[float, float]:
    """Compute pixel resolution (x_res, y_res) in CRS units."""
    minx, miny, maxx, maxy = bounds
    return (maxx - minx) / width, (maxy - miny) / height


def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad a dimension up to the nearest multiple."""
    remainder = size % multiple
    if remainder == 0:
        return size
    return size + (multiple - remainder)


def tile_bounds_grid(
    bounds: BBox, tile_size_geo: float, overlap_geo: float
) -> list[BBox]:
    """Generate a grid of overlapping tile bounding boxes covering the AOI.

    Args:
        bounds: AOI bounding box.
        tile_size_geo: Tile size in CRS units.
        overlap_geo: Overlap in CRS units.

    Returns:
        List of (minx, miny, maxx, maxy) tile bounding boxes.
    """
    minx, miny, maxx, maxy = bounds
    step = tile_size_geo - overlap_geo
    tiles = []

    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tile_maxx = min(x + tile_size_geo, maxx)
            tile_maxy = min(y + tile_size_geo, maxy)
            tiles.append((x, y, tile_maxx, tile_maxy))
            x += step
        y += step

    return tiles
