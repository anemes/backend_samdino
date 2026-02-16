"""Abstract raster source and concrete implementations.

Provides a unified interface for fetching georeferenced raster imagery
from different sources: GeoTIFF files, XYZ/TMS tile services, and
uploaded image chips from QGIS.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

logger = logging.getLogger(__name__)


BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


@dataclass
class RasterChip:
    """A georeferenced image chip."""

    data: np.ndarray  # (C, H, W) uint8
    bounds: BBox  # (minx, miny, maxx, maxy) in the source CRS
    crs: str  # e.g., "EPSG:3857"
    transform: rasterio.Affine  # pixel-to-geo affine transform
    resolution: Tuple[float, float]  # (x_res, y_res) in CRS units


class RasterSource(ABC):
    """Abstract interface for fetching georeferenced raster imagery."""

    @abstractmethod
    def get_chip(
        self, bounds: BBox, width: int, height: int
    ) -> RasterChip:
        """Fetch imagery for the given bounds at specified pixel dimensions.

        Args:
            bounds: (minx, miny, maxx, maxy) in the source CRS.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            RasterChip with image data and geo metadata.
        """
        ...

    @abstractmethod
    def get_bounds(self) -> BBox:
        """Get the full extent of this raster source."""
        ...

    @abstractmethod
    def get_crs(self) -> str:
        """Get the CRS of this raster source."""
        ...

    @abstractmethod
    def get_resolution(self) -> Tuple[float, float]:
        """Get native pixel resolution (x_res, y_res) in CRS units."""
        ...


class GeoTIFFSource(RasterSource):
    """Read from a local GeoTIFF file via rasterio.

    Supports windowed reading for efficient access to large rasters.
    Reads the first 3 bands (RGB) by default.
    """

    def __init__(self, path: str, bands: Tuple[int, ...] = (1, 2, 3)):
        self.path = path
        self.bands = bands

        # Read metadata once
        with rasterio.open(path) as src:
            self._bounds = src.bounds
            self._crs = str(src.crs)
            self._transform = src.transform
            self._width = src.width
            self._height = src.height
            self._res = src.res  # (y_res, x_res)

    def get_chip(
        self, bounds: BBox, width: int, height: int
    ) -> RasterChip:
        """Read a windowed region from the GeoTIFF."""
        with rasterio.open(self.path) as src:
            window = window_from_bounds(*bounds, transform=src.transform)
            data = src.read(
                indexes=list(self.bands),
                window=window,
                out_shape=(len(self.bands), height, width),
            )

        chip_transform = rasterio.transform.from_bounds(*bounds, width, height)
        minx, miny, maxx, maxy = bounds

        return RasterChip(
            data=data.astype(np.uint8),
            bounds=bounds,
            crs=self._crs,
            transform=chip_transform,
            resolution=((maxx - minx) / width, (maxy - miny) / height),
        )

    def get_full_image(self) -> RasterChip:
        """Read the entire raster."""
        return self.get_chip(
            bounds=tuple(self._bounds),
            width=self._width,
            height=self._height,
        )

    def get_bounds(self) -> BBox:
        return tuple(self._bounds)

    def get_crs(self) -> str:
        return self._crs

    def get_resolution(self) -> Tuple[float, float]:
        return (self._res[1], self._res[0])  # (x_res, y_res)


class UploadedChipSource(RasterSource):
    """Wraps an uploaded GeoTIFF file (from QGIS raster capture).

    Used for labeling: the plugin exports a view as GeoTIFF and uploads it.
    This source wraps that file for SAM3 and dataset building.
    """

    def __init__(self, path: str):
        self._source = GeoTIFFSource(path)
        self.path = path

    def get_chip(self, bounds: BBox, width: int, height: int) -> RasterChip:
        return self._source.get_chip(bounds, width, height)

    def get_bounds(self) -> BBox:
        return self._source.get_bounds()

    def get_crs(self) -> str:
        return self._source.get_crs()

    def get_resolution(self) -> Tuple[float, float]:
        return self._source.get_resolution()


class XYZTileSource(RasterSource):
    """Fetch imagery from an XYZ/TMS tile service.

    Uses XYZFetcher to download tiles, stitch into mosaics, and crop
    to requested bounds. All coordinates are in EPSG:3857 (Web Mercator).

    Args:
        url_template: URL with {x}, {y}, {z} placeholders.
        zoom: TMS zoom level (higher = finer resolution).
        cache_dir: Directory for disk cache. None disables caching.
        rate_limit: Max requests per second.
        headers: Extra HTTP headers.
    """

    # Approximate meters/pixel at equator for each zoom level
    _METERS_PER_PIXEL_Z0 = 156543.03

    def __init__(
        self,
        url_template: str,
        zoom: int = 18,
        cache_dir: Optional[str] = None,
        rate_limit: float = 10.0,
        headers: Optional[dict] = None,
    ):
        from .xyz_fetcher import XYZFetcher

        self.url_template = url_template
        self.zoom = zoom
        self._fetcher = XYZFetcher(
            url_template=url_template,
            cache_dir=cache_dir,
            rate_limit=rate_limit,
            headers=headers,
        )
        # Resolution at equator for this zoom level
        self._res = self._METERS_PER_PIXEL_Z0 / (2 ** zoom)

    def get_chip(
        self, bounds: BBox, width: int, height: int
    ) -> RasterChip:
        """Fetch tiles covering bounds, stitch, and crop/resize to requested dimensions.

        Args:
            bounds: (minx, miny, maxx, maxy) in EPSG:3857.
            width: Output width in pixels.
            height: Output height in pixels.
        """
        mosaic = self._fetcher.get_mosaic(bounds, self.zoom)
        if mosaic is None:
            # Return blank chip if fetch failed
            logger.warning("XYZ mosaic fetch returned None for bounds %s", bounds)
            chip_transform = rasterio.transform.from_bounds(*bounds, width, height)
            return RasterChip(
                data=np.zeros((3, height, width), dtype=np.uint8),
                bounds=bounds,
                crs="EPSG:3857",
                transform=chip_transform,
                resolution=(self._res, self._res),
            )

        # mosaic['image'] is (3, H, W), mosaic['bounds'] covers full tile grid
        mosaic_img = mosaic["image"]  # (3, H, W)
        mosaic_bounds = mosaic["bounds"]  # (minx, miny, maxx, maxy)

        # Crop mosaic to requested bounds
        mb_minx, mb_miny, mb_maxx, mb_maxy = mosaic_bounds
        _, mh, mw = mosaic_img.shape

        # Pixel coordinates of requested bounds within the mosaic
        px_per_m_x = mw / (mb_maxx - mb_minx)
        px_per_m_y = mh / (mb_maxy - mb_miny)

        col0 = int((bounds[0] - mb_minx) * px_per_m_x)
        col1 = int((bounds[2] - mb_minx) * px_per_m_x)
        row0 = int((mb_maxy - bounds[3]) * px_per_m_y)  # Y is flipped (top=maxy)
        row1 = int((mb_maxy - bounds[1]) * px_per_m_y)

        # Clamp to mosaic dimensions
        col0 = max(0, min(col0, mw))
        col1 = max(0, min(col1, mw))
        row0 = max(0, min(row0, mh))
        row1 = max(0, min(row1, mh))

        cropped = mosaic_img[:, row0:row1, col0:col1]

        # Resize to requested dimensions if needed
        if cropped.shape[1] != height or cropped.shape[2] != width:
            from PIL import Image as PILImage

            # Transpose to (H, W, C) for PIL
            img = PILImage.fromarray(cropped.transpose(1, 2, 0))
            img = img.resize((width, height), PILImage.BILINEAR)
            cropped = np.array(img).transpose(2, 0, 1)

        chip_transform = rasterio.transform.from_bounds(*bounds, width, height)
        minx, miny, maxx, maxy = bounds

        return RasterChip(
            data=cropped.astype(np.uint8),
            bounds=bounds,
            crs="EPSG:3857",
            transform=chip_transform,
            resolution=((maxx - minx) / width, (maxy - miny) / height),
        )

    def get_bounds(self) -> BBox:
        """Full Web Mercator extent."""
        max_extent = 20037508.342789244
        return (-max_extent, -max_extent, max_extent, max_extent)

    def get_crs(self) -> str:
        return "EPSG:3857"

    def get_resolution(self) -> Tuple[float, float]:
        return (self._res, self._res)

    def close(self):
        """Close the underlying HTTP client."""
        self._fetcher.close()
