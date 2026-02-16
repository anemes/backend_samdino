"""XYZ/TMS tile fetcher with disk cache and rate limiting.

Fetches map tiles from XYZ tile services (Google Satellite, Bing, OpenStreetMap,
custom aerial imagery APIs). Handles TMS math, HTTP fetching, disk caching,
rate limiting, and mosaic stitching.

Usage:
    fetcher = XYZFetcher("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
    mosaic = fetcher.get_mosaic(bounds, zoom=18, crs="EPSG:3857")
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in EPSG:3857


def _lng_lat_to_tile(lng: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Convert WGS84 lng/lat to TMS tile x, y at given zoom level."""
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(x, n - 1)), max(0, min(y, n - 1))


def _tile_to_bounds_3857(x: int, y: int, zoom: int) -> BBox:
    """Get EPSG:3857 bounds for a TMS tile."""
    n = 2 ** zoom
    # WGS84 bounds
    lng_min = x / n * 360.0 - 180.0
    lng_max = (x + 1) / n * 360.0 - 180.0
    lat_max_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_min_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_min = math.degrees(lat_min_rad)
    lat_max = math.degrees(lat_max_rad)
    # Convert to Web Mercator (EPSG:3857)
    return (
        _lng_to_x_3857(lng_min),
        _lat_to_y_3857(lat_min),
        _lng_to_x_3857(lng_max),
        _lat_to_y_3857(lat_max),
    )


def _lng_to_x_3857(lng: float) -> float:
    return lng * 20037508.342789244 / 180.0


def _lat_to_y_3857(lat: float) -> float:
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    return y * 20037508.342789244 / 180.0


def _x_to_lng_3857(x: float) -> float:
    return x * 180.0 / 20037508.342789244


def _y_to_lat_3857(y: float) -> float:
    lat = y * 180.0 / 20037508.342789244
    return 180.0 / math.pi * (2.0 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)


def _covering_tiles(bounds_3857: BBox, zoom: int) -> List[Tuple[int, int]]:
    """Get all TMS tile coordinates that cover the given EPSG:3857 bounds."""
    minx, miny, maxx, maxy = bounds_3857
    # Convert to WGS84
    lng_min = _x_to_lng_3857(minx)
    lat_min = _y_to_lat_3857(miny)
    lng_max = _x_to_lng_3857(maxx)
    lat_max = _y_to_lat_3857(maxy)

    # Get tile range
    x_min, y_max_tile = _lng_lat_to_tile(lng_min, lat_min, zoom)
    x_max, y_min_tile = _lng_lat_to_tile(lng_max, lat_max, zoom)

    tiles = []
    for tx in range(x_min, x_max + 1):
        for ty in range(y_min_tile, y_max_tile + 1):
            tiles.append((tx, ty))

    return tiles


class XYZFetcher:
    """Fetches and caches XYZ/TMS tiles, stitches into mosaics.

    Args:
        url_template: URL with {x}, {y}, {z} placeholders.
        cache_dir: Directory for disk cache. None disables caching.
        rate_limit: Max requests per second (0 = unlimited).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retries on failure.
        headers: Extra HTTP headers (e.g., API keys, User-Agent).
    """

    def __init__(
        self,
        url_template: str,
        cache_dir: Optional[str] = None,
        rate_limit: float = 10.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[dict] = None,
    ):
        self.url_template = url_template
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or {}

        self._last_request_time = 0.0
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self._client.close()

    def get_tile_image(self, x: int, y: int, z: int) -> Optional[np.ndarray]:
        """Fetch a single tile as RGB numpy array (H, W, 3).

        Returns None if the tile cannot be fetched.
        """
        # Check cache
        if self.cache_dir:
            cache_path = self._cache_path(x, y, z)
            if cache_path.exists():
                try:
                    img = Image.open(cache_path).convert("RGB")
                    return np.array(img)
                except Exception:
                    cache_path.unlink(missing_ok=True)

        # Fetch from network
        url = self.url_template.format(x=x, y=y, z=z)

        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            try:
                resp = self._client.get(url, headers=self.headers)
                if resp.status_code == 200:
                    data = resp.content
                    img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
                    arr = np.array(img)

                    # Cache
                    if self.cache_dir:
                        cache_path = self._cache_path(x, y, z)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, "wb") as f:
                            f.write(data)

                    return arr

                elif resp.status_code == 429:
                    # Rate limited — exponential backoff
                    wait = 2 ** attempt
                    logger.warning("Rate limited on %s, waiting %ds", url, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Tile fetch failed: %d %s", resp.status_code, url)
                    return None

            except httpx.TimeoutException:
                logger.warning("Timeout fetching tile %s (attempt %d)", url, attempt + 1)
            except Exception as e:
                logger.warning("Error fetching tile %s: %s", url, e)

        return None

    def get_mosaic(
        self,
        bounds_3857: BBox,
        zoom: int,
    ) -> Optional[dict]:
        """Fetch all tiles covering bounds and stitch into a mosaic.

        Args:
            bounds_3857: (minx, miny, maxx, maxy) in EPSG:3857.
            zoom: TMS zoom level.

        Returns:
            Dict with:
                'image': (3, H, W) uint8 numpy array
                'bounds': actual mosaic bounds in EPSG:3857
                'transform': rasterio Affine transform
                'crs': 'EPSG:3857'
                'resolution': (x_res, y_res)
        """
        tiles = _covering_tiles(bounds_3857, zoom)
        if not tiles:
            return None

        # Determine tile grid extent
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        ncols = x_max - x_min + 1
        nrows = y_max - y_min + 1

        # Fetch all tiles
        tile_size = 256  # Standard TMS tile size
        mosaic = np.zeros((3, nrows * tile_size, ncols * tile_size), dtype=np.uint8)

        fetched = 0
        for tx, ty in tiles:
            img = self.get_tile_image(tx, ty, zoom)
            if img is None:
                continue

            col = tx - x_min
            row = ty - y_min
            h, w = img.shape[:2]

            # Handle non-standard tile sizes
            h = min(h, tile_size)
            w = min(w, tile_size)

            mosaic[
                :,
                row * tile_size : row * tile_size + h,
                col * tile_size : col * tile_size + w,
            ] = img[:h, :w].transpose(2, 0, 1)
            fetched += 1

        if fetched == 0:
            return None

        # Compute actual mosaic bounds in EPSG:3857
        top_left_bounds = _tile_to_bounds_3857(x_min, y_min, zoom)
        bottom_right_bounds = _tile_to_bounds_3857(x_max, y_max, zoom)

        mosaic_bounds = (
            top_left_bounds[0],      # minx
            bottom_right_bounds[1],  # miny
            bottom_right_bounds[2],  # maxx
            top_left_bounds[3],      # maxy
        )

        # Compute resolution
        mosaic_w = mosaic.shape[2]
        mosaic_h = mosaic.shape[1]
        x_res = (mosaic_bounds[2] - mosaic_bounds[0]) / mosaic_w
        y_res = (mosaic_bounds[3] - mosaic_bounds[1]) / mosaic_h

        import rasterio.transform
        transform = rasterio.transform.from_bounds(
            *mosaic_bounds, mosaic_w, mosaic_h
        )

        logger.info(
            "Mosaic: %d/%d tiles fetched, %dx%d pixels, zoom=%d",
            fetched, len(tiles), mosaic_w, mosaic_h, zoom,
        )

        return {
            "image": mosaic,
            "bounds": mosaic_bounds,
            "transform": transform,
            "crs": "EPSG:3857",
            "resolution": (x_res, y_res),
            "num_tiles": fetched,
        }

    def _cache_path(self, x: int, y: int, z: int) -> Path:
        """Get disk cache path for a tile."""
        # Use URL hash as subdirectory to support multiple tile sources
        url_hash = hashlib.md5(self.url_template.encode()).hexdigest()[:8]
        return self.cache_dir / url_hash / str(z) / str(x) / f"{y}.png"

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        if self.rate_limit <= 0:
            return
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
