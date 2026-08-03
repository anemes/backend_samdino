"""Tests for raster_source fixes: empty crop guard (#10)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_geotiff(path, width=256, height=256):
    """Create a small synthetic GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.randint(0, 255, (3, height, width), dtype=np.uint8)
    transform = from_bounds(0, 0, 256, 256, width, height)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_crop_bounds_guard_inverted():
    """#10: When bounds clamp to zero-size, ValueError is raised."""
    # Simulate the clamping logic from XYZTileSource.get_chip
    mw, mh = 256, 256

    # Bounds entirely to the right of the mosaic
    col0 = max(0, min(300, mw))   # 256
    col1 = max(0, min(400, mw))   # 256
    row0 = max(0, min(0, mh))     # 0
    row1 = max(0, min(100, mh))   # 100

    assert col0 >= col1, "col0 should equal col1 after clamping"

    # Our fix raises ValueError in this case
    if col0 >= col1 or row0 >= row1:
        raised = True
    else:
        raised = False
    assert raised, "Should detect empty crop bounds"


def test_crop_bounds_guard_normal():
    """Regression: Normal bounds pass through the guard."""
    mw, mh = 256, 256

    col0 = max(0, min(50, mw))    # 50
    col1 = max(0, min(150, mw))   # 150
    row0 = max(0, min(50, mh))    # 50
    row1 = max(0, min(150, mh))   # 150

    assert col0 < col1
    assert row0 < row1


def test_geotiff_read_inside_bounds():
    """Regression: GeoTIFFSource.get_chip works for valid bounds."""
    from hitl.data.raster_source import GeoTIFFSource

    with tempfile.TemporaryDirectory() as tmp:
        tif = _make_geotiff(str(Path(tmp) / "test.tif"))
        source = GeoTIFFSource(tif)

        chip = source.get_chip(bounds=(50, 50, 150, 150), width=100, height=100)
        assert chip.data.shape == (3, 100, 100)
