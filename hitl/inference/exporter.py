"""Exporter: write prediction results as GeoTIFF and optional vectorized polygons."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping

logger = logging.getLogger(__name__)


def export_prediction(
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    bounds: Tuple[float, float, float, float],
    crs: str,
    output_dir: str | Path,
    job_id: str,
    class_names: Optional[list] = None,
    simplify_tolerance: float = 1.0,
    export_vectors: bool = True,
) -> dict:
    """Export prediction results to files.

    Args:
        class_map: (H, W) uint8 predicted class IDs.
        confidence_map: (H, W) float32 entropy-based confidence.
        bounds: (minx, miny, maxx, maxy) in target CRS.
        crs: CRS string (e.g., "EPSG:3857").
        output_dir: Directory for output files.
        job_id: Unique identifier for this inference job.
        class_names: List of class names for vector attributes.
        simplify_tolerance: Polygon simplification in CRS units.
        export_vectors: Whether to also export vectorized polygons.

    Returns:
        Dict with paths to output files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = class_map.shape
    transform = from_bounds(*bounds, W, H)

    result = {}

    # Write class prediction GeoTIFF
    class_path = output_dir / f"{job_id}_classes.tif"
    with rasterio.open(
        class_path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype=np.uint8,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(class_map, 1)
        if class_names:
            dst.update_tags(**{f"class_{i}": name for i, name in enumerate(class_names)})
    result["class_raster"] = str(class_path)

    # Write confidence heatmap GeoTIFF
    conf_path = output_dir / f"{job_id}_confidence.tif"
    with rasterio.open(
        conf_path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype=np.float32,
        crs=crs, transform=transform,
    ) as dst:
        dst.write(confidence_map, 1)
    result["confidence_raster"] = str(conf_path)

    # Vectorize predictions
    if export_vectors:
        try:
            import geopandas as gpd

            polygons = []
            class_ids = []

            for geom, value in rasterio_shapes(class_map, transform=transform):
                if value == 0:  # skip ignore/background-outside
                    continue
                poly = shape(geom)
                if simplify_tolerance > 0:
                    poly = poly.simplify(simplify_tolerance)
                if not poly.is_empty:
                    polygons.append(poly)
                    class_ids.append(int(value))

            if polygons:
                gdf = gpd.GeoDataFrame(
                    {"class_id": class_ids, "geometry": polygons},
                    crs=crs,
                )
                if class_names:
                    gdf["class_name"] = gdf["class_id"].map(
                        lambda x: class_names[x] if x < len(class_names) else f"class_{x}"
                    )

                vec_path = output_dir / f"{job_id}_predictions.gpkg"
                gdf.to_file(vec_path, driver="GPKG")
                result["vector"] = str(vec_path)

        except Exception as e:
            logger.warning("Vector export failed: %s", e)

    logger.info("Exported prediction results to %s", output_dir)
    return result
