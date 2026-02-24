"""GeoPackage-backed label store for classes, annotations, and annotation regions.

Three layers in one .gpkg:
  - 'classes': class_id (int), name (str), color (str hex)
  - 'annotations': geometry (Polygon), class_id (int), region_id (int),
                   source (str), iteration (int), created_at (str)
  - 'regions': geometry (Polygon), region_id (int), created_at (str)

Annotation regions define exhaustive labeling zones. Everything inside a
region must be either a labeled class polygon or background. Everything
outside all regions is ignore_index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)


@dataclass
class SegClass:
    """Segmentation class definition."""

    class_id: int  # 1 = background (implicit), user classes start at 2
    name: str
    color: str  # hex "#RRGGBB"


class LabelStore:
    """GeoPackage-backed store for segmentation labels.

    Usage:
        store = LabelStore("/path/to/project/labels.gpkg")
        store.set_classes([SegClass(2, "building", "#FF0000"), ...])
        store.add_region(region_geojson, crs="EPSG:4326")
        store.add_annotation(polygon_geojson, class_id=2, region_id=1, crs="EPSG:4326")
    """

    ANNOTATIONS_LAYER = "annotations"
    REGIONS_LAYER = "regions"
    CLASSES_FILE = "classes.json"  # sidecar JSON (GeoPackage can't store non-geo tables cleanly)

    def __init__(self, gpkg_path: str | Path):
        self.path = Path(gpkg_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._classes_path = self.path.parent / self.CLASSES_FILE
        self._ensure_layers()

    def _ensure_layers(self) -> None:
        """Create GeoPackage layers if they don't exist."""
        if not self.path.exists():
            # Create empty annotations layer
            gdf = gpd.GeoDataFrame(
                columns=[
                    "geometry",
                    "class_id",
                    "region_id",
                    "source",
                    "iteration",
                    "created_at",
                    "status",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )
            gdf.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")

            # Create empty regions layer
            rgdf = gpd.GeoDataFrame(
                columns=["geometry", "region_id", "created_at", "status"],
                geometry="geometry",
                crs="EPSG:4326",
            )
            rgdf.to_file(self.path, layer=self.REGIONS_LAYER, driver="GPKG")

    @staticmethod
    def _ensure_status_column(gdf: gpd.GeoDataFrame, default: str) -> gpd.GeoDataFrame:
        """Add status column with default value if missing (backward compat)."""
        if "status" not in gdf.columns:
            gdf["status"] = default
        else:
            gdf["status"] = gdf["status"].fillna(default)
        return gdf

    # --- Class operations ---

    def get_classes(self) -> List[SegClass]:
        """Get all defined classes. Background (id=1) is always implicit."""
        if not self._classes_path.exists():
            return []
        data = json.loads(self._classes_path.read_text())
        return [SegClass(**c) for c in data]

    def set_classes(self, classes: List[SegClass]) -> None:
        """Set the class definitions. Validates that no class uses id=0 or id=1."""
        for c in classes:
            if c.class_id < 2:
                raise ValueError(
                    f"Class id must be >= 2 (0=ignore, 1=background). Got {c.class_id}"
                )
        self._classes_path.write_text(json.dumps([asdict(c) for c in classes], indent=2))

    def get_all_class_ids(self) -> List[int]:
        """Return all class IDs including background (1)."""
        return [1] + [c.class_id for c in self.get_classes()]

    def get_num_classes(self) -> int:
        """Total output channels: max(class_id) + 1.

        Since class IDs include ignore_index=0, background=1, and user
        classes (2, 3, ...), the model needs max_id+1 output channels.
        """
        all_ids = self.get_all_class_ids()  # includes background=1
        if not all_ids:
            return 2  # at minimum: ignore(0) + background(1)
        return max(all_ids) + 1

    # --- Region operations ---

    def add_region(
        self, geometry_geojson: dict, crs: str = "EPSG:4326", status: str = "active",
    ) -> int:
        """Add an annotation region. Returns assigned region_id."""
        existing = self.get_regions(crs=crs)
        region_id = len(existing) + 1

        new_row = gpd.GeoDataFrame(
            [
                {
                    "geometry": shape(geometry_geojson),
                    "region_id": region_id,
                    "created_at": datetime.now().isoformat(),
                    "status": status,
                }
            ],
            crs=crs,
        )

        if len(existing) > 0:
            existing = self._ensure_status_column(existing, "active")
            combined = gpd.GeoDataFrame(
                data=list(existing.drop(columns="geometry").to_dict("records"))
                + list(new_row.drop(columns="geometry").to_dict("records")),
                geometry=list(existing.geometry) + list(new_row.geometry),
                crs=crs,
            )
        else:
            combined = new_row

        combined.to_file(self.path, layer=self.REGIONS_LAYER, driver="GPKG")
        logger.info("Added annotation region %d (status=%s)", region_id, status)
        return region_id

    def get_regions(
        self, crs: str = "EPSG:4326", status: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Get annotation regions, optionally filtered by status."""
        try:
            gdf = gpd.read_file(self.path, layer=self.REGIONS_LAYER)
            gdf = self._ensure_status_column(gdf, "active")
            if len(gdf) > 0 and gdf.crs and gdf.crs.to_epsg() != int(crs.split(":")[1]):
                gdf = gdf.to_crs(crs)
            if status is not None and len(gdf) > 0:
                gdf = gdf[gdf["status"] == status]
            return gdf
        except Exception:
            return gpd.GeoDataFrame(
                columns=["geometry", "region_id", "created_at", "status"],
                geometry="geometry",
                crs=crs,
            )

    # --- Annotation operations ---

    def add_annotation(
        self,
        geometry_geojson: dict,
        class_id: int,
        region_id: int,
        crs: str = "EPSG:4326",
        source: str = "manual",
        iteration: int = 0,
        status: str = "approved",
    ) -> int:
        """Add a labeled polygon annotation. Returns row index."""
        existing = self.get_annotations(crs=crs)

        new_row = gpd.GeoDataFrame(
            [
                {
                    "geometry": shape(geometry_geojson),
                    "class_id": class_id,
                    "region_id": region_id,
                    "source": source,
                    "iteration": iteration,
                    "created_at": datetime.now().isoformat(),
                    "status": status,
                }
            ],
            crs=crs,
        )

        if len(existing) > 0:
            existing = self._ensure_status_column(existing, "approved")
            combined = gpd.GeoDataFrame(
                data=list(existing.drop(columns="geometry").to_dict("records"))
                + list(new_row.drop(columns="geometry").to_dict("records")),
                geometry=list(existing.geometry) + list(new_row.geometry),
                crs=crs,
            )
        else:
            combined = new_row

        combined.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")
        return len(combined) - 1

    def get_annotations(
        self,
        region_id: Optional[int] = None,
        crs: str = "EPSG:4326",
        status: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Get annotations, optionally filtered by region_id and/or status."""
        try:
            gdf = gpd.read_file(self.path, layer=self.ANNOTATIONS_LAYER)
            gdf = self._ensure_status_column(gdf, "approved")
            if len(gdf) > 0 and gdf.crs and gdf.crs.to_epsg() != int(crs.split(":")[1]):
                gdf = gdf.to_crs(crs)
            if region_id is not None and len(gdf) > 0:
                gdf = gdf[gdf["region_id"] == region_id]
            if status is not None and len(gdf) > 0:
                gdf = gdf[gdf["status"] == status]
            return gdf
        except Exception:
            return gpd.GeoDataFrame(
                columns=[
                    "geometry",
                    "class_id",
                    "region_id",
                    "source",
                    "iteration",
                    "created_at",
                    "status",
                ],
                geometry="geometry",
                crs=crs,
            )

    def delete_annotations_in_region(self, region_id: int) -> int:
        """Delete all annotations inside a region. Returns count deleted."""
        gdf = self.get_annotations()
        before = len(gdf)
        gdf = gdf[gdf["region_id"] != region_id]
        after = len(gdf)
        gdf.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")
        deleted = before - after
        logger.info("Deleted %d annotations from region %d", deleted, region_id)
        return deleted

    def delete_annotation(self, annotation_index: int) -> bool:
        """Delete a single annotation by its index. Returns True if deleted."""
        gdf = self.get_annotations()
        if annotation_index < 0 or annotation_index >= len(gdf):
            return False
        gdf = gdf.drop(gdf.index[annotation_index]).reset_index(drop=True)
        gdf.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")
        logger.info("Deleted annotation at index %d", annotation_index)
        return True

    def delete_region(self, region_id: int) -> int:
        """Delete a region and all its annotations. Returns annotations deleted."""
        deleted = self.delete_annotations_in_region(region_id)
        regions = self.get_regions()
        regions = regions[regions["region_id"] != region_id]
        regions.to_file(self.path, layer=self.REGIONS_LAYER, driver="GPKG")
        logger.info("Deleted region %d (and %d annotations)", region_id, deleted)
        return deleted

    def check_annotation_in_region(
        self, geometry_geojson: dict, region_id: int, crs: str = "EPSG:4326"
    ) -> bool:
        """Check if an annotation geometry is inside the specified region."""
        regions = self.get_regions(crs=crs)
        region_rows = regions[regions["region_id"] == region_id]
        if len(region_rows) == 0:
            return False
        region_geom = region_rows.iloc[0].geometry
        annotation_geom = shape(geometry_geojson)
        return region_geom.contains(annotation_geom.centroid)

    def get_stats(self) -> dict:
        """Get label statistics."""
        annotations = self.get_annotations()
        regions = self.get_regions()
        classes = self.get_classes()

        stats = {
            "num_regions": len(regions),
            "num_annotations": len(annotations),
            "num_classes": len(classes),
            "classes": [asdict(c) for c in classes],
        }

        if len(annotations) > 0:
            for c in classes:
                count = len(annotations[annotations["class_id"] == c.class_id])
                stats[f"class_{c.name}_count"] = count

        return stats

    # --- Review workflow ---

    def get_region_status(self, region_id: int) -> str:
        """Return the status of a specific region ('active' or 'in_review')."""
        regions = self.get_regions()
        match = regions[regions["region_id"] == region_id]
        if len(match) == 0:
            return "active"
        return str(match.iloc[0].get("status", "active"))

    def approve_region(self, region_id: int) -> int:
        """Approve an in-review region and all its annotations.

        Sets region status to 'active' and all its annotations to 'approved'.
        Returns the number of annotations approved.
        """
        # Update region status
        regions = self.get_regions()
        mask = regions["region_id"] == region_id
        if mask.sum() == 0:
            raise ValueError(f"Region {region_id} not found")
        regions.loc[mask, "status"] = "active"
        regions.to_file(self.path, layer=self.REGIONS_LAYER, driver="GPKG")

        # Update annotation statuses
        annotations = self.get_annotations()
        ann_mask = annotations["region_id"] == region_id
        count = int(ann_mask.sum())
        if count > 0:
            annotations.loc[ann_mask, "status"] = "approved"
            annotations.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")

        logger.info("Approved region %d (%d annotations)", region_id, count)
        return count

    def add_annotations_bulk(
        self,
        geometries: list,
        class_ids: list,
        region_id: int,
        crs: str = "EPSG:4326",
        source: str = "inference",
        iteration: int = 0,
        status: str = "in_review",
    ) -> int:
        """Add multiple annotations at once. Returns count added."""
        existing = self.get_annotations(crs=crs)
        now = datetime.now().isoformat()

        new_rows = gpd.GeoDataFrame(
            [
                {
                    "geometry": geom,
                    "class_id": cid,
                    "region_id": region_id,
                    "source": source,
                    "iteration": iteration,
                    "created_at": now,
                    "status": status,
                }
                for geom, cid in zip(geometries, class_ids)
            ],
            crs=crs,
        )

        if len(existing) > 0:
            existing = self._ensure_status_column(existing, "approved")
            combined = gpd.GeoDataFrame(
                data=list(existing.drop(columns="geometry").to_dict("records"))
                + list(new_rows.drop(columns="geometry").to_dict("records")),
                geometry=list(existing.geometry) + list(new_rows.geometry),
                crs=crs,
            )
        else:
            combined = new_rows

        combined.to_file(self.path, layer=self.ANNOTATIONS_LAYER, driver="GPKG")
        logger.info("Bulk added %d annotations to region %d", len(geometries), region_id)
        return len(geometries)
