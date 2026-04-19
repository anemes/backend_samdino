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
import math
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, shape

logger = logging.getLogger(__name__)

try:
    import fiona
except Exception:  # pragma: no cover - fallback for minimal runtime images
    fiona = None


_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _shared_path_lock(path: Path) -> threading.RLock:
    """Return a shared in-process lock for one GeoPackage path."""
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


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
    IO_ENGINE = "fiona" if fiona is not None else "pyogrio"
    WRITE_RETRIES = 12
    WRITE_RETRY_DELAY_S = 0.25

    def __init__(self, gpkg_path: str | Path):
        self.path = Path(gpkg_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._classes_path = self.path.parent / self.CLASSES_FILE
        self._io_lock = _shared_path_lock(self.path)
        self._ensure_layers()
        self._set_sqlite_pragmas()

    def _set_sqlite_pragmas(self) -> None:
        """Set SQLite pragmas for network filesystem compatibility.

        WAL mode requires shared memory (-shm file) which doesn't work on
        network mounts like Azure Files SMB. DELETE journal mode uses
        traditional rollback journaling that works everywhere.
        """
        if not self.path.exists():
            return
        try:
            with sqlite3.connect(str(self.path)) as conn:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            logger.warning("Could not set SQLite pragmas on %s", self.path, exc_info=True)

    def _ensure_layers(self) -> None:
        """Create GeoPackage layers if they don't exist."""
        layers = set()
        if self.path.exists() and fiona is not None:
            layers = set(fiona.listlayers(self.path))
        elif self.path.exists():
            # Best-effort fallback when fiona is unavailable.
            layers = {self.ANNOTATIONS_LAYER, self.REGIONS_LAYER}

        if self.ANNOTATIONS_LAYER not in layers:
            self._create_empty_layer(self.ANNOTATIONS_LAYER)
            layers.add(self.ANNOTATIONS_LAYER)

        if self.REGIONS_LAYER not in layers:
            self._create_empty_layer(self.REGIONS_LAYER)

    def _create_empty_layer(self, layer: str) -> None:
        """Create an empty GPKG layer with explicit schema."""
        if fiona is None:
            if layer == self.ANNOTATIONS_LAYER:
                gdf = self._empty_annotations_gdf("EPSG:4326")
            else:
                gdf = self._empty_regions_gdf("EPSG:4326")
            self._write_layer(gdf, layer=layer, mode="w")
            return

        if layer == self.ANNOTATIONS_LAYER:
            schema = {
                "geometry": "MultiPolygon",
                "properties": {
                    "class_id": "int",
                    "region_id": "int",
                    "source": "str",
                    "iteration": "int",
                    "created_at": "str",
                    "status": "str",
                },
            }
        else:
            schema = {
                "geometry": "MultiPolygon",
                "properties": {
                    "region_id": "int",
                    "created_at": "str",
                    "status": "str",
                },
            }

        with self._io_lock:
            with fiona.open(
                self.path,
                mode="w",
                driver="GPKG",
                layer=layer,
                crs="EPSG:4326",
                schema=schema,
            ):
                pass

    @staticmethod
    def _empty_annotations_gdf(crs: str) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "class_id": pd.Series(dtype="int64"),
                "region_id": pd.Series(dtype="int64"),
                "source": pd.Series(dtype="str"),
                "iteration": pd.Series(dtype="int64"),
                "created_at": pd.Series(dtype="str"),
                "status": pd.Series(dtype="str"),
                "geometry": gpd.GeoSeries(dtype="geometry"),
            },
            geometry="geometry",
            crs=crs,
        )

    @staticmethod
    def _empty_regions_gdf(crs: str) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "region_id": pd.Series(dtype="int64"),
                "created_at": pd.Series(dtype="str"),
                "status": pd.Series(dtype="str"),
                "geometry": gpd.GeoSeries(dtype="geometry"),
            },
            geometry="geometry",
            crs=crs,
        )

    @staticmethod
    def _is_missing_layer_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "no such layer" in msg
            or "layer does not exist" in msg
            or "layer not found" in msg
        )

    # CRS used for all GPKG layers — must match _create_empty_layer().
    _STORAGE_CRS = "EPSG:4326"

    def _write_layer(self, gdf: gpd.GeoDataFrame, layer: str, mode: str) -> None:
        gdf = self._promote_to_multi(gdf)
        # Reproject to storage CRS before writing.  Fiona does NOT reproject
        # on append, so EPSG:3857 coordinates would be stored as-is in the
        # EPSG:4326 layer — producing wildly wrong values.
        if (
            len(gdf) > 0
            and gdf.crs is not None
            and gdf.crs.to_epsg() != 4326
        ):
            gdf = gdf.to_crs(self._STORAGE_CRS)
        with self._io_lock:
            last_exc = None
            for attempt in range(1, self.WRITE_RETRIES + 1):
                try:
                    gdf.to_file(
                        self.path,
                        layer=layer,
                        driver="GPKG",
                        mode=mode,
                        engine=self.IO_ENGINE,
                        index=False,
                    )
                    return
                except Exception as exc:
                    last_exc = exc
                    if not self._is_locked_error(exc) or attempt == self.WRITE_RETRIES:
                        raise
                    delay = self.WRITE_RETRY_DELAY_S * attempt
                    logger.warning(
                        "GeoPackage write locked for %s/%s (attempt %d/%d); retrying in %.2fs",
                        self.path,
                        layer,
                        attempt,
                        self.WRITE_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
            if last_exc is not None:
                raise last_exc

    @staticmethod
    def _promote_to_multi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Promote any Polygon geometries to MultiPolygon.

        Layers are created with MultiPolygon schema. Fiona enforces strict
        type matching on append, so all geometries must be MultiPolygon.
        Modifies in-place since callers always pass freshly-constructed frames.
        """
        if len(gdf) == 0:
            return gdf
        needs_promotion = False
        for geom in gdf.geometry:
            if geom is not None and geom.geom_type == "Polygon":
                needs_promotion = True
                break
        if not needs_promotion:
            return gdf
        gdf["geometry"] = [
            MultiPolygon([g]) if g is not None and g.geom_type == "Polygon" else g
            for g in gdf.geometry
        ]
        return gdf

    def _read_layer(self, layer: str, crs: str) -> gpd.GeoDataFrame:
        with self._io_lock:
            try:
                gdf = gpd.read_file(self.path, layer=layer, engine=self.IO_ENGINE)
            except Exception as exc:
                if self._is_missing_layer_error(exc):
                    return (
                        self._empty_annotations_gdf(crs)
                        if layer == self.ANNOTATIONS_LAYER
                        else self._empty_regions_gdf(crs)
                    )
                raise

        if len(gdf) > 0 and gdf.crs and gdf.crs.to_epsg() != int(crs.split(":")[1]):
            gdf = gdf.to_crs(crs)
        return gdf

    def _count_layer(self, layer: str) -> int:
        """Return the number of records in a layer without reading all data."""
        if not self.path.exists():
            return 0
        if fiona is not None:
            try:
                with self._io_lock:
                    with fiona.open(self.path, layer=layer, mode="r") as src:
                        return len(src)
            except Exception:
                return 0
        # Fallback: full read
        try:
            gdf = self._read_layer(layer, crs="EPSG:4326")
            return len(gdf)
        except Exception:
            return 0

    @staticmethod
    def _is_locked_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "database is locked" in msg
            or "failed to commit transaction" in msg
            or "transactionerror" in msg
            or "sqlite busy" in msg
        )

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
        region_id = self._count_layer(self.REGIONS_LAYER) + 1

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
        self._write_layer(new_row, layer=self.REGIONS_LAYER, mode="a")
        logger.info("Added annotation region %d (status=%s)", region_id, status)
        return region_id

    def get_regions(
        self, crs: str = "EPSG:4326", status: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Get annotation regions, optionally filtered by status."""
        gdf = self._read_layer(self.REGIONS_LAYER, crs=crs)
        gdf = self._ensure_status_column(gdf, "active")
        if status is not None and len(gdf) > 0:
            gdf = gdf[gdf["status"] == status]
        return gdf

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
        index = self._count_layer(self.ANNOTATIONS_LAYER)

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
        self._write_layer(new_row, layer=self.ANNOTATIONS_LAYER, mode="a")
        return index

    def get_annotations(
        self,
        region_id: Optional[int] = None,
        crs: str = "EPSG:4326",
        status: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Get annotations, optionally filtered by region_id and/or status."""
        gdf = self._read_layer(self.ANNOTATIONS_LAYER, crs=crs)
        gdf = self._ensure_status_column(gdf, "approved")
        if region_id is not None and len(gdf) > 0:
            gdf = gdf[gdf["region_id"] == region_id]
        if status is not None and len(gdf) > 0:
            gdf = gdf[gdf["status"] == status]
        return gdf

    def delete_annotations_in_region(self, region_id: int) -> int:
        """Delete all annotations inside a region. Returns count deleted."""
        gdf = self.get_annotations()
        before = len(gdf)
        gdf = gdf[gdf["region_id"] != region_id]
        after = len(gdf)
        self._write_layer(gdf, layer=self.ANNOTATIONS_LAYER, mode="w")
        deleted = before - after
        logger.info("Deleted %d annotations from region %d", deleted, region_id)
        return deleted

    def delete_annotation(self, annotation_index: int) -> bool:
        """Delete a single annotation by its index. Returns True if deleted."""
        gdf = self.get_annotations()
        if annotation_index < 0 or annotation_index >= len(gdf):
            return False
        gdf = gdf.drop(gdf.index[annotation_index]).reset_index(drop=True)
        self._write_layer(gdf, layer=self.ANNOTATIONS_LAYER, mode="w")
        logger.info("Deleted annotation at index %d", annotation_index)
        return True

    def delete_region(self, region_id: int) -> int:
        """Delete a region and all its annotations. Returns annotations deleted."""
        deleted = self.delete_annotations_in_region(region_id)
        regions = self.get_regions()
        regions = regions[regions["region_id"] != region_id]
        self._write_layer(regions, layer=self.REGIONS_LAYER, mode="w")
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
        self._write_layer(regions, layer=self.REGIONS_LAYER, mode="w")

        # Update annotation statuses
        annotations = self.get_annotations()
        ann_mask = annotations["region_id"] == region_id
        count = int(ann_mask.sum())
        if count > 0:
            annotations.loc[ann_mask, "status"] = "approved"
            self._write_layer(annotations, layer=self.ANNOTATIONS_LAYER, mode="w")

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
        now = datetime.now().isoformat()

        # Filter out invalid geometries (empty or with non-finite coordinates)
        valid_pairs = []
        for geom, cid in zip(geometries, class_ids):
            if geom is None or geom.is_empty:
                continue
            bounds = geom.bounds
            if not all(math.isfinite(v) for v in bounds):
                continue
            valid_pairs.append((geom, cid))

        skipped = len(geometries) - len(valid_pairs)
        if skipped > 0:
            logger.warning(
                "Skipped %d geometries with invalid/non-finite coordinates", skipped,
            )

        if not valid_pairs:
            logger.warning("No valid geometries to add after filtering")
            return 0

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
                for geom, cid in valid_pairs
            ],
            crs=crs,
        )

        self._write_layer(new_rows, layer=self.ANNOTATIONS_LAYER, mode="a")
        logger.info("Bulk added %d annotations to region %d", len(valid_pairs), region_id)
        return len(valid_pairs)
