"""Label CRUD endpoints: classes, annotations, annotation regions."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


def _round_geojson(geom_dict: dict, prec: int = 6) -> dict:
    """Round coordinates in a GeoJSON geometry dict to *prec* decimal places.

    Prevents 15-digit precision from rasterio reprojection reaching the
    QGIS plugin, where it can overflow the WKT parser.
    """
    def _rc(c):
        if not c:
            return c
        if isinstance(c[0], (int, float)):
            return [round(v, prec) for v in c]
        return [_rc(sub) for sub in c]

    if "coordinates" in geom_dict:
        geom_dict = dict(geom_dict)
        geom_dict["coordinates"] = _rc(geom_dict["coordinates"])
    return geom_dict


class ClassDef(BaseModel):
    class_id: int
    name: str
    color: str


class ClassesRequest(BaseModel):
    classes: List[ClassDef]


class AnnotationRequest(BaseModel):
    geometry_geojson: dict
    class_id: int
    region_id: int
    crs: str = "EPSG:4326"
    source: str = "manual"
    iteration: int = 0


class RegionRequest(BaseModel):
    geometry_geojson: dict
    crs: str = "EPSG:4326"


# --- Dependency ---
def get_label_store():
    from ..app import app_state
    return app_state.label_store


# --- Classes ---

@router.get("/classes")
def get_classes(store=Depends(get_label_store)):
    return {"classes": [{"class_id": c.class_id, "name": c.name, "color": c.color}
                        for c in store.get_classes()]}


@router.post("/classes")
def set_classes(req: ClassesRequest, store=Depends(get_label_store)):
    from ..data.label_store import SegClass
    classes = [SegClass(class_id=c.class_id, name=c.name, color=c.color) for c in req.classes]
    store.set_classes(classes)
    return {"status": "ok", "num_classes": len(classes)}


# --- Regions ---

@router.get("/regions")
def get_regions(crs: str = "EPSG:4326", store=Depends(get_label_store)):
    regions = store.get_regions(crs=crs)
    features = []
    for _, row in regions.iterrows():
        features.append({
            "region_id": int(row["region_id"]),
            "geometry": _round_geojson(row.geometry.__geo_interface__),
            "created_at": row.get("created_at", ""),
        })
    return {"regions": features, "count": len(features)}


@router.post("/regions")
def add_region(req: RegionRequest, store=Depends(get_label_store)):
    region_id = store.add_region(req.geometry_geojson, crs=req.crs)
    return {"region_id": region_id}


# --- Annotations ---

@router.get("/annotations")
def get_annotations(
    region_id: Optional[int] = None,
    crs: str = "EPSG:4326",
    store=Depends(get_label_store),
):
    annotations = store.get_annotations(region_id=region_id, crs=crs)
    features = []
    for _, row in annotations.iterrows():
        features.append({
            "class_id": int(row["class_id"]),
            "region_id": int(row["region_id"]),
            "source": row.get("source", "manual"),
            "iteration": int(row.get("iteration", 0)),
            "geometry": _round_geojson(row.geometry.__geo_interface__),
        })
    return {"annotations": features, "count": len(features)}


@router.post("/annotations")
def add_annotation(req: AnnotationRequest, store=Depends(get_label_store)):
    # Validate annotation is inside the target region
    if not store.check_annotation_in_region(
        req.geometry_geojson, req.region_id, crs=req.crs
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Annotation centroid is outside region {req.region_id}. "
                   "Draw inside the region boundary or select the correct region.",
        )
    idx = store.add_annotation(
        geometry_geojson=req.geometry_geojson,
        class_id=req.class_id,
        region_id=req.region_id,
        crs=req.crs,
        source=req.source,
        iteration=req.iteration,
    )
    return {"index": idx}


@router.delete("/annotations/{annotation_index}")
def delete_annotation(annotation_index: int, store=Depends(get_label_store)):
    ok = store.delete_annotation(annotation_index)
    if not ok:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "ok"}


@router.delete("/regions/{region_id}")
def delete_region(region_id: int, store=Depends(get_label_store)):
    deleted = store.delete_region(region_id)
    return {"status": "ok", "annotations_deleted": deleted}


@router.delete("/annotations/region/{region_id}")
def delete_region_annotations(region_id: int, store=Depends(get_label_store)):
    deleted = store.delete_annotations_in_region(region_id)
    return {"deleted": deleted}


@router.get("/stats")
def get_stats(store=Depends(get_label_store)):
    return store.get_stats()


@router.post("/upload")
async def upload_labels(file: UploadFile = File(...), store=Depends(get_label_store)):
    """Upload a GeoPackage file to replace the current labels."""
    import shutil
    import tempfile
    from pathlib import Path

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Replace current labels
    target = store.path
    shutil.copy2(tmp_path, target)
    Path(tmp_path).unlink()

    return {"status": "ok", "path": str(target)}
