"""Label CRUD endpoints: classes, annotations, annotation regions."""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from .deps import get_current_user, require_active_project_contributor

logger = logging.getLogger(__name__)

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
            return [round(v, prec) if math.isfinite(v) else 0.0 for v in c]
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


# --- Dependencies ---
def get_label_store():
    from ..app import app_state
    return app_state.label_store


def get_state():
    from ..app import app_state
    return app_state


# --- Classes ---

@router.get("/classes")
def get_classes(store=Depends(get_label_store)):
    return {"classes": [{"class_id": c.class_id, "name": c.name, "color": c.color}
                        for c in store.get_classes()]}


@router.post("/classes")
def set_classes(req: ClassesRequest, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
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
            "status": row.get("status", "active"),
        })
    return {"regions": features, "count": len(features)}


@router.post("/regions")
def add_region(req: RegionRequest, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    region_id = store.add_region(req.geometry_geojson, crs=req.crs)
    return {"region_id": region_id}


# --- Annotations ---

@router.get("/annotations")
def get_annotations(
    region_id: Optional[int] = None,
    status: Optional[str] = None,
    crs: str = "EPSG:4326",
    store=Depends(get_label_store),
):
    annotations = store.get_annotations(region_id=region_id, crs=crs, status=status)
    features = []
    for _, row in annotations.iterrows():
        features.append({
            "class_id": int(row["class_id"]),
            "region_id": int(row["region_id"]),
            "source": row.get("source", "manual"),
            "iteration": int(row.get("iteration", 0)),
            "geometry": _round_geojson(row.geometry.__geo_interface__),
            "status": row.get("status", "approved"),
        })
    return {"annotations": features, "count": len(features)}


@router.post("/annotations")
def add_annotation(req: AnnotationRequest, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    if req.class_id < 2:
        raise HTTPException(
            status_code=400,
            detail="class_id must be >= 2 (0=ignore, 1=background are implicit)",
        )
    # Validate annotation is inside the target region
    if not store.check_annotation_in_region(
        req.geometry_geojson, req.region_id, crs=req.crs
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Annotation centroid is outside region {req.region_id}. "
                   "Draw inside the region boundary or select the correct region.",
        )
    # Inherit region status: annotations in an in_review region are also in_review
    status = store.get_region_status(req.region_id)
    ann_status = "in_review" if status == "in_review" else "approved"
    idx = store.add_annotation(
        geometry_geojson=req.geometry_geojson,
        class_id=req.class_id,
        region_id=req.region_id,
        crs=req.crs,
        source=req.source,
        iteration=req.iteration,
        status=ann_status,
    )
    return {"index": idx}


@router.delete("/annotations/{annotation_index}")
def delete_annotation(annotation_index: int, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    ok = store.delete_annotation(annotation_index)
    if not ok:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "ok"}


@router.delete("/regions/{region_id}")
def delete_region(region_id: int, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    deleted = store.delete_region(region_id)
    return {"status": "ok", "annotations_deleted": deleted}


@router.delete("/annotations/region/{region_id}")
def delete_region_annotations(region_id: int, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    deleted = store.delete_annotations_in_region(region_id)
    return {"deleted": deleted}


# --- Review workflow ---


class PromoteInferenceRequest(BaseModel):
    aoi_geojson: dict  # Full AOI polygon GeoJSON geometry
    job_id: str  # Inference job ID to promote from
    project_id: Optional[str] = None  # target project; None = currently active project


@router.post("/promote-inference")
def promote_inference(req: PromoteInferenceRequest, request: Request, state=Depends(get_state)):
    """Promote inference results to in-review annotations.

    Creates a region (status='in_review') from the AOI polygon, then reads the
    prediction GeoPackage and inserts all non-background polygons as annotations
    (status='in_review').

    When *project_id* is supplied (e.g. '_inference' for standalone inference),
    the results are written to that project's GeoPackage without switching the
    globally active project.
    """
    import geopandas as gpd
    from shapely import wkt as shapely_wkt

    user = get_current_user(request)

    inf_state = state.inference_service.state
    if inf_state.job_id != req.job_id:
        raise HTTPException(status_code=404, detail=f"Job '{req.job_id}' not found")
    if inf_state.status != "complete":
        raise HTTPException(
            status_code=409, detail=f"Job status: {inf_state.status}"
        )

    vec_path = inf_state.result_paths.get("vector")
    if not vec_path:
        raise HTTPException(status_code=404, detail="No vector results for this job")

    # Resolve target project, rewriting the legacy '_inference' sentinel
    target_project = req.project_id or state.active_project_id
    if target_project == "_inference":
        target_project = f"_inference_{user['user_id']}"

    if target_project == state.active_project_id:
        store = state.label_store
    else:
        from ..data.label_store import LabelStore
        pm = state.project_manager
        project_dir = pm.get_project_dir(target_project)
        if not project_dir.exists():
            if target_project.startswith("_inference"):
                pm.create_project(target_project, target_project, owner=user["user_id"])
            else:
                raise HTTPException(status_code=404, detail=f"Project '{target_project}' not found")
        store = LabelStore(
            project_dir / "labels.gpkg",
            local_cache_dir=state.config.paths.gpkg_cache_dir,
        )

    # Create the review region from AOI polygon
    region_id = store.add_region(req.aoi_geojson, crs="EPSG:4326", status="in_review")

    # Read prediction GeoPackage
    pred_gdf = gpd.read_file(vec_path)

    # Reproject to EPSG:4326 for storage
    if pred_gdf.crs and pred_gdf.crs.to_epsg() != 4326:
        pred_gdf = pred_gdf.to_crs("EPSG:4326")

    # Filter out class_id < 2 (ignore=0 and background=1)
    pred_gdf = pred_gdf[pred_gdf["class_id"] >= 2]

    if len(pred_gdf) == 0:
        return {
            "region_id": region_id,
            "annotations_created": 0,
            "message": "Region created but no class predictions to promote",
        }

    # Round coordinates to 6 decimal places to avoid WKT parser overflow
    pred_gdf["geometry"] = pred_gdf["geometry"].apply(
        lambda g: shapely_wkt.loads(shapely_wkt.dumps(g, rounding_precision=6))
    )

    # Bulk insert as in_review annotations
    count = store.add_annotations_bulk(
        geometries=list(pred_gdf.geometry),
        class_ids=[int(c) for c in pred_gdf["class_id"]],
        region_id=region_id,
        source="inference",
        status="in_review",
    )

    logger.info(
        "Promoted inference job %s → project '%s': region %d, %d annotations",
        req.job_id, target_project, region_id, count,
    )
    return {"region_id": region_id, "annotations_created": count, "project_id": target_project}


@router.post("/regions/{region_id}/approve")
def approve_region(region_id: int, store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    """Approve an in-review region and all its annotations for training."""
    try:
        count = store.approve_region(region_id)
        return {"status": "ok", "region_id": region_id, "annotations_approved": count}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stats")
def get_stats(store=Depends(get_label_store)):
    return store.get_stats()


@router.post("/upload")
async def upload_labels(file: UploadFile = File(...), store=Depends(get_label_store), _user=Depends(require_active_project_contributor)):
    """Upload a GeoPackage file to replace the current labels."""
    import shutil
    import tempfile
    from pathlib import Path

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gpkg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Replace current labels (write to remote path if using local-copy mode)
    target = store._remote_path if store._using_local_copy else store.path
    shutil.copy2(tmp_path, target)
    Path(tmp_path).unlink()
    if store._using_local_copy:
        store.reload_from_remote()

    return {"status": "ok", "path": str(target)}
