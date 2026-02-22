"""Inference endpoints: predict on AOI, poll status, download results."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    raster_path: Optional[str] = None  # path to source GeoTIFF
    xyz_url: Optional[str] = None  # XYZ tile URL template (alternative to raster_path)
    xyz_zoom: int = 18  # zoom level for XYZ source
    aoi_bounds: List[float]  # [minx, miny, maxx, maxy]
    project_id: str = "default"
    checkpoint_run_id: Optional[str] = None  # which model run to use
    checkpoint_type: str = "best"  # "best" or "latest"


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/predict")
def start_prediction(req: PredictRequest, state=Depends(get_deps)):
    """Start tiled inference on an AOI."""
    from ..data.raster_source import GeoTIFFSource

    if state.inference_service.is_running:
        raise HTTPException(status_code=409, detail="Inference already in progress")

    if req.xyz_url:
        from ..data.raster_source import XYZTileSource
        raster_source = XYZTileSource(
            url_template=req.xyz_url,
            zoom=req.xyz_zoom,
            cache_dir=str(state.config.paths.tile_cache_dir),
        )
    elif req.raster_path:
        raster_source = GeoTIFFSource(req.raster_path)
    else:
        raise HTTPException(status_code=400, detail="Provide raster_path or xyz_url")
    num_classes = state.label_store.get_num_classes()
    # Build class_names list indexed by class_id: [ignore, background, cls2, cls3, ...]
    user_classes = state.label_store.get_classes()
    class_names = ["ignore", "background"]
    class_map = {c.class_id: c.name for c in user_classes}
    for i in range(2, num_classes):
        class_names.append(class_map.get(i, f"class_{i}"))

    # Resolve checkpoint
    checkpoint_path = None
    if req.checkpoint_run_id:
        checkpoint_path = state.registry.get_checkpoint_path(
            req.checkpoint_run_id, req.checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
    else:
        best = state.registry.get_best_checkpoint()
        if best:
            checkpoint_path = best.get("checkpoint_path")

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(req.aoi_bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=req.project_id,
    )
    return {"job_id": job_id, "status": "started"}


@router.post("/predict-upload")
async def start_prediction_upload(
    file: UploadFile = File(...),
    aoi_bounds: str = Form(...),  # JSON string: "[minx, miny, maxx, maxy]"
    project_id: str = Form("default"),
    checkpoint_run_id: Optional[str] = Form(None),
    checkpoint_type: str = Form("best"),
    state=Depends(get_deps),
):
    """Start tiled inference from an uploaded GeoTIFF (e.g. QGIS viewport capture).

    The uploaded file is saved to disk so the background inference thread can
    read it.  It persists until the next inference run overwrites it.
    """
    from ..data.raster_source import GeoTIFFSource

    if state.inference_service.is_running:
        raise HTTPException(status_code=409, detail="Inference already in progress")

    # Parse AOI bounds from JSON string
    try:
        bounds = json.loads(aoi_bounds)
        if len(bounds) != 4:
            raise ValueError("Need exactly 4 values")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid aoi_bounds: {e}")

    # Save uploaded file
    upload_dir = Path(state.config.paths.dataset_cache_dir) / "inference_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix if file.filename else ".tif"
    upload_path = upload_dir / f"upload{suffix}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    raster_source = GeoTIFFSource(str(upload_path))

    # Resolve classes
    num_classes = state.label_store.get_num_classes()
    user_classes = state.label_store.get_classes()
    class_names = ["ignore", "background"]
    class_map = {c.class_id: c.name for c in user_classes}
    for i in range(2, num_classes):
        class_names.append(class_map.get(i, f"class_{i}"))

    # Resolve checkpoint
    checkpoint_path = None
    if checkpoint_run_id:
        checkpoint_path = state.registry.get_checkpoint_path(
            checkpoint_run_id, checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
    else:
        best = state.registry.get_best_checkpoint()
        if best:
            checkpoint_path = best.get("checkpoint_path")

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=project_id,
    )
    return {"job_id": job_id, "status": "started"}


@router.get("/status")
def inference_status(state=Depends(get_deps)):
    """Get current inference state."""
    return state.inference_service.get_state()


@router.get("/result/{job_id}/{file_type}")
def download_result(job_id: str, file_type: str, state=Depends(get_deps)):
    """Download a result file. file_type: 'class_raster', 'confidence_raster', 'vector'."""
    inf_state = state.inference_service.state
    if inf_state.job_id != job_id:
        raise HTTPException(status_code=404, detail="Job not found")
    if inf_state.status != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {inf_state.status}")

    path = inf_state.result_paths.get(file_type)
    if not path:
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found")

    return FileResponse(path, filename=path.split("/")[-1])
