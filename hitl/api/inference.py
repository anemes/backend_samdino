"""Inference endpoints: predict on AOI, poll status, download results."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..data.label_store import SegClass

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


def _resolve_checkpoint_classes(
    checkpoint_record: Optional[dict],
    user_classes: list,
    store,
) -> Tuple[int, List[str], Dict[int, int], List[str]]:
    """Match checkpoint classes to project classes by name.

    Returns:
        ckpt_num_classes: num_classes to build the model with (from checkpoint)
        ckpt_class_names: class name list for the checkpoint model
        class_id_map: checkpoint class index → project class_id
        warnings: list of warning messages
    """
    warnings: List[str] = []

    if not checkpoint_record:
        # No checkpoint — use project classes directly, no remapping
        project_num = store.get_num_classes()
        project_map = {c.class_id: c.name for c in user_classes}
        names = ["ignore", "background"]
        for i in range(2, project_num):
            names.append(project_map.get(i, f"class_{i}"))
        return project_num, names, {}, warnings

    ckpt_class_names = checkpoint_record.get("class_names", [])
    ckpt_num_classes = checkpoint_record.get("num_classes", 0)

    if not ckpt_class_names or ckpt_num_classes < 2:
        # Legacy checkpoint without class metadata — fall back to project
        project_num = store.get_num_classes()
        project_map = {c.class_id: c.name for c in user_classes}
        names = ["ignore", "background"]
        for i in range(2, project_num):
            names.append(project_map.get(i, f"class_{i}"))
        warnings.append("Checkpoint has no class metadata — using project classes (may mismatch)")
        return project_num, names, {}, warnings

    # Build name → project class_id lookup
    project_class_map = {c.name: c.class_id for c in user_classes}
    existing_names = {c.name for c in user_classes}

    # Auto-add checkpoint classes missing from project
    added = []
    for ckpt_idx in range(2, ckpt_num_classes):
        ckpt_name = ckpt_class_names[ckpt_idx] if ckpt_idx < len(ckpt_class_names) else None
        if ckpt_name and ckpt_name not in existing_names:
            new_id = max((c.class_id for c in user_classes), default=1) + 1 + len(added)
            user_classes.append(SegClass(class_id=new_id, name=ckpt_name, color="#888888"))
            added.append(ckpt_name)

    if added:
        store.set_classes(user_classes)
        warnings.append(f"Added checkpoint classes to project: {', '.join(added)}")
        # Rebuild lookup after additions
        project_class_map = {c.name: c.class_id for c in user_classes}

    # Map checkpoint indices → project class_ids by name
    class_id_map: Dict[int, int] = {}
    for ckpt_idx in range(2, ckpt_num_classes):
        ckpt_name = ckpt_class_names[ckpt_idx] if ckpt_idx < len(ckpt_class_names) else None
        if ckpt_name and ckpt_name in project_class_map:
            project_id = project_class_map[ckpt_name]
            class_id_map[ckpt_idx] = project_id
            if ckpt_idx != project_id:
                warnings.append(
                    f"Class '{ckpt_name}': checkpoint index {ckpt_idx} "
                    f"!= project class_id {project_id} (remapped)"
                )

    # Warn about project classes missing from checkpoint
    ckpt_user_names = set(ckpt_class_names[2:ckpt_num_classes])
    for cls in user_classes:
        if cls.name not in ckpt_user_names:
            warnings.append(f"Project class '{cls.name}' not in checkpoint — won't be predicted")

    return ckpt_num_classes, ckpt_class_names, class_id_map, warnings


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

    user_classes = list(state.label_store.get_classes())

    # Resolve checkpoint
    checkpoint_record = None
    checkpoint_path = None
    if req.checkpoint_run_id:
        checkpoint_path = state.registry.get_checkpoint_path(
            req.checkpoint_run_id, req.checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
        # Find matching record for metadata
        for rec in state.registry.list_checkpoints():
            if rec.get("checkpoint_path") == checkpoint_path:
                checkpoint_record = rec
                break
    else:
        checkpoint_record = state.registry.get_best_checkpoint()
        if checkpoint_record:
            checkpoint_path = checkpoint_record.get("checkpoint_path")

    # Match checkpoint classes to project classes
    num_classes, class_names, class_id_map, warnings = _resolve_checkpoint_classes(
        checkpoint_record, user_classes, state.label_store,
    )

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(req.aoi_bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=req.project_id,
        class_id_map=class_id_map,
    )
    return {"job_id": job_id, "status": "started", "warnings": warnings}


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

    user_classes = list(state.label_store.get_classes())

    # Resolve checkpoint
    checkpoint_record = None
    checkpoint_path = None
    if checkpoint_run_id:
        checkpoint_path = state.registry.get_checkpoint_path(
            checkpoint_run_id, checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
        for rec in state.registry.list_checkpoints():
            if rec.get("checkpoint_path") == checkpoint_path:
                checkpoint_record = rec
                break
    else:
        checkpoint_record = state.registry.get_best_checkpoint()
        if checkpoint_record:
            checkpoint_path = checkpoint_record.get("checkpoint_path")

    # Match checkpoint classes to project classes
    num_classes, class_names, class_id_map, warnings = _resolve_checkpoint_classes(
        checkpoint_record, user_classes, state.label_store,
    )

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=project_id,
        class_id_map=class_id_map,
    )
    return {"job_id": job_id, "status": "started", "warnings": warnings}


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
