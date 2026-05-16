"""Inference endpoints: predict on AOI, poll status, download results."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..data.label_store import SegClass
from .deps import get_current_user, require_active_project_contributor, resolve_project_role

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    raster_path: Optional[str] = None  # path to source GeoTIFF
    xyz_url: Optional[str] = None  # XYZ tile URL template (alternative to raster_path)
    xyz_zoom: int = 18  # zoom level for XYZ source
    aoi_bounds: List[float]  # [minx, miny, maxx, maxy]
    project_id: str = "default"  # target project for class resolution / result metadata
    checkpoint_run_id: Optional[str] = None  # which model run to use
    checkpoint_project_id: Optional[str] = None  # project that owns the checkpoint (cross-project)
    checkpoint_type: str = "best"  # "best" or "latest"


def get_deps():
    from ..app import app_state
    return app_state


def _open_label_store(state, project_id: str, user: dict, auto_create: bool = False):
    """Open a LabelStore for *project_id* without switching the active project.

    - Returns the shared store when project_id matches the active project.
    - For _inference_* projects, auto-creates on first access when auto_create=True.
    - For all other projects, enforces visibility: 404 if the caller has no role.
    """
    if project_id == state.active_project_id:
        return state.label_store
    from ..data.label_store import LabelStore
    pm = state.project_manager
    project_dir = pm.get_project_dir(project_id)
    if not project_dir.exists():
        if auto_create and project_id.startswith("_inference"):
            pm.create_project(project_id, project_id, owner=user["user_id"])
        else:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    else:
        # Visibility check for non-inference projects
        if not project_id.startswith("_inference"):
            info = pm.get_project(project_id)
            if info is not None and resolve_project_role(user, info) is None:
                raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return LabelStore(
        project_dir / "labels.gpkg",
        local_cache_dir=state.config.paths.gpkg_cache_dir,
    )


def _open_registry(state, project_id: str, user: dict):
    """Open a ModelRegistry for the given project_id with visibility check."""
    from ..models.registry import ModelRegistry
    if project_id == state.active_project_id:
        return state.registry
    if not project_id.startswith("_inference"):
        pm = state.project_manager
        info = pm.get_project(project_id)
        if info is not None and resolve_project_role(user, info) is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return ModelRegistry(state.config.paths.checkpoint_dir, project_id=project_id)


def _resolve_checkpoint_classes(
    checkpoint_record: Optional[dict],
    user_classes: list,
    store,
) -> Tuple[int, List[str], Dict[int, int], List[str]]:
    """Match checkpoint classes to project classes by name.

    May add missing checkpoint classes to the project's class list and persist
    them via store.set_classes().  Works on a copy of user_classes so that the
    caller's list is never silently mutated — only store is updated on disk.

    Returns:
        ckpt_num_classes: num_classes to build the model with (from checkpoint)
        ckpt_class_names: class name list for the checkpoint model
        class_id_map: checkpoint class index → project class_id
        warnings: list of warning messages
    """
    warnings: List[str] = []
    # Work on a copy so the caller's list is never mutated in place.
    user_classes = list(user_classes)

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
def start_prediction(
    req: PredictRequest,
    request: Request,
    state=Depends(get_deps),
    _user=Depends(require_active_project_contributor),
):
    """Start tiled inference on an AOI."""
    from ..data.raster_source import GeoTIFFSource

    if state.inference_service.is_running:
        raise HTTPException(status_code=409, detail="Inference already in progress")

    user = get_current_user(request)

    effective_project_id = req.project_id

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

    # Resolve label store for the target project (class resolution + auto-add)
    label_store = _open_label_store(state, effective_project_id, user, auto_create=True)
    user_classes = list(label_store.get_classes())

    # Resolve checkpoint registry (may be from a different project)
    ckpt_project = req.checkpoint_project_id or state.active_project_id
    registry = _open_registry(state, ckpt_project, user)

    # Resolve checkpoint
    checkpoint_record = None
    checkpoint_path = None
    if req.checkpoint_run_id:
        checkpoint_path = registry.get_checkpoint_path(
            req.checkpoint_run_id, req.checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
        # Look up record by run_id — path comparison is unreliable when data has
        # moved between environments (local → Docker → ACA changes the base dir).
        checkpoint_record = registry.get_checkpoint_record(
            req.checkpoint_run_id, req.checkpoint_type
        )
    else:
        checkpoint_record = registry.get_best_checkpoint()
        if checkpoint_record:
            checkpoint_path = checkpoint_record.get("checkpoint_path")

    # Match checkpoint classes to target project's classes
    num_classes, class_names, class_id_map, warnings = _resolve_checkpoint_classes(
        checkpoint_record, user_classes, label_store,
    )

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(req.aoi_bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=effective_project_id,
        class_id_map=class_id_map,
    )
    return {"job_id": job_id, "status": "started", "warnings": warnings}


@router.post("/predict-upload")
async def start_prediction_upload(
    request: Request,
    file: UploadFile = File(...),
    aoi_bounds: str = Form(...),  # JSON string: "[minx, miny, maxx, maxy]"
    project_id: str = Form("default"),
    checkpoint_run_id: Optional[str] = Form(None),
    checkpoint_project_id: Optional[str] = Form(None),  # source project for cross-project checkpoints
    checkpoint_type: str = Form("best"),
    state=Depends(get_deps),
    _user=Depends(require_active_project_contributor),
):
    """Start tiled inference from an uploaded GeoTIFF (e.g. QGIS viewport capture).

    The uploaded file is saved to disk so the background inference thread can
    read it.  It persists until the next inference run overwrites it.
    """
    from ..data.raster_source import GeoTIFFSource

    if state.inference_service.is_running:
        raise HTTPException(status_code=409, detail="Inference already in progress")

    user = get_current_user(request)

    effective_project_id = project_id

    # Parse AOI bounds from JSON string
    try:
        bounds = json.loads(aoi_bounds)
        if len(bounds) != 4:
            raise ValueError("Need exactly 4 values")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid aoi_bounds: {e}")

    # Save uploaded file — run in threadpool so blocking file I/O doesn't stall
    # the event loop while the (potentially large) GeoTIFF is being written.
    upload_dir = Path(state.config.paths.dataset_cache_dir) / "inference_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix if file.filename else ".tif"
    upload_path = upload_dir / f"upload{suffix}"

    def _write_upload():
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    await run_in_threadpool(_write_upload)

    raster_source = GeoTIFFSource(str(upload_path))

    # Resolve label store for the target project
    label_store = _open_label_store(state, effective_project_id, user, auto_create=True)
    user_classes = list(label_store.get_classes())

    # Resolve checkpoint registry (may be from a different project)
    ckpt_project = checkpoint_project_id or state.active_project_id
    registry = _open_registry(state, ckpt_project, user)

    # Resolve checkpoint
    checkpoint_record = None
    checkpoint_path = None
    if checkpoint_run_id:
        checkpoint_path = registry.get_checkpoint_path(
            checkpoint_run_id, checkpoint_type
        )
        if checkpoint_path is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        checkpoint_path = str(checkpoint_path)
        checkpoint_record = registry.get_checkpoint_record(
            checkpoint_run_id, checkpoint_type
        )
    else:
        checkpoint_record = registry.get_best_checkpoint()
        if checkpoint_record:
            checkpoint_path = checkpoint_record.get("checkpoint_path")

    # Match checkpoint classes to target project's classes
    num_classes, class_names, class_id_map, warnings = _resolve_checkpoint_classes(
        checkpoint_record, user_classes, label_store,
    )

    job_id = state.inference_service.start_inference(
        raster_source=raster_source,
        aoi_bounds=tuple(bounds),
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id=effective_project_id,
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
