"""Training endpoints: start, stop, status, metrics."""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


class TrainRequest(BaseModel):
    raster_path: str = ""  # path to source GeoTIFF (mutually exclusive with xyz_url)
    xyz_url: str = ""  # XYZ tile URL template (mutually exclusive with raster_path)
    xyz_zoom: int = 18  # zoom level for XYZ tiles
    project_id: str = "default"
    # Training config overrides
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    freeze_backbone: Optional[bool] = None
    mixed_precision: Optional[bool] = None


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/start")
def start_training(req: TrainRequest, state=Depends(get_deps)):
    """Start training in background."""
    if state.train_service.is_running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Build raster source
    if req.raster_path:
        from ..data.raster_source import GeoTIFFSource
        raster_source = GeoTIFFSource(req.raster_path)
    elif req.xyz_url:
        from ..data.raster_source import XYZTileSource
        cache_dir = state.config.paths.tile_cache_dir
        raster_source = XYZTileSource(
            url_template=req.xyz_url,
            zoom=req.xyz_zoom,
            cache_dir=cache_dir,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide raster_path or xyz_url")

    # Build config overrides dict
    overrides = {}
    for field in (
        "epochs", "batch_size", "learning_rate", "weight_decay",
        "warmup_epochs", "early_stopping_patience", "freeze_backbone",
        "mixed_precision",
    ):
        val = getattr(req, field)
        if val is not None:
            overrides[field] = val

    run_id = state.train_service.start_training(
        raster_source=raster_source,
        project_id=req.project_id,
        config_overrides=overrides if overrides else None,
    )
    return {"run_id": run_id, "status": "started"}


@router.post("/stop")
def stop_training(state=Depends(get_deps)):
    """Stop training early."""
    state.train_service.stop_training()
    return {"status": "stopping"}


@router.get("/status")
def training_status(state=Depends(get_deps)):
    """Get current training state."""
    return state.train_service.get_state()


@router.get("/metrics/{run_id}")
def get_metrics(run_id: str, state=Depends(get_deps)):
    """Get all metrics for a training run."""
    return {"metrics": state.registry.get_metrics(run_id=run_id)}


@router.get("/metrics")
def get_all_metrics(state=Depends(get_deps)):
    """Get all metrics across all runs."""
    return {"metrics": state.registry.get_metrics()}
