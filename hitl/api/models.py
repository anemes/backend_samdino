"""Model registry endpoints: list checkpoints, select model."""

from __future__ import annotations

from fastapi import APIRouter, Depends

router = APIRouter()


def get_deps():
    from ..app import app_state
    return app_state


@router.get("/list")
def list_models(state=Depends(get_deps)):
    """List all saved checkpoints."""
    return {
        "checkpoints": state.registry.list_checkpoints(),
        "production_run_id": state.registry.get_production_run(),
    }


@router.get("/best")
def get_best_model(state=Depends(get_deps)):
    """Get the best checkpoint (highest val_mIoU)."""
    best = state.registry.get_best_checkpoint()
    if best is None:
        return {"checkpoint": None, "message": "No checkpoints found"}
    return {"checkpoint": best}
