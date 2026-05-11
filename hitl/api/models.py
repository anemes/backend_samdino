"""Model registry endpoints: list checkpoints, select model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from .deps import get_current_user, resolve_project_role

logger = logging.getLogger(__name__)

router = APIRouter()

# Projects whose IDs start with this prefix are internal and hidden from the catalogue.
_INFERENCE_PREFIX = "_inference"


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


@router.get("/catalogue")
def list_catalogue(request: Request, state=Depends(get_deps)):
    """List all available model checkpoints across all projects plus global models.

    Returns a flat list of entries, each describing one checkpoint with enough
    metadata for the standalone inference panel to show a human-readable label
    and pass the right checkpoint_run_id + checkpoint_project_id back to the
    inference endpoint.

    Only projects visible to the caller (owner, member, or public) appear.
    Internal _inference_* projects are always excluded.
    """
    from ..models.registry import ModelRegistry

    user = get_current_user(request)
    pm = state.project_manager
    config = state.config
    results = []

    # ── Per-project checkpoints ───────────────────────────────────────────────
    for project_id in pm.list_all_project_ids():
        if project_id.startswith(_INFERENCE_PREFIX):
            continue
        # Visibility check
        info = pm.get_project(project_id)
        if info is not None and resolve_project_role(user, info) is None:
            continue
        try:
            registry = ModelRegistry(config.paths.checkpoint_dir, project_id=project_id)
            production_run = registry.get_production_run()
            # Deduplicate by run_id: keep only the best checkpoint per run.
            seen_runs: dict = {}
            for ckpt in registry.list_checkpoints():
                run_id = ckpt.get("run_id", "")
                mIoU = ckpt.get("best_val_mIoU", 0.0)
                if run_id not in seen_runs or mIoU > seen_runs[run_id].get("best_val_mIoU", 0.0):
                    seen_runs[run_id] = ckpt

            for run_id, ckpt in seen_runs.items():
                mIoU = ckpt.get("best_val_mIoU", 0.0)
                display = f"{project_id} / {run_id}"
                if mIoU:
                    display += f" (mIoU {mIoU:.2f})"
                if production_run and run_id == production_run:
                    display += " *"
                results.append({
                    "run_id": run_id,
                    "source": "project",
                    "project_id": project_id,
                    "display_name": display,
                    "class_names": ckpt.get("class_names", []),
                    "num_classes": ckpt.get("num_classes", 0),
                    "best_val_mIoU": mIoU,
                    "checkpoint_path": ckpt.get("checkpoint_path", ""),
                    "timestamp": ckpt.get("timestamp", ""),
                })
        except Exception:
            logger.exception("Failed to read registry for project '%s'", project_id)

    # ── Globally registered models ────────────────────────────────────────────
    # Operator places a registry.json in data/models/global/registry.json.
    # Each entry must have at minimum: run_id, checkpoint_path, class_names,
    # num_classes.  display_name and best_val_mIoU are optional.
    global_registry_path = Path(config.models.dinov3.path).parent / "global" / "registry.json"
    if global_registry_path.exists():
        try:
            global_models = json.loads(global_registry_path.read_text())
            for model in global_models:
                model.setdefault("source", "global")
                model.setdefault("project_id", None)
                model.setdefault("display_name", model.get("run_id", "unknown"))
                model.setdefault("class_names", [])
                model.setdefault("num_classes", 0)
                model.setdefault("best_val_mIoU", None)
                model.setdefault("timestamp", "")
                results.append(model)
        except Exception:
            logger.exception("Failed to read global model registry at %s", global_registry_path)

    return {"catalogue": results}
