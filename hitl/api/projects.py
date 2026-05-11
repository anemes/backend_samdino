"""Project management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .deps import get_current_user, resolve_project_role, require_project_contributor

router = APIRouter()

# Prefixes whose projects are internal — hidden from the user-facing list.
_INFERENCE_PREFIX = "_inference"


class CreateProjectRequest(BaseModel):
    project_id: str
    name: str
    description: str = ""


class SwitchProjectRequest(BaseModel):
    project_id: str


def get_project_manager():
    from ..app import app_state
    return app_state.project_manager


def get_app_state():
    from ..app import app_state
    return app_state


@router.get("/list")
def list_projects(request: Request, pm=Depends(get_project_manager)):
    """List projects visible to the caller (excludes internal _inference_* projects)."""
    user = get_current_user(request)
    projects = []
    for p in pm.list_projects():
        if p.project_id.startswith(_INFERENCE_PREFIX):
            continue
        if resolve_project_role(user, p) is not None:
            projects.append(p)
    return {"projects": [p.to_dict() for p in projects]}


@router.post("/create")
def create_project(req: CreateProjectRequest, request: Request, pm=Depends(get_project_manager)):
    """Create a new project.  The caller becomes the owner."""
    user = get_current_user(request)
    try:
        info = pm.create_project(req.project_id, req.name, req.description, owner=user["user_id"])
        return {"status": "ok", "project": info.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/switch")
def switch_project(req: SwitchProjectRequest, request: Request, state=Depends(get_app_state)):
    """Switch the globally active project.  Requires contributor access on the target."""
    if state is None:
        raise HTTPException(status_code=500, detail="App not initialized")

    info = state.project_manager.get_project(req.project_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Project '{req.project_id}' not found")

    user = get_current_user(request)
    require_project_contributor(user, info, req.project_id)

    state.switch_project(req.project_id)
    return {"status": "ok", "project": info.to_dict()}


@router.get("/active")
def get_active_project(state=Depends(get_app_state)):
    if state is None:
        raise HTTPException(status_code=500, detail="App not initialized")

    project_id = state.active_project_id
    if not project_id:
        return {"active": False}

    info = state.project_manager.get_project(project_id)
    return {
        "active": True,
        "project": info.to_dict() if info else {"project_id": project_id, "name": project_id},
    }


@router.delete("/{project_id}")
def delete_project(project_id: str, request: Request, state=Depends(get_app_state)):
    """Delete a project.  Only the project owner or an admin key can delete."""
    if state is None:
        raise HTTPException(status_code=500, detail="App not initialized")

    if state.active_project_id == project_id:
        raise HTTPException(status_code=400, detail="Cannot delete the active project. Switch to another project first.")

    info = state.project_manager.get_project(project_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Project not found")

    user = get_current_user(request)
    if not user["is_admin"] and info.owner != user["user_id"]:
        raise HTTPException(status_code=403, detail="Only the project owner or an admin can delete this project")

    ok = state.project_manager.delete_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok"}
