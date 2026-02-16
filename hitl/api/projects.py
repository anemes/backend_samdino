"""Project management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


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
def list_projects(pm=Depends(get_project_manager)):
    projects = pm.list_projects()
    return {"projects": [p.to_dict() for p in projects]}


@router.post("/create")
def create_project(req: CreateProjectRequest, pm=Depends(get_project_manager)):
    try:
        info = pm.create_project(req.project_id, req.name, req.description)
        return {"status": "ok", "project": info.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/switch")
def switch_project(req: SwitchProjectRequest, state=Depends(get_app_state)):
    if state is None:
        raise HTTPException(status_code=500, detail="App not initialized")

    info = state.project_manager.get_project(req.project_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Project '{req.project_id}' not found")

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
def delete_project(project_id: str, state=Depends(get_app_state)):
    if state is None:
        raise HTTPException(status_code=500, detail="App not initialized")

    if state.active_project_id == project_id:
        raise HTTPException(status_code=400, detail="Cannot delete the active project. Switch to another project first.")

    ok = state.project_manager.delete_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "ok"}
