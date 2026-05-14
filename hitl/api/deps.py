"""FastAPI dependency factories for authentication and project access control."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from ..data.project_manager import ProjectInfo


def get_app_state():
    from ..app import app_state
    return app_state


def get_current_user(request: Request) -> dict:
    """Return {user_id, is_admin} injected by the auth middleware.

    Falls back to safe defaults so callers always get a valid dict even when
    the open-API (no-key) mode is active.
    """
    return {
        "user_id": getattr(request.state, "user_id", "default"),
        "is_admin": getattr(request.state, "is_admin", True),
    }


def resolve_project_role(user: dict, info: ProjectInfo) -> str | None:
    """Return the caller's effective role on *info*, or None if no access.

    Resolution order:
    1. admin key         → "contributor" (bypass)
    2. project owner     → "contributor"
    3. explicit member   → their listed role ("contributor" | "user")
    4. public project    → "user"
    5. no match          → None  (project invisible to this caller)
    """
    if user["is_admin"]:
        return "contributor"
    if info.owner == user["user_id"]:
        return "contributor"
    role = info.members.get(user["user_id"])
    if role:
        return role
    if info.public:
        return "user"
    return None


def require_project_contributor(user: dict, info: ProjectInfo, project_id: str) -> None:
    """Raise HTTP 403 if *user* does not have contributor access on *info*."""
    role = resolve_project_role(user, info)
    if role != "contributor":
        raise HTTPException(
            status_code=403,
            detail=f"Contributor access required on project '{project_id}'",
        )


def require_active_project_contributor(
    request: Request,
    state=Depends(get_app_state),
) -> dict:
    """Dependency: verify the caller has contributor access to the active project.

    Inject as ``_user=Depends(require_active_project_contributor)`` on any
    write endpoint that operates on the globally active project.  Returns the
    resolved user dict so callers can use it if needed.
    """
    user = get_current_user(request)
    project_id = state.active_project_id
    info = state.project_manager.get_project(project_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Active project '{project_id}' not found")
    require_project_contributor(user, info, project_id)
    return user
