"""Instance session lock: acquire/release/heartbeat/status endpoints.

One user at a time holds the instance lock. All other API calls return 409
until the lock is released or the idle timeout expires.

Typical plugin flow:
    1. POST /api/session/acquire  → receive token
    2. Include X-Session-Token: <token> in every subsequent request
    3. POST /api/session/heartbeat every ~60s to stay alive
    4. POST /api/session/release on disconnect (optional — idle timeout auto-releases)

Admin keys can force-release any session.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from ..app import SESSION_IDLE_TIMEOUT_S
from .deps import get_app_state, get_current_user

router = APIRouter()


@router.post("/acquire")
def acquire_session(request: Request, state=Depends(get_app_state)):
    """Acquire exclusive access to the backend instance.

    Returns the session token that must be sent as X-Session-Token on all
    subsequent requests.  Returns 409 if another user currently holds the lock.
    Re-calling while you already hold the lock refreshes the idle timer and
    returns the same token.
    """
    user = get_current_user(request)
    token = state.acquire_session(user["user_id"])
    if token is None:
        sess = state._instance_session
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Instance is in use by '{sess.user_id}'",
                "held_by": sess.user_id,
                "acquired_at": sess.acquired_at.isoformat(),
                "idle_seconds": (datetime.now() - sess.last_heartbeat).total_seconds(),
                "idle_timeout_seconds": SESSION_IDLE_TIMEOUT_S,
            },
        )
    return {
        "token": token,
        "user_id": user["user_id"],
        "idle_timeout_seconds": SESSION_IDLE_TIMEOUT_S,
    }


@router.post("/release")
def release_session(request: Request, state=Depends(get_app_state)):
    """Release the exclusive session lock.

    The request must carry the matching X-Session-Token header, or the caller
    must have an admin API key (force-release).  Safe to call even when no
    session is held.
    """
    user = get_current_user(request)
    token = request.headers.get("X-Session-Token", "")
    force = user["is_admin"]
    released = state.release_session(token=token, force=force)
    if not released:
        raise HTTPException(
            status_code=403,
            detail="Session token does not match the active session. "
            "Only the session holder or an admin can release the lock.",
        )
    return {"status": "released"}


@router.post("/heartbeat")
def heartbeat(request: Request, state=Depends(get_app_state)):
    """Refresh the session idle timer.

    Call every ~60 seconds to prevent auto-release.  Returns 401 if the
    token is invalid or the session has already expired.
    """
    token = request.headers.get("X-Session-Token", "")
    ok = state.heartbeat_session(token)
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session token. Re-acquire the session.",
        )
    return {"status": "ok"}


@router.get("/status")
def session_status(state=Depends(get_app_state)):
    """Return who (if anyone) currently holds the instance lock."""
    sess = state._instance_session
    if sess is None:
        return {"held": False, "idle_timeout_seconds": SESSION_IDLE_TIMEOUT_S}
    idle = (datetime.now() - sess.last_heartbeat).total_seconds()
    if idle > SESSION_IDLE_TIMEOUT_S:
        return {"held": False, "idle_timeout_seconds": SESSION_IDLE_TIMEOUT_S}
    return {
        "held": True,
        "held_by": sess.user_id,
        "acquired_at": sess.acquired_at.isoformat(),
        "idle_seconds": idle,
        "idle_timeout_seconds": SESSION_IDLE_TIMEOUT_S,
    }
