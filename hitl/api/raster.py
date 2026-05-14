"""Raster source management endpoints.

Register and manage raster sources (XYZ tile services, GeoTIFFs).
Sources are persisted to a JSON sidecar (raster_sources.json) so they survive
server restarts.  Thread-safe via a module-level lock.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .deps import require_active_project_contributor

logger = logging.getLogger(__name__)

router = APIRouter()

_lock = threading.Lock()
_sources: Dict[str, dict] = {}
_next_id: int = 1
_persist_path: Optional[Path] = None  # set once on first write


def _sources_path(state) -> Path:
    return Path(state.config.paths.dataset_cache_dir) / "raster_sources.json"


def _load(state) -> None:
    """Load persisted sources from disk into the in-memory registry."""
    global _next_id
    path = _sources_path(state)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
        for entry in data.get("sources", []):
            sid = entry.get("source_id")
            if sid:
                _sources[sid] = entry
                # Keep _next_id ahead of the highest existing numeric suffix.
                try:
                    num = int(sid.split("_")[-1])
                    if num >= _next_id:
                        _next_id = num + 1
                except ValueError:
                    pass
        logger.info("Loaded %d raster source(s) from %s", len(_sources), path)
    except Exception:
        logger.exception("Failed to load raster_sources.json; starting empty")


def _save(state) -> None:
    path = _sources_path(state)
    try:
        path.write_text(json.dumps({"sources": list(_sources.values())}, indent=2))
    except Exception:
        logger.exception("Failed to persist raster_sources.json")


class RegisterXYZRequest(BaseModel):
    name: str
    url_template: str  # e.g. "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    default_zoom: int = 18
    rate_limit: float = 10.0
    headers: Optional[Dict[str, str]] = None


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/register-xyz")
def register_xyz(
    req: RegisterXYZRequest,
    state=Depends(get_deps),
    _user=Depends(require_active_project_contributor),
):
    """Register an XYZ tile service as a raster source."""
    global _next_id

    with _lock:
        # Lazy-load persisted sources on first write in this process lifetime.
        if not _sources and _persist_path is None:
            _load(state)

        source_id = f"xyz_{_next_id}"
        _next_id += 1

        # Store headers separately from the public record so they are never
        # returned by list/get responses.
        _sources[source_id] = {
            "source_id": source_id,
            "name": req.name,
            "source_type": "xyz",
            "url_template": req.url_template,
            "default_zoom": req.default_zoom,
            "rate_limit": req.rate_limit,
            "_headers": req.headers or {},  # prefixed _ = internal only
            "cache_dir": str(state.config.paths.tile_cache_dir),
        }
        _save(state)

    return {"source_id": source_id, "name": req.name}


@router.get("/sources")
def list_sources(state=Depends(get_deps)):
    """List all registered raster sources."""
    with _lock:
        if not _sources:
            _load(state)
        return {
            "sources": [
                {
                    "source_id": s["source_id"],
                    "name": s["name"],
                    "source_type": s["source_type"],
                    "url_template": s.get("url_template"),
                    "default_zoom": s.get("default_zoom"),
                }
                for s in _sources.values()
            ]
        }


@router.get("/sources/{source_id}")
def get_source(source_id: str, state=Depends(get_deps)):
    """Get details of a registered raster source (credentials excluded)."""
    with _lock:
        if not _sources:
            _load(state)
        if source_id not in _sources:
            raise HTTPException(status_code=404, detail="Source not found")
        s = _sources[source_id]
    return {
        "source_id": s["source_id"],
        "name": s["name"],
        "source_type": s["source_type"],
        "url_template": s.get("url_template"),
        "default_zoom": s.get("default_zoom"),
    }


@router.delete("/sources/{source_id}")
def delete_source(
    source_id: str,
    state=Depends(get_deps),
    _user=Depends(require_active_project_contributor),
):
    """Remove a registered raster source."""
    with _lock:
        if source_id not in _sources:
            raise HTTPException(status_code=404, detail="Source not found")
        del _sources[source_id]
        _save(state)
    return {"deleted": source_id}


def get_raster_source(source_id: str, state=None):
    """Instantiate a RasterSource from a registered ID. Used by other services."""
    with _lock:
        if source_id not in _sources and state is not None:
            _load(state)
        if source_id not in _sources:
            raise ValueError(f"Unknown raster source: {source_id}")
        info = _sources[source_id]

    if info["source_type"] == "xyz":
        from ..data.raster_source import XYZTileSource
        return XYZTileSource(
            url_template=info["url_template"],
            zoom=info["default_zoom"],
            cache_dir=info.get("cache_dir"),
            rate_limit=info.get("rate_limit", 10.0),
            headers=info.get("_headers") or info.get("headers"),  # back-compat
        )

    raise ValueError(f"Unsupported source type: {info['source_type']}")
