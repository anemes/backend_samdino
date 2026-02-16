"""Raster source management endpoints.

Register and manage raster sources (XYZ tile services, GeoTIFFs).
Sources are stored in-memory and referenced by ID in inference requests.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


class RegisterXYZRequest(BaseModel):
    name: str
    url_template: str  # e.g. "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    default_zoom: int = 18
    rate_limit: float = 10.0
    headers: Optional[Dict[str, str]] = None


class RasterSourceInfo(BaseModel):
    source_id: str
    name: str
    source_type: str  # "xyz" or "geotiff"
    url_template: Optional[str] = None
    file_path: Optional[str] = None
    default_zoom: Optional[int] = None


# In-memory registry of raster sources
_sources: Dict[str, dict] = {}
_next_id = 1


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/register-xyz")
def register_xyz(req: RegisterXYZRequest, state=Depends(get_deps)):
    """Register an XYZ tile service as a raster source."""
    global _next_id
    source_id = f"xyz_{_next_id}"
    _next_id += 1

    _sources[source_id] = {
        "source_id": source_id,
        "name": req.name,
        "source_type": "xyz",
        "url_template": req.url_template,
        "default_zoom": req.default_zoom,
        "rate_limit": req.rate_limit,
        "headers": req.headers or {},
        "cache_dir": str(state.config.paths.tile_cache_dir),
    }

    return {"source_id": source_id, "name": req.name}


@router.get("/sources")
def list_sources():
    """List all registered raster sources."""
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
def get_source(source_id: str):
    """Get details of a registered raster source."""
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
def delete_source(source_id: str):
    """Remove a registered raster source."""
    if source_id not in _sources:
        raise HTTPException(status_code=404, detail="Source not found")
    del _sources[source_id]
    return {"deleted": source_id}


def get_raster_source(source_id: str):
    """Instantiate a RasterSource from a registered ID. Used by other services."""
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
            headers=info.get("headers"),
        )

    raise ValueError(f"Unsupported source type: {info['source_type']}")
