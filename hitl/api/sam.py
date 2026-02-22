"""SAM3 interactive segmentation endpoints.

Endpoints:
    POST /sam/set-image     Upload image and start SAM3 session
    POST /sam/prompt        Send point/box prompt, get mask back
    POST /sam/accept        Accept current mask as annotation
    GET  /sam/session       Get current session info
    POST /sam/reset         Clear current session
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Request/Response models ---


class PromptRequest(BaseModel):
    """Point and/or box prompt for SAM3."""

    point_coords: Optional[List[List[float]]] = None  # [[x, y], ...]
    point_labels: Optional[List[int]] = None  # 1=foreground, 0=background
    box: Optional[List[float]] = None  # [x0, y0, x1, y1]
    reset_prompts: bool = False


class AcceptRequest(BaseModel):
    """Accept the current mask as an annotation."""

    class_id: int
    region_id: int
    crs: str = "EPSG:4326"
    simplify_tolerance: float = 1.0


# --- Dependencies ---


def get_sam_service():
    from ..app import app_state

    if app_state is None:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    return app_state.sam_service


def get_label_store():
    from ..app import app_state

    return app_state.label_store


# --- Endpoints ---


@router.post("/set-image")
async def set_image(
    file: UploadFile = File(...),
    sam=Depends(get_sam_service),
):
    """Upload an image and start a SAM3 session.

    The image embeddings are computed and cached. Subsequent prompts
    reuse these cached embeddings for fast inference.
    """
    # Save uploaded file to a persistent location (NOT a temp file).
    # mask_to_polygon() needs to re-open this file later to read the
    # affine transform, so it must survive until the session is reset.
    # The previous session's file is cleaned up when a new image arrives.
    from ..app import app_state

    upload_dir = Path(app_state.config.paths.dataset_cache_dir) / "sam_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Clean up previous upload
    for old in upload_dir.glob("*"):
        old.unlink(missing_ok=True)

    suffix = Path(file.filename).suffix if file.filename else ".tif"
    upload_path = upload_dir / f"sam_image{suffix}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        session = sam.set_image(str(upload_path))
        return {
            "status": "ok",
            "session_id": session.session_id,
            "image_size": [session.image_width, session.image_height],
        }
    except Exception as e:
        logger.exception("SAM set_image failed: %s", e)
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompt")
def prompt(
    req: PromptRequest,
    sam=Depends(get_sam_service),
):
    """Send a point/box prompt and get a mask prediction.

    Points accumulate across calls for iterative refinement.
    Set reset_prompts=True to start fresh on the same image.

    Returns the mask as a base64-encoded PNG and the IoU score.
    """
    if not sam.has_session:
        raise HTTPException(status_code=400, detail="No active SAM session")

    try:
        result = sam.prompt(
            point_coords=req.point_coords,
            point_labels=req.point_labels,
            box=req.box,
            reset_prompts=req.reset_prompts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Encode mask as PNG for transfer
    mask = result["mask"]
    mask_png = _mask_to_png_base64(mask)

    return {
        "session_id": result["session_id"],
        "score": result["score"],
        "mask_png": mask_png,
        "mask_size": [mask.shape[1], mask.shape[0]],  # [width, height]
    }


@router.post("/accept")
def accept_mask(
    req: AcceptRequest,
    sam=Depends(get_sam_service),
    store=Depends(get_label_store),
):
    """Accept the current SAM3 mask and save it as an annotation.

    Vectorizes the binary mask to a geo-referenced polygon (reads the
    affine transform from the session's GeoTIFF) and saves to the label store.
    """
    if not sam.has_session:
        raise HTTPException(status_code=400, detail="No active SAM session")

    polygon = sam.mask_to_polygon(
        crs=req.crs,
        simplify_tolerance=req.simplify_tolerance,
    )

    if polygon is None:
        raise HTTPException(status_code=400, detail="No valid mask to accept")

    # Validate annotation is inside the target region
    if not store.check_annotation_in_region(polygon, req.region_id, crs=req.crs):
        raise HTTPException(
            status_code=400,
            detail=f"SAM3 mask centroid is outside region {req.region_id}. "
                   "Select the correct region or draw inside the region boundary.",
        )

    idx = store.add_annotation(
        geometry_geojson=polygon,
        class_id=req.class_id,
        region_id=req.region_id,
        crs=req.crs,
        source="sam3",
    )

    # Reset prompts for next object (keep image loaded)
    session = sam.session
    session.point_coords = []
    session.point_labels = []
    session.box = None
    session.current_mask = None
    session.current_score = 0.0

    return {
        "status": "ok",
        "annotation_index": idx,
        "geometry": polygon,
    }


@router.get("/session")
def get_session(sam=Depends(get_sam_service)):
    """Get current SAM3 session status."""
    return sam.get_session_info()


@router.post("/reset")
def reset_session(sam=Depends(get_sam_service)):
    """Clear the current SAM3 session."""
    sam.reset()
    return {"status": "ok"}


# --- Helpers ---


def _mask_to_png_base64(mask: np.ndarray) -> str:
    """Encode a binary mask (H, W) as base64 PNG."""
    from PIL import Image

    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
