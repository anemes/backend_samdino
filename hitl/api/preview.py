"""Preview endpoint: quick frozen-feature prototype matching.

Provides instant rough segmentation feedback during labeling without
requiring a trained model. Uses DINOv3 frozen features + cosine similarity.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel

router = APIRouter()


class PreviewRequest(BaseModel):
    image_path: str  # Path to image (GeoTIFF or PNG/JPG)
    prototype_points: Dict[int, List[List[float]]]  # {class_id: [[px_x, px_y], ...]}
    class_names: Optional[Dict[int, str]] = None  # {class_id: name}


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/predict")
def preview_predict(req: PreviewRequest, state=Depends(get_deps)):
    """Run frozen-feature prototype matching for quick preview.

    Returns a base64-encoded class map PNG and metadata.
    """
    # Load image
    try:
        img = Image.open(req.image_path).convert("RGB")
        image_np = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {e}")

    # Convert prototype points format
    prototype_points = {
        int(k): [(p[0], p[1]) for p in v]
        for k, v in req.prototype_points.items()
    }

    # Run preview
    from ..services.preview_service import PreviewService
    preview = PreviewService(state.config, state.gpu_manager)

    result = preview.predict(
        image=image_np,
        prototype_points=prototype_points,
        class_names=req.class_names,
    )

    # Encode class map as base64 PNG (uint8, each pixel = class_id)
    class_map = result["class_map"].astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(class_map).save(buf, format="PNG")
    class_map_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "class_map_b64": class_map_b64,
        "class_map_size": [class_map.shape[1], class_map.shape[0]],  # [W, H]
        "patch_size": result["patch_size"],
        "mean_confidence": float(result["confidence"].mean()),
        "image_size": [image_np.shape[1], image_np.shape[0]],
    }
