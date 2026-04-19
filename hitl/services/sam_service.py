"""SAM3 interactive segmentation service.

Manages image sessions and handles point/box prompts for mask generation.
Uses SAM3's processor for image embedding and predict_inst() for
point/box prompts through the SAM1-compatible interactive predictor.

Session flow:
    1. Plugin captures raster extent → uploads image to backend
    2. set_image() loads image, computes backbone embeddings (cached in state)
    3. User clicks/draws box → prompt() returns binary mask
    4. User adds more clicks to refine → prompt() returns refined mask
    5. User accepts → mask vectorized to polygon → saved as annotation
    6. reset() clears session for next object
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class SAMSession:
    """State for an active SAM3 labeling session on one image."""

    session_id: str
    image_path: str
    image_width: int
    image_height: int
    inference_state: Dict[str, Any] = field(default_factory=dict)
    # Accumulated prompts for iterative refinement
    point_coords: List[List[float]] = field(default_factory=list)
    point_labels: List[int] = field(default_factory=list)  # 1=foreground, 0=background
    box: Optional[List[float]] = None  # [x0, y0, x1, y1] in pixel coords
    # Last prediction
    current_mask: Optional[np.ndarray] = None
    current_score: float = 0.0


class SAMService:
    """Manages SAM3 interactive segmentation sessions.

    Uses Sam3Processor.set_image() to compute and cache backbone features,
    then Sam3Image.predict_inst() for point/box prompt inference.
    """

    def __init__(self, config, gpu_manager):
        self._config = config
        self._gpu = gpu_manager
        self._session: Optional[SAMSession] = None

    @property
    def has_session(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> Optional[SAMSession]:
        return self._session

    def set_image(self, image_path: str) -> SAMSession:
        """Load an image into SAM3 and start a new session.

        Computes backbone embeddings via Sam3Processor.set_image() and
        caches them in the session's inference_state dict.

        Args:
            image_path: Path to the uploaded image (GeoTIFF or PNG/JPG).

        Returns:
            New SAMSession with session_id.
        """
        # Ensure SAM3 is loaded (returns Sam3Processor)
        processor = self._gpu.acquire_sam3(self._config)

        # Load image as PIL (Sam3Processor handles conversion)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Compute and cache backbone embeddings (including sam2_backbone_out)
        state = processor.set_image(img)

        self._session = SAMSession(
            session_id=str(uuid.uuid4())[:8],
            image_path=image_path,
            image_width=w,
            image_height=h,
            inference_state=state,
        )

        logger.info(
            "SAM3 session %s: image loaded (%dx%d)",
            self._session.session_id, w, h,
        )
        return self._session

    def prompt(
        self,
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[List[float]] = None,
        reset_prompts: bool = False,
    ) -> dict:
        """Send a prompt to SAM3 and get a mask prediction.

        Uses Sam3Image.predict_inst() which routes backbone features through
        the SAM1-compatible interactive predictor. Prompts accumulate across
        calls for iterative refinement.

        Args:
            point_coords: [[x, y], ...] in pixel coordinates.
            point_labels: [1, 0, ...] where 1=foreground, 0=background.
            box: [x0, y0, x1, y1] bounding box in pixel coordinates.
            reset_prompts: If True, clear accumulated prompts before adding new ones.

        Returns:
            Dict with 'mask' (binary HxW), 'score' (float), 'session_id'.
        """
        if self._session is None:
            raise RuntimeError("No active SAM session. Call set_image() first.")

        session = self._session

        # Reset if requested
        if reset_prompts:
            session.point_coords = []
            session.point_labels = []
            session.box = None

        # Accumulate prompts
        if point_coords and point_labels:
            session.point_coords.extend(point_coords)
            session.point_labels.extend(point_labels)

        if box is not None:
            session.box = box

        # Build prompt arrays
        pts = np.array(session.point_coords, dtype=np.float32) if session.point_coords else None
        lbls = np.array(session.point_labels, dtype=np.int32) if session.point_labels else None
        bx = np.array(session.box, dtype=np.float32) if session.box is not None else None

        # Use predict_inst() which routes through the main Sam3Image model
        # This uses cached backbone features from set_image()
        model = self._gpu.get_sam3_model()
        masks, iou_predictions, _ = model.predict_inst(
            session.inference_state,
            point_coords=pts,
            point_labels=lbls,
            box=bx,
            multimask_output=len(session.point_coords) <= 1 and session.box is None,
            return_logits=False,
        )

        # Select best mask
        if masks.shape[0] > 1:
            best_idx = int(np.argmax(iou_predictions))
        else:
            best_idx = 0

        mask = masks[best_idx]  # (H, W) bool
        score = float(iou_predictions[best_idx])

        session.current_mask = mask
        session.current_score = score

        logger.info(
            "SAM3 prompt: %d points, box=%s -> score=%.3f",
            len(session.point_coords),
            session.box is not None,
            score,
        )

        return {
            "mask": mask,
            "score": score,
            "session_id": session.session_id,
        }

    def get_current_mask(self) -> Optional[np.ndarray]:
        """Get the current mask from the active session."""
        if self._session is None:
            return None
        return self._session.current_mask

    def mask_to_polygon(
        self,
        crs: str = "EPSG:4326",
    ) -> Optional[dict]:
        """Convert the current mask to a GeoJSON polygon.

        Reads the affine transform and CRS from the session's GeoTIFF image,
        so the polygon is in real geo-coordinates. If `crs` differs from the
        image CRS, the polygon is reprojected.

        The binary mask is smoothed before vectorization to fill small holes,
        remove tiny fragments, and produce smoother polygon edges.
        Simplification is adaptive based on the feature's physical size.

        Args:
            crs: Target CRS for the output polygon.

        Returns:
            GeoJSON geometry dict, or None if no mask.
        """
        if self._session is None or self._session.current_mask is None:
            return None

        mask = self._session.current_mask

        # Read affine transform and CRS from the captured GeoTIFF
        import rasterio
        import rasterio.features
        from pyproj import CRS as ProjCRS, Transformer

        with rasterio.open(self._session.image_path) as src:
            affine = src.transform
            image_crs = str(src.crs)

        logger.info(
            "mask_to_polygon: image CRS=%s, target CRS=%s, transform=%s",
            image_crs, crs, affine,
        )

        mask_uint8 = (mask.astype(np.uint8) * 255)
        pixel_count = int(mask.sum())

        # Only apply morphological smoothing on large masks (>500 pixels).
        # Small masks (dams, narrow structures) are left as-is to preserve detail.
        if pixel_count > 500:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), sigmaX=1.0)
            mask_uint8 = (mask_uint8 > 127).astype(np.uint8)

        mask_bool = mask_uint8.astype(bool)
        logger.info("mask_to_polygon: %d mask pixels (smoothing %s)",
                     pixel_count, "applied" if pixel_count > 500 else "skipped")

        # Extract polygon shapes from binary mask using the real geo-transform
        shapes_iter = list(rasterio.features.shapes(
            mask_uint8,
            mask=mask_bool,
            transform=affine,
        ))

        if not shapes_iter:
            return None

        # Merge all mask regions into one polygon
        polygons = [shape(geom) for geom, val in shapes_iter if val == 1]
        if not polygons:
            return None

        merged = unary_union(polygons)

        # Keep only the largest polygon (discard tiny fragments)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)

        # Simplify large features at 1m tolerance; skip for small ones
        # where morphological smoothing already handles edge quality.
        from pyproj import CRS as ProjCRS
        is_geographic = ProjCRS.from_user_input(image_crs).is_geographic
        bounds = merged.bounds
        extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        extent_m = extent * 111_000 if is_geographic else extent
        if extent_m > 50:
            tol = 1.0 / 111_000 if is_geographic else 1.0
            merged = merged.simplify(tol, preserve_topology=True)

        if merged.is_empty:
            return None

        # Reproject if target CRS differs from image CRS
        src_crs = ProjCRS.from_user_input(image_crs)
        dst_crs = ProjCRS.from_user_input(crs)
        if src_crs != dst_crs:
            from shapely.ops import transform as shapely_transform
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            merged = shapely_transform(transformer.transform, merged)

        # Round coordinates to 6 decimal places (~10 cm in EPSG:4326).
        # rasterio pixel contours + reprojection produce 15-digit precision
        # that bloats the WKT string to sizes that overflow QGIS's internal
        # parser (std::vector larger than max_size()).
        from shapely.wkt import loads as wkt_loads, dumps as wkt_dumps
        merged = wkt_loads(wkt_dumps(merged, rounding_precision=6))

        return mapping(merged)

    def save_mask(self, output_path: str, class_id: int) -> None:
        """Save the current binary mask as a single-band GeoTIFF.

        Pixel values: class_id where mask=True, 0 elsewhere.
        Uses the affine transform and CRS from the session's source image.
        """
        if self._session is None or self._session.current_mask is None:
            return

        import rasterio

        mask = self._session.current_mask
        with rasterio.open(self._session.image_path) as src:
            affine = src.transform
            crs = src.crs

        mask_data = np.where(mask, class_id, 0).astype(np.uint8)
        with rasterio.open(
            output_path, "w", driver="GTiff",
            height=mask_data.shape[0], width=mask_data.shape[1],
            count=1, dtype="uint8", crs=crs, transform=affine,
            compress="deflate",
        ) as dst:
            dst.write(mask_data, 1)

        logger.info("Saved raw SAM mask to %s (%dx%d, class_id=%d)",
                     output_path, mask_data.shape[1], mask_data.shape[0], class_id)

    def reset(self) -> None:
        """Clear the current session (keeps SAM3 model loaded)."""
        if self._session is not None:
            logger.info("SAM3 session %s reset.", self._session.session_id)
        self._session = None

    def get_session_info(self) -> dict:
        """Get current session state for API response."""
        if self._session is None:
            return {"active": False}

        s = self._session
        return {
            "active": True,
            "session_id": s.session_id,
            "image_path": s.image_path,
            "image_size": [s.image_width, s.image_height],
            "num_points": len(s.point_coords),
            "has_box": s.box is not None,
            "has_mask": s.current_mask is not None,
            "mask_score": s.current_score,
        }
