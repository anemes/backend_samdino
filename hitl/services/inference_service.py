"""Inference service: tiled prediction over an AOI with overlap blending.

Runs inference in a background thread, producing GeoTIFF + vector outputs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.raster_source import RasterSource
from ..data.transforms import normalize_for_inference
from ..inference.exporter import export_prediction
from ..inference.stitcher import Stitcher
from ..inference.tiler import Tiler

logger = logging.getLogger(__name__)


@dataclass
class InferenceState:
    """Shared inference state for API polling."""

    job_id: str = ""
    status: str = "idle"  # idle, running, complete, error
    tiles_processed: int = 0
    tiles_total: int = 0
    progress_pct: float = 0.0
    result_paths: Dict[str, str] = field(default_factory=dict)
    error_message: str = ""


class InferenceService:
    """Manages tiled inference lifecycle.

    Usage:
        service = InferenceService(config, gpu_manager)
        job_id = service.start_inference(raster_source, aoi_bounds, model_path, ...)
        state = service.get_state()  # poll progress
    """

    def __init__(self, config, gpu_manager):
        self.config = config
        self.gpu = gpu_manager
        self._state = InferenceState()
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> InferenceState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_inference(
        self,
        raster_source: RasterSource,
        aoi_bounds: Tuple[float, float, float, float],
        num_classes: int,
        class_names: List[str],
        checkpoint_path: Optional[str] = None,
        project_id: str = "default",
    ) -> str:
        """Start inference in a background thread. Returns job_id."""
        if self.is_running:
            raise RuntimeError("Inference already in progress")

        job_id = f"infer_{uuid.uuid4().hex[:8]}"
        self._state = InferenceState(job_id=job_id, status="running")

        self._thread = threading.Thread(
            target=self._inference_loop,
            args=(raster_source, aoi_bounds, num_classes, class_names, checkpoint_path, job_id),
            daemon=True,
        )
        self._thread.start()
        return job_id

    def get_state(self) -> dict:
        return {
            "job_id": self._state.job_id,
            "status": self._state.status,
            "tiles_processed": self._state.tiles_processed,
            "tiles_total": self._state.tiles_total,
            "progress_pct": self._state.progress_pct,
            "result_paths": self._state.result_paths,
            "error_message": self._state.error_message,
        }

    def _inference_loop(
        self,
        raster_source: RasterSource,
        aoi_bounds: Tuple[float, float, float, float],
        num_classes: int,
        class_names: List[str],
        checkpoint_path: Optional[str],
        job_id: str,
    ) -> None:
        """Main inference loop."""
        try:
            inf_cfg = self.config.inference
            dinov3_cfg = self.config.models.dinov3

            # Load model
            model = self.gpu.acquire_segmentor(self.config, num_classes)
            if checkpoint_path:
                model.load_checkpoint(checkpoint_path)
            model.eval()

            # Tile the AOI
            tiler = Tiler(patch_size=inf_cfg.tile_size, overlap=inf_cfg.tile_overlap)
            tiles, output_shape = tiler.tile(raster_source, aoi_bounds)

            self._state.tiles_total = len(tiles)
            logger.info("Inference: %d tiles to process", len(tiles))

            # Create stitcher
            stitcher = Stitcher(
                output_shape=output_shape,
                num_classes=num_classes,
                patch_size=inf_cfg.tile_size,
                overlap=inf_cfg.tile_overlap,
            )

            # Process tiles in batches
            device = self.gpu.device
            batch_size = inf_cfg.batch_size

            with torch.no_grad():
                for i in range(0, len(tiles), batch_size):
                    batch_tiles = tiles[i : i + batch_size]

                    # Prepare batch
                    images = []
                    for tile in batch_tiles:
                        img = tile.image.transpose(1, 2, 0)  # (H, W, C)
                        img_tensor = normalize_for_inference(
                            img,
                            mean=tuple(dinov3_cfg.norm_mean),
                            std=tuple(dinov3_cfg.norm_std),
                        )
                        images.append(img_tensor)

                    batch = torch.cat(images, dim=0).to(device)
                    logits = model(batch)  # (B, C, H, W)
                    logits_np = logits.cpu().numpy()

                    # Add each tile's prediction to stitcher
                    for j, tile in enumerate(batch_tiles):
                        stitcher.add_tile(tile, logits_np[j])

                    self._state.tiles_processed = min(i + batch_size, len(tiles))
                    self._state.progress_pct = self._state.tiles_processed / len(tiles) * 100

            # Finalize
            class_map, confidence_map = stitcher.finalize()

            # Export
            output_dir = Path(self.config.paths.project_dir) / "predictions"
            result_paths = export_prediction(
                class_map=class_map,
                confidence_map=confidence_map,
                bounds=aoi_bounds,
                crs=raster_source.get_crs(),
                output_dir=output_dir,
                job_id=job_id,
                class_names=class_names,
                simplify_tolerance=inf_cfg.vector_simplify_tolerance,
                export_vectors=inf_cfg.output_vectors,
            )

            self._state.result_paths = result_paths
            self._state.status = "complete"
            logger.info("Inference complete: %s", result_paths)

        except Exception as e:
            logger.exception("Inference failed: %s", e)
            self._state.status = "error"
            self._state.error_message = str(e)
