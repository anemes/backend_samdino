"""Inference service: tiled prediction over an AOI with overlap blending.

Runs inference in a background thread, producing GeoTIFF + vector outputs.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
    status: str = "idle"  # idle, running, complete, cancelled, error
    stage: str = ""  # loading_model, fetching_tiles, inferring, exporting
    tiles_processed: int = 0
    tiles_total: int = 0
    progress_pct: float = 0.0
    result_paths: Dict[str, str] = field(default_factory=dict)
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


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
        self._stop_event = threading.Event()

    @property
    def state(self) -> InferenceState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop_inference(self) -> None:
        """Request cancellation of the running inference job.

        Sets the stop event; the inference loop checks it between batches and
        exits cleanly.  Returns immediately — the thread may still be alive
        briefly while it finishes the current batch.
        """
        if self.is_running:
            logger.info("Requesting inference cancellation")
            self._stop_event.set()

    def start_inference(
        self,
        raster_source: RasterSource,
        aoi_bounds: Tuple[float, float, float, float],
        num_classes: int,
        class_names: List[str],
        checkpoint_path: Optional[str] = None,
        project_id: str = "default",
        class_id_map: Optional[Dict[int, int]] = None,
    ) -> str:
        """Start inference in a background thread. Returns job_id."""
        if self.is_running:
            raise RuntimeError("Inference already in progress")

        self._stop_event.clear()
        job_id = f"infer_{uuid.uuid4().hex[:8]}"
        self._state = InferenceState(job_id=job_id, status="running")

        self._thread = threading.Thread(
            target=self._inference_loop,
            args=(raster_source, aoi_bounds, num_classes, class_names,
                  checkpoint_path, job_id, class_id_map or {}),
            daemon=True,
        )
        self._thread.start()
        return job_id

    def get_state(self) -> dict:
        return {
            "job_id": self._state.job_id,
            "status": self._state.status,
            "stage": self._state.stage,
            "tiles_processed": self._state.tiles_processed,
            "tiles_total": self._state.tiles_total,
            "progress_pct": self._state.progress_pct,
            "result_paths": self._state.result_paths,
            "error_message": self._state.error_message,
            "warnings": self._state.warnings,
        }

    def _inference_loop(
        self,
        raster_source: RasterSource,
        aoi_bounds: Tuple[float, float, float, float],
        num_classes: int,
        class_names: List[str],
        checkpoint_path: Optional[str],
        job_id: str,
        class_id_map: Dict[int, int] = None,
    ) -> None:
        """Main inference loop."""
        try:
            inf_cfg = self.config.inference
            dinov3_cfg = self.config.models.dinov3

            # Load model
            self._state.stage = "loading_model"
            model = self.gpu.acquire_segmentor(self.config, num_classes)
            if checkpoint_path:
                model.load_checkpoint(checkpoint_path)
            model.eval()

            # Tile the AOI (fetches imagery with margin for clean edges)
            self._state.stage = "fetching_tiles"
            tiler = Tiler(patch_size=inf_cfg.tile_size, overlap=inf_cfg.tile_overlap)
            tiles, output_shape, crop_box = tiler.tile(
                raster_source, aoi_bounds, margin=inf_cfg.tile_overlap
            )

            if len(tiles) == 0:
                self._state.status = "error"
                self._state.error_message = (
                    "No tiles produced — AOI may be too small or outside raster extent"
                )
                logger.warning("Inference aborted: 0 tiles for bounds %s", aoi_bounds)
                return

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
            self._state.stage = "inferring"
            device = self.gpu.device
            batch_size = inf_cfg.batch_size

            with torch.no_grad():
                for i in range(0, len(tiles), batch_size):
                    if self._stop_event.is_set():
                        logger.info("Inference cancelled after %d/%d tiles", i, len(tiles))
                        self._state.status = "cancelled"
                        return
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

            # Finalize and crop back to original AOI (remove the margin
            # that was added for real edge context)
            class_map, confidence_map = stitcher.finalize()
            t, l, h, w = crop_box
            class_map = class_map[t:t + h, l:l + w]
            confidence_map = confidence_map[t:t + h, l:l + w]

            # Remap checkpoint class indices → project class_ids
            if class_id_map:
                remapped = np.zeros_like(class_map)
                for ckpt_idx, project_id in class_id_map.items():
                    remapped[class_map == ckpt_idx] = project_id
                # Keep background (1) and ignore (0) as-is
                remapped[class_map == 0] = 0
                remapped[class_map == 1] = 1
                class_map = remapped
                # Rebuild class_names indexed by max project class_id
                max_id = max(class_id_map.values(), default=1)
                project_names = ["ignore", "background"] + [""] * (max_id - 1)
                for ckpt_idx, project_id in class_id_map.items():
                    if ckpt_idx < len(class_names):
                        project_names[project_id] = class_names[ckpt_idx]
                class_names = project_names
                logger.info("Remapped class indices: %s", class_id_map)

            # Export — use local disk to avoid SQLite locking on Azure Files (SMB)
            self._state.stage = "exporting"
            cache = self.config.paths.gpkg_cache_dir
            if cache and str(cache).strip():
                output_dir = Path(cache) / "predictions"
            else:
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

            # Save manifest
            manifest = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "aoi_bounds": list(aoi_bounds),
                "crs": raster_source.get_crs(),
                "tiles_processed": self._state.tiles_processed,
                "num_classes": num_classes,
                "class_names": class_names,
                "files": result_paths,
            }
            manifest_path = output_dir / f"{job_id}_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            self._state.result_paths = result_paths
            if inf_cfg.output_vectors and "vector" not in result_paths:
                msg = "Vector export failed (see server logs). Promote-inference will not be available."
                self._state.warnings.append(msg)
                logger.warning(msg)
            self._state.status = "complete"
            logger.info("Inference complete: %s", result_paths)

        except Exception as e:
            logger.exception("Inference failed: %s", e)
            self._state.status = "error"
            self._state.error_message = str(e)
        finally:
            # Release segmentor GPU memory; keep SAM3 loaded if present
            try:
                self.gpu.unload_segmentor()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
