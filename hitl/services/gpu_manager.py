"""GPU model lifecycle manager.

Manages exclusive GPU access for large models that cannot coexist in VRAM.
SAM3 (~6GB) and the segmentor (~16GB) are mutually exclusive on a 24GB GPU.

Usage:
    gpu = GPUManager(config)
    processor = gpu.acquire_sam3()       # loads SAM3, unloads segmentor if needed
    segmentor = gpu.acquire_segmentor()  # loads segmentor, unloads SAM3 if needed
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class ActiveModel(Enum):
    NONE = "none"
    SAM3 = "sam3"
    SEGMENTOR = "segmentor"


class GPUManager:
    """Thread-safe GPU model manager with mutual exclusion.

    Only one large model is loaded at a time. Switching models
    requires unloading the current one first to free VRAM.
    """

    def __init__(self, device: str = "cuda"):
        self._lock = threading.Lock()
        self._active = ActiveModel.NONE
        self._device = device

        # SAM3
        self._sam3_model = None
        self._sam3_processor = None

        # Segmentor
        self._segmentor = None
        self._segmentor_num_classes: Optional[int] = None

    @property
    def active_model(self) -> ActiveModel:
        return self._active

    @property
    def device(self) -> str:
        return self._device

    def acquire_sam3(self, config) -> "Sam3Processor":
        """Load SAM3 onto GPU. Unloads any other model first.

        Args:
            config: AppConfig with sam3 checkpoint path.

        Returns:
            Sam3Processor ready for inference.
        """
        with self._lock:
            if self._active == ActiveModel.SAM3 and self._sam3_processor is not None:
                return self._sam3_processor

            self._unload_all()
            logger.info("Loading SAM3 onto GPU...")

            # Deferred import to avoid loading sam3 at module level
            from sam3 import model_builder as sam3_builder
            from sam3.model.sam3_image_processor import Sam3Processor

            sam3_cfg = config.models.sam3
            bpe_path = sam3_cfg.bpe_path
            if bpe_path is None:
                # Auto-resolve from sam3 package
                from pathlib import Path

                builder_file = Path(sam3_builder.__file__).resolve()
                bpe_path = str(builder_file.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")

            model = sam3_builder.build_sam3_image_model(
                bpe_path=bpe_path,
                device=self._device,
                checkpoint_path=sam3_cfg.checkpoint,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            self._sam3_model = model
            self._sam3_processor = Sam3Processor(
                model,
                resolution=sam3_cfg.resolution,
                device=self._device,
                confidence_threshold=sam3_cfg.confidence_threshold,
            )
            self._active = ActiveModel.SAM3
            logger.info("SAM3 loaded successfully.")
            return self._sam3_processor

    def get_sam3_predictor(self):
        """Get the SAM3 interactive predictor for point/box prompts.

        Requires SAM3 to be loaded via acquire_sam3() first.
        Returns the SAM1-compatible interactive predictor.
        """
        with self._lock:
            if self._active != ActiveModel.SAM3 or self._sam3_model is None:
                raise RuntimeError("SAM3 not loaded. Call acquire_sam3() first.")
            return self._sam3_model.inst_interactive_predictor

    def get_sam3_model(self):
        """Get the raw SAM3 model for predict_inst() calls.

        Thread-safe: holds the lock while checking the model is loaded.
        Requires SAM3 to be loaded via acquire_sam3() first.
        """
        with self._lock:
            if self._active != ActiveModel.SAM3 or self._sam3_model is None:
                raise RuntimeError("SAM3 not loaded. Call acquire_sam3() first.")
            return self._sam3_model

    def acquire_segmentor(self, config, num_classes: int) -> "Segmentor":
        """Load segmentor onto GPU. Unloads any other model first.

        If the segmentor is already loaded with the same num_classes, returns it.
        Otherwise rebuilds with the new class count.

        Args:
            config: AppConfig with DINOv3 path and training config.
            num_classes: Number of segmentation classes (including background).

        Returns:
            Segmentor model on GPU.
        """
        with self._lock:
            if (
                self._active == ActiveModel.SEGMENTOR
                and self._segmentor is not None
                and self._segmentor_num_classes == num_classes
            ):
                return self._segmentor

            self._unload_all()
            logger.info("Loading segmentor onto GPU (num_classes=%d)...", num_classes)

            from .._models_build import build_segmentor

            segmentor = build_segmentor(config, num_classes)
            segmentor = segmentor.to(self._device)

            self._segmentor = segmentor
            self._segmentor_num_classes = num_classes
            self._active = ActiveModel.SEGMENTOR
            logger.info("Segmentor loaded successfully.")
            return self._segmentor

    def get_segmentor(self) -> Optional["Segmentor"]:
        """Get the currently loaded segmentor, or None."""
        with self._lock:
            if self._active == ActiveModel.SEGMENTOR:
                return self._segmentor
            return None

    def _unload_all(self) -> None:
        """Free all GPU models and clear VRAM cache."""
        if self._sam3_model is not None:
            logger.info("Unloading SAM3...")
            del self._sam3_processor
            del self._sam3_model
            self._sam3_processor = None
            self._sam3_model = None

        if self._segmentor is not None:
            logger.info("Unloading segmentor...")
            del self._segmentor
            self._segmentor = None
            self._segmentor_num_classes = None

        self._active = ActiveModel.NONE
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def vram_usage_mb(self) -> float:
        """Current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
