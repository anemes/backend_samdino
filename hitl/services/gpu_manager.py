"""GPU model lifecycle manager.

Manages GPU access for large models with dynamic coexistence support.
SAM3 (~6GB) and the segmentor (~16GB) can coexist on GPUs with enough VRAM,
or are mutually exclusive on smaller GPUs (controlled by max_vram_gb config).

Usage:
    gpu = GPUManager(config)
    processor = gpu.acquire_sam3()       # loads SAM3, evicts segmentor only if needed
    segmentor = gpu.acquire_segmentor()  # loads segmentor, evicts SAM3 only if needed
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


# Approximate VRAM footprint per model (GB)
_MODEL_VRAM_GB = {
    ActiveModel.SAM3: 6.0,
    ActiveModel.SEGMENTOR: 16.0,
}


class GPUManager:
    """Thread-safe GPU model manager with dynamic coexistence.

    On GPUs with enough VRAM (controlled by max_vram_gb), both SAM3 and the
    segmentor stay loaded simultaneously. On smaller GPUs, models are evicted
    as needed to fit within the VRAM budget.
    """

    def __init__(self, config, device: str = "cuda"):
        self._lock = threading.Lock()
        self._loaded: set[ActiveModel] = set()
        self._device = device
        self._max_vram_gb = config.gpu.max_vram_gb
        self._training_overhead_gb = config.gpu.training_vram_overhead_gb

        # SAM3
        self._sam3_model = None
        self._sam3_processor = None

        # Segmentor
        self._segmentor = None
        self._segmentor_num_classes: Optional[int] = None

        # Log coexistence mode
        coexist_cost = _MODEL_VRAM_GB[ActiveModel.SAM3] + _MODEL_VRAM_GB[ActiveModel.SEGMENTOR]
        if self._max_vram_gb >= coexist_cost + self._training_overhead_gb:
            logger.info(
                "VRAM budget %.1fGB: SAM3+Segmentor coexistence enabled (need %.1fGB)",
                self._max_vram_gb, coexist_cost + self._training_overhead_gb,
            )
        else:
            logger.info(
                "VRAM budget %.1fGB: mutual exclusion mode (coexistence needs %.1fGB)",
                self._max_vram_gb, coexist_cost + self._training_overhead_gb,
            )

        if torch.cuda.is_available():
            physical_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if self._max_vram_gb > physical_gb:
                logger.warning(
                    "max_vram_gb (%.1f) exceeds physical VRAM (%.1f) — risk of OOM",
                    self._max_vram_gb, physical_gb,
                )

    @property
    def active_model(self) -> ActiveModel:
        """Backward compat: returns the 'primary' active model."""
        if ActiveModel.SEGMENTOR in self._loaded:
            return ActiveModel.SEGMENTOR
        if ActiveModel.SAM3 in self._loaded:
            return ActiveModel.SAM3
        return ActiveModel.NONE

    @property
    def loaded_models(self) -> frozenset:
        """Set of currently loaded models."""
        return frozenset(self._loaded)

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # VRAM budget helpers
    # ------------------------------------------------------------------

    def _vram_in_use(self) -> float:
        """Estimate VRAM used by currently loaded models (GB)."""
        return sum(_MODEL_VRAM_GB.get(m, 0) for m in self._loaded)

    def _ensure_vram_for(self, target: ActiveModel, extra_overhead: float = 0.0) -> None:
        """Evict models if needed to fit *target* within the VRAM budget.

        Args:
            target: The model we want to load.
            extra_overhead: Additional VRAM needed (e.g. training batches).
        """
        if target in self._loaded:
            return

        needed = _MODEL_VRAM_GB[target] + extra_overhead
        available = self._max_vram_gb - self._vram_in_use()

        if available >= needed:
            logger.info(
                "VRAM budget: %.1fGB available, %.1fGB needed for %s",
                available, needed, target.value,
            )
            return

        # Must evict. Evict models other than the target, largest first.
        evict_candidates = sorted(
            self._loaded,
            key=lambda m: _MODEL_VRAM_GB.get(m, 0),
            reverse=True,
        )
        for model in evict_candidates:
            if model == target:
                continue
            self._unload_model(model)
            available = self._max_vram_gb - self._vram_in_use()
            if available >= needed:
                break

    # ------------------------------------------------------------------
    # Per-model unload
    # ------------------------------------------------------------------

    def _unload_model(self, model: ActiveModel) -> None:
        """Unload a specific model and free its VRAM."""
        if model == ActiveModel.SAM3 and self._sam3_model is not None:
            logger.info("Unloading SAM3...")
            del self._sam3_processor
            del self._sam3_model
            self._sam3_processor = None
            self._sam3_model = None
            self._loaded.discard(ActiveModel.SAM3)

        elif model == ActiveModel.SEGMENTOR and self._segmentor is not None:
            logger.info("Unloading segmentor...")
            del self._segmentor
            self._segmentor = None
            self._segmentor_num_classes = None
            self._loaded.discard(ActiveModel.SEGMENTOR)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _unload_all(self) -> None:
        """Free all GPU models and clear VRAM cache."""
        for model in list(self._loaded):
            self._unload_model(model)
        self._loaded.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Public targeted unload
    # ------------------------------------------------------------------

    def unload_segmentor(self) -> None:
        """Unload just the segmentor, preserving SAM3 if loaded."""
        with self._lock:
            self._unload_model(ActiveModel.SEGMENTOR)

    # ------------------------------------------------------------------
    # SAM3
    # ------------------------------------------------------------------

    def acquire_sam3(self, config) -> "Sam3Processor":
        """Load SAM3 onto GPU. Evicts other models only if VRAM budget requires it.

        Args:
            config: AppConfig with sam3 checkpoint path.

        Returns:
            Sam3Processor ready for inference.
        """
        with self._lock:
            if ActiveModel.SAM3 in self._loaded and self._sam3_processor is not None:
                return self._sam3_processor

            self._ensure_vram_for(ActiveModel.SAM3)
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
            self._loaded.add(ActiveModel.SAM3)
            logger.info("SAM3 loaded successfully.")
            return self._sam3_processor

    def get_sam3_predictor(self):
        """Get the SAM3 interactive predictor for point/box prompts.

        Requires SAM3 to be loaded via acquire_sam3() first.
        Returns the SAM1-compatible interactive predictor.
        """
        with self._lock:
            if ActiveModel.SAM3 not in self._loaded or self._sam3_model is None:
                raise RuntimeError("SAM3 not loaded. Call acquire_sam3() first.")
            return self._sam3_model.inst_interactive_predictor

    def get_sam3_model(self):
        """Get the raw SAM3 model for predict_inst() calls.

        Thread-safe: holds the lock while checking the model is loaded.
        Requires SAM3 to be loaded via acquire_sam3() first.
        """
        with self._lock:
            if ActiveModel.SAM3 not in self._loaded or self._sam3_model is None:
                raise RuntimeError("SAM3 not loaded. Call acquire_sam3() first.")
            return self._sam3_model

    # ------------------------------------------------------------------
    # Segmentor
    # ------------------------------------------------------------------

    def acquire_segmentor(self, config, num_classes: int,
                          training: bool = False) -> "Segmentor":
        """Load segmentor onto GPU. Evicts other models only if VRAM budget requires it.

        If the segmentor is already loaded with the same num_classes, returns it.
        Otherwise rebuilds with the new class count.

        Args:
            config: AppConfig with DINOv3 path and training config.
            num_classes: Number of segmentation classes (including background).
            training: If True, reserves extra VRAM for training batches/gradients.

        Returns:
            Segmentor model on GPU.
        """
        with self._lock:
            if (
                ActiveModel.SEGMENTOR in self._loaded
                and self._segmentor is not None
                and self._segmentor_num_classes == num_classes
            ):
                return self._segmentor

            # If num_classes changed, unload existing segmentor first
            if ActiveModel.SEGMENTOR in self._loaded and self._segmentor is not None:
                self._unload_model(ActiveModel.SEGMENTOR)

            overhead = self._training_overhead_gb if training else 0.0
            self._ensure_vram_for(ActiveModel.SEGMENTOR, extra_overhead=overhead)
            logger.info("Loading segmentor onto GPU (num_classes=%d)...", num_classes)

            from .._models_build import build_segmentor

            segmentor = build_segmentor(config, num_classes)
            segmentor = segmentor.to(self._device)

            self._segmentor = segmentor
            self._segmentor_num_classes = num_classes
            self._loaded.add(ActiveModel.SEGMENTOR)
            logger.info("Segmentor loaded successfully.")
            return self._segmentor

    def get_segmentor(self) -> Optional["Segmentor"]:
        """Get the currently loaded segmentor, or None."""
        with self._lock:
            if ActiveModel.SEGMENTOR in self._loaded:
                return self._segmentor
            return None

    def vram_usage_mb(self) -> float:
        """Current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
