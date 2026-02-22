"""Training service: orchestrates dataset building, model training, and evaluation.

Runs training in a background thread so the API remains responsive.
Streams metrics via a shared state object that the API can poll.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..data.dataset_builder import DatasetBuilder, DatasetStats
from ..data.label_store import LabelStore
from ..data.raster_source import RasterSource
from ..data.tile_dataset import TileDataset
from ..models.registry import ModelRegistry, RunMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    """Shared training state for API polling."""

    run_id: str = ""
    status: str = "idle"  # idle, building_dataset, training, evaluating, complete, error
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    train_mIoU: float = 0.0
    val_loss: float = 0.0
    val_mIoU: float = 0.0
    best_val_mIoU: float = 0.0
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    dataset_stats: Optional[Dict] = None
    error_message: str = ""
    progress_pct: float = 0.0


class TrainService:
    """Manages training lifecycle.

    Usage:
        service = TrainService(config, gpu_manager, label_store, registry)
        service.start_training(raster_source, project_id)
        state = service.get_state()  # poll progress
        service.stop_training()  # early stop
    """

    def __init__(self, config, gpu_manager, label_store: LabelStore, registry: ModelRegistry):
        self.config = config
        self.gpu = gpu_manager
        self.label_store = label_store
        self.registry = registry
        self._state = TrainState()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def state(self) -> TrainState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_training(
        self,
        raster_source: RasterSource,
        project_id: str,
        config_overrides: Optional[dict] = None,
    ) -> str:
        """Start training in a background thread. Returns run_id."""
        if self.is_running:
            raise RuntimeError("Training already in progress")

        run_id = f"{project_id}_{uuid.uuid4().hex[:8]}"
        self._state = TrainState(run_id=run_id, status="building_dataset")
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(raster_source, project_id, run_id, config_overrides),
            daemon=True,
        )
        self._thread.start()
        return run_id

    def stop_training(self) -> None:
        """Request early stop."""
        self._stop_event.set()

    def get_state(self) -> dict:
        """Get current training state as a dict."""
        return {
            "run_id": self._state.run_id,
            "status": self._state.status,
            "epoch": self._state.epoch,
            "total_epochs": self._state.total_epochs,
            "train_loss": self._state.train_loss,
            "train_mIoU": self._state.train_mIoU,
            "val_loss": self._state.val_loss,
            "val_mIoU": self._state.val_mIoU,
            "best_val_mIoU": self._state.best_val_mIoU,
            "per_class_iou": self._state.per_class_iou,
            "dataset_stats": self._state.dataset_stats,
            "error_message": self._state.error_message,
            "progress_pct": self._state.progress_pct,
        }

    def _train_loop(
        self,
        raster_source: RasterSource,
        project_id: str,
        run_id: str,
        config_overrides: Optional[dict],
    ) -> None:
        """Main training loop (runs in background thread)."""
        try:
            cfg = self.config.training
            data_cfg = self.config.data
            dinov3_cfg = self.config.models.dinov3

            # Apply config overrides (e.g., epochs, batch_size)
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(cfg, key):
                        object.__setattr__(cfg, key, value)
                        logger.info("Config override: training.%s = %s", key, value)

            # --- Build dataset ---
            self._state.status = "building_dataset"
            logger.info("Building dataset for run %s...", run_id)

            dataset_dir = Path(self.config.paths.dataset_cache_dir) / run_id
            builder = DatasetBuilder(
                label_store=self.label_store,
                tile_size=data_cfg.tile_size,
                tile_overlap=data_cfg.tile_overlap,
                ignore_index=data_cfg.ignore_index,
                background_class_id=data_cfg.background_class_id,
                min_labeled_fraction=data_cfg.min_labeled_fraction,
                val_fraction=data_cfg.val_fraction,
                test_fraction=data_cfg.test_fraction,
                split_block_size=data_cfg.split_block_size,
            )

            ds_stats = builder.build(raster_source, dataset_dir)
            self._state.dataset_stats = {
                "num_tiles_train": ds_stats.num_tiles_train,
                "num_tiles_val": ds_stats.num_tiles_val,
                "num_tiles_test": ds_stats.num_tiles_test,
                "class_pixel_counts": ds_stats.class_pixel_counts,
            }

            if ds_stats.num_tiles_train == 0:
                self._state.status = "error"
                self._state.error_message = "No training tiles produced. Add more labels."
                return

            # --- Create datasets ---
            num_classes = self.label_store.get_num_classes()
            # Build class_names indexed by class_id: [ignore, background, cls2, cls3, ...]
            user_classes = self.label_store.get_classes()
            class_map = {c.class_id: c.name for c in user_classes}
            class_names = ["ignore", "background"]
            for i in range(2, num_classes):
                class_names.append(class_map.get(i, f"class_{i}"))

            aug_config = {
                "horizontal_flip": cfg.augmentations.horizontal_flip,
                "vertical_flip": cfg.augmentations.vertical_flip,
                "rotate90": cfg.augmentations.rotate90,
                "color_jitter": cfg.augmentations.color_jitter,
                "random_scale": cfg.augmentations.random_scale,
            }

            train_ds = TileDataset(
                root=dataset_dir,
                split="train",
                tile_size=data_cfg.tile_size,
                norm_mean=tuple(dinov3_cfg.norm_mean),
                norm_std=tuple(dinov3_cfg.norm_std),
                augment=True,
                ignore_index=data_cfg.ignore_index,
                min_labeled_fraction=data_cfg.min_labeled_fraction,
                aug_config=aug_config,
            )
            val_ds = TileDataset(
                root=dataset_dir,
                split="val",
                tile_size=data_cfg.tile_size,
                norm_mean=tuple(dinov3_cfg.norm_mean),
                norm_std=tuple(dinov3_cfg.norm_std),
                augment=False,
                ignore_index=data_cfg.ignore_index,
            )

            effective_batch_size = min(cfg.batch_size, len(train_ds))
            train_loader = DataLoader(
                train_ds,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=len(train_ds) > effective_batch_size,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

            # --- Load model ---
            self._state.status = "training"
            model = self.gpu.acquire_segmentor(self.config, num_classes)

            # Compute class weights from distribution
            class_weights = self._compute_class_weights(
                ds_stats.class_pixel_counts, num_classes, data_cfg.ignore_index
            )
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.gpu.device),
                ignore_index=data_cfg.ignore_index,
            )

            optimizer = AdamW(
                model.trainable_parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            # Learning rate scheduler
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, total_iters=cfg.warmup_epochs
            )
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.warmup_epochs],
            )

            scaler = GradScaler(enabled=cfg.mixed_precision)
            self._state.total_epochs = cfg.epochs
            best_val_mIoU = 0.0
            patience_counter = 0

            # --- Training loop ---
            for epoch in range(1, cfg.epochs + 1):
                if self._stop_event.is_set():
                    logger.info("Training stopped early at epoch %d", epoch)
                    break

                self._state.epoch = epoch
                self._state.progress_pct = epoch / cfg.epochs * 100

                # Train one epoch
                train_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer, scaler, cfg.mixed_precision
                )

                # Validate
                val_loss, val_mIoU, per_class_iou = self._validate(
                    model, val_loader, criterion, num_classes, data_cfg.ignore_index
                )

                scheduler.step()

                # Update state
                self._state.train_loss = train_loss
                self._state.val_loss = val_loss
                self._state.val_mIoU = val_mIoU
                self._state.per_class_iou = {
                    class_names[i] if i < len(class_names) else f"class_{i}": iou
                    for i, iou in per_class_iou.items()
                }

                logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_mIoU=%.4f",
                    epoch, cfg.epochs, train_loss, val_loss, val_mIoU,
                )

                # Log metrics
                metrics = RunMetrics(
                    run_id=run_id,
                    iteration=epoch,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_mIoU=0.0,  # computed during training if needed
                    val_loss=val_loss,
                    val_mIoU=val_mIoU,
                    per_class_iou={str(k): v for k, v in per_class_iou.items()},
                    num_train_tiles=len(train_ds),
                    num_val_tiles=len(val_ds),
                    num_classes=num_classes,
                )
                self.registry.log_metrics(metrics)

                # Save checkpoint
                ckpt_dir = Path(self.config.paths.checkpoint_dir) / run_id
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                latest_path = ckpt_dir / "latest.pt"
                model.save_checkpoint(
                    latest_path,
                    metadata={"epoch": epoch, "val_mIoU": val_mIoU, "run_id": run_id},
                )
                self.registry.save_checkpoint(
                    run_id=run_id,
                    iteration=epoch,
                    checkpoint_path=latest_path,
                    val_mIoU=val_mIoU,
                    num_classes=num_classes,
                    class_names=class_names,
                    is_best=False,
                )

                if val_mIoU > best_val_mIoU:
                    best_val_mIoU = val_mIoU
                    self._state.best_val_mIoU = best_val_mIoU
                    best_path = ckpt_dir / "best.pt"
                    model.save_checkpoint(
                        best_path,
                        metadata={"epoch": epoch, "val_mIoU": val_mIoU, "run_id": run_id},
                    )
                    self.registry.save_checkpoint(
                        run_id=run_id,
                        iteration=epoch,
                        checkpoint_path=best_path,
                        val_mIoU=val_mIoU,
                        num_classes=num_classes,
                        class_names=class_names,
                        is_best=True,
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

            self._state.status = "complete"
            logger.info("Training complete. Best val_mIoU=%.4f", best_val_mIoU)

        except Exception as e:
            logger.exception("Training failed: %s", e)
            self._state.status = "error"
            self._state.error_message = str(e)
        finally:
            # Release GPU memory regardless of success or failure
            try:
                self.gpu._unload_all()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_epoch(self, model, loader, criterion, optimizer, scaler, use_amp) -> float:
        """Train for one epoch. Returns mean loss."""
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            if self._stop_event.is_set():
                break

            images = batch["image"].to(self.gpu.device)
            masks = batch["mask"].to(self.gpu.device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, model, loader, criterion, num_classes, ignore_index) -> tuple:
        """Validate and compute per-class IoU. Returns (val_loss, mIoU, per_class_iou)."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        # Confusion matrix for IoU
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        for batch in loader:
            images = batch["image"].to(self.gpu.device)
            masks = batch["mask"].to(self.gpu.device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            # Update confusion matrix (skip ignore pixels)
            valid = targets != ignore_index
            for p, t in zip(preds[valid], targets[valid]):
                if 0 <= p < num_classes and 0 <= t < num_classes:
                    confusion[t, p] += 1

        # Compute per-class IoU
        per_class_iou = {}
        for c in range(num_classes):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            if tp + fp + fn > 0:
                per_class_iou[c] = float(tp / (tp + fp + fn))

        # Mean IoU (exclude classes with no samples)
        iou_values = [v for v in per_class_iou.values()]
        mIoU = float(np.mean(iou_values)) if iou_values else 0.0

        return total_loss / max(num_batches, 1), mIoU, per_class_iou

    @staticmethod
    def _compute_class_weights(
        pixel_counts: Dict[int, int], num_classes: int, ignore_index: int
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        counts = torch.zeros(num_classes)
        for cls_id, count in pixel_counts.items():
            if cls_id != ignore_index and 0 <= cls_id < num_classes:
                counts[cls_id] = count

        # Inverse frequency with smoothing
        total = counts.sum()
        if total == 0:
            return torch.ones(num_classes)

        weights = total / (num_classes * counts.clamp(min=1))
        weights = weights.clamp(max=10.0)  # cap max weight
        return weights
