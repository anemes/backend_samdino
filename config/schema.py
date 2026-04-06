"""Pydantic configuration schema for the HITL segmentation backend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field


class DINOv3Config(BaseModel):
    path: str
    patch_size: int = 16
    num_register_tokens: int = 4
    hidden_size: int = 1024
    num_layers: int = 24
    extract_layers: List[int] = [6, 12, 18, 24]
    norm_mean: List[float] = [0.430, 0.411, 0.296]
    norm_std: List[float] = [0.213, 0.156, 0.143]


class SAM3Config(BaseModel):
    checkpoint: str
    bpe_path: Optional[str] = None
    resolution: int = 1008
    confidence_threshold: float = 0.5


class ModelsConfig(BaseModel):
    dinov3: DINOv3Config
    sam3: SAM3Config


class DataConfig(BaseModel):
    bands: List[str] = ["R", "G", "B"]
    tile_size: int = 512
    tile_overlap: int = 64
    min_labeled_fraction: float = 0.05
    background_class_id: int = 1
    ignore_index: int = 0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_block_size: float = 500.0


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate90: bool = True
    color_jitter: bool = True
    random_scale: List[float] = [0.5, 2.0]


class TrainingConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    mixed_precision: bool = True
    freeze_backbone: bool = True
    fpn_channels: int = 256
    neck_out_channels: List[int] = [256, 512, 1024, 1024]
    neck_scale_factors: List[float] = [4.0, 2.0, 1.0, 0.5]
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_mIoU"
    augmentations: AugmentationConfig = AugmentationConfig()


class InferenceConfig(BaseModel):
    tile_size: int = 512
    tile_overlap: int = 128
    batch_size: int = 8
    output_confidence: bool = True
    output_vectors: bool = True
    vector_simplify_tolerance: float = 1.0


class PathsConfig(BaseModel):
    project_dir: str = "./projects"
    checkpoint_dir: str = "./checkpoints"
    dataset_cache_dir: str = "./dataset_cache"
    tile_cache_dir: str = "./tile_cache"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    dashboard_port: int = 7860
    api_key: Optional[str] = None
    dashboard_user: str = "admin"
    dashboard_password: Optional[str] = None


class GPUConfig(BaseModel):
    device: str = "cuda"
    max_vram_gb: float = 22
    training_vram_overhead_gb: float = 4.0


class AppConfig(BaseModel):
    models: ModelsConfig
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()
    paths: PathsConfig = PathsConfig()
    server: ServerConfig = ServerConfig()
    gpu: GPUConfig = GPUConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        config = cls(**raw)

        # Resolve relative model paths against config file directory
        config_dir = path.parent.resolve()
        for attr in ("path",):
            val = getattr(config.models.dinov3, attr)
            if val and not Path(val).is_absolute():
                resolved = str((config_dir / val).resolve())
                object.__setattr__(config.models.dinov3, attr, resolved)
        for attr in ("checkpoint", "bpe_path"):
            val = getattr(config.models.sam3, attr)
            if val and not Path(val).is_absolute():
                resolved = str((config_dir / val).resolve())
                object.__setattr__(config.models.sam3, attr, resolved)
        # Resolve relative paths in paths config
        for attr in ("project_dir", "checkpoint_dir", "dataset_cache_dir", "tile_cache_dir"):
            val = getattr(config.paths, attr)
            if val and not Path(val).is_absolute():
                resolved = str((config_dir / val).resolve())
                object.__setattr__(config.paths, attr, resolved)

        return config


_config: Optional[AppConfig] = None


def load_config(path: str | Path = None) -> AppConfig:
    """Load config from YAML. Caches after first load."""
    global _config
    if _config is not None:
        return _config
    if path is None:
        path = Path(__file__).parent / "default.yaml"
    _config = AppConfig.from_yaml(path)
    return _config


def get_config() -> AppConfig:
    """Get the loaded config. Raises if not yet loaded."""
    if _config is None:
        return load_config()
    return _config
