"""Tests for GPUManager fixes: null check on unload (#20)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_config():
    config = MagicMock()
    config.gpu.device = "cpu"
    config.gpu.max_vram_gb = 24.0
    config.gpu.training_vram_overhead_gb = 4.0
    return config


def test_unload_partial_load():
    """#20: Unloading SAM3 when processor is None must not raise."""
    from hitl.services.gpu_manager import GPUManager, ActiveModel

    gm = GPUManager(_make_config(), device="cpu")
    gm._sam3_model = MagicMock()
    gm._sam3_processor = None
    gm._loaded = {ActiveModel.SAM3}

    # Should not raise AttributeError
    gm._unload_model(ActiveModel.SAM3)

    assert gm._sam3_model is None
    assert gm._sam3_processor is None
    assert ActiveModel.SAM3 not in gm._loaded


def test_unload_full_load():
    """Regression: unloading with both model and processor set works."""
    from hitl.services.gpu_manager import GPUManager, ActiveModel

    gm = GPUManager(_make_config(), device="cpu")
    gm._sam3_model = MagicMock()
    gm._sam3_processor = MagicMock()
    gm._loaded = {ActiveModel.SAM3}

    gm._unload_model(ActiveModel.SAM3)

    assert gm._sam3_model is None
    assert gm._sam3_processor is None
    assert ActiveModel.SAM3 not in gm._loaded


def test_unload_when_not_loaded():
    """Regression: unloading a model that isn't loaded is a no-op."""
    from hitl.services.gpu_manager import GPUManager, ActiveModel

    gm = GPUManager(_make_config(), device="cpu")
    gm._sam3_model = None
    gm._sam3_processor = None

    # Should be a no-op, not crash
    gm._unload_model(ActiveModel.SAM3)
