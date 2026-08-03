"""Tests for SAMService fixes: reset clears state (#3)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitl.services.sam_service import SAMService, SAMSession


def test_reset_clears_state():
    """#3: reset() must clear inference_state dict and session reference."""
    config = MagicMock()
    gpu = MagicMock()

    service = SAMService(config, gpu)

    # Simulate a session with inference state
    state_dict = {"backbone_out": "tensor_placeholder", "features": [1, 2, 3]}
    service._session = SAMSession(
        session_id="test123",
        image_path="/tmp/test.tif",
        image_width=512,
        image_height=512,
        inference_state=state_dict,
    )

    service.reset()

    assert service._session is None
    # The dict was cleared before dereferencing
    assert len(state_dict) == 0


def test_reset_on_none_session():
    """Regression: reset() on None session doesn't raise."""
    config = MagicMock()
    gpu = MagicMock()
    service = SAMService(config, gpu)

    service._session = None
    service.reset()  # should not raise
    assert service._session is None


def test_reset_prompts_doesnt_touch_inference_state():
    """Regression: reset_prompts() keeps inference_state intact."""
    config = MagicMock()
    gpu = MagicMock()
    service = SAMService(config, gpu)

    state_dict = {"backbone_out": "tensor"}
    service._session = SAMSession(
        session_id="test",
        image_path="/tmp/test.tif",
        image_width=256,
        image_height=256,
        inference_state=state_dict,
    )
    service._session.point_coords = [[10, 20]]
    service._session.point_labels = [1]

    service.reset_prompts()

    assert service._session is not None
    assert len(service._session.inference_state) == 1
    assert service._session.point_coords == []
    assert service._session.point_labels == []
