"""Stitcher: reassemble tiled predictions with cosine-window blending.

In overlap zones, predictions from adjacent tiles are blended using a
2D cosine window to eliminate visible seams. Each pixel's final prediction
is the weighted average of all tiles that cover it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .tiler import InferenceTile


class Stitcher:
    """Accumulate tiled predictions with smooth blending.

    Usage:
        stitcher = Stitcher(output_shape=(1000, 1000), num_classes=5, patch_size=512, overlap=128)
        for tile in tiles:
            logits = model(tile.image)  # (num_classes, H, W)
            stitcher.add_tile(tile, logits)
        class_map, confidence_map = stitcher.finalize()
    """

    def __init__(
        self,
        output_shape: Tuple[int, int],
        num_classes: int,
        patch_size: int,
        overlap: int,
    ):
        self.output_shape = output_shape  # (H, W)
        self.num_classes = num_classes
        self.patch_size = patch_size
        H, W = output_shape

        # Accumulated weighted logits and weight map
        self._logit_accum = np.zeros((num_classes, H, W), dtype=np.float64)
        self._weight_map = np.zeros((H, W), dtype=np.float64)

        # Pre-compute blending window
        self._window = self._build_cosine_window(patch_size)

    @staticmethod
    def _build_cosine_window(size: int) -> np.ndarray:
        """Build a 2D cosine (raised cosine / Hann) blending window.

        Center pixels have weight 1.0, edges taper to 0.0.
        """
        # 1D raised cosine
        x = np.linspace(0, np.pi, size)
        w1d = 0.5 * (1 - np.cos(x))
        # Outer product for 2D
        window = np.outer(w1d, w1d)
        return window

    def add_tile(self, tile: InferenceTile, logits: np.ndarray) -> None:
        """Add a tile's prediction to the accumulator.

        Args:
            tile: InferenceTile with pixel_offset.
            logits: (num_classes, H, W) float32 logits (pre-softmax).
        """
        x_off, y_off = tile.pixel_offset
        _, lh, lw = logits.shape
        H, W = self.output_shape

        # Compute valid region (may be clipped at image edges)
        y_end = min(y_off + lh, H)
        x_end = min(x_off + lw, W)
        valid_h = y_end - y_off
        valid_w = x_end - x_off

        # Slice the window and logits to valid region
        window_slice = self._window[:valid_h, :valid_w]
        logits_slice = logits[:, :valid_h, :valid_w]

        # Accumulate
        self._logit_accum[:, y_off:y_end, x_off:x_end] += logits_slice * window_slice[None, :, :]
        self._weight_map[y_off:y_end, x_off:x_end] += window_slice

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Finalize: divide by weights, compute class map and confidence.

        Returns:
            class_map: (H, W) uint8 with predicted class IDs.
            confidence_map: (H, W) float32 with per-pixel entropy (lower = more confident).
        """
        # Avoid division by zero
        safe_weights = np.where(self._weight_map > 0, self._weight_map, 1.0)

        # Normalize logits
        normalized = self._logit_accum / safe_weights[None, :, :]

        # Softmax
        exp_logits = np.exp(normalized - normalized.max(axis=0, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        # Class map
        class_map = probs.argmax(axis=0).astype(np.uint8)

        # Confidence as entropy (lower = more confident)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=0)
        # Normalize to [0, 1]
        max_entropy = np.log(self.num_classes)
        confidence_map = (entropy / max_entropy).astype(np.float32)

        # Mark unvisited pixels
        unvisited = self._weight_map == 0
        class_map[unvisited] = 0
        confidence_map[unvisited] = 1.0

        return class_map, confidence_map
