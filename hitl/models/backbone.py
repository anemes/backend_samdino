"""DINOv3-sat backbone with multi-scale feature extraction for UperNet.

Extracts intermediate hidden states from a DINOv3 ViT-L model and reshapes
patch tokens into spatial feature maps. Outputs 4 feature maps (one per
selected layer) for use with the NeckAdapter → UperNet pipeline.

Model details:
  - DINOv3 ViT-L pretrained on 493M satellite images
  - Hidden size: 1024, 24 transformer layers, patch size 16
  - 4 register tokens (skipped during feature extraction)
  - Uses RoPE positional encoding (resolution-agnostic)
  - Normalization: mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143]
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Backbone(nn.Module):
    """Frozen DINOv3-sat ViT backbone producing multi-scale features.

    Extracts features from intermediate layers (default: [6, 12, 18, 24])
    and reshapes from (B, N_patches, D) → (B, D, H_p, W_p).

    All output feature maps have the same spatial resolution and channel count
    (1024). The NeckAdapter handles projection to different channel widths
    and scale simulation for UperNet's FPN.
    """

    def __init__(
        self,
        model_path: str,
        extract_layers: Tuple[int, ...] = (6, 12, 18, 24),
        patch_size: int = 16,
        num_register_tokens: int = 4,
        freeze: bool = True,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.extract_layers = extract_layers
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        # Number of tokens to skip: 1 CLS + register tokens
        self._skip_tokens = 1 + num_register_tokens
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze_last_n(self, n: int) -> None:
        """Unfreeze the last N transformer layers for fine-tuning."""
        total = len(self.model.encoder.layer)
        for i, layer in enumerate(self.model.encoder.layer):
            if i >= total - n:
                for param in layer.parameters():
                    param.requires_grad = True

    @property
    def num_scales(self) -> int:
        return len(self.extract_layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from intermediate layers.

        Args:
            x: (B, 3, H, W) input images, pre-normalized.
               H and W must be divisible by patch_size.

        Returns:
            List of tensors, each (B, hidden_size, H_p, W_p) where
            H_p = H // patch_size, W_p = W // patch_size.
        """
        B, _, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        outputs = self.model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, 1+reg+N, D)

        features = []
        for layer_idx in self.extract_layers:
            hs = hidden_states[layer_idx]
            # Remove CLS + register tokens
            patch_tokens = hs[:, self._skip_tokens :, :]  # (B, N_patches, D)
            # Reshape to spatial grid
            spatial = patch_tokens.transpose(1, 2).reshape(
                B, self.hidden_size, h_patches, w_patches
            )
            features.append(spatial)

        return features

    def train(self, mode: bool = True) -> "DINOv3Backbone":
        """Override to keep backbone in eval mode when frozen."""
        super().train(mode)
        # If all backbone params are frozen, keep it in eval mode
        # (affects dropout, batchnorm behavior)
        if not any(p.requires_grad for p in self.model.parameters()):
            self.model.eval()
        return self
