"""Quick preview via frozen DINOv3 prototype matching.

Produces a rough segmentation map without any training by:
1. Extracting frozen DINOv3 patch features from an image
2. Using labeled pixel locations as prototypes (mean feature per class)
3. Classifying every patch by cosine similarity to prototypes

This is much lower quality than a trained UperNet but gives instant
feedback during the labeling phase — useful for checking if labels
are reasonable before committing to a training run.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PreviewService:
    """Frozen-feature prototype matching for quick segmentation preview.

    Usage:
        service = PreviewService(config, gpu_manager)
        result = service.predict(image_np, prototypes)
    """

    def __init__(self, config, gpu_manager):
        self._config = config
        self._gpu = gpu_manager

    def predict(
        self,
        image: np.ndarray,
        prototype_points: Dict[int, List[Tuple[float, float]]],
        class_names: Optional[Dict[int, str]] = None,
    ) -> dict:
        """Run prototype matching on an image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            prototype_points: {class_id: [(px_x, px_y), ...]} — pixel coords
                of labeled examples per class.
            class_names: Optional {class_id: name} for the response.

        Returns:
            Dict with:
                'class_map': (H_out, W_out) int array of predicted class IDs
                'confidence': (H_out, W_out) float array of max similarity
                'patch_size': int — spatial resolution factor
        """
        dinov3_cfg = self._config.models.dinov3

        # Get backbone (will load if needed — reuses segmentor's backbone)
        backbone = self._get_backbone()
        device = self._gpu.device

        H, W, _ = image.shape
        patch_size = dinov3_cfg.patch_size

        # Ensure dimensions are divisible by patch_size
        H_pad = (H // patch_size) * patch_size
        W_pad = (W // patch_size) * patch_size
        image_crop = image[:H_pad, :W_pad]

        # Normalize
        mean = np.array(dinov3_cfg.norm_mean, dtype=np.float32)
        std = np.array(dinov3_cfg.norm_std, dtype=np.float32)
        img_norm = (image_crop.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Extract features from last layer only (highest quality)
        with torch.no_grad():
            features = backbone(img_tensor)  # list of (1, D, H_p, W_p)
            feat = features[-1]  # last scale: (1, 1024, H_p, W_p)
            feat = F.normalize(feat, dim=1)  # L2 normalize along channel dim

        _, D, H_p, W_p = feat.shape

        # Build prototypes from labeled points
        prototypes = {}  # class_id -> (D,) normalized feature vector
        feat_flat = feat[0]  # (D, H_p, W_p)

        for class_id, points in prototype_points.items():
            class_features = []
            for px_x, px_y in points:
                # Map pixel coords to patch coords
                py = int(px_y) // patch_size
                px = int(px_x) // patch_size
                py = max(0, min(py, H_p - 1))
                px = max(0, min(px, W_p - 1))
                class_features.append(feat_flat[:, py, px])

            if class_features:
                proto = torch.stack(class_features).mean(dim=0)
                prototypes[class_id] = F.normalize(proto, dim=0)

        if not prototypes:
            logger.warning("No valid prototypes — returning empty prediction")
            return {
                "class_map": np.zeros((H_p, W_p), dtype=np.int32),
                "confidence": np.zeros((H_p, W_p), dtype=np.float32),
                "patch_size": patch_size,
            }

        # Compute cosine similarity to each prototype
        class_ids = sorted(prototypes.keys())
        proto_stack = torch.stack([prototypes[c] for c in class_ids])  # (K, D)

        # Reshape features to (H_p*W_p, D)
        feat_2d = feat_flat.permute(1, 2, 0).reshape(-1, D)  # (N, D)

        # Similarity: (N, K)
        sim = feat_2d @ proto_stack.T

        # Classify
        best_sim, best_idx = sim.max(dim=1)
        class_map = torch.tensor([class_ids[i] for i in best_idx.cpu()], dtype=torch.int32)
        class_map = class_map.reshape(H_p, W_p).numpy()
        confidence = best_sim.reshape(H_p, W_p).cpu().numpy()

        logger.info(
            "Preview: %dx%d patches, %d classes, mean confidence=%.3f",
            W_p, H_p, len(class_ids), float(confidence.mean()),
        )

        return {
            "class_map": class_map,
            "confidence": confidence,
            "patch_size": patch_size,
        }

    def _get_backbone(self):
        """Get the DINOv3 backbone, loading if necessary."""
        # If segmentor is loaded, reuse its backbone
        if self._gpu._segmentor is not None:
            return self._gpu._segmentor.backbone

        # Otherwise load backbone standalone
        from ..models.backbone import DINOv3Backbone

        dinov3_cfg = self._config.models.dinov3
        backbone = DINOv3Backbone(
            model_path=dinov3_cfg.path,
            extract_layers=tuple(dinov3_cfg.extract_layers),
            patch_size=dinov3_cfg.patch_size,
            num_register_tokens=dinov3_cfg.num_register_tokens,
            freeze=True,
        )
        backbone = backbone.to(self._gpu.device)
        backbone.eval()
        return backbone
