#!/usr/bin/env python3
"""Download required model weights for the HITL segmentation backend.

Downloads:
  - DINOv3-sat ViT-L (493M satellite images pretrain) from HuggingFace
  - SAM3 checkpoint from Meta

Usage:
    python scripts/download_models.py              # download all
    python scripts/download_models.py --dinov3     # DINOv3 only
    python scripts/download_models.py --sam3       # SAM3 only
    python scripts/download_models.py --models-dir /path/to/models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MODELS = {
    "dinov3": {
        "repo_id": "facebook/dinov3-vitl16-pretrain-sat493m",
        "local_subdir": "dinov3-vitl16-pretrain-sat493m",
        "description": "DINOv3 ViT-L pretrained on 493M satellite images",
    },
    "sam3": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "local_subdir": "sam3",
        "filename": "sam3.pt",
        "description": "SAM3 (SAM2.1) Hiera Large checkpoint",
    },
}

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def download_dinov3(models_dir: Path, token: str | None = None) -> None:
    """Download DINOv3-sat from HuggingFace."""
    from huggingface_hub import snapshot_download

    info = MODELS["dinov3"]
    local_dir = models_dir / info["local_subdir"]

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  DINOv3 already exists at {local_dir}, skipping.")
        return

    print(f"  Downloading {info['repo_id']}...")
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(local_dir),
        token=token,
        resume_download=True,
    )
    print(f"  Done: {local_dir}")


def download_sam3(models_dir: Path) -> None:
    """Download SAM3 checkpoint."""
    import urllib.request

    info = MODELS["sam3"]
    local_dir = models_dir / info["local_subdir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    output_path = local_dir / info["filename"]

    if output_path.exists():
        print(f"  SAM3 already exists at {output_path}, skipping.")
        return

    print(f"  Downloading SAM3 from {info['url']}...")
    urllib.request.urlretrieve(info["url"], str(output_path))
    print(f"  Done: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights for HITL backend.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help=f"Target directory (default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--dinov3", action="store_true", help="Download DINOv3 only")
    parser.add_argument("--sam3", action="store_true", help="Download SAM3 only")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    # If no specific model requested, download all
    download_all = not args.dinov3 and not args.sam3

    print(f"Models directory: {models_dir}\n")

    if download_all or args.dinov3:
        print(f"[DINOv3] {MODELS['dinov3']['description']}")
        download_dinov3(models_dir, token=args.token)
        print()

    if download_all or args.sam3:
        print(f"[SAM3] {MODELS['sam3']['description']}")
        download_sam3(models_dir)
        print()

    print("All requested models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
