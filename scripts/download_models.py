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
        "repo_id": "facebook/sam3",
        "local_subdir": "sam3",
        "description": "SAM3 checkpoint from HuggingFace",
    },
}

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _has_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _find_any_weight_file(root: Path) -> bool:
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    for pattern in patterns:
        for f in root.rglob(pattern):
            if _has_nonempty_file(f):
                return True
    return False


def verify_dinov3(local_dir: Path) -> bool:
    """Return True when the DINOv3 snapshot looks complete enough to use."""
    config_ok = _has_nonempty_file(local_dir / "config.json")
    weights_ok = _find_any_weight_file(local_dir)
    return config_ok and weights_ok


def verify_sam3(local_dir: Path) -> bool:
    """Return True when the SAM3 snapshot looks complete enough to use."""
    checkpoint_ok = _has_nonempty_file(local_dir / "sam3.pt")
    config_ok = _has_nonempty_file(local_dir / "config.json")
    return checkpoint_ok and config_ok


def download_dinov3(models_dir: Path, token: str | None = None) -> None:
    """Download DINOv3-sat from HuggingFace."""
    from huggingface_hub import snapshot_download

    info = MODELS["dinov3"]
    local_dir = models_dir / info["local_subdir"]

    if verify_dinov3(local_dir):
        print(f"  DINOv3 already verified at {local_dir}, skipping.")
        return
    if local_dir.exists():
        print(f"  Existing DINOv3 directory is incomplete at {local_dir}; re-downloading...")

    print(f"  Downloading {info['repo_id']}...")
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(local_dir),
        token=token,
        resume_download=True,
    )
    if not verify_dinov3(local_dir):
        raise RuntimeError(
            "DINOv3 download appears incomplete. "
            "Check Hugging Face authentication/permissions and retry."
        )
    print(f"  Done: {local_dir}")


def download_sam3(models_dir: Path, token: str | None = None) -> None:
    """Download SAM3 snapshot from HuggingFace."""
    from huggingface_hub import snapshot_download

    info = MODELS["sam3"]
    local_dir = models_dir / info["local_subdir"]

    if verify_sam3(local_dir):
        print(f"  SAM3 already verified at {local_dir}, skipping.")
        return
    if local_dir.exists():
        print(f"  Existing SAM3 directory is incomplete at {local_dir}; re-downloading...")

    print(f"  Downloading {info['repo_id']}...")
    snapshot_download(
        repo_id=info["repo_id"],
        local_dir=str(local_dir),
        token=token,
        resume_download=True,
    )
    if not verify_sam3(local_dir):
        raise RuntimeError(
            "SAM3 download appears incomplete. "
            "Check Hugging Face authentication/permissions and retry."
        )
    print(f"  Done: {local_dir}")


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

    try:
        if download_all or args.dinov3:
            print(f"[DINOv3] {MODELS['dinov3']['description']}")
            download_dinov3(models_dir, token=args.token)
            print()

        if download_all or args.sam3:
            print(f"[SAM3] {MODELS['sam3']['description']}")
            download_sam3(models_dir, token=args.token)
            print()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print("All requested models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
