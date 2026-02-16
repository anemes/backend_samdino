# HITL Segmentation System — Architecture & Model Justifications

## 1. Overview

A human-in-the-loop (HITL) semantic segmentation system for geospatial imagery. The user labels interactively in QGIS using SAM3, trains a real segmentation model (DINOv3-sat + UperNet) on the backend, runs tiled inference on large areas, corrects predictions, and retrains iteratively.

### Components
- **QGIS Plugin** (`qgis_plugin/hitl_sketcher/`): Frontend for labeling, class management, raster capture, inference triggering, and prediction viewing
- **Backend Service** (`backend/hitl/`): FastAPI REST API + training/inference engine + Gradio dashboard
- **SAM3**: Interactive mask generation for efficient labeling
- **DINOv3-sat + UperNet**: Trainable segmentation model

---

## 2. Model Choices & Justifications

### 2.1 Backbone: DINOv3 ViT-L pretrained on satellite imagery

**Model**: `dinov3-vitl16-pretrain-sat493m` (1024 hidden, 24 layers, patch size 16)

**Why this backbone:**
- Pretrained on 493 million satellite images via self-supervised learning (DINOv3 method)
- Produces dense spatial features optimized for remote sensing patterns (textures, structures, spatial relationships)
- ViT-L provides 1024-dimensional features with 24 transformer layers — high-capacity representation
- Uses RoPE (Rotary Position Embedding) — supports arbitrary input resolutions without interpolation artifacts
- Fits on 24GB GPU (3090 Ti) when frozen, with room for the trainable head

**Why not other backbones:**
- ResNet/ConvNeXt: Lower representation quality for RS tasks; no satellite pretraining at this scale
- ViT-B (smaller): Insufficient capacity for diverse custom classes
- ViT-H (larger): Won't fit on 24GB with a trainable head
- Prithvi-EO-2.0: Better for multiband/multitemporal data, but we start RGB-only. Architecture supports swapping to Prithvi later.

**Fine-tuning strategy:**
- Default: backbone frozen, only neck + head trained (decoder-only fine-tuning)
- With enough data (>1000 tiles): unfreeze last 2-4 backbone layers for partial fine-tuning
- This is the standard approach for ViT-based segmentation (DINOv2 paper, mmsegmentation)

### 2.2 Segmentation Head: UperNet

**Architecture**: Pyramid Pooling Module (PPM) + Feature Pyramid Network (FPN) + classifier

**Why UperNet:**
- **Standard pairing**: UperNet is the segmentation eval head used by the DINOv2/v3 papers themselves
- **Multi-scale fusion**: FPN handles objects at different scales (vehicles at 1m vs land cover at 100m)
- **Fast training**: Simple conv layers, no transformer decoder, no Hungarian matching
  - 10-20 min per HITL round on 3090 Ti (vs 30-60 min for Mask2Former)
- **HITL iteration speed is critical**: In HITL, you retrain every round. 3x faster iterations means 3x more improvement cycles in the same time.

**Why not Mask2Former:**
- Mask2Former uses a transformer decoder with learned queries + Hungarian matching for loss assignment
- ~2-3% mIoU improvement on benchmarks, but 3x slower training
- More complex implementation and debugging
- For HITL where you iterate frequently, training speed matters more than marginal quality
- If needed later, it's a head swap — same backbone, same dataset

**Why not SegFormer:**
- SegFormer uses its own encoder (MiT). Using DINOv3 features with SegFormer requires replacing its encoder, which defeats the purpose
- SegFormer-as-head-only is essentially a simpler version of UperNet

### 2.3 Multi-Scale Feature Extraction (ViT → UperNet Adapter)

DINOv3 ViT outputs single-resolution features from all layers. UperNet expects a multi-scale pyramid. The NeckAdapter bridges this:

1. Extract features from layers [6, 12, 18, 24] (every 6th layer)
2. Project each to different channel widths: [256, 512, 1024, 1024]
3. Upsample/downsample to simulate scale hierarchy: [4x, 2x, 1x, 0.5x]
4. Feed to UperNet's FPN

This is the standard approach (mmsegmentation's MLN, ViT-Adapter paper).

### 2.4 Interactive Labeling: SAM3

**Why SAM3:**
- Pixel-accurate masks from single click or bounding box — 10-50x faster than manual polygon tracing
- Handles complex boundaries (trees, irregular buildings, terrain edges)
- User corrects SAM3 output rather than drawing from scratch — lower effort, higher quality
- SAM3 supports text prompts, point prompts, box prompts, and mask prompts

**SAM3 is NOT the segmenter** — it's a labeling tool. The real segmenter is DINOv3+UperNet.

---

## 3. Labeling Strategy: Exhaustive Regions

### The Problem
With pure ignore_index masking (label only positives, ignore everything else), the model has no negative examples. It predicts the target class everywhere.

### The Solution: Exhaustive Regions
- User draws **annotation region** polygons defining exhaustive labeling zones
- **Inside region + class polygon** → class_id (2, 3, 4, ...)
- **Inside region + no polygon** → background (class_id=1)
- **Outside all regions** → ignore_index (0), excluded from loss

This provides true positives AND true negatives from each labeled region, while allowing partial coverage of the AOI.

### Training Loss
```
CrossEntropyLoss(ignore_index=0)
```
- Only pixels inside annotation regions contribute to gradients
- Background pixels inside regions provide negative signal
- Tiles with <5% non-ignore pixels are skipped (insufficient gradient signal)

### Spatial Train/Val/Test Split
- AOI divided into geographic blocks (configurable, default 500m)
- Blocks assigned to train (70%) / val (15%) / test (15%)
- Prevents spatial leakage (adjacent tiles can't leak between splits)
- Test set is pure holdout — never used for corrections

---

## 4. Data Pipeline

### Raster Sources
- **For labeling**: QGIS plugin exports visible extent as GeoTIFF via QgsRasterPipe
- **For inference**: Backend has RasterSource abstraction
  - Phase 1: GeoTIFF files
  - Phase 3: XYZ/TMS tile services (Google Satellite, aerial APIs), WMS
- **XYZ tile fetching**: Calculate covering TMS tiles at zoom level, fetch with disk cache + rate limiting, stitch into mosaic

### Dataset Building
1. Read annotation regions and polygons from GeoPackage
2. Fetch raster imagery for each region
3. Rasterize: class polygons → class IDs, region interior → background, exterior → ignore
4. Tile into overlapping patches (512x512, 64px overlap)
5. Filter tiles with <5% labeled pixels
6. Spatial block split → train/val/test directories

### Augmentations (training only)
- Random horizontal/vertical flip
- Random 90° rotation
- Color jitter (brightness, contrast, saturation, hue)
- Random scale (0.5x-2.0x) with crop
- DINOv3-sat normalization: mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143]

---

## 5. Inference Pipeline

### Tiled Inference with Overlap Blending
1. Tile the AOI into overlapping patches (512x512, 128px overlap)
2. Batch inference on GPU (batch_size=8)
3. **Cosine-window blending**: each pixel in overlap zone gets predictions from 2-4 tiles, weighted by a 2D raised-cosine window that's 1.0 at center and tapers to 0.0 at edges
4. Finalize: divide accumulated logits by weight map, softmax, argmax

### Outputs
- **Class raster** (GeoTIFF, uint8): Predicted class IDs per pixel
- **Confidence raster** (GeoTIFF, float32): Per-pixel entropy (lower = more confident)
- **Vector polygons** (GeoPackage): Vectorized class predictions with simplification

---

## 6. GPU Management

Single 3090 Ti with 24GB VRAM:
- SAM3: ~6GB (3.45GB weights + activations)
- DINOv3 + UperNet training: ~12-16GB (frozen backbone + trainable head + batch)
- **Cannot coexist** — mutual exclusion via GPUManager

The GPUManager:
- Tracks which model is loaded (SAM3 vs Segmentor vs None)
- Unloads one before loading the other
- Clears CUDA cache between swaps
- Thread-safe with mutex lock
- Switch latency: ~5-10 seconds (acceptable for HITL)

Mixed precision (torch.cuda.amp) used throughout to maximize batch size.

---

## 7. HITL Loop Flow

```
Round 0: User creates classes, draws annotation regions, labels objects with SAM3
         → Dataset: 50-200 tiles, 2-5% of AOI covered
         → Train: decoder-only, 50 epochs, ~15 min

Round 1: Model predicts full AOI → user loads predictions + confidence heatmap
         → User draws correction regions around worst predictions
         → Labels inside correction regions
         → Retrain (now more data, better coverage)

Round 2+: Iterate until val_mIoU plateaus or user is satisfied
          → Each round: annotation regions accumulate, model improves
          → Optional: active learning (entropy-ranked tile suggestions)
```

---

## 8. Future Extensions

- **Multiband support**: Swap DINOv3-sat for Prithvi-EO-2.0 backbone (same head)
- **GPU offload**: Backend already designed for remote deployment (change backend_url to SSH tunnel)
- **Mask2Former head**: Swap UperNet → Mask2Former (same backbone, same dataset) for quality mode
- **Active learning**: Rank tiles by prediction entropy, suggest highest-uncertainty regions for labeling
- **Multi-class instances**: Mask2Former natively supports instance segmentation if needed later
