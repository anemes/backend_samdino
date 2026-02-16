"""End-to-end pipeline test: labels → dataset → train → infer.

Creates synthetic GeoTIFF imagery and labels, builds a tiled dataset,
trains the DINOv3+UperNet segmentor for a few epochs, runs inference,
and validates outputs at each stage.

Run with:
    cd backend
    python -m pytest tests/test_e2e_pipeline.py -v -s 2>&1 | head -200
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box, mapping

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_synthetic_geotiff(path: str, width: int = 1024, height: int = 1024):
    """Create a synthetic 3-band GeoTIFF with colored regions.

    Generates distinct colored patches that represent different "classes":
    - Top-left quadrant: green (class 2 - vegetation)
    - Top-right quadrant: gray (class 3 - buildings)
    - Bottom-left: blue (class 4 - water)
    - Bottom-right: brown (background)

    CRS: EPSG:3857, placed near (0, 0) for simplicity.
    """
    # Create image with distinct regions
    img = np.zeros((3, height, width), dtype=np.uint8)

    h2, w2 = height // 2, width // 2

    # Green (vegetation) - top-left
    img[0, :h2, :w2] = 30
    img[1, :h2, :w2] = 150
    img[2, :h2, :w2] = 30

    # Gray (buildings) - top-right
    img[0, :h2, w2:] = 140
    img[1, :h2, w2:] = 140
    img[2, :h2, w2:] = 140

    # Blue (water) - bottom-left
    img[0, h2:, :w2] = 20
    img[1, h2:, :w2] = 50
    img[2, h2:, :w2] = 180

    # Brown (background) - bottom-right
    img[0, h2:, w2:] = 160
    img[1, h2:, w2:] = 120
    img[2, h2:, w2:] = 60

    # Add some noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place at small EPSG:3857 coordinates (10m resolution)
    minx, miny = 1000.0, 1000.0
    res = 10.0  # 10 meters/pixel
    maxx = minx + width * res
    maxy = miny + height * res
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype=np.uint8, crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(img)

    return (minx, miny, maxx, maxy), res


def test_e2e_pipeline():
    """Full end-to-end: labels → dataset → train → infer."""
    tmpdir = tempfile.mkdtemp(prefix="hitl_e2e_")
    print(f"\n=== E2E Test Directory: {tmpdir} ===\n")

    try:
        _run_e2e(tmpdir)
    finally:
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_e2e(tmpdir: str):
    tmpdir = Path(tmpdir)

    # ----------------------------------------------------------------
    # Stage 1: Create synthetic GeoTIFF
    # ----------------------------------------------------------------
    print("--- Stage 1: Creating synthetic GeoTIFF ---")
    raster_path = str(tmpdir / "test_image.tif")
    bounds, res = make_synthetic_geotiff(raster_path, width=1024, height=1024)
    minx, miny, maxx, maxy = bounds
    print(f"  Image: 1024x1024, bounds={bounds}, res={res}m")

    # Verify it's readable
    from hitl.data.raster_source import GeoTIFFSource
    src = GeoTIFFSource(raster_path)
    assert src.get_crs() == "EPSG:3857"
    chip = src.get_chip(bounds, 1024, 1024)
    assert chip.data.shape == (3, 1024, 1024)
    print("  ✓ GeoTIFF readable via GeoTIFFSource")

    # ----------------------------------------------------------------
    # Stage 2: Set up label store with classes, regions, annotations
    # ----------------------------------------------------------------
    print("\n--- Stage 2: Setting up label store ---")
    from hitl.data.label_store import LabelStore, SegClass

    gpkg_path = tmpdir / "labels.gpkg"
    store = LabelStore(gpkg_path)

    # Define classes
    classes = [
        SegClass(class_id=2, name="vegetation", color="#00FF00"),
        SegClass(class_id=3, name="buildings", color="#888888"),
        SegClass(class_id=4, name="water", color="#0000FF"),
    ]
    store.set_classes(classes)
    print(f"  Classes: {[c.name for c in classes]}")

    # Create annotation regions (two large regions covering the image)
    # Region 1: left half
    region1_geom = mapping(box(minx, miny, minx + (maxx - minx) / 2, maxy))
    region1_id = store.add_region(region1_geom, crs="EPSG:3857")

    # Region 2: right half
    region2_geom = mapping(box(minx + (maxx - minx) / 2, miny, maxx, maxy))
    region2_id = store.add_region(region2_geom, crs="EPSG:3857")

    print(f"  Regions: {region1_id}, {region2_id}")

    # Add annotation polygons matching the colored patches
    h2_geo = (maxy - miny) / 2
    w2_geo = (maxx - minx) / 2

    # Vegetation (top-left of left half) → region 1
    veg_geom = mapping(box(minx + 10, miny + h2_geo + 10, minx + w2_geo - 10, maxy - 10))
    store.add_annotation(veg_geom, class_id=2, region_id=region1_id, crs="EPSG:3857")

    # Water (bottom-left of left half) → region 1
    water_geom = mapping(box(minx + 10, miny + 10, minx + w2_geo - 10, miny + h2_geo - 10))
    store.add_annotation(water_geom, class_id=4, region_id=region1_id, crs="EPSG:3857")

    # Buildings (top-right of right half) → region 2
    bldg_geom = mapping(box(minx + w2_geo + 10, miny + h2_geo + 10, maxx - 10, maxy - 10))
    store.add_annotation(bldg_geom, class_id=3, region_id=region2_id, crs="EPSG:3857")

    # Bottom-right stays unlabeled → background (class 1) within region 2
    stats = store.get_stats()
    print(f"  Annotations: {stats['num_annotations']}")
    print(f"  ✓ Label store configured")

    # ----------------------------------------------------------------
    # Stage 3: Build dataset
    # ----------------------------------------------------------------
    print("\n--- Stage 3: Building dataset ---")
    from hitl.data.dataset_builder import DatasetBuilder

    dataset_dir = tmpdir / "dataset"
    builder = DatasetBuilder(
        label_store=store,
        tile_size=512,
        tile_overlap=64,
        ignore_index=0,
        background_class_id=1,
        min_labeled_fraction=0.05,
        val_fraction=0.15,
        test_fraction=0.15,
        split_block_size=500.0,
    )

    ds_stats = builder.build(
        raster_source=src,
        output_dir=dataset_dir,
        target_crs="EPSG:3857",
    )

    print(f"  Regions processed: {ds_stats.num_regions}")
    print(f"  Annotations: {ds_stats.num_annotations}")
    print(f"  Tiles: {ds_stats.num_tiles_total} (train={ds_stats.num_tiles_train}, val={ds_stats.num_tiles_val}, test={ds_stats.num_tiles_test})")
    print(f"  Skipped (low label): {ds_stats.num_skipped_low_label}")
    print(f"  Class distribution: {ds_stats.class_pixel_counts}")

    assert ds_stats.num_tiles_total > 0, "No tiles produced!"
    assert ds_stats.num_tiles_train > 0, "No training tiles!"
    print("  ✓ Dataset built successfully")

    # ----------------------------------------------------------------
    # Stage 4: Verify TileDataset loads correctly
    # ----------------------------------------------------------------
    print("\n--- Stage 4: Verifying TileDataset ---")
    from hitl.data.tile_dataset import TileDataset

    train_ds = TileDataset(
        root=dataset_dir,
        split="train",
        tile_size=512,
        norm_mean=(0.430, 0.411, 0.296),
        norm_std=(0.213, 0.156, 0.143),
        augment=False,
        ignore_index=0,
    )

    print(f"  Train tiles loaded: {len(train_ds)}")
    assert len(train_ds) > 0, "No training tiles in dataset!"

    sample = train_ds[0]
    print(f"  Sample image shape: {sample['image'].shape}")
    print(f"  Sample mask shape: {sample['mask'].shape}")
    print(f"  Sample tile_id: {sample['tile_id']}")
    assert sample['image'].shape[0] == 3, "Expected 3 channels"
    assert sample['image'].shape[1] == 512 and sample['image'].shape[2] == 512
    assert sample['mask'].shape[0] == 512 and sample['mask'].shape[1] == 512

    # Check mask contains expected classes
    mask_classes = set(sample['mask'].unique().numpy().tolist())
    print(f"  Mask classes in sample: {mask_classes}")
    print("  ✓ TileDataset works correctly")

    # ----------------------------------------------------------------
    # Stage 5: Build and forward-pass the model
    # ----------------------------------------------------------------
    print("\n--- Stage 5: Model forward pass ---")
    import torch
    from config.schema import load_config

    config = load_config()
    num_classes = len(classes) + 2  # ignore(0) + background(1) + 3 user classes = 5

    from hitl._models_build import build_segmentor
    model = build_segmentor(config, num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = sample['image'].unsqueeze(0).to(device)
        logits = model(x)
        print(f"  Input: {x.shape}")
        print(f"  Logits: {logits.shape}")
        assert logits.shape == (1, num_classes, 512, 512)

    print("  ✓ Model forward pass OK")

    # ----------------------------------------------------------------
    # Stage 6: Quick training (2 epochs)
    # ----------------------------------------------------------------
    print("\n--- Stage 6: Training (2 epochs) ---")

    from hitl.services.gpu_manager import GPUManager
    from hitl.models.registry import ModelRegistry
    from hitl.services.train_service import TrainService

    gpu_mgr = GPUManager(device=device)
    registry = ModelRegistry(str(tmpdir / "checkpoints"), project_id="test")
    train_svc = TrainService(config, gpu_mgr, store, registry)

    # Override config for quick test
    overrides = {
        "epochs": 2,
        "batch_size": 2,
        "warmup_epochs": 0,
        "early_stopping_patience": 100,
    }

    run_id = train_svc.start_training(
        raster_source=src,
        project_id="test",
        config_overrides=overrides,
    )

    print(f"  Run ID: {run_id}")

    # Wait for training to complete
    import time
    max_wait = 300  # 5 min max
    start = time.time()
    while train_svc.is_running and (time.time() - start) < max_wait:
        state = train_svc.get_state()
        if state["status"] == "training":
            print(f"    Epoch {state['epoch']}/{state['total_epochs']}, loss={state.get('train_loss', '?')}")
        elif state["status"] == "building_dataset":
            print(f"    Building dataset...")
        time.sleep(2)

    final_state = train_svc.get_state()
    print(f"  Final status: {final_state['status']}")

    if final_state["status"] == "error":
        print(f"  ERROR: {final_state['error_message']}")
        raise RuntimeError(f"Training failed: {final_state['error_message']}")

    assert final_state["status"] == "complete", f"Expected complete, got {final_state['status']}"
    print(f"  Train loss: {final_state.get('train_loss', 'N/A')}")
    print(f"  Val mIoU: {final_state.get('val_mIoU', 'N/A')}")
    print("  ✓ Training completed")

    # ----------------------------------------------------------------
    # Stage 7: Check model registry
    # ----------------------------------------------------------------
    print("\n--- Stage 7: Model registry ---")
    checkpoints = registry.list_checkpoints()
    print(f"  Checkpoints saved: {len(checkpoints)}")
    assert len(checkpoints) > 0, "No checkpoints saved!"

    best = registry.get_best_checkpoint()
    if best:
        print(f"  Best checkpoint: val_mIoU={best.get('best_val_mIoU', 'N/A')}")

    metrics = registry.get_metrics(run_id)
    print(f"  Metrics logged: {len(metrics)} entries")
    print("  ✓ Registry works")

    # ----------------------------------------------------------------
    # Stage 8: Tiled inference
    # ----------------------------------------------------------------
    print("\n--- Stage 8: Tiled inference ---")
    from hitl.services.inference_service import InferenceService

    infer_svc = InferenceService(config, gpu_mgr)

    # Use the best checkpoint
    checkpoint_path = None
    if best:
        checkpoint_path = best.get("checkpoint_path")
        print(f"  Using checkpoint: {checkpoint_path}")

    class_names = ["ignore", "background", "vegetation", "buildings", "water"]

    infer_job = infer_svc.start_inference(
        raster_source=src,
        aoi_bounds=bounds,
        num_classes=num_classes,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        project_id="test",
    )

    print(f"  Job ID: {infer_job}")

    # Wait for inference
    start = time.time()
    while infer_svc.is_running and (time.time() - start) < 120:
        state = infer_svc.get_state()
        pct = state.get("progress_pct", 0)
        tiles = state.get("tiles_processed", 0)
        total = state.get("tiles_total", 0)
        print(f"    Progress: {pct:.1f}% ({tiles}/{total} tiles)")
        time.sleep(2)

    infer_state = infer_svc.get_state()
    print(f"  Final status: {infer_state['status']}")

    if infer_state["status"] == "error":
        print(f"  ERROR: {infer_state['error_message']}")
        raise RuntimeError(f"Inference failed: {infer_state['error_message']}")

    assert infer_state["status"] == "complete", f"Expected complete, got {infer_state['status']}"

    # Check output files
    result_paths = infer_state.get("result_paths", {})
    print(f"  Result files: {list(result_paths.keys())}")

    if "class_raster" in result_paths:
        with rasterio.open(result_paths["class_raster"]) as ds:
            pred = ds.read(1)
            print(f"  Class raster: {pred.shape}, classes={np.unique(pred)}")

    if "confidence_raster" in result_paths:
        with rasterio.open(result_paths["confidence_raster"]) as ds:
            conf = ds.read(1)
            print(f"  Confidence raster: {conf.shape}, range=[{conf.min():.3f}, {conf.max():.3f}]")

    print("  ✓ Inference completed")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("E2E PIPELINE TEST PASSED")
    print("=" * 60)
    print(f"  Synthetic image: 1024x1024, 3 bands")
    print(f"  Labels: {stats['num_annotations']} annotations in {ds_stats.num_regions} regions")
    print(f"  Dataset: {ds_stats.num_tiles_total} tiles")
    print(f"  Training: 2 epochs, final loss={final_state.get('train_loss', 'N/A')}")
    print(f"  Inference: {infer_state.get('tiles_total', 0)} tiles processed")
    print(f"  Output files: {len(result_paths)}")
    print("=" * 60)


if __name__ == "__main__":
    test_e2e_pipeline()
