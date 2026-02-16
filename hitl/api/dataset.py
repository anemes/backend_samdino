"""Dataset building endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


class BuildRequest(BaseModel):
    raster_path: str  # path to the source GeoTIFF
    target_crs: str = ""  # empty = use source CRS


def get_deps():
    from ..app import app_state
    return app_state


@router.post("/build")
def build_dataset(req: BuildRequest, state=Depends(get_deps)):
    """Build a tiled dataset from current labels + raster source."""
    from ..data.raster_source import GeoTIFFSource
    from ..data.dataset_builder import DatasetBuilder
    from pathlib import Path

    raster_source = GeoTIFFSource(req.raster_path)
    target_crs = req.target_crs if req.target_crs else None

    builder = DatasetBuilder(
        label_store=state.label_store,
        tile_size=state.config.data.tile_size,
        tile_overlap=state.config.data.tile_overlap,
        ignore_index=state.config.data.ignore_index,
        background_class_id=state.config.data.background_class_id,
        min_labeled_fraction=state.config.data.min_labeled_fraction,
        val_fraction=state.config.data.val_fraction,
        test_fraction=state.config.data.test_fraction,
        split_block_size=state.config.data.split_block_size,
    )

    dataset_dir = Path(state.config.paths.dataset_cache_dir) / "latest"
    stats = builder.build(raster_source, dataset_dir, target_crs=target_crs)

    return {
        "status": "ok",
        "dataset_dir": str(dataset_dir),
        "stats": {
            "num_tiles_train": stats.num_tiles_train,
            "num_tiles_val": stats.num_tiles_val,
            "num_tiles_test": stats.num_tiles_test,
            "num_skipped": stats.num_skipped_low_label,
            "class_pixel_counts": stats.class_pixel_counts,
        },
    }


@router.get("/stats")
def dataset_stats(state=Depends(get_deps)):
    """Get label statistics."""
    return state.label_store.get_stats()
