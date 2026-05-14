"""REST API endpoints."""

from fastapi import APIRouter

from . import dataset, inference, labels, models, preview, projects, raster, sam, session, training

router = APIRouter(prefix="/api")
router.include_router(session.router, prefix="/session", tags=["session"])
router.include_router(projects.router, prefix="/projects", tags=["projects"])
router.include_router(labels.router, prefix="/labels", tags=["labels"])
router.include_router(dataset.router, prefix="/dataset", tags=["dataset"])
router.include_router(training.router, prefix="/training", tags=["training"])
router.include_router(inference.router, prefix="/inference", tags=["inference"])
router.include_router(models.router, prefix="/models", tags=["models"])
router.include_router(sam.router, prefix="/sam", tags=["sam"])
router.include_router(raster.router, prefix="/raster", tags=["raster"])
router.include_router(preview.router, prefix="/preview", tags=["preview"])
