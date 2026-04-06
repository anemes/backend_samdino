"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .data.label_store import LabelStore
from .data.project_manager import ProjectManager
from .models.registry import ModelRegistry
from .services.gpu_manager import GPUManager
from .services.inference_service import InferenceService
from .services.sam_service import SAMService
from .services.train_service import TrainService
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Shared application state across all endpoints."""

    config: object
    gpu_manager: GPUManager
    label_store: LabelStore
    registry: ModelRegistry
    train_service: TrainService
    inference_service: InferenceService
    sam_service: SAMService
    project_manager: Optional["ProjectManager"] = None
    active_project_id: Optional[str] = None

    def switch_project(self, project_id: str) -> None:
        """Switch to a different project, updating label_store and registry.

        Stops any running training before switching to prevent orphaned
        threads and GPU memory leaks.
        """
        pm = self.project_manager
        if pm is None:
            raise RuntimeError("ProjectManager not initialized")

        project_dir = pm.get_project_dir(project_id)
        if not project_dir.exists():
            raise ValueError(f"Project '{project_id}' does not exist")

        # Stop running training on the old project before switching
        if self.train_service.is_running:
            logger.warning(
                "Training is running on project '%s' — stopping before switch.",
                self.active_project_id,
            )
            self.train_service.stop_training()
            # Wait for the training thread to finish (up to 10s)
            if self.train_service._thread is not None:
                self.train_service._thread.join(timeout=10)
                if self.train_service._thread.is_alive():
                    logger.warning("Training thread did not stop within 10s")

        self.active_project_id = project_id
        self.label_store = LabelStore(project_dir / "labels.gpkg")
        self.registry = ModelRegistry(
            self.config.paths.checkpoint_dir, project_id=project_id
        )
        self.train_service = TrainService(
            self.config, self.gpu_manager, self.label_store, self.registry
        )
        logger.info("Switched to project '%s'", project_id)


# Global state accessible by endpoints
app_state: Optional[AppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    global app_state

    from config.schema import load_config

    setup_logging()
    config = load_config()

    # Ensure directories exist
    Path(config.paths.project_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.dataset_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.tile_cache_dir).mkdir(parents=True, exist_ok=True)

    # Initialize services
    gpu_manager = GPUManager(config, device=config.gpu.device)

    # Project manager
    project_manager = ProjectManager(config.paths.project_dir)

    # Ensure a "default" project exists
    if project_manager.get_project("default") is None:
        project_manager.create_project("default", "Default Project", "Auto-created default project")

    default_project_dir = project_manager.get_project_dir("default")
    label_store = LabelStore(default_project_dir / "labels.gpkg")
    registry = ModelRegistry(config.paths.checkpoint_dir, project_id="default")
    train_service = TrainService(config, gpu_manager, label_store, registry)
    inference_service = InferenceService(config, gpu_manager)
    sam_service = SAMService(config, gpu_manager)

    app_state = AppState(
        config=config,
        gpu_manager=gpu_manager,
        label_store=label_store,
        registry=registry,
        train_service=train_service,
        inference_service=inference_service,
        sam_service=sam_service,
        project_manager=project_manager,
        active_project_id="default",
    )

    logger.info("Backend initialized. GPU device: %s", config.gpu.device)

    # Preload SAM3 so first interaction is instant
    logger.info("Preloading SAM3 weights...")
    gpu_manager.acquire_sam3(config)

    # Launch Gradio dashboard on separate port
    dashboard_thread = None
    try:
        from .dashboard.app import create_dashboard
        dashboard = create_dashboard()
        if dashboard is not None:
            dashboard_port = config.server.dashboard_port
            launch_kwargs = {
                "server_name": "0.0.0.0",
                "server_port": dashboard_port,
                "share": False,
                "quiet": True,
            }
            if config.server.dashboard_password:
                launch_kwargs["auth"] = (
                    config.server.dashboard_user,
                    config.server.dashboard_password,
                )
            dashboard_thread = threading.Thread(
                target=dashboard.launch,
                kwargs=launch_kwargs,
                daemon=True,
            )
            dashboard_thread.start()
            logger.info("Dashboard launched at http://localhost:%d", dashboard_port)
        else:
            logger.warning("Dashboard creation returned None (gradio not installed?)")
    except Exception:
        logger.exception("Dashboard launch failed")

    yield

    # Cleanup
    logger.info("Shutting down...")
    app_state = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="HITL Segmentation Backend",
        description="DINOv3-sat + UperNet segmentation with SAM3 interactive labeling",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for QGIS plugin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key auth middleware
    from ..config.schema import get_config
    _config = get_config()
    _api_key = _config.server.api_key

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not _api_key:
            return await call_next(request)
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {_api_key}":
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing API key"}
            )
        return await call_next(request)

    # Mount API routes
    from .api import router as api_router
    app.include_router(api_router)

    # Health check
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "gpu_active": app_state.gpu_manager.active_model.value if app_state else "unknown",
            "gpu_vram_mb": app_state.gpu_manager.vram_usage_mb() if app_state else 0,
        }

    return app


# Entry point
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "hitl.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
