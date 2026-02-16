"""Gradio dashboard for training configuration, metrics, and model management.

Tabs:
1. Status     - GPU, backend health, active project
2. Training   - Configure & launch training, live loss/mIoU curves, per-class IoU
3. Models     - Browse checkpoints, compare runs, class schema compatibility
4. Dataset    - Build dataset from raster source, view class distribution
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_state():
    """Get the global AppState (lazy import to avoid circular deps)."""
    from ..app import app_state
    return app_state


def create_dashboard():
    """Create the full Gradio dashboard application."""
    try:
        import gradio as gr
    except ImportError:
        logger.warning("Gradio not installed. Dashboard disabled. pip install gradio")
        return None

    # ========================== HELPER FUNCTIONS ==========================

    def refresh_status():
        state = _get_state()
        if not state:
            return "Backend not initialized", "Unknown", "No project"

        gpu = state.gpu_manager
        gpu_info = (
            f"Device: {gpu.device}\n"
            f"Active model: {gpu.active_model.value}\n"
            f"VRAM usage: {gpu.vram_usage_mb():.0f} MB"
        )

        proj = state.active_project_id or "none"
        proj_info = f"Active project: {proj}"

        if state.project_manager:
            projects = state.project_manager.list_projects()
            proj_info += f"\nTotal projects: {len(projects)}"

        train_status = "idle"
        if state.train_service.is_running:
            ts = state.train_service.get_state()
            train_status = (
                f"{ts['status']} | Epoch {ts['epoch']}/{ts['total_epochs']} | "
                f"val_mIoU: {ts['val_mIoU']:.4f}"
            )

        return train_status, gpu_info, proj_info

    def get_project_list():
        state = _get_state()
        if not state or not state.project_manager:
            return []
        return [p.project_id for p in state.project_manager.list_projects()]

    def get_class_info(project_id):
        state = _get_state()
        if not state:
            return "No backend"

        pm = state.project_manager
        if not pm:
            return "No project manager"

        project_dir = pm.get_project_dir(project_id)
        gpkg_path = project_dir / "labels.gpkg"
        if not gpkg_path.exists():
            return "No labels yet"

        from ..data.label_store import LabelStore
        store = LabelStore(gpkg_path)
        classes = store.get_classes()
        if not classes:
            return "No classes defined"

        lines = [f"{'ID':>4}  {'Name':<20}  {'Color':<8}"]
        lines.append("-" * 36)
        lines.append(f"   0  {'ignore':<20}  {'---':<8}")
        lines.append(f"   1  {'background':<20}  {'#888888':<8}")
        for c in classes:
            lines.append(f"{c.class_id:>4}  {c.name:<20}  {c.color:<8}")
        return "\n".join(lines)

    def get_label_stats(project_id):
        state = _get_state()
        if not state or not state.project_manager:
            return "No backend"

        project_dir = state.project_manager.get_project_dir(project_id)
        gpkg_path = project_dir / "labels.gpkg"
        if not gpkg_path.exists():
            return "No labels yet"

        from ..data.label_store import LabelStore
        store = LabelStore(gpkg_path)
        stats = store.get_stats()
        return (
            f"Regions: {stats.get('num_regions', 0)}\n"
            f"Annotations: {stats.get('num_annotations', 0)}\n"
            f"Classes: {stats.get('num_classes', 0)}"
        )

    # --- Training helpers ---

    def start_training(
        project_id, raster_path, xyz_url, xyz_zoom,
        epochs, batch_size, lr, weight_decay,
        warmup_epochs, patience, freeze_backbone, mixed_precision,
    ):
        state = _get_state()
        if not state:
            return "Backend not initialized"

        if state.train_service.is_running:
            return "Training already in progress"

        # Switch to target project first
        if project_id and project_id != state.active_project_id:
            state.switch_project(project_id)

        # Build raster source
        raster_path = raster_path.strip() if raster_path else ""
        xyz_url = xyz_url.strip() if xyz_url else ""

        if raster_path:
            from ..data.raster_source import GeoTIFFSource
            raster_source = GeoTIFFSource(raster_path)
        elif xyz_url:
            from ..data.raster_source import XYZTileSource
            raster_source = XYZTileSource(
                url_template=xyz_url,
                zoom=int(xyz_zoom),
                cache_dir=state.config.paths.tile_cache_dir,
            )
        else:
            return "Error: provide a raster path or XYZ URL"

        # Build overrides
        overrides = {}
        if epochs: overrides["epochs"] = int(epochs)
        if batch_size: overrides["batch_size"] = int(batch_size)
        if lr: overrides["learning_rate"] = float(lr)
        if weight_decay: overrides["weight_decay"] = float(weight_decay)
        if warmup_epochs: overrides["warmup_epochs"] = int(warmup_epochs)
        if patience: overrides["early_stopping_patience"] = int(patience)
        if freeze_backbone is not None:
            overrides["freeze_backbone"] = bool(freeze_backbone)
        if mixed_precision is not None:
            overrides["mixed_precision"] = bool(mixed_precision)

        try:
            run_id = state.train_service.start_training(
                raster_source=raster_source,
                project_id=project_id or state.active_project_id or "default",
                config_overrides=overrides if overrides else None,
            )
            return f"Training started! Run ID: {run_id}"
        except Exception as e:
            return f"Error: {e}"

    def stop_training():
        state = _get_state()
        if state and state.train_service.is_running:
            state.train_service.stop_training()
            return "Stop requested"
        return "No training in progress"

    def poll_training():
        """Returns (status_text, per_class_text, dataset_info)."""
        state = _get_state()
        if not state:
            return "No backend", "", ""

        ts = state.train_service.get_state()
        status = (
            f"Status: {ts['status']}\n"
            f"Run: {ts['run_id']}\n"
            f"Epoch: {ts['epoch']}/{ts['total_epochs']}\n"
            f"Progress: {ts['progress_pct']:.1f}%\n"
            f"Train Loss: {ts['train_loss']:.4f}\n"
            f"Val Loss: {ts['val_loss']:.4f}\n"
            f"Val mIoU: {ts['val_mIoU']:.4f}\n"
            f"Best mIoU: {ts['best_val_mIoU']:.4f}"
        )
        if ts['error_message']:
            status += f"\nError: {ts['error_message']}"

        # Per-class IoU from latest epoch
        per_class = ts.get("per_class_iou", {})
        per_class_text = ""
        if per_class:
            lines = [f"{'Class':<20} {'IoU':>8}"]
            lines.append("-" * 30)
            for name, iou in sorted(per_class.items(), key=lambda x: -x[1]):
                bar = "#" * int(iou * 30)
                lines.append(f"{name:<20} {iou:>7.4f}  {bar}")
            per_class_text = "\n".join(lines)

        # Dataset info
        ds = ts.get("dataset_stats") or {}
        ds_text = ""
        if ds:
            ds_text = (
                f"Train tiles: {ds.get('num_tiles_train', 0)}\n"
                f"Val tiles: {ds.get('num_tiles_val', 0)}\n"
                f"Test tiles: {ds.get('num_tiles_test', 0)}\n"
            )
            pixel_counts = ds.get("class_pixel_counts", {})
            if pixel_counts:
                total = sum(pixel_counts.values()) or 1
                ds_text += "\nClass distribution:\n"
                for cls_id, count in sorted(pixel_counts.items(), key=lambda x: str(x[0])):
                    pct = count / total * 100
                    ds_text += f"  Class {cls_id}: {count:,} px ({pct:.1f}%)\n"

        return status, per_class_text, ds_text

    # --- Model helpers ---

    def list_all_models():
        """List checkpoints across all projects, show class schema info."""
        state = _get_state()
        if not state or not state.project_manager:
            return "No backend"

        from ..models.registry import ModelRegistry

        pm = state.project_manager
        ckpt_dir = Path(state.config.paths.checkpoint_dir)

        lines = []
        for proj in pm.list_projects():
            proj_ckpt_dir = ckpt_dir / proj.project_id
            registry_path = proj_ckpt_dir / "registry.json"
            if not registry_path.exists():
                continue

            reg = ModelRegistry(ckpt_dir, proj.project_id)
            checkpoints = reg.list_checkpoints()
            best = [c for c in checkpoints if "best" in c.get("checkpoint_path", "")]

            if not best:
                continue

            # Group by run_id, keep best per run
            runs = {}
            for c in best:
                rid = c.get("run_id", "?")
                if rid not in runs or c.get("best_val_mIoU", 0) > runs[rid].get("best_val_mIoU", 0):
                    runs[rid] = c

            lines.append(f"=== Project: {proj.name} ({proj.project_id}) ===")
            for rid, c in runs.items():
                classes = c.get("class_names", [])
                classes_str = ", ".join(classes[2:]) if len(classes) > 2 else "none"
                lines.append(
                    f"  Run: {rid}\n"
                    f"    mIoU: {c.get('best_val_mIoU', 0):.4f}\n"
                    f"    Classes ({c.get('num_classes', 0)}): {classes_str}\n"
                    f"    Path: {c.get('checkpoint_path', '?')}\n"
                    f"    Time: {c.get('timestamp', '?')}"
                )
            lines.append("")

        return "\n".join(lines) if lines else "No trained models yet"

    def get_run_metrics(run_id):
        """Get detailed metrics for a specific run."""
        state = _get_state()
        if not state:
            return "No backend", None, None

        # Search across all project registries
        from ..models.registry import ModelRegistry

        pm = state.project_manager
        ckpt_dir = Path(state.config.paths.checkpoint_dir)
        metrics = []

        if pm:
            for proj in pm.list_projects():
                reg = ModelRegistry(ckpt_dir, proj.project_id)
                m = reg.get_metrics(run_id=run_id)
                if m:
                    metrics = m
                    break

        if not metrics:
            # Try current project's registry
            metrics = state.registry.get_metrics(run_id=run_id)

        if not metrics:
            return f"No metrics for run '{run_id}'", None, None

        # Summary
        best_epoch = max(metrics, key=lambda m: m.get("val_mIoU", 0))
        summary = (
            f"Run: {run_id}\n"
            f"Epochs trained: {len(metrics)}\n"
            f"Best epoch: {best_epoch.get('epoch', '?')}\n"
            f"Best val mIoU: {best_epoch.get('val_mIoU', 0):.4f}\n"
            f"Final train loss: {metrics[-1].get('train_loss', 0):.4f}\n"
            f"Final val loss: {metrics[-1].get('val_loss', 0):.4f}\n"
            f"Train tiles: {best_epoch.get('num_train_tiles', 0)}\n"
            f"Val tiles: {best_epoch.get('num_val_tiles', 0)}\n"
            f"Num classes: {best_epoch.get('num_classes', 0)}"
        )

        # Per-class IoU at best epoch
        per_class = best_epoch.get("per_class_iou", {})
        if per_class:
            summary += "\n\nPer-class IoU (best epoch):\n"
            for name, iou in sorted(per_class.items(), key=lambda x: -float(x[1])):
                bar = "#" * int(float(iou) * 30)
                summary += f"  {name:<20} {float(iou):>7.4f}  {bar}\n"

        # Build plot data
        loss_data = None
        miou_data = None
        try:
            import pandas as pd

            df = pd.DataFrame(metrics)
            if "epoch" in df.columns:
                loss_data = df[["epoch", "train_loss", "val_loss"]].melt(
                    id_vars="epoch",
                    value_vars=["train_loss", "val_loss"],
                    var_name="type", value_name="loss"
                )
                miou_data = df[["epoch", "val_mIoU"]].copy()
        except ImportError:
            pass

        return summary, loss_data, miou_data

    def check_model_compatibility(run_id):
        """Check which projects a model's class schema is compatible with."""
        state = _get_state()
        if not state or not state.project_manager:
            return "No backend"

        from ..models.registry import ModelRegistry

        # Find the model's class schema from any project registry
        pm = state.project_manager
        ckpt_dir = Path(state.config.paths.checkpoint_dir)
        model_classes = None

        for proj in pm.list_projects():
            reg = ModelRegistry(ckpt_dir, proj.project_id)
            checkpoints = reg.list_checkpoints()
            for c in checkpoints:
                if c.get("run_id") == run_id:
                    model_classes = c.get("class_names", [])
                    break
            if model_classes:
                break

        if not model_classes:
            return f"Run '{run_id}' not found in any project registry"

        # Check each project
        lines = [
            f"Model class schema: {model_classes}",
            f"Num classes: {len(model_classes)}",
            "",
            "Compatibility:",
        ]

        from ..data.label_store import LabelStore
        for proj in pm.list_projects():
            project_dir = pm.get_project_dir(proj.project_id)
            gpkg_path = project_dir / "labels.gpkg"
            if not gpkg_path.exists():
                lines.append(f"  {proj.name}: no labels")
                continue

            store = LabelStore(gpkg_path)
            user_classes = store.get_classes()
            proj_class_names = ["ignore", "background"] + [c.name for c in user_classes]

            if proj_class_names == model_classes:
                lines.append(f"  {proj.name}: COMPATIBLE (exact match)")
            elif len(proj_class_names) == len(model_classes):
                lines.append(f"  {proj.name}: PARTIAL (same count, different names)")
                for i, (mc, pc) in enumerate(zip(model_classes, proj_class_names)):
                    if mc != pc:
                        lines.append(f"    class {i}: model='{mc}' vs project='{pc}'")
            else:
                lines.append(
                    f"  {proj.name}: INCOMPATIBLE "
                    f"(model has {len(model_classes)} classes, project has {len(proj_class_names)})"
                )

        return "\n".join(lines)

    # --- Dataset helpers ---

    def build_dataset(project_id, raster_path, xyz_url, xyz_zoom):
        state = _get_state()
        if not state:
            return "No backend"

        if project_id and project_id != state.active_project_id:
            state.switch_project(project_id)

        raster_path = raster_path.strip() if raster_path else ""
        xyz_url = xyz_url.strip() if xyz_url else ""

        if raster_path:
            from ..data.raster_source import GeoTIFFSource
            raster_source = GeoTIFFSource(raster_path)
        elif xyz_url:
            from ..data.raster_source import XYZTileSource
            raster_source = XYZTileSource(
                url_template=xyz_url,
                zoom=int(xyz_zoom),
                cache_dir=state.config.paths.tile_cache_dir,
            )
        else:
            return "Provide a raster path or XYZ URL"

        try:
            from ..data.dataset_builder import DatasetBuilder
            data_cfg = state.config.data

            dataset_dir = Path(state.config.paths.dataset_cache_dir) / f"preview_{project_id}"
            builder = DatasetBuilder(
                label_store=state.label_store,
                tile_size=data_cfg.tile_size,
                tile_overlap=data_cfg.tile_overlap,
                ignore_index=data_cfg.ignore_index,
                background_class_id=data_cfg.background_class_id,
                min_labeled_fraction=data_cfg.min_labeled_fraction,
                val_fraction=data_cfg.val_fraction,
                test_fraction=data_cfg.test_fraction,
                split_block_size=data_cfg.split_block_size,
            )

            stats = builder.build(raster_source, dataset_dir)
            total = sum(stats.class_pixel_counts.values()) or 1

            result = (
                f"Dataset built at: {dataset_dir}\n\n"
                f"Train tiles: {stats.num_tiles_train}\n"
                f"Val tiles: {stats.num_tiles_val}\n"
                f"Test tiles: {stats.num_tiles_test}\n\n"
                f"Class distribution:\n"
            )
            for cls_id, count in sorted(stats.class_pixel_counts.items()):
                pct = count / total * 100
                bar = "#" * int(pct)
                result += f"  Class {cls_id}: {count:>10,} px ({pct:>5.1f}%) {bar}\n"

            return result
        except Exception as e:
            return f"Error: {e}"

    # ========================== BUILD DASHBOARD ==========================

    with gr.Blocks(title="HITL Segmentation Dashboard") as demo:
        gr.Markdown("# HITL Segmentation Dashboard")
        gr.Markdown("DINOv3-sat + UperNet segmentation training and model management")

        # ==================== STATUS TAB ====================
        with gr.Tab("Status"):
            with gr.Row():
                with gr.Column():
                    train_status_box = gr.Textbox(
                        label="Training Status", value="idle", lines=2, interactive=False
                    )
                with gr.Column():
                    gpu_box = gr.Textbox(
                        label="GPU Status", value="--", lines=3, interactive=False
                    )
                with gr.Column():
                    proj_box = gr.Textbox(
                        label="Project Info", value="--", lines=3, interactive=False
                    )

            status_refresh = gr.Button("Refresh Status", variant="secondary")
            status_refresh.click(
                refresh_status,
                outputs=[train_status_box, gpu_box, proj_box],
            )

        # ==================== TRAINING TAB ====================
        with gr.Tab("Training"):
            with gr.Row():
                # Left column: configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")

                    train_project = gr.Dropdown(
                        label="Project",
                        choices=get_project_list(),
                        value=None,
                        allow_custom_value=True,
                    )
                    refresh_proj_btn = gr.Button("Refresh Projects", size="sm")
                    refresh_proj_btn.click(
                        lambda: gr.update(choices=get_project_list()),
                        outputs=[train_project],
                    )

                    gr.Markdown("#### Raster Source")
                    train_raster = gr.Textbox(
                        label="GeoTIFF Path",
                        placeholder="/path/to/imagery.tif",
                    )
                    train_xyz = gr.Textbox(
                        label="XYZ Tile URL (alternative)",
                        placeholder="https://tile.server/{z}/{x}/{y}.png",
                    )
                    train_zoom = gr.Slider(
                        label="XYZ Zoom Level", minimum=10, maximum=20,
                        value=18, step=1,
                    )

                    gr.Markdown("#### Hyperparameters")
                    with gr.Row():
                        train_epochs = gr.Number(label="Epochs", value=50)
                        train_bs = gr.Number(label="Batch Size", value=4)
                    with gr.Row():
                        train_lr = gr.Number(label="Learning Rate", value=1e-4)
                        train_wd = gr.Number(label="Weight Decay", value=0.01)
                    with gr.Row():
                        train_warmup = gr.Number(label="Warmup Epochs", value=5)
                        train_patience = gr.Number(label="Early Stop Patience", value=10)
                    with gr.Row():
                        train_freeze = gr.Checkbox(label="Freeze Backbone", value=True)
                        train_amp = gr.Checkbox(label="Mixed Precision", value=True)

                    with gr.Row():
                        start_btn = gr.Button("Start Training", variant="primary")
                        stop_btn = gr.Button("Stop Training", variant="stop")

                    launch_msg = gr.Textbox(label="", interactive=False, lines=1)

                    start_btn.click(
                        start_training,
                        inputs=[
                            train_project, train_raster, train_xyz, train_zoom,
                            train_epochs, train_bs, train_lr, train_wd,
                            train_warmup, train_patience, train_freeze, train_amp,
                        ],
                        outputs=[launch_msg],
                    )
                    stop_btn.click(stop_training, outputs=[launch_msg])

                # Right column: live metrics
                with gr.Column(scale=1):
                    gr.Markdown("### Live Metrics")

                    poll_btn = gr.Button("Refresh Metrics", variant="secondary")

                    live_status = gr.Textbox(
                        label="Training Progress", lines=10, interactive=False,
                        elem_classes=["mono"],
                    )

                    per_class_box = gr.Textbox(
                        label="Per-Class IoU (current epoch)", lines=8,
                        interactive=False, elem_classes=["mono"],
                    )

                    dataset_box = gr.Textbox(
                        label="Dataset Stats", lines=8, interactive=False,
                        elem_classes=["mono"],
                    )

                    poll_btn.click(
                        poll_training,
                        outputs=[live_status, per_class_box, dataset_box],
                    )

                    # Auto-refresh every 5 seconds during training
                    timer = gr.Timer(value=5)
                    timer.tick(
                        poll_training,
                        outputs=[live_status, per_class_box, dataset_box],
                    )

        # ==================== MODELS TAB ====================
        with gr.Tab("Models"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### All Trained Models")
                    models_box = gr.Textbox(
                        label="Checkpoints by Project", lines=20,
                        interactive=False, elem_classes=["mono"],
                    )
                    models_refresh = gr.Button("Refresh Models")
                    models_refresh.click(list_all_models, outputs=[models_box])

                with gr.Column():
                    gr.Markdown("### Run Details")
                    run_id_input = gr.Textbox(
                        label="Run ID",
                        placeholder="project_abc12345",
                    )
                    run_detail_btn = gr.Button("Load Run Metrics")

                    run_summary = gr.Textbox(
                        label="Summary", lines=15, interactive=False,
                        elem_classes=["mono"],
                    )
                    run_loss_plot = gr.LinePlot(
                        x="epoch", y="loss", color="type",
                        title="Loss Curves",
                        height=250,
                    )
                    run_miou_plot = gr.LinePlot(
                        x="epoch", y="val_mIoU",
                        title="Validation mIoU",
                        height=250,
                    )

                    run_detail_btn.click(
                        get_run_metrics,
                        inputs=[run_id_input],
                        outputs=[run_summary, run_loss_plot, run_miou_plot],
                    )

            gr.Markdown("### Model Compatibility Check")
            with gr.Row():
                compat_run_id = gr.Textbox(
                    label="Run ID to check",
                    placeholder="Enter run_id",
                )
                compat_btn = gr.Button("Check Compatibility")

            compat_result = gr.Textbox(
                label="Compatibility Report", lines=10, interactive=False,
                elem_classes=["mono"],
            )
            compat_btn.click(
                check_model_compatibility,
                inputs=[compat_run_id],
                outputs=[compat_result],
            )

        # ==================== DATASET TAB ====================
        with gr.Tab("Dataset"):
            gr.Markdown("### Build Training Dataset")
            gr.Markdown(
                "Build a tiled dataset from a raster source + project labels. "
                "This rasterizes vector annotations into segmentation masks, "
                "tiles the imagery, and splits into train/val/test."
            )

            with gr.Row():
                with gr.Column():
                    ds_project = gr.Dropdown(
                        label="Project",
                        choices=get_project_list(),
                        allow_custom_value=True,
                    )
                    ds_refresh_proj = gr.Button("Refresh Projects", size="sm")
                    ds_refresh_proj.click(
                        lambda: gr.update(choices=get_project_list()),
                        outputs=[ds_project],
                    )

                    ds_raster = gr.Textbox(
                        label="GeoTIFF Path",
                        placeholder="/path/to/imagery.tif",
                    )
                    ds_xyz = gr.Textbox(
                        label="XYZ Tile URL (alternative)",
                        placeholder="https://tile.server/{z}/{x}/{y}.png",
                    )
                    ds_zoom = gr.Slider(
                        label="XYZ Zoom Level", minimum=10, maximum=20,
                        value=18, step=1,
                    )

                    ds_build_btn = gr.Button("Build Dataset", variant="primary")

                with gr.Column():
                    ds_result = gr.Textbox(
                        label="Build Result", lines=15, interactive=False,
                        elem_classes=["mono"],
                    )

            ds_build_btn.click(
                build_dataset,
                inputs=[ds_project, ds_raster, ds_xyz, ds_zoom],
                outputs=[ds_result],
            )

            gr.Markdown("### Project Label Stats")
            with gr.Row():
                stats_project = gr.Dropdown(
                    label="Project",
                    choices=get_project_list(),
                    allow_custom_value=True,
                )
                stats_refresh = gr.Button("Refresh", size="sm")
                stats_refresh.click(
                    lambda: gr.update(choices=get_project_list()),
                    outputs=[stats_project],
                )

            with gr.Row():
                with gr.Column():
                    class_info_box = gr.Textbox(
                        label="Class Definitions", lines=10, interactive=False,
                        elem_classes=["mono"],
                    )
                with gr.Column():
                    label_stats_box = gr.Textbox(
                        label="Label Statistics", lines=5, interactive=False,
                    )

            stats_load_btn = gr.Button("Load Stats")
            stats_load_btn.click(
                get_class_info, inputs=[stats_project], outputs=[class_info_box]
            )
            stats_load_btn.click(
                get_label_stats, inputs=[stats_project], outputs=[label_stats_box]
            )

    return demo
