# HITL Segmentation Backend

Backend server for human-in-the-loop semantic segmentation. This repo acts as a server that offers some cool features:

- A project-based API that allows multiple projects with their own classes, datasets and checkpoints
- A SAM3 labelling API that accepts images and +/- coordinates to quickly label segmentations for curating a training dataset
- A DinoV3 segmentation training API that can train new checkpoints for a project based on datasets attached to a project
- A DinoV3 segmentation inference API that provides inferences
- A dashboard for kicking off training runs, viewing training outputs, and performing batch inference

Currently this has a nice [demo frontend client](https://github.com/anemes/qgis_sam3plugin) in the form of a QGIS plugin that can be used to rapidly build out geospatial segmentation models.

The DinoV3 backbone is currently [DINOv3-sat ViT-L backbone](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m) but easy to switch out.

**Stack**: DINOv3-sat ViT-L backbone + UperNet head, SAM3 interactive labeling, GeoPackage label store, Gradio training dashboard.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-capable GPU with 16+ GB VRAM (tested on RTX 3090 Ti 24 GB)
- Git (for submodules)

## Install

SAM3 is vendored as a git submodule at `vendor/sam3` and declared as a
[uv path source](https://docs.astral.sh/uv/concepts/dependencies/#path-dependencies)
in `pyproject.toml` (`[tool.uv.sources]`), so `uv sync` installs it automatically.

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd hitl-seg-backend

# 2. Install all dependencies (creates .venv, installs SAM3 from vendor/sam3)
uv sync

# 3. (GPU only) Replace CPU PyTorch with CUDA wheels
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you already cloned without `--recurse-submodules`, init the submodule first:

```bash
git submodule update --init --recursive
uv sync
```

## Configure

Edit `config/default.yaml`:

- `models.dinov3.path` — path to DINOv3-sat weights directory
- `models.sam3.checkpoint` — path to SAM3 checkpoint (`.pt`)
- `server.port` — API port (default 8000)
- `gpu.device` — `cuda` or `cpu`

Model weights are not included in the repo. Place them in `data/models/` (gitignored) or point the config to an external path.

## Run (local, no Docker)

```bash
python -m hitl.app
```

- REST API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`

## Run With Docker (local test)

### Prerequisites

- Docker Engine + Compose plugin (or Docker Desktop)
- NVIDIA Container Toolkit (for GPU profile)
- Model weights under `./data/models/`:
  - `./data/models/dinov3-vitl16-pretrain-sat493m/...`
  - `./data/models/sam3/sam3.pt`

### First-time setup

```bash
mkdir -p data/models
# Place model weights in data/models/ before building
```

### Build and run

```bash
docker compose --profile gpu up --build
```

All runtime data (projects, checkpoints, caches) is stored under `data/` and shared between local and Docker runs.

### Endpoints

- REST API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`

### Verify it works

```bash
# Health check
curl http://localhost:8000/health

# Open dashboard in browser
xdg-open http://localhost:8000/dashboard  # Linux
```

Then connect the QGIS plugin to `http://localhost:8000`, create a project, draw a region, and test SAM labeling.

## Remote GPU (SSH tunnel)

Run the backend on a GPU machine, access from a local QGIS client:

```bash
# Local machine
ssh -L 8000:localhost:8000 gpu-machine

# GPU machine
cd backend_samdino && docker compose --profile gpu up -d
```

Then point the QGIS plugin at `http://localhost:8000`.

## Azure Container Apps (GPU)

Repeatable deployment scripts for ACA with GPU workload profiles and Azure Files storage are in [`scripts/aca/`](scripts/aca/README.md).

Quick start:

1. Copy `scripts/aca/aca.env.sh.example` → `scripts/aca/aca.env.sh` and edit
2. `01_setup_once.sh` — one-time Azure Files + environment setup
3. `02_upload_data.sh` — upload model weights and data
4. `03_deploy.sh` — build image and deploy (or `--skip-build` for config-only changes)

See [`scripts/aca/README.md`](scripts/aca/README.md) for full first-deploy, redeployment, auth setup, and troubleshooting instructions.

## Project Structure

```
config/           Configuration (default.yaml)
hitl/
  api/            FastAPI routes (projects, labels, sam, training, inference)
  data/           GeoPackage label store
  models/         DINOv3 backbone + UperNet head
  services/       SAM service, training engine, inference engine
  app.py          Application entry point
vendor/sam3/      SAM3 submodule (facebook/sam3)
scripts/          Utility scripts (model download)
tests/            Backend tests
docs/             Architecture docs, setup guide, changelog
```

## Runtime Directories (gitignored)

All runtime data lives under `data/`. Subdirectories are created automatically on first run:

| Directory | Contents |
|-----------|----------|
| `data/models/` | Model weight files (download separately) |
| `data/projects/` | GeoPackage label stores and captured images |
| `data/checkpoints/` | Saved model weights during training |
| `data/dataset_cache/` | Preprocessed tile datasets |
| `data/tile_cache/` | Cached inference output tiles |

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC-4.0). See [LICENSE](LICENSE) for the full text.

You are free to use, modify, and share this software for non-commercial purposes with attribution. Commercial use requires separate permission.

**Dependency licenses**: SAM3 and DINOv3-sat are licensed under Apache 2.0 by Meta AI Research.
