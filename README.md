# HITL Segmentation Backend

FastAPI backend for human-in-the-loop semantic segmentation of geospatial imagery.

**Stack**: DINOv3-sat ViT-L backbone + UperNet head, SAM3 interactive labeling, GeoPackage label store, Gradio training dashboard.

## Requirements

- Python 3.11+
- CUDA-capable GPU with 16+ GB VRAM (tested on RTX 3090 Ti 24 GB)
- Git (for submodules)

## Install

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd hitl-seg-backend

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM3 from vendored submodule
pip install -e vendor/sam3

# 5. Install backend
pip install -e .
```

## Configure

Edit `config/default.yaml`:

- `models.dinov3.path` — path to DINOv3-sat weights directory
- `models.sam3.checkpoint` — path to SAM3 checkpoint (`.pt`)
- `server.port` — API port (default 8000)
- `gpu.device` — `cuda` or `cpu`

Model weights are not included in the repo. Place them in a `models/` directory (gitignored) or point the config to an external path.

## Run

```bash
python -m hitl.app
```

- REST API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Training dashboard: `http://localhost:7860`

## Run With Docker

### Prerequisites

- Docker Desktop (Mac/Windows) or Docker Engine + Compose plugin (Linux)
- Model weights present on host under `./models`:
  - `./models/dinov3-vitl16-pretrain-sat493m/...`
  - `./models/sam3/sam3.pt`
- Docker image dependencies are installed with `uv sync --frozen` from `uv.lock`

### CPU profile (Mac/Linux/Windows)

```bash
docker compose --profile cpu up --build
```

`docker-compose.yml` defaults containers to `linux/amd64` via `DOCKER_PLATFORM` so SAM3 works on Apple Silicon Macs.
CPU profile installs PyTorch CPU wheels only and skips `nvidia-*` CUDA packages.

If you want to force platform explicitly:

```bash
DOCKER_PLATFORM=linux/amd64 docker compose --profile cpu up --build
```

### GPU profile (Linux/Windows with NVIDIA)

Requires NVIDIA Container Toolkit / WSL2 GPU integration.  
GPU profile installs CUDA 12.1 PyTorch wheels.

```bash
docker compose --profile gpu up --build
```

If needed, force platform explicitly:

```bash
DOCKER_PLATFORM=linux/amd64 docker compose --profile gpu up --build
```

### Endpoints

- REST API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Training dashboard: `http://localhost:7860`

### Persistent data

The compose setup mounts these host directories for persistence:

- `./models` -> `/app/models`
- `./projects` -> `/app/projects`
- `./checkpoints` -> `/app/checkpoints`
- `./dataset_cache` -> `/app/dataset_cache`
- `./tile_cache` -> `/app/tile_cache`

## Remote GPU (SSH tunnel)

Run the backend on a GPU machine, access from a local QGIS client:

```bash
# Local machine
ssh -L 8000:localhost:8000 gpu-machine

# GPU machine (containerized)
cd backend_samdino && docker compose --profile gpu up -d
```

Then point the QGIS plugin at `http://localhost:8000`.

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

These are created automatically on first run:

| Directory | Contents |
|-----------|----------|
| `checkpoints/` | Saved model weights during training |
| `dataset_cache/` | Preprocessed tile datasets |
| `tile_cache/` | Cached inference output tiles |
| `projects/` | GeoPackage label stores and captured images |
| `models/` | Model weight files (download separately) |

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC-4.0). See [LICENSE](LICENSE) for the full text.

You are free to use, modify, and share this software for non-commercial purposes with attribution. Commercial use requires separate permission.

**Dependency licenses**: SAM3 and DINOv3-sat are licensed under Apache 2.0 by Meta AI Research.
