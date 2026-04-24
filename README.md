# EasySegment Backend

FastAPI inference and training server for the [EasySegment QGIS plugin](https://github.com/anemes/qgis_sam3plugin). Provides SAM3-powered interactive segmentation, DINOv3-sat + UperNet model training, and tiled batch inference — all accessible over a REST API from the QGIS plugin.

- **SAM3** — interactive mask generation via point/box prompts on captured viewport images
- **DINOv3-sat backbone** — ViT-L pretrained on 493M satellite images (frozen); UperNet head trained per project
- **Tiled inference** — predict over any AOI with overlap blending and confidence heatmap output
- **GeoPackage label store** — per-project annotation storage with class definitions and region management
- **Gradio dashboard** — monitor training runs and trigger batch inference from the browser

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CUDA-capable GPU with 16+ GB VRAM (tested on RTX 3090 Ti 24 GB); CPU mode available but slow
- Git (for submodules)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Docker GPU only)

---

## Model Weights

Weights are not included in the repo. Download from HuggingFace using the provided script:

```bash
# Download both models
python scripts/download_models.py --dinov3 --sam3

# With a HuggingFace token (required if models are gated)
HF_TOKEN=hf_... python scripts/download_models.py --dinov3 --sam3
# or
python scripts/download_models.py --dinov3 --sam3 --token hf_...

# Custom output directory (default: data/models)
python scripts/download_models.py --dinov3 --sam3 --models-dir /path/to/models
```

Downloaded weights land in `data/models/`:
```
data/models/
├── dinov3-vitl16-pretrain-sat493m/   ← DINOv3-sat ViT-L
└── sam3/
    └── sam3.pt                        ← SAM3 checkpoint
```

The default config already points to these paths. If you put weights elsewhere, update `config/default.yaml` accordingly (see [Configuration](#configuration)).

---

## Install

SAM3 is vendored as a git submodule at `vendor/sam3` and declared as a
[uv path source](https://docs.astral.sh/uv/concepts/dependencies/#path-dependencies)
in `pyproject.toml`, so `uv sync` installs it automatically.

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/anemes/backend_samdino
cd backend_samdino

# Install all dependencies (creates .venv, installs SAM3 from vendor/sam3)
uv sync

# GPU only: replace CPU PyTorch with CUDA wheels
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
uv sync
```

---

## Configuration

### Config YAML

The primary config is `config/default.yaml`. Key sections:

```yaml
paths:
  project_dir: ./projects        # GeoPackage label stores (per-project)
  checkpoint_dir: ./checkpoints  # saved model checkpoints
  dataset_cache_dir: ./dataset_cache
  tile_cache_dir: ./tile_cache

server:
  host: "0.0.0.0"
  port: 8000
  api_key: null                  # set via HITL_API_KEY env var instead

training:
  epochs: 50
  batch_size: 4
  learning_rate: 1.0e-4
  freeze_backbone: true          # DINOv3 backbone is always frozen
  mixed_precision: true          # set false for CPU

inference:
  tile_size: 512
  tile_overlap: 128
  output_vectors: true           # vectorise predictions to GeoPackage
```

Relative paths in the config are resolved relative to the config file's directory.

To use a custom config:
```bash
export HITL_CONFIG_PATH=/path/to/my-config.yaml
```

### Environment variables

All secrets and deployment-specific values should be set via environment variables, not in the config file.

| Variable | Default | Description |
|---|---|---|
| `HITL_API_KEY` | _(none)_ | Bearer token required on all API endpoints. If unset, the API is open. |
| `HITL_CONFIG_PATH` | `config/default.yaml` | Path to config YAML |
| `HITL_DASHBOARD_PASSWORD` | _(none)_ | Basic auth password for `/dashboard`. Falls back to `HITL_API_KEY` if unset. |
| `HITL_DASHBOARD_USER` | `admin` | Username for dashboard basic auth |
| `HITL_ENABLE_DASHBOARD` | `true` | Set `false` to disable the Gradio dashboard entirely |
| `HOST` | from config | Override server bind host |
| `PORT` | from config | Override server port |
| `HITL_RELOAD` | `false` | Enable uvicorn auto-reload (development only) |
| `HF_TOKEN` | _(none)_ | HuggingFace token for downloading gated model weights |

### `.env` file

Python does not auto-read `.env` files. For local dev without auth, no env vars are needed — just run the server directly.

For Docker, pass a `.env` file via `--env-file` (see [Docker](#docker)). For remote deployments, inject secrets via your platform (systemd environment, GitHub Actions secrets, Azure Key Vault, etc.).

Example `.env` for Docker / remote use:

```env
# Protect the API with a bearer token (leave unset to run open)
HITL_API_KEY=change-me-in-production

# Separate dashboard password (optional — falls back to HITL_API_KEY)
HITL_DASHBOARD_PASSWORD=dashboard-password

# HuggingFace token for downloading model weights
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

If you need to set vars for a local run, export them directly:

```bash
export HITL_API_KEY=my-key
export HITL_RELOAD=true
```

---

## Running Locally

```bash
uv run python -m hitl.app
```

- REST API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`

Health check:
```bash
curl http://localhost:8000/health
```

Authenticated status check:
```bash
curl -H "Authorization: Bearer $HITL_API_KEY" http://localhost:8000/api/status
```

---

## Docker

Mounts `./data` (models, projects, checkpoints) and `./config/default.yaml` into the container. Runtime data is shared between local and Docker runs.

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### First-time setup

```bash
mkdir -p data/models
python scripts/download_models.py --dinov3 --sam3
```

### Run

```bash
docker compose --profile gpu up --build
```

### With an env file

```bash
docker compose --env-file .env --profile gpu up --build
```

### Building the image manually

```bash
docker build \
  --build-arg CPU_TORCH_ONLY=0 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t easysegment-backend:gpu .
```

---

## Remote GPU (SSH tunnel)

Run the backend on a remote GPU machine and access it from your local QGIS client:

```bash
# On the remote GPU machine
cd backend_samdino && docker compose --profile gpu up -d

# On your local machine — forward port 8000
ssh -L 8000:localhost:8000 your-gpu-machine
```

Then point the QGIS plugin at `http://localhost:8000`.

---

## API

Interactive docs at `http://localhost:8000/docs`.

### Authentication

If `HITL_API_KEY` is configured, include the token on all requests:

```
Authorization: Bearer <your-api-key>
```

Exempt: `GET /health`, `GET /docs`, `GET /openapi.json`.

### Endpoint groups

| Prefix | Description |
|---|---|
| `/api/projects/` | Create, list, switch, delete projects |
| `/api/labels/` | Classes, regions, annotations, review workflow |
| `/api/sam/` | Set image, send prompts, accept mask |
| `/api/training/` | Start/stop training, poll status, training metrics |
| `/api/inference/` | Start inference, poll status, download results |
| `/api/models/` | List checkpoints, get best model |
| `/api/raster/` | Register XYZ tile sources |
| `/api/dataset/` | Build training dataset, label stats |
| `/api/status` | GPU info, VRAM usage, active project |

---

## Dashboard

A Gradio dashboard for kicking off training runs, monitoring progress, and running batch inference is available at `http://localhost:8000/dashboard`.

- Enable/disable: `HITL_ENABLE_DASHBOARD=true|false`
- Credentials: `HITL_DASHBOARD_USER` (default: `admin`) and `HITL_DASHBOARD_PASSWORD`
- If no password is set, the dashboard falls back to `HITL_API_KEY` for basic auth (or is open if neither is set)

---

## Project Structure

```
config/           Configuration YAML files
  default.yaml                  Local dev (relative paths)
  default.docker.cpu.yaml       Docker CPU profile
  default.docker.gpu.yaml       Docker GPU profile
  schema.py                     Pydantic config schema + env var overrides
hitl/
  api/            FastAPI routes (projects, labels, sam, training, inference)
  data/           GeoPackage label store, project manager, dataset builder
  models/         DINOv3 backbone + UperNet head
  services/       SAM service, training engine, inference engine, GPU manager
  dashboard/      Gradio dashboard
  app.py          Application entry point, AppState singleton, auth middleware
vendor/sam3/      SAM3 (git submodule — facebook/sam3)
scripts/
  download_models.py            Download DINOv3-sat + SAM3 from HuggingFace
  aca/                          Azure Container Apps deployment scripts
tests/            Backend tests
```

### Runtime directories (gitignored, auto-created)

| Directory | Contents |
|---|---|
| `data/models/` | Model weight files (download separately) |
| `data/projects/` | Per-project GeoPackage label stores and SAM mask cache |
| `data/checkpoints/` | Saved model checkpoints per project |
| `data/dataset_cache/` | Preprocessed training tile datasets |
| `data/tile_cache/` | Cached inference tile outputs |

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License** (CC BY-NC-4.0). See [LICENSE](LICENSE) for the full text.

You are free to use, modify, and share this software for non-commercial purposes with attribution. Commercial use requires separate permission.

**Dependency licenses**: SAM3 and DINOv3-sat are licensed under Apache 2.0 by Meta AI Research.
