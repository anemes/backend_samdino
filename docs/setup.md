# Setup Guide

## Prerequisites

- Python 3.11+
- CUDA-capable GPU with 16+ GB VRAM (tested on RTX 3090 Ti 24GB)
- QGIS 3.28+ (for the plugin)
- Git

## Backend Setup

### 1. Create virtual environment

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Initialize submodules and install SAM3

```bash
git submodule update --init --recursive
pip install -e vendor/sam3
```

### 4. Install backend dependencies

```bash
pip install -e .
```

### 5. Configure

Edit `config/default.yaml`:
- Set `models.dinov3.path` to point to your DINOv3-sat weights directory
- Set `models.sam3.checkpoint` to point to your SAM3 checkpoint
- Adjust `training` and `inference` settings as needed

### 6. Start the backend

```bash
cd backend
python -m hitl.app
```

The API will be available at `http://localhost:8000`.
API docs: `http://localhost:8000/docs`

## QGIS Plugin Setup

### 1. Link the plugin to QGIS

```bash
# Find your QGIS plugins directory
# Linux: ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
# Mac: ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/
# Windows: %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\

ln -s /path/to/qgis_plugin/hitl_sketcher ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/hitl_sketcher
```

### 2. Enable in QGIS

1. Open QGIS
2. Go to Plugins → Manage and Install Plugins
3. Find "HITL Sketcher" and enable it
4. The toolbar should appear

### 3. Connect to backend

1. Click "Backend Connection" in the toolbar
2. Enter the backend URL (default: `http://localhost:8000`)
3. Click "Connect" — should show green "Connected"

## Remote GPU Setup (SSH Tunnel)

To use a remote GPU machine:

```bash
# On your local machine, create an SSH tunnel
ssh -L 8000:localhost:8000 your-gpu-machine

# On the GPU machine, start the backend
cd backend && python -m hitl.app
```

Then configure the QGIS plugin to connect to `http://localhost:8000`.

## Quick Test

1. Start the backend
2. Open QGIS with an aerial/satellite raster layer
3. Connect the plugin to the backend
4. Add classes (e.g., "building", "road")
5. Draw an annotation region
6. Draw label polygons inside the region
7. Click "Sync with Backend"
8. Use the API to trigger training:

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"raster_path": "/path/to/your/raster.tif"}'
```

8. Monitor at `http://localhost:8000/api/training/status`
