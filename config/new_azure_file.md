## ACA Repeatable Workflow (cmd.exe)

This repo now includes repeatable deployment scripts under `scripts/aca/`:
- `01_setup_once.cmd` -> create/register Azure Files (once)
- `02_upload_data.cmd` -> upload local model/data folders
- `03_deploy.cmd` -> build image + create/update Container App

The flow targets your existing managed environment:
- `rg-simpro`
- `managedEnvironment-rgsimpro-b142`

### Repo changes made for ACA compatibility

1. Fixed app import bug in [`hitl/app.py`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\hitl\app.py) (`..config` -> `config`) that breaks startup.
2. Added runtime toggles in [`hitl/app.py`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\hitl\app.py):
   - `HITL_ENABLE_DASHBOARD` (disable dashboard in ACA)
   - `HITL_API_KEY` (inject API key from env/secret)
   - `PORT`/`HOST`/`HITL_RELOAD`
3. Added `HITL_CONFIG_PATH` support in [`config/schema.py`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\config\schema.py).
4. Docker entrypoint now runs module mode ([`Dockerfile`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\Dockerfile)) so env-based host/port behavior is respected.
5. Added `.dockerignore` rule for `data/` so runtime data isn’t uploaded during image builds.
6. Added reusable ACA scripts and env template:
   - [`aca.env.cmd.example`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\scripts\aca\aca.env.cmd.example)
   - [`01_setup_once.cmd`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\scripts\aca\01_setup_once.cmd)
   - [`02_upload_data.cmd`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\scripts\aca\02_upload_data.cmd)
   - [`03_deploy.cmd`](C:\Users\nemesa\Documents\projects\qgis\backend_samdino\scripts\aca\03_deploy.cmd)

### 0) One-time local config

Create `scripts\aca\aca.env.cmd` from the example:

```bat
copy scripts\aca\aca.env.cmd.example scripts\aca\aca.env.cmd
```

Edit `scripts\aca\aca.env.cmd` and set:
- `ACR`
- `WORKLOAD_PROFILE`
- `STG` (must be a valid storage account name: lowercase letters/numbers only, no `-`)

### a) Set up Azure Files (once)

```bat
call scripts\aca\01_setup_once.cmd
echo Exit code: %ERRORLEVEL%
```

What it does:
- Validates `managedEnvironment-rgsimpro-b142`
- Creates storage account if missing (`Premium_LRS`, `FileStorage`)
- Enforces storage account `minimumTlsVersion=TLS1_2` (required by common org policies)
- Creates file share
- Registers share with Container Apps env (`env storage set`)
- Creates base directories: `models`, `projects`, `checkpoints`, `dataset_cache`, `tile_cache`

### b) Upload local `/data` directories

```bat
call scripts\aca\02_upload_data.cmd
echo Exit code: %ERRORLEVEL%
```

What it uploads:
- Preferred: full `.\data\` tree -> share root (`/`)
- Plus back-compat: if `.\models` exists outside `.\data\models`, uploads that to `/models`

If your local data lives elsewhere, set `DATA_ROOT` in `aca.env.cmd`.
`02_upload_data.cmd` defaults `DATA_ROOT` to repo root (inferred from script path), so it works even when launched from `scripts\aca`.

### c) Deploy Container App

```bat
call scripts\aca\03_deploy.cmd
echo Exit code: %ERRORLEVEL%
```

Fast retry modes (skip long build loop):

```bat
call scripts\aca\03_deploy.cmd --skip-build
call scripts\aca\03_deploy.cmd --apply-only
```

What it does:
- Builds/pushes GPU image to ACR (`cu121` wheels)
- Generates deployment YAML with:
  - system-assigned identity
  - ACR registry identity
  - Azure Files mount at `/app/data`
  - ingress on `8000`
  - target workload profile
- Creates app if missing, otherwise updates app
- Ensures `AcrPull` role assignment
- Prints app FQDN and `/health` URL

`03_deploy.cmd` defaults `REPO_ROOT` to repo root (inferred from script path), so ACR build context is correct even when launched from `scripts\aca`.

### Re-deploy cycle

For normal updates:

1. `scripts\aca\02_upload_data.cmd` (only if model/data changed)
2. `scripts\aca\03_deploy.cmd`
