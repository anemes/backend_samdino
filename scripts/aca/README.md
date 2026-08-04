# ACA Deployment Scripts

Deploy the HITL backend to Azure Container Apps with GPU and Azure Files storage.

All commands below show both **bash** and **cmd.exe** variants.

## First Deployment

### 1. Create env config

```bash
cp scripts/aca/aca.env.sh.example scripts/aca/aca.env.sh
```

```cmd
copy scripts\aca\aca.env.cmd.example scripts\aca\aca.env.cmd
```

Edit the file and set at minimum:
- `ACR` — your Azure Container Registry name
- `WORKLOAD_PROFILE` — GPU workload profile name (must exist in the managed environment)
- `STG` — storage account name (lowercase letters/numbers only, 3-24 chars)

### 2. One-time Azure Files setup

Creates storage account, file share, registers with Container Apps environment, creates base directories.

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/01_setup_once.sh
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\01_setup_once.cmd
```

### 3. Upload model weights and data

Uploads `./data/` tree (models, projects, checkpoints) to Azure Files.

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/02_upload_data.sh
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\02_upload_data.cmd
```

If your data lives elsewhere, set `DATA_ROOT` in the env file.

### 4. Build and deploy

Builds GPU Docker image in ACR, creates/updates the Container App.

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/03_deploy.sh
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\03_deploy.cmd
```

### 5. Set API key and dashboard auth

```bash
az containerapp secret set -g $RG -n $APP \
  --secrets hitl-api-key=<strong-key> hitl-dashboard-password=<strong-password>

az containerapp update -g $RG -n $APP --set-env-vars \
  HITL_API_KEY=secretref:hitl-api-key \
  HITL_DASHBOARD_USER=admin \
  HITL_DASHBOARD_PASSWORD=secretref:hitl-dashboard-password \
  HITL_ENABLE_DASHBOARD=true
```

```cmd
az containerapp secret set -g %RG% -n %APP% --secrets hitl-api-key=<strong-key> hitl-dashboard-password=<strong-password>
az containerapp update -g %RG% -n %APP% --set-env-vars HITL_API_KEY=secretref:hitl-api-key HITL_DASHBOARD_USER=admin HITL_DASHBOARD_PASSWORD=secretref:hitl-dashboard-password HITL_ENABLE_DASHBOARD=true
```

If `HITL_DASHBOARD_PASSWORD` is unset but `HITL_API_KEY` is set, the backend falls back to using the API key as the dashboard password.

### 6. Verify

```bash
FQDN=$(az containerapp show -g $RG -n $APP --query properties.configuration.ingress.fqdn -o tsv)
curl -i https://$FQDN/health
# Dashboard: https://$FQDN/dashboard
```

```cmd
for /f "delims=" %i in ('az containerapp show -g %RG% -n %APP% --query properties.configuration.ingress.fqdn -o tsv') do @set FQDN=%i
curl.exe --ssl-no-revoke -i https://%FQDN%/health
REM Dashboard: https://%FQDN%/dashboard
```

---

## Redeployment (after code changes)

### Code or dependency changes

Rebuilds the Docker image and deploys:

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/03_deploy.sh
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\03_deploy.cmd
```

### Config/env changes only (no code change)

Deploys without rebuilding:

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/03_deploy.sh --skip-build
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\03_deploy.cmd --skip-build
```

### Swap to an existing image tag

```bash
az containerapp update -g $RG -n $APP --image <acr-login-server>/$IMAGE_REPO:$IMAGE_TAG
```

```cmd
az containerapp update -g %RG% -n %APP% --image <acr-login-server>/%IMAGE_REPO%:%IMAGE_TAG%
```

### Upload new data (model weights changed)

```bash
source scripts/aca/aca.env.sh && bash scripts/aca/02_upload_data.sh
```

```cmd
call scripts\aca\aca.env.cmd && call scripts\aca\02_upload_data.cmd
```

Secrets and env vars persist across redeployments unless explicitly overwritten.

---

## Troubleshooting

### Check app status

```bash
az containerapp show -g $RG -n $APP \
  --query "properties.{state:provisioningState,revision:latestRevisionName}" -o table
az containerapp revision list -g $RG -n $APP -o table
```

```cmd
az containerapp show -g %RG% -n %APP% --query "properties.{state:provisioningState,revision:latestRevisionName}" -o table
az containerapp revision list -g %RG% -n %APP% -o table
```

### View logs

```bash
az containerapp logs show -g $RG -n $APP --tail 100                # app logs
az containerapp logs show -g $RG -n $APP --type system --tail 100  # system logs
```

```cmd
az containerapp logs show -g %RG% -n %APP% --tail 100
az containerapp logs show -g %RG% -n %APP% --type system --tail 100
```

### Registry auth

- If `ACR_USERNAME` + `ACR_PASSWORD` are set in the env file, deploy uses username/password auth.
- If blank, deploy uses managed identity + `AcrPull` role (recommended).

---

## Multi-user access

The backend supports multiple API keys, each with its own user identity and per-project roles.
See [`keys.json.example`](keys.json.example) and [`project.json.example`](project.json.example)
for annotated examples of both files.

### Key registry — `data/keys.json`

Create this file on the Azure Files share **before** starting the container.
Each entry maps a bearer token to a `user_id` and optional `admin` flag:

```json
[
  {"key": "sk-alice-long-random-string", "user_id": "alice", "label": "Alice Smith"},
  {"key": "sk-bob-long-random-string",   "user_id": "bob",   "label": "Bob Jones"},
  {"key": "sk-ops-secret",               "user_id": "ops",   "label": "Operator", "admin": true}
]
```

Upload to the share (bash):

```bash
source scripts/aca/aca.env.sh
az storage file upload \
  --account-name $STG --account-key $STG_KEY \
  --share-name $SHARE \
  --source data/keys.json --path data/keys.json
```

```cmd
call scripts\aca\aca.env.cmd
az storage file upload --account-name %STG% --account-key %STG_KEY% --share-name %SHARE% --source data\keys.json --path data/keys.json
```

**Key registry changes require a container restart** (the file is read once at startup):

```bash
az containerapp revision restart -g $RG -n $APP \
  --revision $(az containerapp revision list -g $RG -n $APP --query "[0].name" -o tsv)
```

**Backward compatibility**: if `data/keys.json` does not exist, `HITL_API_KEY` env var is used
as a single admin key for `user_id="default"` — identical to the previous single-key behaviour.

### Project membership — `data/projects/<id>/project.json`

Each project's `project.json` controls who can access it:

```json
{
  "project_id": "buildings",
  "name": "Building Detection",
  "owner": "alice",
  "public": false,
  "members": {
    "bob": "contributor",
    "carol": "user"
  }
}
```

Roles:

| Role | Read labels/metrics | Run inference | Label / train / SAM |
|---|---|---|---|
| `contributor` | ✓ | ✓ | ✓ |
| `user` | ✓ | ✓ | ✗ |
| (public project) | ✓ | ✓ | ✗ |
| (no access / private) | ✗ | ✗ | ✗ |

Edit `project.json` directly on the share (or upload a replacement).
**Membership changes take effect immediately** — no restart required.

### Per-user inference results

Standalone inference results are stored in a dedicated project per user:
`_inference_{user_id}` (e.g. `_inference_alice`). These projects are auto-created on first use
and are invisible in the project catalogue — they will never appear in another user's list.

---

## Scaling

### Scale-to-zero

`MIN_REPLICAS=0` (the default) enables scale-to-zero: the container shuts down after ~5 minutes
of idle HTTP traffic and restarts automatically when the next request arrives.

**Cold-start warning — GPU profiles**: On Dedicated GPU workload profiles (`NC*`), a cold start
requires Azure to provision a GPU node, pull the image, and start the container. Expect **2–4
minutes** before the first request is served. Set `MIN_REPLICAS=1` in `aca.env.sh` to keep the
instance always-on and avoid this latency (at the cost of continuous GPU billing).

SAM3 weights (~6 GB) load on the first `/api/sam/set-image` call after each cold start, adding
a few extra seconds to the first interactive segmentation session.

### Replica count

`MAX_REPLICAS` must stay at `1` — see [Storage constraints](#storage-constraints) below.

---

## Storage constraints

GeoPackage uses SQLite which is single-writer. **`MAX_REPLICAS` must stay at `1`** — do not increase it or you will get `database is locked` errors. SQLite journal mode is set to `DELETE` (not WAL) for Azure Files SMB compatibility.
