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

## Storage constraints

GeoPackage uses SQLite which is single-writer. **`MAX_REPLICAS` must stay at `1`** — do not increase it or you will get `database is locked` errors. SQLite journal mode is set to `DELETE` (not WAL) for Azure Files SMB compatibility.
