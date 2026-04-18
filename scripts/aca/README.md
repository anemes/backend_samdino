# ACA Scripts: Post-Deploy Changes

This guide is for **changes after initial ACA setup**.
Commands below are for **Command Prompt (`cmd.exe`)**.

## 0) Load env vars

```cmd
call scripts\aca\aca.env.cmd
```

## 1) Which command to run for which change

- **Code/dependency change** (Python code, Dockerfile, requirements):
  - rebuild image + deploy:
  ```cmd
  call scripts\aca\03_deploy.cmd
  ```
- **No code change** (just ACA config/env/secrets/ingress/image already in ACR):
  - deploy without rebuild:
  ```cmd
  call scripts\aca\03_deploy.cmd --skip-build
  ```
- **Only swap to an image tag that already exists in ACR**:
  ```cmd
  az containerapp update -g %RG% -n %APP% --image acrjacobsdigitalsolutionsada.azurecr.io/%IMAGE_REPO%:%IMAGE_TAG%
  ```

## 2) API key (no image rebuild)

Create/update secret:

```cmd
az containerapp secret set -g %RG% -n %APP% --secrets hitl-api-key=<strong_api_key>
```

Bind env var to secret:

```cmd
az containerapp update -g %RG% -n %APP% --set-env-vars HITL_API_KEY=secretref:hitl-api-key
```

## 3) Dashboard auth username/password (no image rebuild)

The backend supports:
- `HITL_DASHBOARD_USER`
- `HITL_DASHBOARD_PASSWORD`

Set username directly and password via secret (used on `/dashboard`):

```cmd
az containerapp secret set -g %RG% -n %APP% --secrets hitl-dashboard-password=<strong_password>
az containerapp update -g %RG% -n %APP% --set-env-vars HITL_DASHBOARD_USER=<dashboard_user> HITL_DASHBOARD_PASSWORD=secretref:hitl-dashboard-password HITL_ENABLE_DASHBOARD=true
```

If `HITL_DASHBOARD_PASSWORD` is unset but `HITL_API_KEY` is set, the backend falls back to using the API key as the dashboard basic-auth password.

## 4) Dashboard exposure model

Current default deploy exposes FastAPI ingress on `targetPort=8000`.
Dashboard is mounted on the same app at:

```text
https://<FQDN>/dashboard
```

No ingress port switch is required for normal usage.

If you want strict separation (separate public URL and scaling policy), deploy a second Container App for dashboard traffic.
Use the same image, mount, and env, but a different `%APP%` name.

## 5) Quick verification

```cmd
az containerapp show -g %RG% -n %APP% --query "properties.{provisioningState:provisioningState,runningStatus:runningStatus,latestRevision:latestRevisionName,latestReadyRevision:latestReadyRevisionName}" -o table
az containerapp revision list -g %RG% -n %APP% -o table
for /f "delims=" %i in ('az containerapp show -g %RG% -n %APP% --query properties.configuration.ingress.fqdn -o tsv') do @set FQDN=%i
curl.exe --ssl-no-revoke -i https://%FQDN%/health
curl.exe --ssl-no-revoke -I https://%FQDN%/dashboard
```

For startup/deploy issues:

```cmd
az containerapp logs show -g %RG% -n %APP% --type system --tail 100
az containerapp logs show -g %RG% -n %APP% --tail 100
```

## 6) Registry auth mode in `03_deploy.cmd`

- If `ACR_USERNAME` + `ACR_PASSWORD` are set in `aca.env.cmd`, deploy uses username/password registry auth.
- If they are blank, deploy uses managed identity + `AcrPull` role path.
