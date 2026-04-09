@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%aca.env.cmd"

if not exist "%ENV_FILE%" (
  echo Missing "%ENV_FILE%".
  echo Create it from "scripts\aca\aca.env.cmd.example" first.
  exit /b 1
)

call "%ENV_FILE%"
if errorlevel 1 exit /b 1

if not defined RG (
  echo Missing required variable: RG
  exit /b 1
)
if not defined ENV (
  echo Missing required variable: ENV
  exit /b 1
)
if not defined APP (
  echo Missing required variable: APP
  exit /b 1
)
if not defined ACR (
  echo Missing required variable: ACR
  exit /b 1
)
if not defined IMAGE_REPO (
  echo Missing required variable: IMAGE_REPO
  exit /b 1
)
if not defined IMAGE_TAG (
  echo Missing required variable: IMAGE_TAG
  exit /b 1
)
if not defined WORKLOAD_PROFILE (
  echo Missing required variable: WORKLOAD_PROFILE
  exit /b 1
)
if not defined STORAGE_NAME (
  echo Missing required variable: STORAGE_NAME
  exit /b 1
)

if not defined CPU set "CPU=4.0"
if not defined MEMORY set "MEMORY=16.0Gi"
if not defined MIN_REPLICAS set "MIN_REPLICAS=1"
if not defined MAX_REPLICAS set "MAX_REPLICAS=10"
if not defined ENABLE_DASHBOARD set "ENABLE_DASHBOARD=true"
if not defined ALLOW_INSECURE set "ALLOW_INSECURE=false"
if not defined REPO_ROOT (
  for %%i in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fi"
)

echo [0/7] Validating Azure CLI login and subscription...
call az account show --query "{subscription:name, id:id, tenantId:tenantId}" -o table
if errorlevel 1 (
  echo ERROR: Azure CLI session is not available.
  echo Run: az login
  exit /b 1
)
if /I not "%ENABLE_DASHBOARD%"=="true" if /I not "%ENABLE_DASHBOARD%"=="false" (
  echo ENABLE_DASHBOARD must be true or false.
  exit /b 1
)
if /I not "%ALLOW_INSECURE%"=="true" if /I not "%ALLOW_INSECURE%"=="false" (
  echo ALLOW_INSECURE must be true or false.
  exit /b 1
)

echo [1/7] Resolving environment metadata...
for /f "delims=" %%i in ('az containerapp env show -g "%RG%" -n "%ENV%" --query id -o tsv') do set "ENV_ID=%%i"
for /f "delims=" %%i in ('az containerapp env show -g "%RG%" -n "%ENV%" --query location -o tsv') do set "LOC=%%i"
if not defined ENV_ID (
  echo Unable to resolve managed environment id.
  exit /b 1
)
if not defined LOC (
  echo Unable to resolve managed environment location.
  exit /b 1
)

echo [2/7] Building GPU image in ACR...
pushd "%REPO_ROOT%"
if errorlevel 1 (
  echo Unable to enter repo root "%REPO_ROOT%".
  exit /b 1
)
call az acr build -r "%ACR%" -t "%IMAGE_REPO%:%IMAGE_TAG%" ^
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ^
  --build-arg CPU_TORCH_ONLY=0 ^
  .
set "BUILD_RC=%ERRORLEVEL%"
popd
if not "%BUILD_RC%"=="0" exit /b %BUILD_RC%
if errorlevel 1 exit /b 1

echo [3/7] Resolving image URL...
for /f "delims=" %%i in ('az acr show -n "%ACR%" --query loginServer -o tsv') do set "ACR_LOGIN_SERVER=%%i"
if not defined ACR_LOGIN_SERVER (
  echo Unable to resolve ACR login server.
  exit /b 1
)
set "IMAGE=%ACR_LOGIN_SERVER%/%IMAGE_REPO%:%IMAGE_TAG%"
echo Using image: %IMAGE%

echo [4/7] Generating deployment YAML...
set "TMP_YAML=%TEMP%\aca-%APP%-deploy.yaml"
(
  echo location: %LOC%
  echo name: %APP%
  echo type: Microsoft.App/containerApps
  echo identity:
  echo   type: SystemAssigned
  echo properties:
  echo   managedEnvironmentId: %ENV_ID%
  echo   workloadProfileName: %WORKLOAD_PROFILE%
  echo   configuration:
  echo     activeRevisionsMode: Single
  echo     ingress:
  echo       external: true
  echo       allowInsecure: %ALLOW_INSECURE%
  echo       targetPort: 8000
  echo       transport: auto
  echo     registries:
  echo       - server: %ACR_LOGIN_SERVER%
  echo         identity: system
  echo   template:
  echo     containers:
  echo       - name: %APP%
  echo         image: %IMAGE%
  echo         env:
  echo           - name: HITL_ENABLE_DASHBOARD
  echo             value: "%ENABLE_DASHBOARD%"
  echo         resources:
  echo           cpu: %CPU%
  echo           memory: %MEMORY%
  echo         volumeMounts:
  echo           - volumeName: data
  echo             mountPath: /app/data
  echo     scale:
  echo       minReplicas: %MIN_REPLICAS%
  echo       maxReplicas: %MAX_REPLICAS%
  echo     volumes:
  echo       - name: data
  echo         storageType: AzureFile
  echo         storageName: %STORAGE_NAME%
) > "%TMP_YAML%"

echo [5/7] Creating/updating Container App...
call az containerapp show -g "%RG%" -n "%APP%" >nul 2>nul
if errorlevel 1 (
  call az containerapp create -g "%RG%" -n "%APP%" --yaml "%TMP_YAML%"
) else (
  call az containerapp update -g "%RG%" -n "%APP%" --yaml "%TMP_YAML%"
)
if errorlevel 1 exit /b 1

echo [6/7] Ensuring AcrPull role assignment...
for /f "delims=" %%i in ('az containerapp show -g "%RG%" -n "%APP%" --query identity.principalId -o tsv') do set "APP_PRINCIPAL_ID=%%i"
for /f "delims=" %%i in ('az acr show -n "%ACR%" --query id -o tsv') do set "ACR_ID=%%i"
if defined APP_PRINCIPAL_ID if defined ACR_ID (
  call az role assignment create ^
    --assignee-object-id "%APP_PRINCIPAL_ID%" ^
    --assignee-principal-type ServicePrincipal ^
    --role AcrPull ^
    --scope "%ACR_ID%" >nul 2>nul
)

echo [7/7] Deployment info...
for /f "delims=" %%i in ('az containerapp show -g "%RG%" -n "%APP%" --query "properties.configuration.ingress.fqdn" -o tsv') do set "FQDN=%%i"
if defined FQDN (
  echo App URL: https://%FQDN%
  echo Health:  https://%FQDN%/health
)
echo Done.
exit /b 0
