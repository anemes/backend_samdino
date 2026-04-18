@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "ENV_FILE=%SCRIPT_DIR%aca.env.cmd"

if not exist "%ENV_FILE%" (
  echo Missing "%ENV_FILE%".
  echo Create scripts\aca\aca.env.cmd and set required values first.
  exit /b 1
)

call "%ENV_FILE%"
if errorlevel 1 exit /b 1

set "SKIP_BUILD=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--skip-build" (
  set "SKIP_BUILD=1"
  shift
  goto :parse_args
)
echo Unknown argument: %~1
echo Supported argument: --skip-build
exit /b 1

:args_done
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
if not defined MEMORY set "MEMORY=16Gi"
if not defined MIN_REPLICAS set "MIN_REPLICAS=1"
if not defined MAX_REPLICAS set "MAX_REPLICAS=10"
if not defined ENABLE_DASHBOARD set "ENABLE_DASHBOARD=true"
if not defined ALLOW_INSECURE set "ALLOW_INSECURE=false"
if not defined FORCE_NEW_REVISION set "FORCE_NEW_REVISION=true"

if /I not "%ENABLE_DASHBOARD%"=="true" if /I not "%ENABLE_DASHBOARD%"=="false" (
  echo ENABLE_DASHBOARD must be true or false.
  exit /b 1
)
if /I not "%ALLOW_INSECURE%"=="true" if /I not "%ALLOW_INSECURE%"=="false" (
  echo ALLOW_INSECURE must be true or false.
  exit /b 1
)
if /I not "%FORCE_NEW_REVISION%"=="true" if /I not "%FORCE_NEW_REVISION%"=="false" (
  echo FORCE_NEW_REVISION must be true or false.
  exit /b 1
)

set "REGISTRY_MODE=identity"
if defined ACR_USERNAME (
  if defined ACR_PASSWORD (
    set "REGISTRY_MODE=password"
  ) else (
    echo ACR_PASSWORD is required when ACR_USERNAME is set.
    exit /b 1
  )
) else (
  if defined ACR_PASSWORD (
    echo ACR_USERNAME is required when ACR_PASSWORD is set.
    exit /b 1
  )
)

if not defined REPO_ROOT (
  for %%i in ("%SCRIPT_DIR%..\..") do set "REPO_ROOT=%%~fi"
)
set "YAML_PATH=%REPO_ROOT%\config\aca-%APP%-deploy.yaml"

echo [0/7] Validating Azure CLI login and subscription...
call az account show --query "{subscription:name, id:id, tenantId:tenantId}" -o table
if errorlevel 1 (
  echo ERROR: Azure CLI session is not available.
  echo Run: az login
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

if "%SKIP_BUILD%"=="1" (
  echo [2/7] Skipping ACR build via --skip-build.
) else (
  echo [2/7] Building GPU image in ACR...
  pushd "%REPO_ROOT%"
  if errorlevel 1 (
    echo Unable to enter repo root "%REPO_ROOT%".
    exit /b 1
  )
  call az acr build -r "%ACR%" -t "%IMAGE_REPO%:%IMAGE_TAG%" --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 --build-arg CPU_TORCH_ONLY=0 .
  set "BUILD_RC=%ERRORLEVEL%"
  popd
  if not "%BUILD_RC%"=="0" exit /b %BUILD_RC%
)

echo [3/7] Resolving image URL...
for /f "delims=" %%i in ('az acr show -n "%ACR%" --query loginServer -o tsv') do set "ACR_LOGIN_SERVER=%%i"
if not defined ACR_LOGIN_SERVER (
  echo Unable to resolve ACR login server.
  exit /b 1
)
set "IMAGE=%ACR_LOGIN_SERVER%/%IMAGE_REPO%:%IMAGE_TAG%"
set "REV_SUFFIX=r%RANDOM%%RANDOM%"
echo Using image: %IMAGE%
echo Registry auth mode: %REGISTRY_MODE%
if /I "%FORCE_NEW_REVISION%"=="true" (
  echo Forced revision rollout enabled. Suffix: %REV_SUFFIX%
)

echo [4/7] Generating deployment YAML...
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
) > "%YAML_PATH%"
if errorlevel 1 (
  echo Failed to write deploy YAML: "%YAML_PATH%"
  exit /b 1
)

if /I "%REGISTRY_MODE%"=="identity" (
  (
    echo     registries:
    echo       - server: %ACR_LOGIN_SERVER%
    echo         identity: system
  ) >> "%YAML_PATH%"
)

(
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
) >> "%YAML_PATH%"
if errorlevel 1 (
  echo Failed to finalize deploy YAML: "%YAML_PATH%"
  exit /b 1
)

echo Deploy YAML: %YAML_PATH%

echo [5/7] Creating or updating Container App...
call az containerapp show -g "%RG%" -n "%APP%" >nul 2>nul
if errorlevel 1 (
  call az containerapp create -g "%RG%" -n "%APP%" --yaml "%YAML_PATH%"
) else (
  call az containerapp update -g "%RG%" -n "%APP%" --yaml "%YAML_PATH%"
)
if errorlevel 1 exit /b 1

if /I "%REGISTRY_MODE%"=="password" (
  echo [6/7] Applying ACR username/password credentials...
  call az containerapp registry set -g "%RG%" -n "%APP%" --server "%ACR_LOGIN_SERVER%" --username "%ACR_USERNAME%" --password "%ACR_PASSWORD%" --output none
  if errorlevel 1 exit /b 1
  echo [6/7] Triggering a revision with configured image...
  if /I "%FORCE_NEW_REVISION%"=="true" (
    call az containerapp update -g "%RG%" -n "%APP%" --image "%IMAGE%" --revision-suffix "%REV_SUFFIX%" --output none
  ) else (
    call az containerapp update -g "%RG%" -n "%APP%" --image "%IMAGE%" --output none
  )
  if errorlevel 1 exit /b 1
  goto :step7
)

echo [6/7] Ensuring AcrPull role for managed identity...
for /f "delims=" %%i in ('az containerapp show -g "%RG%" -n "%APP%" --query identity.principalId -o tsv') do set "APP_PRINCIPAL_ID=%%i"
for /f "delims=" %%i in ('az acr show -n "%ACR%" --query id -o tsv') do set "ACR_ID=%%i"

if not defined APP_PRINCIPAL_ID (
  echo ERROR: Container App managed identity principalId is empty.
  exit /b 1
)
if not defined ACR_ID (
  echo ERROR: Unable to resolve ACR id for "%ACR%".
  exit /b 1
)

set "HAS_ACRPULL=0"
for /f "delims=" %%i in ('az role assignment list --assignee-object-id "%APP_PRINCIPAL_ID%" --scope "%ACR_ID%" --query "[?roleDefinitionName=='AcrPull'] | length(@)" -o tsv') do set "HAS_ACRPULL=%%i"

if "%HAS_ACRPULL%"=="0" (
  echo AcrPull role not found. Attempting assignment...
  call az role assignment create --assignee-object-id "%APP_PRINCIPAL_ID%" --assignee-principal-type ServicePrincipal --role AcrPull --scope "%ACR_ID%" --output none
  if errorlevel 1 (
    echo ERROR: Failed to assign AcrPull.
    echo If you do not have role assignment permissions, set ACR_USERNAME and ACR_PASSWORD in scripts\aca\aca.env.cmd and rerun with --skip-build.
    exit /b 1
  )
  echo Waiting 30 seconds for RBAC propagation...
  timeout /t 30 /nobreak >nul
)

echo [6/7] Triggering image rollout...
if /I "%FORCE_NEW_REVISION%"=="true" (
  call az containerapp update -g "%RG%" -n "%APP%" --image "%IMAGE%" --revision-suffix "%REV_SUFFIX%" --output none
) else (
  call az containerapp update -g "%RG%" -n "%APP%" --image "%IMAGE%" --output none
)
if errorlevel 1 exit /b 1

:step7
echo [7/7] Deployment info...
call az containerapp show -g "%RG%" -n "%APP%" --query "properties.{provisioningState:provisioningState,runningStatus:runningStatus,latestRevision:latestRevisionName,latestReadyRevision:latestReadyRevisionName}" -o table
call az containerapp revision list -g "%RG%" -n "%APP%" -o table
for /f "delims=" %%i in ('az containerapp show -g "%RG%" -n "%APP%" --query "properties.configuration.ingress.fqdn" -o tsv') do set "FQDN=%%i"
if defined FQDN (
  echo App URL: https://%FQDN%
  echo Health:  https://%FQDN%/health
)
echo Done.
exit /b 0
