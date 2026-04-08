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
if not defined STG (
  echo Missing required variable: STG
  exit /b 1
)
if not defined SHARE (
  echo Missing required variable: SHARE
  exit /b 1
)
if not defined STORAGE_NAME (
  echo Missing required variable: STORAGE_NAME
  exit /b 1
)

echo [0/6] Validating Azure CLI login and subscription...
call az account show --query "{subscription:name, id:id, tenantId:tenantId}" -o table
if errorlevel 1 (
  echo ERROR: Azure CLI session is not available.
  echo Run: az login
  exit /b 1
)
echo [0/6] OK

echo [1/6] Validating Container Apps environment...
call az containerapp env show -g "%RG%" -n "%ENV%" --query "{name:name,location:location,resourceGroup:resourceGroup}" -o table
if errorlevel 1 (
  echo ERROR: Unable to read managed environment "%ENV%" in resource group "%RG%".
  echo Check current subscription, then run:
  echo   az account set --subscription YOUR_SUBSCRIPTION
  exit /b 1
)
echo [1/6] OK

if not defined LOC (
  for /f "delims=" %%i in ('az containerapp env show -g "%RG%" -n "%ENV%" --query location -o tsv') do set "LOC=%%i"
)
if not defined LOC (
  echo Unable to resolve environment location.
  exit /b 1
)
echo Using location: %LOC%

echo [2/6] Ensuring storage account "%STG%" exists...
call az storage account show -g "%RG%" -n "%STG%" >nul 2>nul
if errorlevel 1 (
  echo Storage account not found. Creating...
  call az storage account create ^
    -g "%RG%" ^
    -n "%STG%" ^
    -l "%LOC%" ^
    --sku Premium_LRS ^
    --kind FileStorage ^
    --min-tls-version TLS1_2
  if errorlevel 1 exit /b 1
) else (
  echo Storage account already exists.
  echo Enforcing TLS1_2 policy on existing storage account...
  call az storage account update ^
    -g "%RG%" ^
    -n "%STG%" ^
    --min-tls-version TLS1_2 >nul
  if errorlevel 1 exit /b 1
)
echo [2/6] OK

echo [3/6] Ensuring file share "%SHARE%" exists...
call az storage share-rm create ^
  -g "%RG%" ^
  --storage-account "%STG%" ^
  --name "%SHARE%" ^
  --quota 1024 ^
  --enabled-protocols SMB
if errorlevel 1 exit /b 1
echo [3/6] OK

echo [4/6] Reading storage account key...
for /f "delims=" %%i in ('az storage account keys list -g "%RG%" -n "%STG%" --query "[0].value" -o tsv') do set "KEY=%%i"
if not defined KEY (
  echo Failed to read storage account key.
  exit /b 1
)
echo [4/6] OK

echo [5/6] Registering Azure Files on managed environment...
call az containerapp env storage set ^
  -g "%RG%" ^
  -n "%ENV%" ^
  --storage-name "%STORAGE_NAME%" ^
  --access-mode ReadWrite ^
  --azure-file-account-name "%STG%" ^
  --azure-file-account-key "%KEY%" ^
  --azure-file-share-name "%SHARE%"
if errorlevel 1 exit /b 1
echo [5/6] OK

echo [6/6] Creating base directories in the share...
for %%d in (models projects checkpoints dataset_cache tile_cache) do (
  call az storage directory create ^
    --account-name "%STG%" ^
    --account-key "%KEY%" ^
    --share-name "%SHARE%" ^
    --name "%%d" >nul
  if errorlevel 1 exit /b 1
)
echo [6/6] OK

echo Setup complete.
exit /b 0
