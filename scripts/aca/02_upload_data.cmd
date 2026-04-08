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
if not defined STG (
  echo Missing required variable: STG
  exit /b 1
)
if not defined SHARE (
  echo Missing required variable: SHARE
  exit /b 1
)

echo [0/1] Validating Azure CLI login and subscription...
call az account show --query "{subscription:name, id:id, tenantId:tenantId}" -o table
if errorlevel 1 (
  echo ERROR: Azure CLI session is not available.
  echo Run: az login
  exit /b 1
)

if not defined DATA_ROOT (
  for %%i in ("%SCRIPT_DIR%..\..") do set "DATA_ROOT=%%~fi"
)
echo Using DATA_ROOT=%DATA_ROOT%

for /f "delims=" %%i in ('az storage account keys list -g "%RG%" -n "%STG%" --query "[0].value" -o tsv') do set "KEY=%%i"
if not defined KEY (
  echo Failed to read storage account key.
  exit /b 1
)

REM Preferred path: upload the whole local data tree to preserve all runtime state.
if exist "%DATA_ROOT%\data" (
  echo Uploading full "%DATA_ROOT%\data" -^> "%SHARE%/"
  call az storage file upload-batch ^
    --account-name "%STG%" ^
    --account-key "%KEY%" ^
    --source "%DATA_ROOT%\data" ^
    --destination "%SHARE%"
  if errorlevel 1 exit /b 1

  REM Back-compat: if models are outside data/, upload them too.
  if exist "%DATA_ROOT%\models" (
    if exist "%DATA_ROOT%\data\models" (
      echo Local "%DATA_ROOT%\data\models" exists; skipping "%DATA_ROOT%\models".
    ) else (
      echo Uploading "%DATA_ROOT%\models" -^> "%SHARE%/models"
      call az storage file upload-batch ^
        --account-name "%STG%" ^
        --account-key "%KEY%" ^
        --source "%DATA_ROOT%\models" ^
        --destination "%SHARE%" ^
        --destination-path "models"
      if errorlevel 1 exit /b 1
    )
  )

  echo Upload complete.
  exit /b 0
)

REM Fallback path when there is no local data/ directory yet.
if exist "%DATA_ROOT%\models" (
  echo Uploading "%DATA_ROOT%\models" -^> "%SHARE%/models"
  call az storage file upload-batch ^
    --account-name "%STG%" ^
    --account-key "%KEY%" ^
    --source "%DATA_ROOT%\models" ^
    --destination "%SHARE%" ^
    --destination-path "models"
  if errorlevel 1 exit /b 1
) else (
  echo WARNING: no data/ or models/ directory found at "%DATA_ROOT%".
)

echo Upload complete.
exit /b 0
