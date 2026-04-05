@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "FORGE_ROOT=%%~fI"
set "FORGE_PYTHON=%FORGE_ROOT%\venv\Scripts\python.exe"

if not exist "%FORGE_PYTHON%" (
  echo [sd-forge-llm-prompt-gen-yoiko] Forge venv python was not found:
  echo   %FORGE_PYTHON%
  echo Run this script from an installed Forge extension directory.
  exit /b 1
)

echo [sd-forge-llm-prompt-gen-yoiko] bootstrapping Forge environment...
"%FORGE_PYTHON%" "%SCRIPT_DIR%install.py"
set "BOOTSTRAP_EXIT=%ERRORLEVEL%"

if not "%BOOTSTRAP_EXIT%"=="0" (
  echo [sd-forge-llm-prompt-gen-yoiko] bootstrap failed with exit code %BOOTSTRAP_EXIT%.
  exit /b %BOOTSTRAP_EXIT%
)

echo [sd-forge-llm-prompt-gen-yoiko] bootstrap complete.
echo You can now restart Forge. If you usually launch with --skip-install, keep using that after this one-time bootstrap.
exit /b 0
