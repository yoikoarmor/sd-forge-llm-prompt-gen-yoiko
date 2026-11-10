@echo off
rem Full pipeline: generate -> score -> report
rem Usage: run_pipeline.bat [count]
setlocal
rem Activate the desired Python environment first, or set PY explicitly.
if not defined PY set "PY=python"
set COUNT=%1
if "%COUNT%"=="" set COUNT=20
cd /d "%~dp0"
"%PY%" runner.py --count %COUNT% --score
if errorlevel 1 goto :end
"%PY%" report.py
:end
endlocal
