@echo off
rem Full pipeline: generate -> score -> report
rem Usage: run_pipeline.bat [count]
setlocal
set PY=D:\stablematrix\Data\Packages\Stable Diffusion WebUI Forge - Neo\venv\Scripts\python.exe
set COUNT=%1
if "%COUNT%"=="" set COUNT=20
cd /d "%~dp0"
"%PY%" runner.py --count %COUNT% --score
if errorlevel 1 goto :end
"%PY%" report.py
:end
endlocal
