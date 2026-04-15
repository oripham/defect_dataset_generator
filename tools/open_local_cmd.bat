@echo off
REM Mo Command Prompt moi: da set HF_HOME, PYTHONPATH, SDXL_LOW_VRAM
call "%~dp0run_local_env.bat"
if errorlevel 1 exit /b 1
title HondaPlus defect_dataset_generator
cd /d "%ROOT%"
cmd /k
