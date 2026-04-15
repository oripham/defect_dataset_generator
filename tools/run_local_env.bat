@echo off
REM Khong dung setlocal — de "call run_local_env.bat" giu HF_HOME / PYTHONPATH o CMD cha.

REM Repo root = thu muc cha cua tools/
pushd "%~dp0.." 2>nul || (
  echo [ERROR] Khong vao duoc thu muc goc project.
  exit /b 1
)
set "ROOT=%CD%"
popd

cd /d "%ROOT%" 2>nul || (
  echo [ERROR] cd /d failed: %ROOT%
  exit /b 1
)

set "HF_HOME=%ROOT%\models"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "PYTHONPATH=%ROOT%;%ROOT%\scripts"

REM 6GB VRAM: SDXLRefiner dung CPU offload (cham hon, tranh OOM). May RTX 12GB+: set SDXL_LOW_VRAM=0
if not defined SDXL_LOW_VRAM set "SDXL_LOW_VRAM=1"

echo.
echo === HondaPlus local env ===
echo ROOT=%ROOT%
echo HF_HOME=%HF_HOME%
echo SDXL_LOW_VRAM=%SDXL_LOW_VRAM%
echo PYTHONPATH=%PYTHONPATH%
echo.
echo Cach dung (cung cua so CMD):
echo   call tools\run_local_env.bat
echo   python tools\test_pipeline.py
echo   python tools\test_pipeline.py --sdxl
echo.
echo Mo CMD moi da set san bien: chay tools\open_local_cmd.bat
echo.
