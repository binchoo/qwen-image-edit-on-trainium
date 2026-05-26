@echo off
REM templates/launchers/train_xpu.bat
REM
REM Launches qwen-image-edit fine-tuning on Intel XPU on Windows.
REM Activates oneAPI runtime, conda env, and runs the training entry point.
REM
REM Usage:
REM   train_xpu.bat <config-path> [--cache]
REM     <config-path>: path to YAML config (typically output of recommend_config.py)
REM     --cache:       optional. If present, runs in cache-build mode (one-time
REM                    per dataset). Otherwise runs fit (training) mode.
REM
REM Example:
REM   train_xpu.bat configs/my_run.yaml --cache
REM   train_xpu.bat configs/my_run.yaml
REM
REM Prerequisites:
REM   - SKILL.md §2 pre-flight done (oneAPI installed at default path)
REM   - SKILL.md §4 conda env created (default name: qwen-image-edit-xpu)
REM   - SKILL.md §6 framework adaptation applied to qwen-image-finetune
REM
REM This launcher assumes the conda env name is qwen-image-edit-xpu and the
REM oneAPI install is at the Windows default path. Edit if your setup differs.

setlocal

if "%~1"=="" (
    echo ERROR: config path required
    echo Usage: train_xpu.bat ^<config^> [--cache]
    exit /b 1
)
set CONFIG=%~1
set MODE=%~2

REM NoDefaultCurrentDirectoryInExePath guards against DLL loading from
REM the current directory. Cleared only for Intel oneAPI setvars.bat;
REM restored below — DO NOT remove the restore line.
set "NoDefaultCurrentDirectoryInExePath="
REM Default oneAPI install path — update this line if installed elsewhere.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
if errorlevel 1 (
    echo ERROR: setvars.bat failed -- check oneAPI install path in this script
    exit /b 1
)

REM Activate conda env
call conda activate qwen-image-edit-xpu
if errorlevel 1 (
    echo ERROR: conda activate failed -- expected env name 'qwen-image-edit-xpu'
    exit /b 1
)

REM Skip HF login on package import (assume token configured separately or
REM model is on disk per pre-flight)
set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1

REM Cache compiled SYCL kernels for faster subsequent runs
set SYCL_CACHE_PERSISTENT=1


if /i "%MODE%"=="--cache" (
    echo [train_xpu] cache-build mode
    python -m qflux.main --config %CONFIG% --cache
) else (
    echo [train_xpu] fit mode
    accelerate launch --config_file accelerate_config_xpu.yaml -m qflux.main --config %CONFIG%
)

endlocal
