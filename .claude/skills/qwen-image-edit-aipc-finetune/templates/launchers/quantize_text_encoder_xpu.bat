@echo off
REM templates/launchers/quantize_text_encoder_xpu.bat
REM
REM One-time NF4 pre-quantization of the Qwen2.5-VL text_encoder.
REM Wraps templates/quantize_text_encoder_nf4.py with oneAPI activation.
REM
REM Usage:
REM   quantize_text_encoder_xpu.bat <src-pipeline-dir> <dst-nf4-dir>
REM
REM Example:
REM   quantize_text_encoder_xpu.bat C:\models\Qwen-Image-Edit-2511 C:\models\Qwen-Image-Edit-2511-nf4-text-encoder
REM
REM After completion, set model.text_encoder_path in your training config.
REM Prerequisites: §2 pre-flight done; §4 conda env created.

setlocal

if "%~1"=="" (
    echo ERROR: source pipeline directory required.
    echo Usage: quantize_text_encoder_xpu.bat ^<src^> ^<dst^>
    exit /b 1
)
if "%~2"=="" (
    echo ERROR: destination directory required.
    echo Usage: quantize_text_encoder_xpu.bat ^<src^> ^<dst^>
    exit /b 1
)

set SRC=%~1
set DST=%~2

REM NoDefaultCurrentDirectoryInExePath guards against DLL loading from
REM the current directory. Cleared only for Intel oneAPI setvars.bat;
REM restored below — DO NOT remove the restore line.
set "NoDefaultCurrentDirectoryInExePath="
REM Default oneAPI install path — update this line if installed elsewhere.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" --force
set "NoDefaultCurrentDirectoryInExePath=1"
if errorlevel 1 (
    echo ERROR: setvars.bat failed
    exit /b 1
)

REM Activate conda env
call conda activate qwen-image-edit-xpu
if errorlevel 1 (
    echo ERROR: conda activate failed — expected env name 'qwen-image-edit-xpu'
    exit /b 1
)

set QFLUX_DOTENV_LOADED=1
set PYTHONUTF8=1
set SYCL_CACHE_PERSISTENT=1

echo [quantize_te] Quantizing text_encoder: %SRC% ^→ %DST%
python templates\quantize_text_encoder_nf4.py --src "%SRC%" --dst "%DST%"

echo [quantize_te] Done. Exit code: %ERRORLEVEL%
endlocal
