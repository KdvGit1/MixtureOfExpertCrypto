@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title MarketMoE - Veri, CUDA Egitim ve Backtest

echo ================================================================
echo MarketMoE tam CUDA arastirma akisi
echo Ucretsiz veri ^> egitim ^> kilitli test ^> maliyetli backtest
echo Gercek emir GONDERILMEZ ve model otomatik production'a ALINMAZ.
echo ================================================================
echo.

if not defined MARKET_MOE_SKIP_SETUP (
    echo [1/4] Python ortami ve proje bagimliliklari hazirlaniyor...
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
    if errorlevel 1 goto :failure
) else (
    echo [1/4] Kurulum MARKET_MOE_SKIP_SETUP nedeniyle atlandi.
)

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo HATA: .venv Python bulunamadi: "%PYTHON_EXE%"
    goto :failure
)

if not defined MARKET_MOE_TORCH_INDEX_URL set "MARKET_MOE_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
if not defined MARKET_MOE_SKIP_CUDA_INSTALL (
    echo [2/4] NVIDIA CUDA ve PyTorch kontrol ediliyor...
    where nvidia-smi.exe >nul 2>nul
    if errorlevel 1 (
        echo HATA: nvidia-smi bulunamadi. NVIDIA surucusunu kurup bilgisayari yeniden baslatin.
        goto :failure
    )
    "%PYTHON_EXE%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo CUDA destekli PyTorch kuruluyor. Bu indirme birkac GB olabilir...
        "%PYTHON_EXE%" -m pip install --upgrade --force-reinstall --no-deps torch --index-url "%MARKET_MOE_TORCH_INDEX_URL%"
        if errorlevel 1 goto :failure
    )
    "%PYTHON_EXE%" -c "import torch,sys; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK'); sys.exit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo HATA: PyTorch CUDA GPU'yu goremedi. NVIDIA surucusunu ve CUDA wheel adresini kontrol edin.
        goto :failure
    )
) else (
    echo [2/4] CUDA kurulumu MARKET_MOE_SKIP_CUDA_INSTALL nedeniyle atlandi.
)

echo [3/4] Yapilandirma dogrulaniyor...
"%PYTHON_EXE%" "%~dp0scripts\windows_full_pipeline.py" --config "%~dp0configs\pipelines\windows_cuda_12gb.yaml" --check
if errorlevel 1 goto :failure

echo [4/4] Veri indirme, egitim ve backtest baslatiliyor...
echo Kesinti olursa ayni BAT dosyasini yeniden calistirin; checkpoint'ten devam eder.
echo.
"%PYTHON_EXE%" "%~dp0scripts\windows_full_pipeline.py" --config "%~dp0configs\pipelines\windows_cuda_12gb.yaml" %*
set "PIPELINE_EXIT=%ERRORLEVEL%"
if not "%PIPELINE_EXIT%"=="0" goto :pipeline_failure

echo.
echo BASARILI: Tum etkin egitim ve backtest isleri tamamlandi.
echo Ozet: "%~dp0artifacts\pipeline_runs\windows_cuda_full\summary.html"
echo Log:  "%~dp0artifacts\pipeline_runs\windows_cuda_full\pipeline.log"
goto :finish

:pipeline_failure
echo.
echo UYARI: Akis bir veya daha fazla hata ile tamamlandi. Cikis kodu: %PIPELINE_EXIT%
echo Ayrinti: "%~dp0artifacts\pipeline_runs\windows_cuda_full\pipeline.log"
echo Sorunu giderdikten sonra ayni BAT dosyasini yeniden calistirarak devam edebilirsiniz.
goto :finish_with_error

:failure
set "PIPELINE_EXIT=1"
echo.
echo HATA: Hazirlik tamamlanamadi. Yukaridaki mesaji inceleyin.
goto :finish_with_error

:finish
if not defined MARKET_MOE_NO_PAUSE pause
exit /b 0

:finish_with_error
if not defined MARKET_MOE_NO_PAUSE pause
exit /b %PIPELINE_EXIT%
