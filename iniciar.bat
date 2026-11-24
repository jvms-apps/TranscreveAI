@echo off
chcp 65001 >nul
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%LOCALAPPDATA%\TranscricaoAudios\venv\Scripts\python.exe"
set "PORT=8765"

if not exist "%VENV_PYTHON%" (
    echo Ambiente local nao encontrado. Executando instalacao inicial...
    powershell -ExecutionPolicy Bypass -File "%PROJECT_DIR%instalar.ps1"
    if errorlevel 1 (
        echo Falha na instalacao. Verifique as mensagens acima.
        pause
        exit /b 1
    )
)

echo ============================================================
echo   TranscreveAI - Servidor Local
echo   Acesse: http://127.0.0.1:%PORT%
echo   Pressione CTRL+C nesta janela para encerrar.
echo ============================================================

cd /d "%PROJECT_DIR%"
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:%PORT%"
"%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%

if errorlevel 1 (
    echo.
    echo O servidor encerrou com erro.
    pause
)
