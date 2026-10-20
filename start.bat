@echo off
title VOICE FORGE STUDIO - Unificado
rem Antes do primeiro uso rode PREPARAR-MODELO.bat (aceite da licenca do modelo feito por VOCE).
cd /d "%~dp0"

echo ========================================================
echo        🎙️  VOICE FORGE STUDIO - XTTS v2 Lab
echo ========================================================
echo [1/2] Ativando ambiente virtual...
if not exist ".venv\Scripts\activate.bat" (
    echo [!] Erro: Ambiente .venv nao encontrado!
    pause
    exit
)
call .venv\Scripts\activate

echo [2/2] Iniciando servidor FastAPI (Porta 8000)...
echo Interface disponivel em: http://localhost:8000
echo.
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --log-level info

pause
