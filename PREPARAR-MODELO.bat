@echo off
rem Baixa o modelo XTTS v2 pelo mecanismo OFICIAL da biblioteca TTS.
rem A biblioteca mostra a licenca (CPML, uso NAO comercial) e pergunta [y/n]. Responda VOCE.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [!] .venv nao encontrado.
  exit /b 1
)
echo Leia LICENCAS.md e o texto da CPML: https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE
".venv\Scripts\python.exe" -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')"
