#!/usr/bin/env bash
# Baixa o modelo XTTS v2 pelo mecanismo OFICIAL da biblioteca TTS.
# A biblioteca mostra a licenca (CPML, uso NAO comercial) e pergunta [y/n] no terminal. Responda VOCE.
# Este script nao define COQUI_TOS_AGREED e nao responde a pergunta por voce.
set -eu
cd "$(dirname "$(readlink -f "$0")")"
[ -x .venv/bin/python ] || { echo "[!] .venv nao encontrado. Veja o guia de instalacao."; exit 1; }
echo "Leia LICENCAS.md e o texto da CPML: https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE"
exec .venv/bin/python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')"
