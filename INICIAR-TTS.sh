#!/usr/bin/env bash
# ============================================================
#  🎙️  VOICE FORGE STUDIO — XTTS v2   (clique pra rodar)
#  Sobe o servidor FastAPI e abre no navegador (localhost:8000)
# ============================================================
set -u
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$DIR"

# Se foi dado duplo-clique (sem terminal), reabre num terminal pra ver os logs
if [ ! -t 1 ] && [ -z "${TTS_IN_TERM:-}" ]; then
  export TTS_IN_TERM=1
  for T in konsole gnome-terminal xfce4-terminal alacritty kitty xterm; do
    if command -v "$T" >/dev/null 2>&1; then
      case "$T" in
        konsole)        exec "$T" --hold -e bash "$0" ;;
        gnome-terminal) exec "$T" -- bash "$0" ;;
        *)              exec "$T" -e bash "$0" ;;
      esac
    fi
  done
fi

# Se o servidor já estiver rodando, só abre o navegador (não sobe outro)
if curl -s -o /dev/null http://localhost:8000/ ; then
  echo "Servidor já está no ar — abrindo o navegador."
  xdg-open http://localhost:8000 >/dev/null 2>&1 &
  exit 0
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[!] Ambiente .venv nao encontrado. Rode a instalacao primeiro."
  read -r -p "Enter pra fechar..."; exit 1
fi

source .venv/bin/activate

# Licenca do modelo (CPML): este script NAO a aceita por voce.
# O aceite e feito por VOCE, uma vez, em ./PREPARAR-MODELO.sh (pergunta oficial da biblioteca TTS).
MODELO_DIR="$(python -c 'from TTS.utils.generic_utils import get_user_data_dir as g; print(g("tts"))')/tts_models--multilingual--multi-dataset--xtts_v2"
if [ ! -f "$MODELO_DIR/tos_agreed.txt" ]; then
  echo "[!] O modelo XTTS ainda nao foi baixado/aceito neste computador."
  echo "    Leia LICENCAS.md e rode primeiro:  ./PREPARAR-MODELO.sh"
  read -r -p "Enter pra fechar..."; exit 1
fi

echo "========================================================"
echo "   Iniciando Voice Forge Studio (XTTS v2) na porta 8000"
echo "   (no 1o uso baixa o modelo ~1.8GB — pode demorar)"
echo "========================================================"
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --log-level info &   # somente esta maquina (server.py sozinho escuta em 0.0.0.0)
SRV=$!

echo "Aguardando o servidor responder..."
OPENED=0
for i in $(seq 1 900); do
  if curl -s -o /dev/null http://localhost:8000/ ; then
    echo "✅ Servidor no ar: http://localhost:8000"
    xdg-open http://localhost:8000 >/dev/null 2>&1 &
    OPENED=1; break
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "[!] O servidor encerrou antes de subir — veja o erro acima."
    read -r -p "Enter pra fechar..."; exit 1
  fi
  sleep 1
done
[ "$OPENED" = 0 ] && echo "[!] Demorou demais pra subir; deixe rodando ou veja os logs."

echo "Feche esta janela (ou Ctrl+C) para PARAR o servidor."
wait "$SRV"
