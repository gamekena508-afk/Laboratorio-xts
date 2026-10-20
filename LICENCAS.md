# Licenças e avisos

Isto não é aconselhamento jurídico. São três coisas diferentes:

| O quê | Licença |
|---|---|
| **Código deste repositório** (`server.py`, `app.html`, scripts) | MIT. Veja [`LICENSE`](LICENSE). |
| **Pesos do modelo XTTS v2** (não estão aqui; você baixa) | Coqui Public Model License 1.0.0 (CPML): **somente uso não comercial**. Texto: <https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE> |
| **Áudio gerado pelo modelo** | A CPML também cobre o resultado gerado: sem uso comercial. Quem receber o áudio deve receber os termos (ou a URL). Vozes de terceiros: você só pode clonar vozes que tem direito de usar. |

A licença MIT do código **não muda** a licença do modelo. Usar este laboratório com o XTTS v2 continua sendo uso não comercial.

## Aceite da licença do modelo

O projeto **não aceita** a CPML por você: não define `COQUI_TOS_AGREED`, não cria `tos_agreed.txt`, não responde `y`. O `PREPARAR-MODELO` só dispara a pergunta oficial da biblioteca TTS e **você** responde.

## Componentes que você instala (licenças segundo os metadados de cada projeto)

TTS 0.22.0 (MPL-2.0) · transformers (Apache-2.0) · fastapi e pydantic (MIT) · uvicorn (BSD-3) · librosa (ISC) · torch (BSD-style) · yt-dlp (Unlicense) · silero-vad (MIT; baixado via `torch.hub`). A licença de `audio-separator` e dos modelos que ele baixa **não foi verificada**; confira na fonte antes de redistribuir. O `ffmpeg` tem licença própria.

## Atribuição

XTTS © Coqui. Silero VAD © Silero. Respeite os termos de cada projeto citado.
