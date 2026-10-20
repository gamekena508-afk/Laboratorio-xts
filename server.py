# -*- coding: utf-8 -*-
"""
VOICE FORGE STUDIO — Servidor Unificado v1.0
=============================================
✅ Patch PyTorch 2.6+ (safe_globals)
✅ Cache de Speaker Latents (RAM + disco)
✅ Síntese direta sem I/O de disco
✅ SLERP para speaker_embedding (fusão correta)
✅ LERP para gpt_cond_latent (prosódia)
✅ Dual Engine alternado + semáforo GPU
✅ Biometria vocal (registro + identificação via WAV)
✅ Conversor de áudio via ffmpeg
✅ Upload de novas vozes
✅ Serve app.html na raiz
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import io
import json
import pickle
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, cast

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse, StreamingResponse
from pydantic import BaseModel

from separador_engine import safe_audio_name, separate_audio_stems

if TYPE_CHECKING:
    from TTS.api import TTS


class XttsModelLike(Protocol):
    def get_conditioning_latents(self, audio_path: Any, **kwargs: Any) -> Tuple[Any, Any]:
        ...

    def inference(self, text: str, language: str, gpt_cond_latent: Any,
                  speaker_embedding: Any, **kwargs: Any) -> Dict[str, Any]:
        ...

    def inference_stream(self, text: str, language: str, gpt_cond_latent: Any,
                         speaker_embedding: Any, **kwargs: Any) -> Any:
        ...

    def eval(self) -> Any:
        ...

    def parameters(self) -> Any:
        ...


# ─── PATCH PyTorch 2.6+ ─────────────────────────────────────────────────────
def _patch_torch_safe_globals():
    # Força fallback weights_only=False logo de cara, sem depender de add_safe_globals
    _orig = torch.load
    def _patched(f, map_location=None, pickle_module=None, **kw):
        kw.pop("weights_only", None)
        if pickle_module is not None:
            return _orig(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kw)
        return _orig(f, map_location=map_location, weights_only=False, **kw)
    torch.load = _patched
    try:
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            from TTS.tts.configs.shared_configs import CharactersConfig
            add_safe_globals([
                XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig, CharactersConfig
            ])
            print("[OK] Patch PyTorch 2.6+ aplicado (safe_globals + fallback)")
        else:
            print("[OK] PyTorch < 2.6; fallback weights_only=False aplicado")
    except Exception as e:
        print(f"[OK] Fallback weights_only=False ativo ({e})")

# ─────────────────────────────────────────────────────────────────────────────

# ─── Caminhos ────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
VOICES_DIR        = BASE_DIR / "reference_voices"
FUSED_DIR         = VOICES_DIR / "fused_embeddings"
VOICE_DB_DIR      = BASE_DIR / "voice_database"
CACHE_DIR         = BASE_DIR / "cache_embeddings"
OUTPUTS_DIR       = BASE_DIR / "outputs"
AUDIO_TREATMENT_DIR = OUTPUTS_DIR / "tratamento_audio"
VOICE_OR_SINGING_DIR = AUDIO_TREATMENT_DIR / "voz_ou_canto"
BACKGROUND_OR_TRACK_DIR = AUDIO_TREATMENT_DIR / "fundo_ou_trilha"
APP_HTML          = BASE_DIR / "app.html"
ENGINE_CONFIG     = BASE_DIR / "engine_config.json"
XTTS_MODEL_NAME   = "tts_models/multilingual/multi-dataset/xtts_v2"
MIN_ENGINE_COUNT  = 1
MAX_ENGINE_COUNT  = 5

for d in [VOICES_DIR, FUSED_DIR, VOICE_DB_DIR, CACHE_DIR, OUTPUTS_DIR,
          AUDIO_TREATMENT_DIR, VOICE_OR_SINGING_DIR, BACKGROUND_OR_TRACK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Estado global ───────────────────────────────────────────────────────────
tts_engine: Optional[TTS] = None
tts_engines: List[TTS] = []
engine_locks: List[asyncio.Lock] = []
engine_stats: List[Dict[str, Any]] = []
engine_rr_index = 0
executor = ThreadPoolExecutor(max_workers=8)
speaker_cache: Dict[str, Tuple] = {}   # hash → (gpt_cond_latent, speaker_embedding)


def require_tts_engine() -> TTS:
    if not tts_engines:
        raise HTTPException(503, "Engine não carregado")
    return tts_engines[0]


def clamp_engine_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = MIN_ENGINE_COUNT
    return max(MIN_ENGINE_COUNT, min(MAX_ENGINE_COUNT, count))


def load_engine_config() -> Dict[str, Any]:
    default = {"engine_count": MIN_ENGINE_COUNT}
    if not ENGINE_CONFIG.exists():
        return default
    try:
        data = json.loads(ENGINE_CONFIG.read_text(encoding="utf-8"))
        data["engine_count"] = clamp_engine_count(data.get("engine_count"))
        return data
    except Exception:
        return default


def save_engine_config(engine_count: int) -> Dict[str, Any]:
    data = {"engine_count": clamp_engine_count(engine_count)}
    ENGINE_CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def get_vram_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "engines": len(tts_engines),
            "engine_count_config": load_engine_config()["engine_count"],
        }
    props = torch.cuda.get_device_properties(0)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "engines": len(tts_engines),
        "engine_count_config": load_engine_config()["engine_count"],
        "allocated_gb": round(allocated / 1024**3, 2),
        "reserved_gb": round(reserved / 1024**3, 2),
        "total_gb": round(props.total_memory / 1024**3, 1),
        "free_estimated_gb": round((props.total_memory - reserved) / 1024**3, 2),
        "per_engine_estimated_gb": round((allocated / max(1, len(tts_engines))) / 1024**3, 2),
        "slots": engine_stats,
    }


def next_engine_slot() -> Tuple[int, TTS, asyncio.Lock]:
    global engine_rr_index
    if not tts_engines:
        raise HTTPException(503, "Nenhum engine XTTS carregado")
    idx = engine_rr_index % len(tts_engines)
    engine_rr_index += 1
    return idx, tts_engines[idx], engine_locks[idx]


def get_xtts_model(engine: TTS) -> XttsModelLike:
    synthesizer = getattr(engine, "synthesizer", None)
    model = getattr(synthesizer, "tts_model", None)
    if model is None:
        raise RuntimeError("Modelo XTTS não disponível no engine")
    return cast(XttsModelLike, model)

# ─── Modelos Pydantic ────────────────────────────────────────────────────────
class SynthRequest(BaseModel):
    text: str
    language: str = "pt"
    speaker_wav: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stream_chunk_size: Optional[int] = None
    overlap_wav_len: Optional[int] = None


def _clamp_float(value: Optional[float], default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    return max(minimum, min(maximum, float(value)))


def _clamp_int(value: Optional[int], default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))

class SynthFusedRequest(BaseModel):
    text: str
    language: str = "pt"
    embedding_name: str        # nome do .pkl em fused_embeddings/

class FuseRequest(BaseModel):
    voice_a: str               # nome do .wav em reference_voices/
    voice_b: str
    alpha: float = 0.5         # 1.0 = 100% voz A, 0.0 = 100% voz B
    name: str
    preview_text: str = "Olá, esta é minha nova voz criada por fusão neural."

class EngineConfigRequest(BaseModel):
    engine_count: int

class VoiceFuseTripleRequest(BaseModel):
    name: str
    voice_a: str
    voice_b: str
    voice_c: str
    weight_a: float = 0.33
    weight_b: float = 0.33
    weight_c: float = 0.34
    preview_text: str = "Olá, esta é minha nova voz criada por fusão tripla."

class VoiceFuseManyRequest(BaseModel):
    name: str
    voices: List[str]
    weights: List[float]
    preview_text: str = "Olá, esta é minha nova voz criada por fusão neural."
    preview_language: str = "pt"

class VoiceGenerateRequest(BaseModel):
    """Cria uma voz sintetica a partir de parametros - SEM audio de exemplo."""
    name: str                          # nome do .pkl a salvar
    gender: float = 0.5               # 0.0 = feminino maximo, 1.0 = masculino maximo
    age: float = 0.5                  # 0.0 = jovem, 1.0 = velho
    pitch: float = 0.5                # 0.0 = fino/agudo, 1.0 = grave
    expressiveness: float = 0.5       # 0.0 = neutro, 1.0 = expressivo
    style_intensity: float = 0.0      # NOVO: Textura/Intensidade de Estilo
    speed: float = 0.5                # 0.0 = lento, 1.0 = rapido
    seed: int = 42                    # NOVO: DNA Vocal Unico
    preview_text: str = "Ola, esta e minha nova voz criada diretamente da matriz neural."
    preview_language: str = "pt"

class BiometryRegisterRequest(BaseModel):
    user_name: str

class YTDownloadRequest(BaseModel):
    url: str
    name: str = "minha_voz"
    max_duration: int = 60
    target_voice: str | None = None
    similarity_threshold: float = 0.65
    separate_audio: bool = False


def _youtube_download_with_fallback(
    url: str,
    ydl_opts: Dict[str, Any],
    *,
    extract_info: bool = False,
) -> Any:
    """Baixa mídia do YouTube com fallback de sessão para bloqueios HTTP 403."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp não está instalado") from exc

    attempts: List[Tuple[str, Optional[Tuple[str, ...]]]] = [
        ("anonimo", None),
        ("chrome", ("chrome",)),
        ("firefox", ("firefox",)),
    ]
    last_error: Optional[Exception] = None

    for label, cookies_from_browser in attempts:
        opts = dict(ydl_opts)
        opts.setdefault("noplaylist", True)
        opts.setdefault("retries", 3)
        opts.setdefault("fragment_retries", 3)
        opts.setdefault("extractor_retries", 3)
        opts.setdefault("socket_timeout", 20)
        opts.setdefault("concurrent_fragment_downloads", 1)
        if cookies_from_browser is not None:
            opts["cookiesfrombrowser"] = cookies_from_browser

        try:
            with yt_dlp.YoutubeDL(cast(dict, opts)) as ydl:  # type: ignore
                if extract_info:
                    return ydl.extract_info(url, download=True)
                ydl.download([url])
                return None
        except Exception as exc:
            last_error = exc
            message = str(exc)
            retryable = (
                "403" in message
                or "Forbidden" in message
                or "Sign in" in message
                or "bot" in message.lower()
                or "cookies" in message.lower()
            )
            if not retryable:
                raise
            print(f"[YouTube] tentativa {label} bloqueada; tentando fallback de sessão")

    if last_error is not None:
        raise last_error
    raise RuntimeError("Falha desconhecida ao baixar mídia do YouTube")


# ─── Utilitários de texto ────────────────────────────────────────────────────
# ─── Normalização de números, horas e unidades ───────────────────────────────
# Medido em 2026-08-09: o XTTS engolia trechos com número cru. Em
# "Percorri 45 km em 2h30 a 90 km/h" o pedaço "em 2h30" simplesmente NÃO era
# falado, e "1.500 reais" saía como "R$ 1.500" com "gastei" virando "vestei".
# Escrever por extenso resolveu nos testes. Siglas (VRAM, TTS, GPU) NÃO são
# soletradas: soletrar com hífen piorou a confiança de 0,88 para 0,69.
_UNIDADES = {
    "km/h": "quilômetros por hora", "km": "quilômetros", "cm": "centímetros",
    "mm": "milímetros", "kg": "quilos", "mb": "megabytes", "gb": "gigabytes",
    "tb": "terabytes", "kb": "kilobytes", "ms": "milissegundos",
    "khz": "quilohertz", "hz": "hertz",
    # area e volume: sem isso, "480 m²" saia falado como "MIE" (varredura
    # grande, hermes13). O expoente aceita tanto o caractere quanto o digito.
    "km²": "quilômetros quadrados", "km2": "quilômetros quadrados",
    "m²": "metros quadrados", "m2": "metros quadrados",
    "cm²": "centímetros quadrados", "cm2": "centímetros quadrados",
    "m³": "metros cúbicos", "m3": "metros cúbicos",
    "m/s": "metros por segundo",
}
_UNID_SING = {
    "quilômetros": "quilômetro", "centímetros": "centímetro",
    "milímetros": "milímetro", "quilos": "quilo", "megabytes": "megabyte",
    "gigabytes": "gigabyte", "terabytes": "terabyte", "kilobytes": "kilobyte",
    "milissegundos": "milissegundo",
    "quilômetros por hora": "quilômetro por hora",
    "quilômetros quadrados": "quilômetro quadrado",
    "metros quadrados": "metro quadrado",
    "centímetros quadrados": "centímetro quadrado",
    "metros cúbicos": "metro cúbico",
    "metros por segundo": "metro por segundo",
}


def _num_valor(s: str):
    """'1.500' -> 1500 · '2,5' -> 2.5 · '45' -> 45 · '1.2' -> 1.2

    O ponto só é separador de milhar quando o último grupo tem exatamente 3
    dígitos. Sem essa checagem, '1.2 mil reais' virava 'doze mil reais' e o
    valor falado ficava 10x errado - medido na varredura grande (hermes10).
    """
    s = s.strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s and len(s.rsplit(".", 1)[1]) == 3:
        s = s.replace(".", "")          # 1.500 / 1.234.567 -> milhar
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return 0


def _por_extenso(n) -> str:
    try:
        from num2words import num2words
        # a vírgula do num2words vira pausa na fala; remover
        return num2words(n, lang="pt_BR").replace(",", "")
    except Exception:
        return str(n)


def _feminino(s: str) -> str:
    """'um hora' -> 'uma hora'; 'dois horas' -> 'duas horas'."""
    return re.sub(r"\bdois\b", "duas", re.sub(r"\bum\b", "uma", s))


def normalize_numbers(text: str) -> str:
    """Expande número, hora, unidade, percentual e dinheiro para extenso.
    Não soletra sigla e não força acentuação fonética."""
    if not re.search(r"\d", text):
        return text

    def _hora(m):
        h, mi = int(m.group(1)), m.group(2)
        s = f"{_feminino(_por_extenso(h))} {'hora' if h == 1 else 'horas'}"
        if mi and int(mi) > 0:
            s += f" e {_por_extenso(int(mi))}"
        return s
    # o \s* dos minutos fica DENTRO do grupo opcional, senão "14h e saiu"
    # perde o espaço e vira "catorze horase saiu"
    text = re.sub(r"\b(\d{1,2})\s*h(?:\s*(\d{2}))?\b", _hora, text, flags=re.IGNORECASE)

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%",
                  lambda m: f"{_por_extenso(_num_valor(m.group(1)))} por cento", text)
    # a escala entra no grupo: sem ela, "R$ 1.2 mil" virava "doze reais mil",
    # com o "mil" orfao depois da unidade monetaria.
    text = re.sub(r"R\$\s*([\d.,]+)(\s*(?:mil|milh(?:ão|ões)|bilh(?:ão|ões)))?",
                  lambda m: f"{_por_extenso(_num_valor(m.group(1)))}"
                            f"{m.group(2) or ''} reais", text)

    _u = "|".join(sorted((re.escape(u) for u in _UNIDADES), key=len, reverse=True))

    def _unid(m):
        v = _num_valor(m.group(1))
        nome = _UNIDADES[m.group(2).lower()]
        if v == 1:
            nome = _UNID_SING.get(nome, nome)
        return f"{_por_extenso(v)} {nome}"
    text = re.sub(rf"\b([\d.,]+)\s*({_u})\b(?![a-zA-Z])", _unid, text, flags=re.IGNORECASE)

    text = re.sub(r"\b\d[\d.,]*\b", lambda m: _por_extenso(_num_valor(m.group(0))), text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Dicionario de pronuncia.
#
# O XTTS le algumas grafias errado de forma sistematica. A troca abaixo e
# TEXTUAL e acontece antes da sintese: o usuario continua escrevendo certo, e
# o modelo recebe a grafia que ele consegue falar.
#
# REGRA PARA ENTRAR AQUI: a variante precisa ter acertado 3/3 na medicao com
# Whisper (consertar.py). Palavra que nao atingiu 3/3 fica de fora - remendo
# nao medido e pior que defeito conhecido.
#
# Medido e REPROVADO, por isso NAO esta aqui: gnose (0/12 variantes testadas),
# lhama (max 1/3), nhambu (max 1/3), SSD e HDMI (0/15). Nenhuma reescrita e
# nenhum preset de amostragem resolveu esses quatro.
# --------------------------------------------------------------------------
_PRONUNCIA = {
    # encontro consonantal inicial mn- : o modelo insere vogal ("menemonico").
    # "nemonico" e a pronuncia brasileira corrente da palavra.
    "mnemônico":  "nemônico",
    "mnemônica":  "nemônica",
    "mnemônicos": "nemônicos",
    "mnemônicas": "nemônicas",
    "mnemonico":  "nemonico",
    "mnemonica":  "nemonica",
    # xs- : o modelo parte a palavra em duas ("ex-sodato").
    "exsudato":  "exudato",
    "exsudatos": "exudatos",
    "exsudação": "exudação",
}
_PRONUNCIA_RX = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _PRONUNCIA), key=len, reverse=True)) + r")\b",
    re.IGNORECASE)


def aplica_pronuncia(text: str) -> str:
    """Troca grafias que o modelo le errado. Preserva a capitalizacao original."""
    def _sub(m):
        orig = m.group(0)
        novo = _PRONUNCIA.get(orig.lower())
        if novo is None:
            return orig
        if orig.isupper():
            return novo.upper()
        if orig[:1].isupper():
            return novo[:1].upper() + novo[1:]
        return novo
    return _PRONUNCIA_RX.sub(_sub, text)


def preprocess_text(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)
    text = text.replace("*", "").replace("...", " ").replace("..", " ")
    text = re.sub(r"^\s*(ponto|vírgula|virgula|exclamação|exclamacao|interrogação|interrogacao)\s*$", "", text, flags=re.IGNORECASE)
    text = normalize_numbers(text)
    text = aplica_pronuncia(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    pattern = r"(?<=[.!?])\s+"
    segs = [sanitize_tts_segment(s) for s in re.split(pattern, text)]
    segs = [s for s in segs if s]
    return segs or [text]

# EXPERIMENTO 2026-08-10 - TESTADO E REPROVADO, deixado como registro.
#
# Hipotese: a ULTIMA palavra da frase e fragil ("no natal." saia "Natali" em
# 4 de 6 quando vinha depois de palavra dificil). Sem pontuacao final o modelo
# nao saberia que a frase acabou.
#
# Medicao A/B, 5 condicoes x 6 repeticoes:
#   ultima palavra depois de palavra dificil : 2/6 -> 6/6   (melhorou muito)
#   palavra no meio, vizinho facil           : 6/6 -> 4/6   (piorou)
#   palavra no meio, vizinho dificil         : 6/6 -> 5/6   (piorou)
#   palavra no meio, frase simples           : 6/6 -> 3/6   (piorou, chegou a
#                                              colar palavras: "natalpassado")
#   TOTAL                                    : 26/30 -> 24/30
#
# Conserta o fim da frase e estraga o meio. Saldo negativo -> desligado.
MANTER_PONTO_FINAL = False


def sanitize_tts_segment(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[\s\.,;:!?¡¿]+", "", text)
    if MANTER_PONTO_FINAL:
        # preserva um unico . ! ? no fim; remove o resto da sujeira
        m = re.search(r"([\.!?])[\s\.,;:!?¡¿]*$", text)
        fim = m.group(1) if m else ""
        text = re.sub(r"[\s\.,;:!?¡¿]+$", "", text) + fim
    else:
        text = re.sub(r"[\s\.,;:!?¡¿]+$", "", text)
    text = re.sub(r"\s+([\.,;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─── Cache de latents ────────────────────────────────────────────────────────
def _speaker_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_latents(engine: TTS, speaker_path: str) -> Tuple:
    h = _speaker_hash(speaker_path)
    if h in speaker_cache:
        return speaker_cache[h]
    cache_file = CACHE_DIR / f"{h}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            latents = pickle.load(f)
        speaker_cache[h] = latents
        return latents
    print(f"  [LOAD] Calculando latents para {Path(speaker_path).name}...")
    model = get_xtts_model(engine)
    with torch.no_grad():
        gpt, spk = model.get_conditioning_latents(audio_path=[speaker_path])
    latents = (gpt, spk)
    speaker_cache[h] = latents
    with open(cache_file, "wb") as f:
        pickle.dump(latents, f)
    return latents

# ─── SLERP (fusão correta do speaker_embedding) ──────────────────────────────
def slerp(emb_a: torch.Tensor, emb_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolação Esférica Linear — mantém magnitude da voz."""
    a = emb_a.squeeze().float()
    b = emb_b.squeeze().float()
    a_n = a / a.norm()
    b_n = b / b.norm()
    dot = torch.clamp(torch.dot(a_n, b_n), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        return (alpha * emb_a + (1 - alpha) * emb_b)
    sin_t = torch.sin(theta)
    result = (torch.sin(alpha * theta) / sin_t) * emb_a + \
             (torch.sin((1 - alpha) * theta) / sin_t) * emb_b
    # Garantir formato [1, 512, 1] para o decodificador HiFiGAN do XTTS v2
    if result.ndim == 1:
        return result.unsqueeze(0).unsqueeze(-1)
    if result.ndim == 2:
        return result.unsqueeze(-1)
    return result

# ─── Síntese de áudio ────────────────────────────────────────────────────────
def _generate_sync(engine: TTS, text: str, latents: Tuple, language: str,
                   cancel: asyncio.Event) -> Tuple[np.ndarray, int]:
    text = sanitize_tts_segment(text)
    if not text.strip() or cancel.is_set():
        return np.array([]), 24000
    gpt_cond, spk_emb = latents
    # Ajuste de segurança: XTTS v2 HiFiGAN espera [1, 512, 1]
    if spk_emb.ndim == 2:
        spk_emb = spk_emb.unsqueeze(-1)
    elif spk_emb.ndim == 3 and spk_emb.shape[1] == 1:
        # Caso tenha vindo como [1, 1, 512], corrigimos para [1, 512, 1]
        spk_emb = spk_emb.transpose(1, 2)

    try:
        with torch.no_grad():
            out = get_xtts_model(engine).inference(
                text=text, language=language,
                gpt_cond_latent=gpt_cond, speaker_embedding=spk_emb,
                temperature=0.65, top_k=50, repetition_penalty=2.0,
            )
        wav = out["wav"]
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        return wav.flatten().astype(np.float32), 24000
    except Exception as e:
        import traceback
        print(f"  [ERROR] Erro na inferencia: {e}")
        traceback.print_exc()
        raise

def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


# limiar relativo ao pico do proprio audio. Absoluto nao serve: cada geracao
# tem volume diferente, e o ruido de fundo do XTTS nao e silencio digital.
_SIL_JANELA = 0.02      # 20 ms
_SIL_LIMIAR = 0.02      # 2% do pico
_SIL_MAX    = 1.00      # pausa acima disto e encurtada
_SIL_DEIXA  = 0.30      # sobra uma pausa natural de frase


def encurta_silencio(audio: np.ndarray, sr: int = 24000,
                     max_sil: float = _SIL_MAX,
                     deixa: float = _SIL_DEIXA) -> np.ndarray:
    """Encurta pausas internas longas, preservando as pausas naturais.

    Medido em 355 audios: 37% tinham pausa interna >= 0.5s e 31.5% de toda a
    duracao gerada era silencio interno - houve caso de 12.5s de pausa dentro
    de um audio de 18.9s.

    O que isto NAO faz: nao conserta a palavra. Pausa longa e sintoma de uma
    geracao que degradou (correlacao +0.37 com a taxa de erro); cortar o
    silencio melhora a escuta, nao a pronuncia.
    """
    a = np.asarray(audio, dtype=np.float32).flatten()
    h = max(1, int(_SIL_JANELA * sr))
    nj = len(a) // h
    if nj < 3:
        return a
    rms = np.sqrt(np.mean(a[:nj * h].reshape(nj, h) ** 2, axis=1))
    pico = float(rms.max())
    if pico <= 1e-6:
        return a
    quieto = rms < (pico * _SIL_LIMIAR)

    manter = np.ones(nj, dtype=bool)
    jan_max = int(round(max_sil / _SIL_JANELA))
    jan_deixa = int(round(deixa / _SIL_JANELA))
    ini = None
    for i in range(nj + 1):
        q = quieto[i] if i < nj else False
        if q and ini is None:
            ini = i
        elif not q and ini is not None:
            if (i - ini) > jan_max:
                # mantem metade da sobra em cada ponta: corta o miolo e
                # preserva o decaimento natural da fala nas bordas
                borda = jan_deixa // 2
                manter[ini + borda: i - borda] = False
            ini = None

    if manter.all():
        return a
    fim = nj * h
    keep = np.repeat(manter, h)
    saida = np.concatenate([a[:fim][keep], a[fim:]])
    # rampa curta nas emendas evita clique; 5 ms e inaudivel
    r = max(1, int(0.005 * sr))
    if len(saida) > 2 * r:
        saida[:r] *= np.linspace(0, 1, r, dtype=np.float32)
        saida[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return saida.astype(np.float32)


def _prepare_speaker_embedding(spk_emb: Any) -> Any:
    if spk_emb.ndim == 2:
        return spk_emb.unsqueeze(-1)
    if spk_emb.ndim == 3 and spk_emb.shape[1] == 1:
        return spk_emb.transpose(1, 2)
    return spk_emb


def _float_audio_to_pcm16(audio: Any) -> bytes:
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    wav = np.asarray(audio, dtype=np.float32).flatten()
    if wav.size == 0:
        return b""
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


async def synth_segments(latents: Tuple, text: str, language: str) -> bytes:
    clean = preprocess_text(text)
    
    # Se o texto for apenas pontuação (ex: o usuário digitou apenas "."), gerar 1 seg de silêncio
    if not re.search(r"[a-zA-Z0-9À-ÿ]", clean):
        silence = np.zeros(24000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, 24000, format="WAV")
        return buf.getvalue()

    segs = split_sentences(clean)
    cancel = asyncio.Event()
    loop = asyncio.get_event_loop()
    all_audio: List[np.ndarray] = []
    async def generate_one(index: int, segment: str) -> Tuple[int, np.ndarray]:
        engine_idx, engine, lock = next_engine_slot()
        async with lock:
            wav, _sr = await loop.run_in_executor(
                executor, _generate_sync, engine, segment, latents, language, cancel
            )
        if len(wav) > 0:
            print(f"  Seg {index+1}/{len(segs)} OK (engine {engine_idx+1})")
        return index, wav

    try:
        results = await asyncio.gather(
            *(generate_one(i, seg) for i, seg in enumerate(segs))
        )
        for _, wav in sorted(results, key=lambda item: item[0]):
            if len(wav) > 0:
                all_audio.append(wav)

        if not all_audio:
            raise ValueError("Nenhum áudio gerado")

        combined = _normalize(encurta_silencio(np.concatenate(all_audio)))
        buf = io.BytesIO()
        sf.write(buf, combined, 24000, format="WAV")
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        import traceback
        print(f"  [ERROR] Erro em synth_segments: {e}")
        traceback.print_exc()
        raise

# ─── ffmpeg ──────────────────────────────────────────────────────────────────
def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def convert_to_wav(input_path: Path, output_path: Path, sr: int = 24000) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-ar", str(sr), "-ac", "1", "-y", str(output_path)],
            capture_output=True, text=True
        )
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False

def convert_audio_format(input_path: Path, output_path: Path,
                          fmt: str = "wav", speed: float = 1.0) -> bool:
    try:
        cmd = ["ffmpeg", "-i", str(input_path), "-ar", "24000", "-ac", "1"]
        if speed != 1.0 and 0.5 <= speed <= 2.0:
            cmd += ["-filter:a", f"atempo={speed}"]
        if fmt == "mp3":
            cmd += ["-b:a", "192k"]
        elif fmt == "ogg":
            cmd += ["-c:a", "libvorbis", "-q:a", "5"]
        cmd += ["-y", str(output_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and output_path.exists()
    except Exception:
        return False

# ─── Lifespan ────────────────────────────────────────────────────────────────
def require_cuda_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU obrigatoria: CUDA nao esta disponivel para este Python/Torch. "
            "Instale uma build CUDA do PyTorch e confirme com nvidia-smi antes de iniciar."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("GPU obrigatoria: nenhuma GPU CUDA foi encontrada.")
    return torch.cuda.get_device_name(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine, tts_engines, engine_locks, engine_stats
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)

    print("\n" + "="*60)
    print("VOICE FORGE STUDIO - iniciando...")
    print("="*60)
    gpu_name = require_cuda_device()
    print(f"Dispositivo: CUDA - {gpu_name}")
    print("[LOAD] Importando Coqui TTS... isso pode levar alguns segundos.")
    _patch_torch_safe_globals()
    from TTS.api import TTS
    desired_engines = load_engine_config()["engine_count"]
    tts_engines = []
    engine_locks = []
    engine_stats = []

    for i in range(desired_engines):
        start_t = time.time()
        before = torch.cuda.memory_allocated()
        try:
            print(f"[LOAD] Carregando XTTS engine {i+1}/{desired_engines}...")
            engine = TTS(XTTS_MODEL_NAME, gpu=True, progress_bar=False)
            m = get_xtts_model(engine)
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            after = torch.cuda.memory_allocated()
            tts_engines.append(engine)
            engine_locks.append(asyncio.Lock())
            engine_stats.append({
                "index": i + 1,
                "status": "ready",
                "load_seconds": round(time.time() - start_t, 1),
                "vram_delta_gb": round((after - before) / 1024**3, 2),
            })
            print(f"[OK] Engine {i+1} pronto em {time.time()-start_t:.1f}s")
        except RuntimeError as e:
            engine_stats.append({
                "index": i + 1,
                "status": "failed",
                "error": str(e),
                "load_seconds": round(time.time() - start_t, 1),
            })
            print(f"[WARN] Engine {i+1} falhou: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            break

    if not tts_engines:
        raise RuntimeError("Nenhum engine XTTS carregou. Verifique CUDA/VRAM.")
    tts_engine = tts_engines[0]
    
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_info = f"[VRAM]: {vram_used:.2f}/{vram_total:.2f} GB"
    
    print(vram_info)
    print("="*60)
    print("[URL] Interface: http://localhost:8000...")
    print("[DOCS] API Docs:  http://localhost:8000/docs")
    print("="*60 + "\n")
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[STOP] Servidor encerrado.")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Forge Studio", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    if APP_HTML.exists():
        return HTMLResponse(content=APP_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>app.html não encontrado</h1>")

@app.get("/health")
async def health():
    gpu = get_vram_snapshot()
    return {
        "status": "ok",
        "engine_loaded": bool(tts_engines),
        "engine_count": len(tts_engines),
        "engine_count_config": load_engine_config()["engine_count"],
        "engines": engine_stats,
        "pytorch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": gpu,
        "latent_cache_entries": len(speaker_cache),
        "ffmpeg": has_ffmpeg(),
    }

@app.get("/engines/status")
async def engines_status():
    return {
        "loaded": len(tts_engines),
        "configured": load_engine_config()["engine_count"],
        "max": MAX_ENGINE_COUNT,
        "slots": engine_stats,
        "vram": get_vram_snapshot(),
    }

@app.post("/engines/config")
async def engines_config(req: EngineConfigRequest):
    data = save_engine_config(req.engine_count)
    return {
        "status": "ok",
        "configured": data["engine_count"],
        "loaded": len(tts_engines),
        "restart_required": True,
    }

@app.get("/vram")
async def vram_status():
    return get_vram_snapshot()

@app.get("/voices")
async def list_voices():
    """Lista todas as vozes WAV e embeddings fundidos disponíveis."""
    wavs = []
    for f in sorted(VOICES_DIR.glob("*.wav")):
        try:
            audio, sr = sf.read(str(f))  # type: ignore
            duration = round(len(audio) / sr, 1)
        except Exception:
            duration = 0
        wavs.append({"name": f.stem, "file": f.name,
                      "path": str(f), "duration": duration, "type": "wav"})
    fused = []
    for f in sorted(FUSED_DIR.glob("*.pkl")):
        fused.append({"name": f.stem, "file": f.name,
                      "path": str(f), "type": "fused"})
    return {"wav": wavs, "fused": fused}

@app.get("/voices/play/{type}/{name}")
async def play_voice(type: str, name: str):
    """Retorna o arquivo de áudio para preview no frontend."""
    if type == "wav":
        path = VOICES_DIR / f"{name}.wav"
    elif type == "fused":
        path = FUSED_DIR / f"{name}_preview.wav"
        if not path.exists():
            path = FUSED_DIR / f"{name}_sample.wav"
    else:
        raise HTTPException(400, "Tipo inválido")
        
    if not path.exists():
        # Retorna 1 segundo de silêncio para evitar erro no frontend
        buf = io.BytesIO()
        sf.write(buf, np.zeros(24000, dtype=np.float32), 24000, format="WAV")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="audio/wav")
        
    return FileResponse(path, media_type="audio/wav")

@app.get("/audio_treatment/play/{kind}/{name}")
async def play_audio_treatment(kind: str, name: str):
    """Retorna stems tratados para preview no frontend."""
    safe_name = safe_audio_name(Path(name).stem)
    if kind == "voz_ou_canto":
        path = VOICE_OR_SINGING_DIR / f"{safe_name}.wav"
    elif kind == "fundo_ou_trilha":
        path = BACKGROUND_OR_TRACK_DIR / f"{safe_name}.wav"
    else:
        raise HTTPException(400, "Tipo de tratamento invalido")

    if not path.exists():
        raise HTTPException(404, "Audio tratado nao encontrado")
    return Response(content=path.read_bytes(), media_type="audio/wav")

@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    """TTS: texto → áudio usando voz de referência WAV."""
    engine = require_tts_engine()
    if not req.text:
        raise HTTPException(400, "Texto vazio")
    speaker = req.speaker_wav or str(next(VOICES_DIR.glob("*.wav"), None))
    if not speaker or not Path(speaker).exists():
        raise HTTPException(404, f"Voz não encontrada: {speaker}")
    try:
        latents = get_latents(engine, speaker)
        wav_bytes = await synth_segments(latents, req.text, req.language)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/synthesize_stream")
async def synthesize_stream(req: SynthRequest):
    """TTS streaming: texto -> PCM s16le 24kHz usando XTTS ja carregado."""
    if not req.text:
        raise HTTPException(400, "Texto vazio")

    clean = preprocess_text(req.text)
    if not re.search(r"[a-zA-Z0-9À-ÿ]", clean):
        raise HTTPException(400, "Texto sem fala")

    speaker = req.speaker_wav or str(next(VOICES_DIR.glob("*.wav"), None))
    if not speaker or not Path(speaker).exists():
        raise HTTPException(404, f"Voz não encontrada: {speaker}")

    engine_idx, engine, lock = next_engine_slot()
    latents = get_latents(engine, speaker)
    temperature = _clamp_float(req.temperature, 0.65, 0.1, 1.2)
    top_k = _clamp_int(req.top_k, 50, 1, 100)
    top_p = _clamp_float(req.top_p, 0.85, 0.1, 1.0)
    repetition_penalty = _clamp_float(req.repetition_penalty, 2.0, 1.0, 10.0)
    stream_chunk_size = _clamp_int(req.stream_chunk_size, 20, 5, 80)
    overlap_wav_len = _clamp_int(req.overlap_wav_len, 1024, 256, 4096)

    async def audio_stream():
        gpt_cond, spk_emb = latents
        spk_emb = _prepare_speaker_embedding(spk_emb)
        model = get_xtts_model(engine)
        async with lock:
            try:
                for wav_chunk in model.inference_stream(
                    text=sanitize_tts_segment(clean),
                    language=req.language,
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=spk_emb,
                    stream_chunk_size=stream_chunk_size,
                    overlap_wav_len=overlap_wav_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    enable_text_splitting=True,
                ):
                    pcm = _float_audio_to_pcm16(wav_chunk)
                    if pcm:
                        yield pcm
                    await asyncio.sleep(0)
                print(f"  Stream OK (engine {engine_idx+1})")
            except AttributeError as e:
                if "_pad_token_tensor" not in str(e):
                    raise
                print("  [WARN] inference_stream incompatível com transformers; usando fallback chunked PCM")
                with torch.no_grad():
                    out = model.inference(
                        text=sanitize_tts_segment(clean),
                        language=req.language,
                        gpt_cond_latent=gpt_cond,
                        speaker_embedding=spk_emb,
                        temperature=temperature,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                pcm = _float_audio_to_pcm16(out["wav"])
                for offset in range(0, len(pcm), 8192):
                    yield pcm[offset : offset + 8192]
                    await asyncio.sleep(0)
                print(f"  Stream fallback OK (engine {engine_idx+1})")
            except Exception as e:
                print(f"  [ERROR] Erro em synthesize_stream: {e}")
                raise

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
            "X-Audio-Format": "s16le",
        },
    )

@app.post("/synthesize_fused")
async def synthesize_fused(req: SynthFusedRequest):
    """TTS: texto → áudio usando embedding fundido (chimera .pkl)."""
    require_tts_engine()
    pkl = FUSED_DIR / f"{req.embedding_name}.pkl"
    if not pkl.exists():
        raise HTTPException(404, f"Embedding não encontrado: {req.embedding_name}")
    try:
        with open(pkl, "rb") as f:
            latents = pickle.load(f)
        # Suporte a PKL salvo como (gpt, spk) ou só spk
        if not isinstance(latents, tuple):
            raise HTTPException(400, "Formato de PKL inválido")
        wav_bytes = await synth_segments(latents, req.text, req.language)
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/voices/fuse")
async def fuse_voices(req: FuseRequest):
    """Funde duas vozes usando SLERP (speaker_embedding) + LERP (gpt_cond_latent)."""
    engine = require_tts_engine()
    wav_a = VOICES_DIR / req.voice_a
    wav_b = VOICES_DIR / req.voice_b
    if not wav_a.exists():
        raise HTTPException(404, f"Voz A não encontrada: {req.voice_a}")
    if not wav_b.exists():
        raise HTTPException(404, f"Voz B não encontrada: {req.voice_b}")
    alpha = max(0.0, min(1.0, req.alpha))
    try:
        gpt_a, spk_a = get_latents(engine, str(wav_a))
        gpt_b, spk_b = get_latents(engine, str(wav_b))
        # Normalização de segurança para evitar "inchaço" de norm em fusões sucessivas
        gpt_fused = (alpha * gpt_a + (1 - alpha) * gpt_b)
        gpt_fused = gpt_fused / gpt_fused.norm() if gpt_fused.norm() > 0 else gpt_fused
        
        # SLERP para identidade (speaker_embedding)
        spk_fused = slerp(spk_a, spk_b, alpha)
        spk_fused = spk_fused / spk_fused.norm() if spk_fused.norm() > 0 else spk_fused
        latents = (gpt_fused, spk_fused)
        FUSED_DIR.mkdir(parents=True, exist_ok=True)
        out_pkl = FUSED_DIR / f"{req.name}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(latents, f)
        # Gerar preview WAV
        preview_text = req.preview_text
        wav_bytes = await synth_segments(latents, preview_text, "pt")
        preview_path = FUSED_DIR / f"{req.name}_preview.wav"
        preview_path.write_bytes(wav_bytes)
        return {"status": "ok", "name": req.name,
                "pkl": str(out_pkl), "preview": str(preview_path),
                "method": "SLERP(speaker) + LERP(gpt)", "alpha": alpha}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/voices/fuse_triple")
async def fuse_triple_voices(req: VoiceFuseTripleRequest):
    """Funde 3 vozes com pesos customizados (A, B, C)."""
    engine = require_tts_engine()
    
    try:
        # 1. Carregar latentes das 3 vozes
        path_a = VOICES_DIR / req.voice_a
        path_b = VOICES_DIR / req.voice_b
        path_c = VOICES_DIR / req.voice_c

        for p, n in [(path_a, req.voice_a), (path_b, req.voice_b), (path_c, req.voice_c)]:
            if not p.exists():
                raise HTTPException(404, f"Voz não encontrada: {n}")

        g_a, s_a = get_latents(engine, str(path_a))
        g_b, s_b = get_latents(engine, str(path_b))
        g_c, s_c = get_latents(engine, str(path_c))
        
        # 2. Normalizar pesos para garantir que somem 1.0
        total_w = req.weight_a + req.weight_b + req.weight_c
        wa = req.weight_a / total_w
        wb = req.weight_b / total_w
        wc = req.weight_c / total_w
        
        # 3. Fusão Linear (Matriz DNA)
        # Speaker Embedding
        fused_spk = wa * s_a + wb * s_b + wc * s_c
        fused_spk = fused_spk / fused_spk.norm()
        # Garantir formato [1, 512, 1]
        if fused_spk.ndim == 1:
            fused_spk = fused_spk.unsqueeze(0).unsqueeze(-1)
        
        # GPT Latent (Estilo de fala)
        fused_gpt = wa * g_a + wb * g_b + wc * g_c
        
        latents = (fused_gpt.cpu(), fused_spk.cpu())
        
        # 4. Salvar .pkl
        FUSED_DIR.mkdir(parents=True, exist_ok=True)
        out_pkl = FUSED_DIR / f"{req.name}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(latents, f)
            
        # 5. Gerar preview
        wav_bytes = await synth_segments(latents, req.preview_text, "pt")
        preview_path = FUSED_DIR / f"{req.name}_preview.wav"
        preview_path.write_bytes(wav_bytes)
        
        return {
            "status": "ok", 
            "name": req.name,
            "pkl": str(out_pkl), 
            "preview": str(preview_path),
            "weights": {"a": wa, "b": wb, "c": wc}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

# ─── Geração de voz sintética SEM referência ────────────────────────────────

@app.post("/voices/fuse_many")
async def fuse_many_voices(req: VoiceFuseManyRequest):
    """Funde N vozes com pesos customizados."""
    engine = require_tts_engine()
    voices = [v.strip() for v in req.voices if v and v.strip()]
    if len(voices) < 2:
        raise HTTPException(400, "Selecione pelo menos 2 vozes")
    if len(req.weights) != len(req.voices):
        raise HTTPException(400, "Quantidade de pesos diferente da quantidade de vozes")

    weights = [max(0.0, float(w)) for w in req.weights]
    total_w = sum(weights)
    if total_w <= 0:
        raise HTTPException(400, "A soma dos pesos precisa ser maior que zero")
    weights = [w / total_w for w in weights]

    try:
        gpt_fused = None
        spk_fused = None
        normalized_weights: Dict[str, float] = {}

        for voice_name, weight in zip(req.voices, weights):
            file_name = voice_name if voice_name.endswith(".wav") else f"{voice_name}.wav"
            path = VOICES_DIR / file_name
            if not path.exists():
                raise HTTPException(404, f"Voz não encontrada: {file_name}")
            gpt, spk = get_latents(engine, str(path))
            gpt_fused = (weight * gpt) if gpt_fused is None else gpt_fused + weight * gpt
            spk_fused = (weight * spk) if spk_fused is None else spk_fused + weight * spk
            normalized_weights[Path(file_name).stem] = round(weight, 4)

        if gpt_fused is None or spk_fused is None:
            raise HTTPException(400, "Nenhuma voz válida para fusão")

        if spk_fused.ndim == 1:
            spk_fused = spk_fused.unsqueeze(0).unsqueeze(-1)
        elif spk_fused.ndim == 2:
            spk_fused = spk_fused.unsqueeze(-1)
        norm = spk_fused.norm()
        if norm > 0:
            spk_fused = spk_fused / norm

        latents = (gpt_fused, spk_fused)
        FUSED_DIR.mkdir(parents=True, exist_ok=True)
        out_pkl = FUSED_DIR / f"{req.name}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(latents, f)

        wav_bytes = await synth_segments(latents, req.preview_text, req.preview_language)
        preview_path = FUSED_DIR / f"{req.name}_preview.wav"
        preview_path.write_bytes(wav_bytes)

        return {
            "status": "ok",
            "name": req.name,
            "pkl": str(out_pkl),
            "preview": str(preview_path),
            "weights": normalized_weights,
            "voice_count": len(voices),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

def _build_synthetic_voice(
    engine: TTS,
    gender: float,
    age: float,
    pitch: float,
    expressiveness: float,
    style_intensity: float,
    speed: float,
    seed: int = 42
) -> Tuple:
    """
    Gera DNA Vocal Unico:
    1. Base Bilinear (Genero x Idade)
    2. DNA Offset (Baseado na Seed)
    3. Style Injection (Texturas Intensas/Suaves)
    """
    def get_safe_latents(name, default_seed):
        try:
            p = VOICES_DIR / f"{name}.wav" if name else None
            if p is not None and p.exists():
                g, s = get_latents(engine, str(p))
                return g.cpu(), s.squeeze().cpu()
        except Exception:
            pass
        gen = torch.Generator().manual_seed(default_seed)
        return torch.randn(1, 32, 1024, generator=gen), torch.randn(512, generator=gen)

    # --- ANCORAS BASE ---
    # This synthetic corner uses seeded noise, without a bundled speaker identity.
    g_FY, s_FY = get_safe_latents(None, 101)
    g_FO, s_FO = get_safe_latents("celiav2_fixed", 102)
    g_MY, s_MY = get_safe_latents("voice_preview_lucas - engaging, captivating and kind_converted", 103)
    g_MO, s_MO = get_safe_latents("voice_preview_paulo becker - strong and sentimental_converted", 104)
    
    # --- ANCORAS DE ESTILO (TEXTURA) ---
    # Usamos Blondie para intensidade e Minerva para algo mais suave/noturno se existirem
    g_ST1, s_ST1 = get_safe_latents("voice_preview_blondie - intense woman", 201)
    g_ST2, s_ST2 = get_safe_latents("voice_preview_minerva nightshade", 202)

    # 1. Mistura Base (Genero x Idade)
    w_FY = (1-gender)*(1-age)
    w_FO = (1-gender)*age
    w_MY = gender*(1-age)
    w_MO = gender*age
    
    spk = w_FY*s_FY + w_FO*s_FO + w_MY*s_MY + w_MO*s_MO
    gpt = w_FY*g_FY + w_FO*g_FO + w_MY*g_MY + w_MO*g_MO

    # 2. DNA VOCAL (A Semente cria um desvio unico no universo de vozes)
    dna_gen = torch.Generator().manual_seed(seed)
    dna_offset_spk = torch.randn(512, generator=dna_gen) * 0.15
    dna_offset_gpt = torch.randn(1, 32, 1024, generator=dna_gen) * 0.05
    
    spk = spk + dna_offset_spk
    gpt = gpt + dna_offset_gpt

    # 3. Injecao de Estilo/Textura
    if style_intensity > 0:
        # Mistura o estilo escolhido (Blondie se Fem, talvez algo Masc no futuro)
        style_weight = style_intensity * 0.3
        spk = (1-style_weight)*spk + style_weight*s_ST1
        gpt = (1-style_weight)*gpt + style_weight*g_ST1

    # 4. Ajustes Finais (Pitch e Expressividade)
    pitch_mod = (pitch - 0.5) * 1.2
    spk[::2] += pitch_mod 
    
    exp_gen = torch.Generator().manual_seed(int(expressiveness * 10000))
    spk = spk + torch.randn(512, generator=exp_gen) * (expressiveness * 0.4)

    # Normalizar
    spk = spk / spk.norm()
    spk = spk.unsqueeze(0).unsqueeze(-1).float()
    gpt = gpt.float()

    return (gpt, spk)


@app.post("/voices/generate")
async def generate_voice(req: VoiceGenerateRequest):
    """Cria uma voz sintética totalmente nova SEM referência (manipulação direta do espaço latente)."""
    engine = require_tts_engine()

    # Clamp dos parametros
    gender          = max(0.0, min(1.0, req.gender))
    age             = max(0.0, min(1.0, req.age))
    pitch           = max(0.0, min(1.0, req.pitch))
    expressiveness  = max(0.0, min(1.0, req.expressiveness))
    style_intensity = max(0.0, min(1.0, req.style_intensity))
    speed           = max(0.0, min(1.0, req.speed))
    seed            = req.seed

    try:
        model = get_xtts_model(engine)
        loop  = asyncio.get_event_loop()

        latents = await loop.run_in_executor(
            executor,
            _build_synthetic_voice,
            engine, gender, age, pitch, expressiveness, style_intensity, speed, seed
        )

        # Mover para o device correto
        device = next(model.parameters()).device
        gpt_cond, spk_emb = latents
        gpt_cond = gpt_cond.to(device)
        spk_emb  = spk_emb.to(device)
        latents  = (gpt_cond, spk_emb)

        # Salvar .pkl
        FUSED_DIR.mkdir(parents=True, exist_ok=True)
        out_pkl = FUSED_DIR / f"{req.name}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(latents, f)

        # Gerar preview
        wav_bytes = await synth_segments(latents, req.preview_text, req.preview_language)
        preview_path = FUSED_DIR / f"{req.name}_preview.wav"
        preview_path.write_bytes(wav_bytes)

        return {
            "status": "ok",
            "name": req.name,
            "pkl": str(out_pkl),
            "preview": str(preview_path),
            "params": {
                "gender": gender, "age": age, "pitch": pitch,
                "expressiveness": expressiveness, "style_intensity": style_intensity,
                "speed": speed, "seed": seed
            },
            "method": "synthetic_latent_interpolation"
        }
    except Exception as e:
        import traceback
        print(f"  [ERROR] Erro em generate_voice: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/voices/upload")
async def upload_voice(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    separate_audio: bool = Form(False)
):
    """Upload de voz (qualquer formato) → converte para WAV 24kHz mono e salva."""
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    suffix = Path(file.filename).suffix.lower()
    allowed = {".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wav", ".mp4", ".mkv", ".webm", ".avi", ".mov"}
    if suffix not in allowed:
        raise HTTPException(400, f"Formato não suportado: {suffix}")
    content = await file.read()
    stem = safe_audio_name(name if name else Path(file.filename).stem)
    tmp_in = Path(tempfile.mktemp(suffix=suffix))
    tmp_in.write_bytes(content)
    out_path = VOICES_DIR / f"{stem}.wav"
    treatment = None
    if separate_audio:
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        ok = convert_to_wav(tmp_in, tmp_wav)
        if ok:
            try:
                treatment = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    separate_audio_stems,
                    tmp_wav,
                    VOICE_OR_SINGING_DIR,
                    BACKGROUND_OR_TRACK_DIR,
                    stem,
                )
                ok = convert_to_wav(treatment.voice_or_singing_path, out_path)
            except Exception as e:
                tmp_wav.unlink(missing_ok=True)
                tmp_in.unlink(missing_ok=True)
                raise HTTPException(500, f"Falha na separacao do audio: {e}")
        tmp_wav.unlink(missing_ok=True)
    else:
        ok = convert_to_wav(tmp_in, out_path)
    tmp_in.unlink(missing_ok=True)
    if not ok:
        raise HTTPException(500, "Falha na conversão do áudio")
    response: Dict[str, Any] = {"status": "ok", "name": stem, "path": str(out_path)}
    if treatment is not None:
        response.update({
            "voice_or_singing_path": str(treatment.voice_or_singing_path),
            "background_or_track_path": str(treatment.background_or_track_path),
            "used_audio_treatment": True,
        })
    return response

@app.post("/voices/download_yt")
async def download_yt_voice(req: YTDownloadRequest):
    """Baixa um áudio do YouTube, remove ruídos (Silero VAD) e salva como WAV referencial."""
    engine = require_tts_engine()
    
    url = req.url
    name = safe_audio_name(req.name)
    max_duration = req.max_duration

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        ydl_opts: dict = {
            'format': 'bestaudio/best',
            'outtmpl': str(tmp_dir_path / 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
        }
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                lambda: _youtube_download_with_fallback(url, ydl_opts),
            )
            
            tmp_yt = tmp_dir_path / 'audio.wav'
            if not tmp_yt.exists():
                raise HTTPException(500, "Download falhou, áudio não encontrado.")
                
            treatment = None
            vad_source = tmp_yt
            if req.separate_audio:
                try:
                    treatment = await loop.run_in_executor(
                        executor,
                        separate_audio_stems,
                        tmp_yt,
                        VOICE_OR_SINGING_DIR,
                        BACKGROUND_OR_TRACK_DIR,
                        name,
                    )
                    vad_source = treatment.voice_or_singing_path
                except Exception as e:
                    raise HTTPException(500, f"Falha na separacao do audio: {e}")

            _model_vad_result = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad', force_reload=False)
            model_vad = _model_vad_result[0] if isinstance(_model_vad_result, tuple) else _model_vad_result
            utils_vad = _model_vad_result[1] if isinstance(_model_vad_result, tuple) and len(_model_vad_result) > 1 else _model_vad_result
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils_vad  # type: ignore

            tmp_16k = tmp_dir_path / "16k.wav"
            convert_to_wav(vad_source, tmp_16k, sr=16000)

            wav_16k = read_audio(str(tmp_16k), sampling_rate=16000)
            speech_timestamps = get_speech_timestamps(wav_16k, model_vad, sampling_rate=16000)
            
            if not speech_timestamps:
                 raise HTTPException(400, "Nenhuma voz detectada no áudio (Silero VAD retornou vazio).")

            if req.target_voice:
                target_spk = None
                # 1. Tentar WAV em reference_voices/
                ref_path = VOICES_DIR / req.target_voice
                if ref_path.exists():
                    _, target_spk = get_latents(engine, str(ref_path))
                else:
                    # 2. Tentar PKL em voice_database/
                    pkl_name = req.target_voice if req.target_voice.endswith(".pkl") else f"{req.target_voice}.pkl"
                    pkl_path = VOICE_DB_DIR / pkl_name
                    if pkl_path.exists():
                        with open(pkl_path, "rb") as f:
                            target_spk = pickle.load(f)
                
                if target_spk is not None:
                    # Garantir que target_spk seja [1, 512] para similaridade
                    if target_spk.ndim == 3:
                        target_spk = target_spk.squeeze().unsqueeze(0)
                    elif target_spk.ndim == 1:
                        target_spk = target_spk.unsqueeze(0)

                    filtered_timestamps = []
                    for i, ts in enumerate(speech_timestamps):
                        chunk = wav_16k[ts['start']:ts['end']]
                        if len(chunk) < 16000 * 0.8:
                            continue
                        
                        chunk_path = tmp_dir_path / f"chunk_{i}.wav"
                        save_audio(str(chunk_path), chunk, sampling_rate=16000)
                        chunk_24k = tmp_dir_path / f"chunk_{i}_24k.wav"
                        convert_to_wav(chunk_path, chunk_24k, sr=24000)
                        
                        try:
                            _, chunk_spk = get_latents(engine, str(chunk_24k))
                            # Garantir [1, 512]
                            chunk_spk = chunk_spk.squeeze().unsqueeze(0)
                            
                            sim = F.cosine_similarity(target_spk, chunk_spk).item()
                            if sim >= req.similarity_threshold:
                                filtered_timestamps.append(ts)
                        except Exception:
                            pass
                            
                    if not filtered_timestamps:
                        raise HTTPException(400, "Nenhum segmento detectado que corresponda à voz alvo.")
                    speech_timestamps = filtered_timestamps

            audio_clean = collect_chunks(speech_timestamps, wav_16k)
            max_samples = max_duration * 16000
            if audio_clean.shape[0] > max_samples:
                audio_clean = audio_clean[:max_samples]
                
            tmp_clean = tmp_dir_path / "clean.wav"
            save_audio(str(tmp_clean), audio_clean, sampling_rate=16000)
            
            out_path = VOICES_DIR / f"{name}.wav"
            convert_to_wav(tmp_clean, out_path, sr=24000)
            
            response = {"status": "ok", "name": name, "path": str(out_path), "duration_seconds": round(audio_clean.shape[0] / 16000, 1)}
            if treatment is not None:
                response.update({
                    "voice_or_singing_path": str(treatment.voice_or_singing_path),
                    "background_or_track_path": str(treatment.background_or_track_path),
                    "used_audio_treatment": True,
                })
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Erro interno: {str(e)}")


class YTRawRequest(BaseModel):
    url: str
    format: str = "mp3"


@app.post("/youtube/raw")
async def youtube_raw_download(req: YTRawRequest):
    """Baixa video ou audio do YouTube SEM processamento (sem VAD, sem voice tracking)."""
    fmt = (req.format or "mp3").lower()
    if fmt not in {"mp3", "mp4"}:
        raise HTTPException(400, "Formato deve ser 'mp3' ou 'mp4'")

    tmpdir_obj = tempfile.TemporaryDirectory()
    tmp_dir_path = Path(tmpdir_obj.name)

    if fmt == "mp3":
        ydl_opts: dict = {
            'format': 'bestaudio/best',
            'outtmpl': str(tmp_dir_path / '%(title).80s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'noplaylist': True,
        }
        target_ext = '.mp3'
        media_type = 'audio/mpeg'
    else:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(tmp_dir_path / '%(title).80s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'noplaylist': True,
        }
        target_ext = '.mp4'
        media_type = 'video/mp4'

    try:
        loop = asyncio.get_event_loop()

        def _run_download():
            return _youtube_download_with_fallback(req.url, ydl_opts, extract_info=True)

        info = await loop.run_in_executor(executor, _run_download)

        matches = list(tmp_dir_path.glob(f"*{target_ext}"))
        if not matches:
            matches = sorted(tmp_dir_path.iterdir(), key=lambda p: p.stat().st_size, reverse=True)
            if not matches:
                raise HTTPException(500, "Arquivo nao gerado")

        out_file = matches[0]
        title = (info or {}).get("title") or out_file.stem
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:80] or "youtube"
        download_name = f"{safe_title}{target_ext}"

        def _iterfile():
            try:
                with open(out_file, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                tmpdir_obj.cleanup()

        headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
        return StreamingResponse(_iterfile(), media_type=media_type, headers=headers)

    except HTTPException:
        tmpdir_obj.cleanup()
        raise
    except Exception as e:
        tmpdir_obj.cleanup()
        raise HTTPException(500, f"Falha no download: {str(e)}")


# ─── Creative Sampling Maker ─────────────────────────────────────────────────

CREATIVE_SAMPLES_DIR = Path("outputs") / "creative_samples"
CREATIVE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def _butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str, order: int = 4) -> np.ndarray:
    """Filtro Butterworth high-pass ou low-pass."""
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * sr
    norm = max(min(cutoff / nyq, 0.999), 0.001)
    sos = butter(order, norm, btype=btype, output='sos')
    return sosfiltfilt(sos, audio).astype(np.float32)


def _find_loudest_window(audio: np.ndarray, sr: int, win_seconds: float) -> int:
    """Retorna o índice de início da janela com maior RMS.

    Hop adaptativo: ~1% da janela (mínimo 0.05s) — mantém a busca rápida mesmo
    em janelas grandes (Macro-Sampling de 30-45s).
    """
    win = int(win_seconds * sr)
    if len(audio) <= win:
        return 0
    hop = max(int(win * 0.01), int(0.05 * sr), 1)
    best_idx = 0
    best_rms = -1.0
    sq = audio.astype(np.float64) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    for start in range(0, len(audio) - win + 1, hop):
        end = start + win
        rms = float(np.sqrt((csum[end] - csum[start]) / win))
        if rms > best_rms:
            best_rms = rms
            best_idx = start
    return best_idx


def _soft_saturation(audio: np.ndarray, drive: float = 1.8) -> np.ndarray:
    """Saturação por tanh (drives altos = esmagamento agressivo de transientes)."""
    return (np.tanh(audio * drive) / np.tanh(drive)).astype(np.float32)


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Brown noise (1/f²): integral cumulativa de white noise, com leak para evitar drift DC."""
    white = rng.standard_normal(n).astype(np.float32)
    brown = np.empty(n, dtype=np.float32)
    leak = 0.997
    acc = 0.0
    for i in range(n):
        acc = leak * acc + white[i]
        brown[i] = acc
    brown -= float(np.mean(brown))
    peak = float(np.max(np.abs(brown)) + 1e-9)
    return (brown / peak).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink noise (1/f) via filtros IIR em cascata (Paul Kellet)."""
    white = rng.standard_normal(n).astype(np.float32)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    pink = np.empty(n, dtype=np.float32)
    for i in range(n):
        w = float(white[i])
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        out = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
        b6 = w * 0.115926
        pink[i] = out * 0.11
    pink -= float(np.mean(pink))
    peak = float(np.max(np.abs(pink)) + 1e-9)
    return (pink / peak).astype(np.float32)


def _inject_tape_noise(
    audio: np.ndarray,
    sr: int,
    mix: float,
    noise_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mistura ruído sintético (brown/pink) ao áudio como 'sujeira de fita cassete'."""
    n = len(audio)
    noise = _brown_noise(n, rng) if noise_type == "brown" else _pink_noise(n, rng)
    sig_peak = float(np.max(np.abs(audio)) + 1e-9)
    noise *= sig_peak
    return ((1.0 - mix) * audio + mix * noise).astype(np.float32)


def _bitcrush_decimate(audio: np.ndarray, sr: int, target_sr: int = 8000, bits: int = 8) -> np.ndarray:
    """Bitcrushing + sample-rate reduction (artefatos digitais agressivos)."""
    target_sr = max(int(target_sr), 1000)
    if target_sr < sr:
        factor = max(int(round(sr / target_sr)), 1)
        # Decimação grosseira: pega 1 a cada N amostras e segura (sample-and-hold)
        ds = audio[::factor]
        held = np.repeat(ds, factor)[:len(audio)]
        if len(held) < len(audio):
            held = np.pad(held, (0, len(audio) - len(held)), mode='edge')
    else:
        held = audio.copy()

    # Quantização para 'bits' (signed)
    levels = max(2 ** (bits - 1) - 1, 1)
    crushed = np.round(held * levels) / levels
    return np.clip(crushed, -1.0, 1.0).astype(np.float32)


def _wow_flutter(audio: np.ndarray, sr: int, wow_hz: float = 0.5, wow_depth: float = 0.012,
                 flutter_hz: float = 7.0, flutter_depth: float = 0.004) -> np.ndarray:
    """Wow & flutter: modulação de pitch lenta (wow) + rápida (flutter), tipo fita esticada/gasta."""
    n = len(audio)
    t = np.arange(n, dtype=np.float64) / sr
    # Posição de leitura modulada por dois osciladores
    mod = wow_depth * np.sin(2 * np.pi * wow_hz * t) + flutter_depth * np.sin(2 * np.pi * flutter_hz * t)
    read_pos = np.arange(n, dtype=np.float64) + mod * sr
    read_pos = np.clip(read_pos, 0.0, n - 1.0001)
    idx0 = read_pos.astype(np.int64)
    frac = (read_pos - idx0).astype(np.float32)
    out = (1.0 - frac) * audio[idx0] + frac * audio[np.minimum(idx0 + 1, n - 1)]
    return out.astype(np.float32)


def _tape_dropouts(audio: np.ndarray, sr: int, rng: np.random.Generator,
                   rate_per_sec: float = 1.5, min_ms: float = 40.0, max_ms: float = 140.0,
                   attenuation: float = 0.15) -> np.ndarray:
    """Simula tape dropouts: trechos curtos com volume despencando (fita ruim)."""
    out = audio.copy()
    duration = len(audio) / sr
    n_drops = int(duration * rate_per_sec)
    for _ in range(n_drops):
        center = rng.uniform(0.0, duration)
        length_ms = rng.uniform(min_ms, max_ms)
        length = int(length_ms * sr / 1000.0)
        start = max(int(center * sr) - length // 2, 0)
        end = min(start + length, len(out))
        if end <= start:
            continue
        # Envelope cosseno (suave nas bordas) que atenua o trecho
        ramp = np.linspace(0.0, np.pi, end - start, dtype=np.float32)
        env = (1.0 - attenuation) + attenuation * (1.0 - np.sin(ramp))
        env = 1.0 - (1.0 - env)  # mantém forma
        # Simplifica: gain decai pra `attenuation` no centro, volta a 1
        gain = attenuation + (1.0 - attenuation) * (1.0 - np.sin(ramp))
        out[start:end] *= gain
    return out


def _chorus_flanger(
    audio: np.ndarray,
    sr: int,
    rate_hz: float = 0.6,
    base_delay_ms: float = 3.0,
    depth_ms: float = 4.0,
    feedback: float = 0.7,
    mix: float = 0.55,
) -> np.ndarray:
    """Modulação de delay curto com feedback alto (comb filtering / phase smearing)."""
    n = len(audio)
    base_delay = int(base_delay_ms * sr / 1000.0)
    depth = int(depth_ms * sr / 1000.0)
    max_delay = base_delay + depth + 2

    buf = np.zeros(n + max_delay, dtype=np.float32)
    buf[max_delay:] = audio
    out = np.zeros(n, dtype=np.float32)

    t = np.arange(n, dtype=np.float32) / sr
    lfo = np.sin(2.0 * np.pi * rate_hz * t).astype(np.float32)

    for i in range(n):
        d = base_delay + depth * (0.5 + 0.5 * lfo[i])
        d_int = int(d)
        d_frac = d - d_int
        idx0 = max_delay + i - d_int
        idx1 = idx0 - 1
        if idx1 < 0:
            delayed = 0.0
        else:
            delayed = (1.0 - d_frac) * buf[idx0] + d_frac * buf[idx1]
        wet = delayed + feedback * out[i - 1] if i > 0 else delayed
        out[i] = wet

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0:
        out /= peak
    return ((1.0 - mix) * audio + mix * out).astype(np.float32)


def _room_reverb(audio: np.ndarray, sr: int, wet: float = 0.15, decay: float = 0.35) -> np.ndarray:
    """Reverb de sala curta por convolução com IR sintético (ruído com envelope exponencial)."""
    ir_len = int(decay * sr)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(ir_len).astype(np.float32)
    env = np.exp(-np.linspace(0.0, 6.0, ir_len)).astype(np.float32)
    ir = noise * env
    ir /= (np.max(np.abs(ir)) + 1e-9)
    from scipy.signal import fftconvolve
    wet_sig = fftconvolve(audio, ir, mode='full')[:len(audio)].astype(np.float32)
    wet_sig /= (np.max(np.abs(wet_sig)) + 1e-9)
    peak = np.max(np.abs(audio)) + 1e-9
    wet_sig *= peak
    return ((1.0 - wet) * audio + wet * wet_sig).astype(np.float32)


def _resample_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    """Estilo TikTok 'Sped Up/Slowed': muda tempo E pitch juntos (resample puro)."""
    if factor == 1.0 or len(audio) < 2:
        return audio
    new_len = max(int(len(audio) / factor), 2)
    new_x = np.linspace(0.0, len(audio) - 1.0, new_len)
    return np.interp(new_x, np.arange(len(audio)), audio).astype(np.float32)


def _apply_8d_panning(audio: np.ndarray, sr: int, rate_hz: float = 0.15) -> np.ndarray:
    """8D Audio: pan automation com LFO senoidal. Retorna estéreo (n,2)."""
    n = len(audio)
    t = np.arange(n, dtype=np.float32) / sr
    lfo = np.sin(2.0 * np.pi * rate_hz * t).astype(np.float32)
    # pan -1..+1 → ganhos L/R com equal-power
    angle = (lfo + 1.0) * (np.pi / 4.0)  # 0..pi/2
    left = audio * np.cos(angle).astype(np.float32)
    right = audio * np.sin(angle).astype(np.float32)
    return np.stack([left, right], axis=1).astype(np.float32)


def _apply_remix_effect(audio: np.ndarray, sr: int, effect: str, rng: np.random.Generator) -> tuple:
    """Aplica preset de remix. Retorna (audio_out, eh_estereo, meta)."""
    import librosa
    eff = (effect or "none").lower()
    meta: Dict[str, Any] = {"effect": eff}

    if eff == "sped_up":
        out = _resample_speed(audio, factor=1.25)
        meta.update({"speed_factor": 1.25, "style": "tiktok_sped_up"})
        return out, False, meta

    if eff == "nightcore":
        # Acelera e ainda sobe pitch +1 semitom em cima → mais agudo ainda
        sped = _resample_speed(audio, factor=1.35)
        out = librosa.effects.pitch_shift(sped, sr=sr, n_steps=1)
        meta.update({"speed_factor": 1.35, "extra_pitch": "+1 semitom"})
        return out.astype(np.float32), False, meta

    if eff == "slowed":
        out = _resample_speed(audio, factor=0.85)
        meta.update({"speed_factor": 0.85})
        return out, False, meta

    if eff == "slowed_reverb":
        slowed = _resample_speed(audio, factor=0.82)
        # Reverb extra pesado em cima
        out = _room_reverb(slowed, sr, wet=0.45, decay=0.7)
        meta.update({"speed_factor": 0.82, "extra_reverb_wet": 0.45})
        return out, False, meta

    if eff == "chipmunk":
        out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=5)
        meta.update({"pitch_semitones": 5})
        return out.astype(np.float32), False, meta

    if eff == "demonic":
        out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-5)
        meta.update({"pitch_semitones": -5})
        return out.astype(np.float32), False, meta

    if eff == "8d":
        out = _apply_8d_panning(audio, sr, rate_hz=0.15)
        meta.update({"pan_rate_hz": 0.15, "stereo": True})
        return out, True, meta

    if eff == "reverse":
        out = np.flip(audio).copy()
        meta.update({"reversed": True})
        return out, False, meta

    if eff == "phonk":
        # Phonk drift: slowed + bass boost (low-shelf via filtro passa-baixa misturado)
        slowed = _resample_speed(audio, factor=0.88)
        bass = _butter_filter(slowed, sr, cutoff=180.0, btype='lowpass', order=2)
        out = (slowed + 0.6 * bass).astype(np.float32)
        peak = float(np.max(np.abs(out)) + 1e-9)
        if peak > 0.95:
            out = (out * (0.95 / peak)).astype(np.float32)
        meta.update({"speed_factor": 0.88, "bass_boost": 0.6})
        return out, False, meta

    # none ou desconhecido
    return audio, False, {"effect": "none"}


def _process_creative_sample(
    in_path: Path,
    out_path: Path,
    seed: int | None = None,
    pollution_level: str = "none",
    mix_path: Path | None = None,
    mix_amount: float = 0.0,
    remix_effect: str = "none",
) -> Dict[str, Any]:
    """Pipeline 'Extreme Lo-Fi / Cassete Tape':
    time/pitch agressivo -> bandpass telefone -> janela 16s -> saturação pesada
    -> reverb 30-35% -> injeção de noise floor (brown/pink).
    """
    import librosa

    rng = np.random.default_rng(seed)

    audio, sr = librosa.load(str(in_path), sr=None, mono=True)
    audio = audio.astype(np.float32)
    if len(audio) == 0:
        raise HTTPException(400, "Arquivo de áudio vazio.")

    # 1. Extreme Time & Pitch Shift
    rate = float(rng.uniform(0.85, 0.88))                  # bem mais lento
    tempo_change_pct = round((rate - 1.0) * 100.0, 2)
    pitch_semitones = int(rng.choice([-4, -3]))            # queda drástica

    audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
    audio_shifted = librosa.effects.pitch_shift(audio_stretched, sr=sr, n_steps=pitch_semitones)

    # 2. EQ de Telefone (Bandpass extremo: HPF 300Hz + LPF 6kHz)
    audio_hp = _butter_filter(audio_shifted, int(sr), cutoff=300.0, btype='highpass')
    audio_eq = _butter_filter(audio_hp, int(sr), cutoff=6000.0, btype='lowpass')

    # 3. Janela 35-45s da região de maior RMS.
    # Se o áudio for menor que 35s, processa o áudio inteiro.
    MIN_LOOP_SECONDS = 35.0
    MAX_LOOP_SECONDS = 45.0
    total_seconds = len(audio_eq) / sr

    if total_seconds < MIN_LOOP_SECONDS:
        loop_audio = audio_eq.copy()
        loop_seconds = total_seconds
        start_idx = 0
    else:
        loop_seconds = float(rng.uniform(MIN_LOOP_SECONDS, MAX_LOOP_SECONDS))
        if total_seconds < loop_seconds:
            loop_seconds = total_seconds
        start_idx = _find_loudest_window(audio_eq, int(sr), loop_seconds)
        end_idx = min(start_idx + int(loop_seconds * sr), len(audio_eq))
        loop_audio = audio_eq[start_idx:end_idx].copy()

    # Fade in/out 30ms pra remover cliques na borda do loop
    fade = min(int(0.03 * sr), len(loop_audio) // 4)
    if fade > 0:
        loop_audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        loop_audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    # 4. Texturização pesada: saturação agressiva
    # Drive e reverb por nível (extreme suavizado; heavy meio-termo; ultra mantém).
    level_preview = pollution_level if pollution_level in ("none", "extreme", "heavy", "savage", "ultra") else "none"
    if level_preview == "extreme":
        drive = float(rng.uniform(3.15, 3.60))
        reverb_wet = float(rng.uniform(0.27, 0.31))
    elif level_preview == "heavy":
        drive = float(rng.uniform(3.32, 3.80))
        reverb_wet = float(rng.uniform(0.30, 0.34))
    elif level_preview == "savage":
        drive = float(rng.uniform(3.41, 3.90))
        reverb_wet = float(rng.uniform(0.31, 0.355))
    else:
        drive = float(rng.uniform(3.5, 4.0))
        reverb_wet = float(rng.uniform(0.30, 0.35))
    saturated = _soft_saturation(loop_audio, drive=drive)

    # 4.5 Modos de Poluição — ANTES do reverb
    # none    = pipeline limpo (lo-fi normal)
    # extreme = bitcrush 8kHz/8bit + chorus fb 0.7 + pink 35%
    # ultra   = bitcrush 4kHz/4bit + chorus fb 0.85 mix 0.7 + wow/flutter + tape dropouts + pink 50%
    level = pollution_level if pollution_level in ("none", "extreme", "heavy", "savage", "ultra") else "none"
    pollution_meta: Dict[str, Any] = {"level": level, "enabled": level != "none"}

    if level == "extreme":
        # ~10% mais suave: bitcrush menos pesado, chorus com menos feedback e menos mix
        crushed = _bitcrush_decimate(saturated, int(sr), target_sr=8800, bits=9)
        smeared = _chorus_flanger(crushed, int(sr), rate_hz=0.6, base_delay_ms=3.0,
                                  depth_ms=3.6, feedback=0.63, mix=0.50)
        pre_reverb = smeared
        pollution_meta.update({
            "bitcrush_target_sr_hz": 8800,
            "bitcrush_bits": 9,
            "phase_smearing": {"rate_hz": 0.6, "base_delay_ms": 3.0, "depth_ms": 3.6,
                               "feedback": 0.63, "mix": 0.50},
        })
    elif level == "heavy":
        # Meio-termo entre "extreme" e "ultra": acréscimo de wow/flutter e dropouts
        # muito sutis, sem chegar no peso de fita queimada do ultra.
        crushed = _bitcrush_decimate(saturated, int(sr), target_sr=8150, bits=8)
        smeared = _chorus_flanger(crushed, int(sr), rate_hz=0.62, base_delay_ms=3.1,
                                  depth_ms=4.0, feedback=0.68, mix=0.54)
        warbled = _wow_flutter(smeared, int(sr), wow_hz=0.5, wow_depth=0.003,
                               flutter_hz=6.0, flutter_depth=0.001)
        dropped = _tape_dropouts(warbled, int(sr), rng, rate_per_sec=0.2,
                                 min_ms=25.0, max_ms=70.0, attenuation=0.75)
        pre_reverb = dropped
        pollution_meta.update({
            "bitcrush_target_sr_hz": 8150,
            "bitcrush_bits": 8,
            "phase_smearing": {"rate_hz": 0.62, "base_delay_ms": 3.1, "depth_ms": 4.0,
                               "feedback": 0.68, "mix": 0.54},
            "wow_flutter": {"wow_hz": 0.5, "wow_depth": 0.003,
                            "flutter_hz": 6.0, "flutter_depth": 0.001},
            "tape_dropouts": {"rate_per_sec": 0.2, "min_ms": 25.0, "max_ms": 70.0,
                              "attenuation": 0.75},
        })
    elif level == "savage":
        # Meio-termo entre "heavy" e "ultra" — wow/flutter um pouco mais perceptível
        # e dropouts menos raros, sem chegar ao caráter "fita queimada" do ultra.
        crushed = _bitcrush_decimate(saturated, int(sr), target_sr=7825, bits=7)
        smeared = _chorus_flanger(crushed, int(sr), rate_hz=0.635, base_delay_ms=3.15,
                                  depth_ms=4.25, feedback=0.705, mix=0.56)
        warbled = _wow_flutter(smeared, int(sr), wow_hz=0.5, wow_depth=0.004,
                               flutter_hz=6.0, flutter_depth=0.0015)
        dropped = _tape_dropouts(warbled, int(sr), rng, rate_per_sec=0.25,
                                 min_ms=25.0, max_ms=70.0, attenuation=0.70)
        pre_reverb = dropped
        pollution_meta.update({
            "bitcrush_target_sr_hz": 7825,
            "bitcrush_bits": 7,
            "phase_smearing": {"rate_hz": 0.635, "base_delay_ms": 3.15, "depth_ms": 4.25,
                               "feedback": 0.705, "mix": 0.56},
            "wow_flutter": {"wow_hz": 0.5, "wow_depth": 0.004,
                            "flutter_hz": 6.0, "flutter_depth": 0.0015},
            "tape_dropouts": {"rate_per_sec": 0.25, "min_ms": 25.0, "max_ms": 70.0,
                              "attenuation": 0.70},
        })
    elif level == "ultra":
        # Apenas ~25% mais agressivo que "extreme" — adiciona caráter de fita gasta
        # sem destruir o áudio. Bitcrush quase idêntico ao extreme; wow/flutter e
        # dropouts bem sutis pra dar a sensação de "tape antiga", não de "fita queimada".
        # 1) Bitcrush levemente mais sujo
        crushed = _bitcrush_decimate(saturated, int(sr), target_sr=7500, bits=7)
        # 2) Phase smearing um pouco mais profundo
        smeared = _chorus_flanger(crushed, int(sr), rate_hz=0.65, base_delay_ms=3.2,
                                  depth_ms=4.5, feedback=0.73, mix=0.58)
        # 3) Wow & flutter MUITO sutil
        warbled = _wow_flutter(smeared, int(sr), wow_hz=0.5, wow_depth=0.005,
                               flutter_hz=6.0, flutter_depth=0.002)
        # 4) Tape dropouts raros e quase imperceptíveis
        dropped = _tape_dropouts(warbled, int(sr), rng, rate_per_sec=0.3,
                                 min_ms=25.0, max_ms=70.0, attenuation=0.65)
        pre_reverb = dropped
        pollution_meta.update({
            "bitcrush_target_sr_hz": 7500,
            "bitcrush_bits": 7,
            "phase_smearing": {"rate_hz": 0.65, "base_delay_ms": 3.2, "depth_ms": 4.5,
                               "feedback": 0.73, "mix": 0.58},
            "wow_flutter": {"wow_hz": 0.5, "wow_depth": 0.005,
                            "flutter_hz": 6.0, "flutter_depth": 0.002},
            "tape_dropouts": {"rate_per_sec": 0.3, "min_ms": 25.0, "max_ms": 70.0,
                              "attenuation": 0.65},
        })
        # Reverb só um tiquinho maior que extreme
        reverb_wet = float(rng.uniform(0.32, 0.37))
    else:
        pre_reverb = saturated

    reverbed = _room_reverb(pre_reverb, int(sr), wet=reverb_wet, decay=0.5 if level == "ultra" else 0.4)

    # 5. Injeção de Noise Floor
    if level == "ultra":
        noise_type = "pink"
        noise_mix = 0.38  # só ~10% acima do extreme (0.35)
    elif level == "savage":
        noise_type = "pink"
        noise_mix = 0.36  # entre heavy (0.34) e ultra (0.38)
    elif level == "heavy":
        noise_type = "pink"
        noise_mix = 0.34  # entre extreme (0.31) e ultra (0.38)
    elif level == "extreme":
        noise_type = "pink"
        noise_mix = 0.31  # ~10% menos chiado que antes (era 0.35)
    else:
        noise_type = str(rng.choice(["brown", "pink"]))
        noise_mix = float(rng.uniform(0.10, 0.15))
    dirty = _inject_tape_noise(reverbed, int(sr), mix=noise_mix, noise_type=noise_type, rng=rng)

    # 5.5 Efeito de Remix (Sped Up / Nightcore / Slowed+Reverb / etc.)
    remix_audio, is_stereo, remix_meta = _apply_remix_effect(dirty, int(sr), remix_effect, rng)
    dirty = remix_audio

    # 6. Blend opcional com 2º áudio (layer/mistura criativa) — só em mono
    mix_meta: Dict[str, Any] = {"enabled": False}
    if mix_path is not None and mix_amount > 0.0 and not is_stereo:
        try:
            mix_audio, _mix_sr = librosa.load(str(mix_path), sr=sr, mono=True)
            mix_audio = mix_audio.astype(np.float32)
            if len(mix_audio) > 0:
                target_len = len(dirty)
                # Se mais curto, faz loop; se mais longo, corta
                if len(mix_audio) < target_len:
                    reps = int(np.ceil(target_len / len(mix_audio)))
                    mix_audio = np.tile(mix_audio, reps)
                mix_audio = mix_audio[:target_len]

                # Iguala o peak do mix ao peak do dirty pra não dominar
                dirty_peak = float(np.max(np.abs(dirty)) + 1e-9)
                mix_peak = float(np.max(np.abs(mix_audio)) + 1e-9)
                mix_audio *= (dirty_peak / mix_peak)

                amount = float(np.clip(mix_amount, 0.0, 1.0))
                dirty = ((1.0 - amount) * dirty + amount * mix_audio).astype(np.float32)
                mix_meta = {"enabled": True, "amount": round(amount, 3),
                            "loop_to_fit": len(mix_audio) <= target_len * 2}
        except Exception as e:
            mix_meta = {"enabled": False, "error": str(e)}

    # Normalize final pra não estourar
    peak = float(np.max(np.abs(dirty)) + 1e-9)
    if peak > 0.95:
        dirty = (dirty * (0.95 / peak)).astype(np.float32)

    # soundfile aceita (n,) mono ou (n,2) estéreo
    sf.write(str(out_path), dirty, sr, subtype='PCM_16')

    return {
        "tempo_change_pct": tempo_change_pct,
        "rate": round(rate, 3),
        "pitch_semitones": pitch_semitones,
        "highpass_hz": 300,
        "lowpass_hz": 6000,
        "loop_seconds": round(loop_seconds, 2),
        "loop_start_seconds": round(start_idx / sr, 2),
        "saturation_drive": round(drive, 2),
        "reverb_wet": round(reverb_wet, 3),
        "noise_type": noise_type,
        "noise_mix": round(noise_mix, 3),
        "sample_rate": sr,
        "pollution_mode": pollution_meta,
        "mix_layer": mix_meta,
        "remix_effect": remix_meta,
    }


def _spectral_denoise(audio: np.ndarray, sr: int, strength: float = 0.7,
                      n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Denoise por subtração espectral.

    Estima o noise floor pelo quantil 10% de magnitude em cada bin de frequência,
    subtrai com fator `strength * over_sub` e aplica floor para não estourar.
    """
    import librosa
    if len(audio) < n_fft:
        return audio
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag = np.abs(stft)
    phase = np.angle(stft)
    # Noise floor por bin = quantil 10% das magnitudes (regiões mais silenciosas)
    noise_profile = np.quantile(mag, 0.10, axis=1, keepdims=True)
    over_sub = 1.0 + strength * 1.5
    clean_mag = np.maximum(mag - over_sub * noise_profile, mag * (1.0 - strength) * 0.05)
    clean = clean_mag * np.exp(1j * phase)
    out = librosa.istft(clean, hop_length=hop, length=len(audio))
    return out.astype(np.float32)


def _eq_expansion(audio: np.ndarray, sr: int, bass_gain: float = 0.5,
                  treble_gain: float = 0.5) -> np.ndarray:
    """Compensa o bandpass do pipeline: realça graves (<200Hz) e agudos (>4kHz)."""
    bass = _butter_filter(audio, sr, cutoff=200.0, btype='lowpass', order=2)
    treble = _butter_filter(audio, sr, cutoff=4000.0, btype='highpass', order=2)
    out = audio + bass_gain * bass + treble_gain * treble
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0:
        out = (out / peak * 0.95).astype(np.float32)
    return out.astype(np.float32)


def _restore_audio(
    in_path: Path,
    out_path: Path,
    denoise_strength: float = 0.7,
    eq_expand: bool = True,
    pitch_correction: float = 0.0,
    time_correction: float = 1.0,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
    linear_mode: bool = False,
) -> Dict[str, Any]:
    """Tenta reverter a poluição: corte opcional + denoise + EQ expansion + correções.

    linear_mode: se True, pula TODO o processamento e só corta o trecho (start/end).
    """
    import librosa
    audio, sr = librosa.load(str(in_path), sr=None, mono=True)
    audio = audio.astype(np.float32)
    if len(audio) == 0:
        raise HTTPException(400, "Arquivo de áudio vazio.")

    total_seconds = len(audio) / sr
    meta: Dict[str, Any] = {"input_seconds": round(total_seconds, 2), "sample_rate": sr,
                            "linear_mode": bool(linear_mode)}

    # 0) Corte de trecho linear (start/end em segundos)
    start_s = max(float(start_seconds), 0.0)
    end_s = float(end_seconds) if end_seconds and end_seconds > 0 else total_seconds
    end_s = min(end_s, total_seconds)
    if end_s <= start_s:
        end_s = total_seconds
        start_s = 0.0
    if start_s > 0.0 or end_s < total_seconds:
        a = int(start_s * sr)
        b = int(end_s * sr)
        audio = audio[a:b]
        meta["trim_start_seconds"] = round(start_s, 2)
        meta["trim_end_seconds"] = round(end_s, 2)

    if linear_mode:
        # Modo linear: NADA de processamento. Só salva o trecho cortado.
        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0.95:
            audio = (audio * (0.95 / peak)).astype(np.float32)
        sf.write(str(out_path), audio.astype(np.float32), sr, subtype='PCM_16')
        meta["output_seconds"] = round(len(audio) / sr, 2)
        meta["processing_skipped"] = True
        return meta

    # 1) Denoise espectral
    if denoise_strength > 0.0:
        audio = _spectral_denoise(audio, int(sr), strength=float(np.clip(denoise_strength, 0.0, 1.0)))
        meta["denoise_strength"] = round(denoise_strength, 3)

    # 2) EQ expansion (reverte o bandpass)
    if eq_expand:
        audio = _eq_expansion(audio, int(sr), bass_gain=0.5, treble_gain=0.5)
        meta["eq_expand"] = True

    # 3) Correção de pitch (manual)
    if pitch_correction and abs(pitch_correction) > 0.05:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(pitch_correction))
        meta["pitch_correction_semitones"] = round(float(pitch_correction), 2)

    # 4) Correção de tempo (manual) — só aplica se !=1.0
    if time_correction and abs(time_correction - 1.0) > 0.01:
        rate = max(float(time_correction), 0.25)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        meta["time_correction_rate"] = round(rate, 3)

    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.95:
        audio = (audio * (0.95 / peak)).astype(np.float32)

    sf.write(str(out_path), audio, sr, subtype='PCM_16')
    meta["output_seconds"] = round(len(audio) / sr, 2)
    return meta


@app.post("/creative/restore")
async def creative_restore(
    file: UploadFile = File(...),
    custom_name: str | None = Form(None),
    save_to_library: bool = Form(False),
    denoise_strength: float = Form(0.7),
    eq_expand: bool = Form(True),
    pitch_correction: float = Form(0.0),
    time_correction: float = Form(1.0),
    start_seconds: float = Form(0.0),
    end_seconds: float = Form(0.0),
    linear_mode: bool = Form(False),
):
    """Reverte a poluição de um áudio: denoise + EQ expansion + correções opcionais."""
    if not file.filename:
        raise HTTPException(400, "Arquivo inválido")

    stem = safe_audio_name(custom_name) if custom_name else safe_audio_name(Path(file.filename).stem)
    if not stem:
        stem = "restored"
    stamp = int(time.time())
    out_name = f"{stem}_restored_{stamp}.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        suffix = Path(file.filename).suffix or ".wav"
        tmp_in = tmp_dir / f"input{suffix}"
        with open(tmp_in, "wb") as f:
            f.write(await file.read())

        tmp_wav = tmp_dir / "input_norm.wav"
        if not convert_to_wav(tmp_in, tmp_wav):
            raise HTTPException(500, "Falha ao normalizar entrada para WAV")

        if save_to_library:
            out_path = VOICES_DIR / out_name
        else:
            out_path = CREATIVE_SAMPLES_DIR / out_name

        try:
            loop = asyncio.get_event_loop()
            meta = await loop.run_in_executor(
                executor,
                _restore_audio,
                tmp_wav, out_path, denoise_strength, eq_expand, pitch_correction,
                time_correction, start_seconds, end_seconds, linear_mode,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Falha na restauração: {str(e)}")

        if save_to_library:
            return {"status": "ok", "name": stem, "path": str(out_path),
                    "saved_to_library": True, **meta}

        def _iterfile():
            with open(out_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Restore-Meta": json.dumps(meta),
        }
        return StreamingResponse(_iterfile(), media_type="audio/wav", headers=headers)


@app.post("/creative/sample")
async def creative_sample(
    file: UploadFile = File(...),
    custom_name: str | None = Form(None),
    save_to_library: bool = Form(False),
    pollution_mode: bool = Form(False),
    pollution_level: str = Form("none"),
    remove_vocals: bool = Form(False),
    mix_file: UploadFile | None = File(None),
    mix_amount: float = Form(0.0),
    remix_effect: str = Form("none")
):
    """Pipeline Creative Sampling Maker:
    1) Time stretch -3% a -5% + pitch shift ±1 semitom
    2) HPF 100Hz + LPF 12kHz (textura lo-fi)
    3) Micro-loop 5-8s da parte mais alta
    4) Saturação leve + reverb sala 15% wet
    """
    if not file.filename:
        raise HTTPException(400, "Arquivo inválido")

    stem = safe_audio_name(custom_name) if custom_name else safe_audio_name(Path(file.filename).stem)
    if not stem:
        stem = "creative_sample"
    stamp = int(time.time())
    out_name = f"{stem}_{stamp}.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        suffix = Path(file.filename).suffix or ".wav"
        tmp_in = tmp_dir / f"input{suffix}"
        with open(tmp_in, "wb") as f:
            f.write(await file.read())

        tmp_wav = tmp_dir / "input_norm.wav"
        if not convert_to_wav(tmp_in, tmp_wav):
            raise HTTPException(500, "Falha ao normalizar entrada para WAV")

        # Remoção de voz: separa voz/instrumental e usa só a faixa de fundo/trilha.
        pipeline_input = tmp_wav
        vocals_removed = False
        if remove_vocals:
            try:
                loop_sep = asyncio.get_event_loop()
                treatment = await loop_sep.run_in_executor(
                    executor,
                    separate_audio_stems,
                    tmp_wav,
                    VOICE_OR_SINGING_DIR,
                    BACKGROUND_OR_TRACK_DIR,
                    f"creative_{stem}_{stamp}",
                )
                pipeline_input = Path(treatment.background_or_track_path)
                vocals_removed = True
            except Exception as e:
                raise HTTPException(500, f"Falha ao separar voz/instrumental: {e}")

        # 2º áudio opcional pra blend
        mix_path: Path | None = None
        if mix_file is not None and mix_file.filename:
            mix_suffix = Path(mix_file.filename).suffix or ".wav"
            mix_raw = tmp_dir / f"mix_input{mix_suffix}"
            with open(mix_raw, "wb") as f:
                f.write(await mix_file.read())
            mix_norm = tmp_dir / "mix_norm.wav"
            if convert_to_wav(mix_raw, mix_norm):
                mix_path = mix_norm

        if save_to_library:
            out_path = VOICES_DIR / out_name
        else:
            out_path = CREATIVE_SAMPLES_DIR / out_name

        # Compat: se vier `pollution_mode=true` sem `pollution_level`, trata como "extreme"
        effective_level = pollution_level
        if effective_level not in ("none", "extreme", "ultra"):
            effective_level = "extreme" if pollution_mode else "none"
        elif effective_level == "none" and pollution_mode:
            effective_level = "extreme"

        try:
            loop = asyncio.get_event_loop()
            meta = await loop.run_in_executor(
                executor,
                _process_creative_sample,
                pipeline_input, out_path, None, effective_level, mix_path, mix_amount, remix_effect,
            )
            meta["vocals_removed"] = vocals_removed
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Falha no pipeline: {str(e)}")

        if save_to_library:
            return {
                "status": "ok",
                "name": stem,
                "path": str(out_path),
                "saved_to_library": True,
                **meta,
            }

        def _iterfile():
            with open(out_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        headers = {
            "Content-Disposition": f'attachment; filename="{out_name}"',
            "X-Sample-Meta": json.dumps(meta),
        }
        return StreamingResponse(_iterfile(), media_type="audio/wav", headers=headers)


@app.post("/convert")
async def convert_audio(
    file: UploadFile = File(...),
    format: str = Form("wav"),
    speed: float = Form(1.0),
    save_to_library: bool = Form(False),
    custom_name: str | None = Form(None),
    separate_audio: bool = Form(False)
):
    """Conversor de áudio: qualquer formato → WAV/MP3/OGG/FLAC."""
    if not has_ffmpeg():
        raise HTTPException(500, "ffmpeg não encontrado")
    if format not in ["wav", "mp3", "ogg", "flac"]:
        raise HTTPException(400, "Formato inválido")
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    content = await file.read()
    suffix = Path(file.filename).suffix
    stem = safe_audio_name(Path(file.filename).stem)
    tmp_in = Path(tempfile.mktemp(suffix=suffix))
    tmp_out = Path(tempfile.mktemp(suffix=f".{format}"))
    tmp_in.write_bytes(content)
    ok = convert_audio_format(tmp_in, tmp_out, format, speed)
    tmp_in.unlink(missing_ok=True)
    if not ok:
        tmp_out.unlink(missing_ok=True)
        raise HTTPException(500, "Falha na conversão")

    if save_to_library:
        out_name = safe_audio_name(custom_name if custom_name else stem)
        final_path = VOICES_DIR / f"{out_name}.wav"
        treatment = None
        if separate_audio:
            try:
                treatment = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    separate_audio_stems,
                    tmp_out,
                    VOICE_OR_SINGING_DIR,
                    BACKGROUND_OR_TRACK_DIR,
                    out_name,
                )
                convert_to_wav(treatment.voice_or_singing_path, final_path)
            except Exception as e:
                tmp_out.unlink(missing_ok=True)
                raise HTTPException(500, f"Falha na separacao do audio: {e}")
        else:
            # Converte para WAV 24k fixo se for para a biblioteca
            convert_to_wav(tmp_out, final_path)
        tmp_out.unlink(missing_ok=True)
        response: Dict[str, Any] = {"status": "ok", "name": out_name, "path": str(final_path)}
        if treatment is not None:
            response.update({
                "voice_or_singing_path": str(treatment.voice_or_singing_path),
                "background_or_track_path": str(treatment.background_or_track_path),
                "used_audio_treatment": True,
            })
        return response

    if separate_audio:
        out_name = safe_audio_name(custom_name if custom_name else stem)
        try:
            treatment = await asyncio.get_event_loop().run_in_executor(
                executor,
                separate_audio_stems,
                tmp_out,
                VOICE_OR_SINGING_DIR,
                BACKGROUND_OR_TRACK_DIR,
                out_name,
            )
        except Exception as e:
            tmp_out.unlink(missing_ok=True)
            raise HTTPException(500, f"Falha na separacao do audio: {e}")
        tmp_out.unlink(missing_ok=True)
        return {
            "status": "ok",
            "name": out_name,
            "voice_or_singing_path": str(treatment.voice_or_singing_path),
            "background_or_track_path": str(treatment.background_or_track_path),
            "used_audio_treatment": True,
        }

    data = tmp_out.read_bytes()
    tmp_out.unlink(missing_ok=True)
    mt = {"wav": "audio/wav", "mp3": "audio/mpeg",
          "ogg": "audio/ogg", "flac": "audio/flac"}
    return Response(content=data, media_type=mt[format],
                    headers={"Content-Disposition":
                             f'attachment; filename="{stem}_converted.{format}"'})

@app.post("/convert/trim")
async def trim_audio(
    file: UploadFile = File(...),
    start_time: str = Form(""),
    end_time: str = Form(""),
    custom_name: str | None = Form(None),
    remove_silence: bool = Form(False),
    action_type: str = Form("keep")
):
    """Corta um trecho do áudio (mantém ou remove a seleção), limpa silêncios e salva."""
    if not has_ffmpeg():
        raise HTTPException(500, "ffmpeg não encontrado")
    
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    content = await file.read()
    suffix = Path(file.filename).suffix.lower()
    stem = Path(file.filename).stem
    
    tmp_in = Path(tempfile.mktemp(suffix=suffix))
    tmp_out = Path(tempfile.mktemp(suffix=".wav"))
    tmp_in.write_bytes(content)
    
    s_time = float(start_time) if start_time else 0.0
    e_time = float(end_time) if end_time else 999999.0
    
    cmd = ["ffmpeg", "-i", str(tmp_in)]
    
    if action_type == "remove":
        if s_time <= 0.1:
            cmd += ["-ss", str(e_time)]
        elif e_time >= 999998.0:
            cmd += ["-to", str(s_time)]
        else:
            cmd += [
                "-filter_complex", 
                f"[0:a]atrim=0:{s_time},asetpts=PTS-STARTPTS[p1];[0:a]atrim={e_time},asetpts=PTS-STARTPTS[p2];[p1][p2]concat=n=2:v=0:a=1[out]", 
                "-map", "[out]"
            ]
    else:
        cmd += ["-ss", str(s_time)]
        if e_time < 999998.0:
            cmd += ["-to", str(e_time)]
    
    cmd += ["-ar", "24000", "-ac", "1", "-y", str(tmp_out)]
    
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise HTTPException(500, f"Falha no corte: {r.stderr}")
            
        if remove_silence:
            _model_vad_result = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                          model='silero_vad', force_reload=False)
            model_vad = _model_vad_result[0] if isinstance(_model_vad_result, tuple) else _model_vad_result
            utils_vad = _model_vad_result[1] if isinstance(_model_vad_result, tuple) and len(_model_vad_result) > 1 else _model_vad_result
            (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils_vad  # type: ignore

            tmp_16k = Path(tempfile.mktemp(suffix=".wav"))
            convert_to_wav(tmp_out, tmp_16k, sr=16000)
            
            wav_16k = read_audio(str(tmp_16k), sampling_rate=16000)
            speech_timestamps = get_speech_timestamps(wav_16k, model_vad, sampling_rate=16000)
            tmp_16k.unlink(missing_ok=True)
            
            if not speech_timestamps:
                 raise HTTPException(400, "Nenhuma voz detectada após o corte/remoção de silêncio.")
                 
            audio_clean = collect_chunks(speech_timestamps, wav_16k)
            tmp_clean_16k = Path(tempfile.mktemp(suffix=".wav"))
            save_audio(str(tmp_clean_16k), audio_clean, sampling_rate=16000)
            
            convert_to_wav(tmp_clean_16k, tmp_out, sr=24000)
            tmp_clean_16k.unlink(missing_ok=True)

        out_name = custom_name if custom_name else f"{stem}_cortado"
        final_path = VOICES_DIR / f"{out_name}.wav"
        
        import shutil
        shutil.move(str(tmp_out), str(final_path))
        
        return {"status": "ok", "name": out_name, "path": str(final_path)}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)

@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Remove uma voz (WAV ou PKL fundido)."""
    wav = VOICES_DIR / f"{name}.wav"
    pkl = FUSED_DIR / f"{name}.pkl"
    preview = FUSED_DIR / f"{name}_preview.wav"
    deleted = []
    for p in [wav, pkl, preview]:
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    if not deleted:
        raise HTTPException(404, f"Voz não encontrada: {name}")
    return {"status": "ok", "deleted": deleted}

# ─── Biometria ───────────────────────────────────────────────────────────────

@app.get("/biometry/users")
async def biometry_list():
    """Lista todos os locutores registrados."""
    users = []
    for f in sorted(VOICE_DB_DIR.glob("*.pkl")):
        users.append({"name": f.stem, "file": f.name})
    return {"users": users}

@app.post("/biometry/register")
async def biometry_register(
    file: UploadFile = File(...),
    user_name: str = Form(...)
):
    """Registra um locutor: extrai speaker_embedding do WAV enviado."""
    engine = require_tts_engine()
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    content = await file.read()
    suffix = Path(file.filename).suffix.lower()
    tmp_in = Path(tempfile.mktemp(suffix=suffix))
    tmp_in.write_bytes(content)
    # Converter para WAV se necessário
    if suffix != ".wav":
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        ok = convert_to_wav(tmp_in, tmp_wav)
        tmp_in.unlink(missing_ok=True)
        if not ok:
            raise HTTPException(500, "Falha na conversão do áudio")
        tmp_in = tmp_wav
    try:
        model = get_xtts_model(engine)
        with torch.no_grad():
            _, spk_emb = model.get_conditioning_latents(audio_path=[str(tmp_in)])
        save_path = VOICE_DB_DIR / f"{user_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(spk_emb, f)
        return {"status": "ok", "user": user_name, "path": str(save_path)}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_in.unlink(missing_ok=True)

@app.post("/biometry/isolate")
async def biometry_isolate(
    user_name: str = Form(...), 
    file: UploadFile = File(...),
    confidence_target: float = Form(90.0) # Alvo de confiança (ex: 100)
):
    """
    Isola a voz de um usuário específico em um áudio misto.
    Remove (muta) tudo o que não for identificado como a voz do usuário.
    """
    engine = require_tts_engine()
    
    # 1. Carregar Biometria Salva
    pkl_path = VOICE_DB_DIR / f"{user_name}.pkl"
    if not pkl_path.exists():
        raise HTTPException(404, f"Biometria de '{user_name}' não encontrada.")
    
    with open(pkl_path, "rb") as f:
        target_emb = pickle.load(f).squeeze().unsqueeze(0)
    
    # 2. Processar Áudio Enviado
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    content = await file.read()
    tmp_in = Path(tempfile.mktemp(suffix=Path(file.filename).suffix))
    tmp_in.write_bytes(content)
    
    # Garantir WAV
    if tmp_in.suffix.lower() != ".wav":
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        convert_to_wav(tmp_in, tmp_wav)
        tmp_in.unlink(missing_ok=True)
        tmp_in = tmp_wav

    try:
        import librosa
        audio, sr = librosa.load(str(tmp_in), sr=24000)
        
        # Segmentos de 1.2 segundos para análise de identidade
        seg_len = int(1.2 * sr)
        final_audio = np.zeros_like(audio)
        
        print(f"🕵️ Isolando voz de {user_name} com alvo de {confidence_target}%...")
        
        model = get_xtts_model(engine)
        
        for i in range(0, len(audio), seg_len):
            chunk = audio[i : i + seg_len]
            if len(chunk) < int(0.5 * sr):
                continue # Pula trechos muito pequenos
            
            # Extrair embedding temporário do chunk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as c_tmp:
                sf.write(c_tmp.name, chunk, sr)
                c_path = Path(c_tmp.name)
            
            try:
                with torch.no_grad():
                    _, live_emb = model.get_conditioning_latents(audio_path=[str(c_path)])
                
                # Comparar com o alvo
                sim = torch.nn.functional.cosine_similarity(
                    live_emb.squeeze().unsqueeze(0), 
                    target_emb
                ).item() * 100
                
                # Se bater com a confiança desejada, mantemos o áudio original
                if sim >= confidence_target:
                    final_audio[i : i + seg_len] = chunk
                    print(f"  [MATCH] Seg {i//seg_len}: {sim:.1f}%")
            finally:
                if c_path.exists():
                    c_path.unlink()

        # Exportar resultado
        out_buf = io.BytesIO()
        sf.write(out_buf, final_audio, 24000, format="WAV")
        out_buf.seek(0)
        
        return Response(
            content=out_buf.getvalue(), 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=isolated_{user_name}.wav"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        tmp_in.unlink(missing_ok=True)

@app.post("/biometry/identify")
async def biometry_identify(file: UploadFile = File(...)):
    """Identifica quem está falando comparando com o banco de biometria."""
    engine = require_tts_engine()
    # Carregar banco
    db: Dict[str, torch.Tensor] = {}
    for pkl_f in VOICE_DB_DIR.glob("*.pkl"):
        with open(pkl_f, "rb") as f:
            db[pkl_f.stem] = pickle.load(f)
    if not db:
        raise HTTPException(400, "Banco de biometria vazio. Registre locutores primeiro.")
    if not file.filename:
        raise HTTPException(400, "Nome do arquivo não fornecido")
    content = await file.read()
    suffix = Path(file.filename).suffix.lower()
    tmp_in = Path(tempfile.mktemp(suffix=suffix))
    tmp_in.write_bytes(content)
    if suffix != ".wav":
        tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
        convert_to_wav(tmp_in, tmp_wav)
        tmp_in.unlink(missing_ok=True)
        tmp_in = tmp_wav
    try:
        model = get_xtts_model(engine)
        with torch.no_grad():
            _, live_emb = model.get_conditioning_latents(audio_path=[str(tmp_in)])
        scores = {}
        for name, saved_emb in db.items():
            sim = F.cosine_similarity(
                live_emb.squeeze().unsqueeze(0),
                saved_emb.squeeze().unsqueeze(0)
            ).item()
            scores[name] = round(sim * 100, 1)
        best = max(scores, key=lambda x: scores[x])
        return {
            "best_match": best,
            "confidence": scores[best],
            "all_scores": scores,
            "identified": scores[best] >= 85.0,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        tmp_in.unlink(missing_ok=True)

@app.delete("/biometry/{user_name}")
async def biometry_delete(user_name: str):
    """Remove um locutor do banco de biometria."""
    pkl = VOICE_DB_DIR / f"{user_name}.pkl"
    if not pkl.exists():
        raise HTTPException(404, f"Usuário não encontrado: {user_name}")
    pkl.unlink()
    return {"status": "ok", "deleted": user_name}

@app.get("/voices/list_all")
async def list_all_voices():
    """Retorna todas as vozes originais e quimeras."""
    wavs = [f.name for f in VOICES_DIR.glob("*.wav")]
    pkls = [f.name for f in FUSED_DIR.glob("*.pkl")]
    return {"wavs": wavs, "pkls": pkls}

# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")  # somente esta maquina
