# -*- coding: utf-8 -*-
"""Separacao de audio em voz/canto e fundo/trilha."""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class SeparationResult:
    voice_or_singing_path: Path
    background_or_track_path: Path


def safe_audio_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "audio"


def _find_named_stem(output_dir: Path, files: Iterable[str], labels: tuple[str, ...]) -> Path:
    candidates = [output_dir / f for f in files]
    for path in candidates:
        lower_name = path.name.lower()
        if path.exists() and any(label in lower_name for label in labels):
            return path
    raise RuntimeError(f"Stem nao gerado pelo separador: {'/'.join(labels)}")


def _pad_short_audio(input_path: Path, work_dir: Path, minimum_seconds: float = 8.0) -> tuple[Path, float | None]:
    info = sf.info(str(input_path))
    duration = info.frames / float(info.samplerate)
    if duration >= minimum_seconds:
        return input_path, None

    audio, sr = sf.read(str(input_path), always_2d=True)
    min_frames = int(minimum_seconds * sr)
    if audio.shape[0] < min_frames:
        padding = np.zeros((min_frames - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
        audio = np.vstack([audio, padding])

    padded_path = work_dir / f"{input_path.stem}_padded.wav"
    sf.write(str(padded_path), audio, sr)
    return padded_path, duration


def _trim_to_duration(audio_path: Path, duration_seconds: float) -> None:
    audio, sr = sf.read(str(audio_path), always_2d=False)
    max_frames = int(duration_seconds * sr)
    sf.write(str(audio_path), audio[:max_frames], sr)


def separate_audio_stems(
    input_path: Path,
    voice_or_singing_dir: Path,
    background_or_track_dir: Path,
    output_name: str,
) -> SeparationResult:
    """Separa um arquivo em Vocals e Instrumental e grava nas pastas finais."""
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "audio-separator[gpu] nao esta instalado. Rode pip install -r requirements.txt."
        ) from exc

    if not input_path.exists():
        raise FileNotFoundError(f"Audio de entrada nao encontrado: {input_path}")

    voice_or_singing_dir.mkdir(parents=True, exist_ok=True)
    background_or_track_dir.mkdir(parents=True, exist_ok=True)

    safe_name = safe_audio_name(output_name)
    voice_final = voice_or_singing_dir / f"{safe_name}.wav"
    background_final = background_or_track_dir / f"{safe_name}.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        separator_input, trim_duration = _pad_short_audio(input_path, tmp_out)
        separator = Separator(output_dir=str(tmp_out), output_format="WAV")
        separator.load_model()
        output_files = separator.separate(str(separator_input))

        voice_tmp = _find_named_stem(tmp_out, output_files, ("vocals", "vocal"))
        background_tmp = _find_named_stem(tmp_out, output_files, ("instrumental", "instrument"))
        shutil.copyfile(voice_tmp, voice_final)
        shutil.copyfile(background_tmp, background_final)
        if trim_duration is not None:
            _trim_to_duration(voice_final, trim_duration)
            _trim_to_duration(background_final, trim_duration)

    return SeparationResult(
        voice_or_singing_path=voice_final,
        background_or_track_path=background_final,
    )
