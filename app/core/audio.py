"""Extração e normalização de áudio a partir de arquivos de vídeo e áudio via FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS


@dataclass
class AudioMetadata:
    source_path: Path
    wav_path: Path
    duration: float
    sample_rate: int
    channels: int
    is_video: bool
    file_size_bytes: int


def get_ffmpeg_executable() -> str:
    """Retorna o caminho para o executável FFmpeg (sistema ou embutido via imageio-ffmpeg)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg não encontrado. Execute instalar.ps1 para instalar 'imageio-ffmpeg' "
            "ou instale o FFmpeg no sistema."
        ) from exc


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def extract_audio_to_wav(
    input_path: Path | str,
    output_wav_path: Path | str,
    sample_rate: int = 16000,
    enhance_audio: bool = False,
) -> AudioMetadata:
    """Converte qualquer vídeo ou áudio suportado para WAV PCM 16-bit mono 16kHz.

    Quando enhance_audio=True, aplica filtro de banda de voz (70Hz–7800Hz) e normalização
    EBU R128 (loudnorm) para recuperar vozes baixas e reduzir ruídos de fundo.
    """
    src = Path(input_path)
    dst = Path(output_wav_path)

    if not src.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_executable()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",  # descarta vídeo
    ]

    if enhance_audio:
        cmd.extend(["-af", "highpass=f=70,lowpass=f=7800,loudnorm=I=-16:TP=-1.5:LRA=11"])

    cmd.extend([
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(dst),
    ])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        err_msg = (result.stderr or "").strip()
        if "does not contain any stream" in err_msg.lower() or "output file does not contain" in err_msg.lower():
            raise ValueError("O arquivo selecionado não possui faixa de áudio.")
        raise RuntimeError(
            f"Falha ao extrair áudio com FFmpeg (código {result.returncode}): {err_msg or 'erro desconhecido'}"
        )

    duration, actual_sr, actual_ch = read_wav_info(dst)
    if duration <= 0.05:
        raise ValueError("A faixa de áudio extraída está vazia ou corrompida.")

    return AudioMetadata(
        source_path=src,
        wav_path=dst,
        duration=duration,
        sample_rate=actual_sr,
        channels=actual_ch,
        is_video=is_video_file(src),
        file_size_bytes=src.stat().st_size,
    )


def read_wav_info(wav_path: Path) -> tuple[float, int, int]:
    """Lê duração (em segundos), taxa de amostragem e canais de um arquivo WAV."""
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        channels = wf.getnchannels()
        duration = frames / float(rate) if rate > 0 else 0.0
        return duration, rate, channels


def format_timestamp(seconds: float, force_hours: Optional[bool] = None) -> str:
    """Formata segundos em HH:MM:SS ou MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)

    if force_hours is True or (force_hours is None and hours > 0):
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration_human(seconds: float) -> str:
    """Formata duração de forma legível em português (ex: 1h 12min 05s)."""
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}min")
    parts.append(f"{secs}s")
    return " ".join(parts)
