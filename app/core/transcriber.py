"""Motor de transcrição acelerado por GPU/CPU usando Faster-Whisper (CTranslate2)."""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Garante que DLLs CUDA sejam registradas antes de importar faster_whisper / ctranslate2
from app.config import (
    DEFAULT_QUALITY_PRESET,
    DEFAULT_WHISPER_MODEL,
    WHISPER_MODELS_DIR,
    detect_hardware,
)

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: List[WordTiming] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    model_name: str
    device_used: str
    compute_type: str
    segments: List[TranscriptSegment]


ProgressCallback = Callable[[float, str], None]


class WhisperTranscriber:
    """Wrapper thread-safe com cache de modelos Faster-Whisper e fallback automático CUDA -> CPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_key: Optional[Tuple[str, str, str]] = None
        self._cached_model = None

    def _resolve_device_and_compute(
        self,
        model_name: str,
        device: str = "auto",
        quality_preset: str = DEFAULT_QUALITY_PRESET,
    ) -> Tuple[str, str]:
        hw = detect_hardware()
        if device == "auto":
            use_cuda = bool(hw["cuda_available"])
        else:
            use_cuda = device == "cuda" and bool(hw["cuda_available"])

        if use_cuda:
            if quality_preset == "high":
                # Evita Sysmem Fallback (lentidão extrema) quando VRAM livre estiver baixa na GPU
                free_vram = hw.get("cuda_free_vram_mb")
                min_fp16_vram = 4200 if model_name in ("large-v3", "large-v2") else 3200
                if isinstance(free_vram, int) and free_vram < min_fp16_vram:
                    logger.info(
                        "VRAM livre (%d MiB) abaixo de %d MiB para FP16 em '%s'; usando int8_float16.",
                        free_vram,
                        min_fp16_vram,
                        model_name,
                    )
                    return "cuda", "int8_float16"
                return "cuda", "float16"
            return "cuda", "int8_float16"
        return "cpu", "int8"

    def _clear_cached_model_unlocked(self) -> None:
        if self._cached_model is not None:
            self._cached_model = None
            self._cached_key = None
            gc.collect()

    def _get_model(
        self,
        model_name: str,
        device: str,
        quality_preset: str = DEFAULT_QUALITY_PRESET,
        compute_override: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        from faster_whisper import WhisperModel

        target_device, compute_type = self._resolve_device_and_compute(model_name, device, quality_preset)
        if compute_override:
            compute_type = compute_override

        with self._lock:
            key = (model_name, target_device, compute_type)
            if self._cached_model is not None and self._cached_key == key:
                return self._cached_model, target_device, compute_type

            # Libera modelo anterior antes de carregar um novo para economizar VRAM/RAM
            self._clear_cached_model_unlocked()

            if progress_callback:
                progress_callback(
                    0.02,
                    f"Carregando modelo Whisper '{model_name}' em {target_device.upper()} ({compute_type}) "
                    f"(na 1ª vez o download pode levar alguns instantes)...",
                )

            cpu_threads = max(4, min(12, (detect_hardware()["cpu_threads"] or 4) // 2))

            try:
                model = WhisperModel(
                    model_name,
                    device=target_device,
                    compute_type=compute_type,
                    download_root=str(WHISPER_MODELS_DIR),
                    cpu_threads=cpu_threads,
                )
                self._cached_model = model
                self._cached_key = key
                return model, target_device, compute_type
            except Exception as exc:
                # Se float16 falhou na GPU (ex: VRAM insuficiente), tenta int8_float16 antes de cair para CPU
                if target_device == "cuda" and compute_type == "float16":
                    logger.warning("Falha ao carregar em CUDA float16 (%s). Tentando CUDA int8_float16...", exc)
                    self._clear_cached_model_unlocked()
                    try:
                        model = WhisperModel(
                            model_name,
                            device="cuda",
                            compute_type="int8_float16",
                            download_root=str(WHISPER_MODELS_DIR),
                            cpu_threads=cpu_threads,
                        )
                        self._cached_model = model
                        self._cached_key = (model_name, "cuda", "int8_float16")
                        return model, "cuda", "int8_float16"
                    except Exception:
                        pass

                if target_device == "cuda":
                    logger.warning("Falha ao inicializar Whisper em CUDA (%s). Tentando fallback para CPU...", exc)
                    self._clear_cached_model_unlocked()
                    if progress_callback:
                        progress_callback(0.04, "GPU indisponível no momento; alternando automaticamente para CPU...")
                    fallback_device, fallback_compute = "cpu", "int8"
                    model = WhisperModel(
                        model_name,
                        device=fallback_device,
                        compute_type=fallback_compute,
                        download_root=str(WHISPER_MODELS_DIR),
                        cpu_threads=cpu_threads,
                    )
                    self._cached_model = model
                    self._cached_key = (model_name, fallback_device, fallback_compute)
                    return model, fallback_device, fallback_compute
                raise

    def transcribe(
        self,
        wav_path: Path | str,
        model_name: str = DEFAULT_WHISPER_MODEL,
        language: Optional[str] = "auto",
        device: str = "auto",
        quality_preset: str = DEFAULT_QUALITY_PRESET,
        initial_prompt: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        _compute_override: Optional[str] = None,
    ) -> TranscriptionResult:
        """Executa transcrição com timestamps por palavra, filtro VAD e controle de qualidade."""
        lang_arg = None if (not language or language == "auto") else language
        prompt_arg = initial_prompt.strip() if initial_prompt and initial_prompt.strip() else None
        is_high_quality = quality_preset == "high"

        model, used_device, used_compute = self._get_model(
            model_name,
            device,
            quality_preset=quality_preset,
            compute_override=_compute_override,
            progress_callback=progress_callback,
        )

        if progress_callback:
            preset_label = "Qualidade Máxima" if is_high_quality else "Equilibrado"
            progress_callback(
                0.08,
                f"Transcrevendo áudio com Whisper '{model_name}' ({used_device.upper()} · {used_compute} · {preset_label})...",
            )

        # Carrega diretamente como array float32 mono 16kHz (evita dupla decodificação e incompatibilidades do PyAV)
        audio_array, sr = sf.read(str(wav_path), dtype="float32")
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1).astype("float32")
        if sr != 16000:
            raise ValueError(f"WAV esperado em 16000 Hz, mas recebido {sr} Hz.")

        def _run_transcribe(active_model):
            # Na GPU usamos beam_size=5 com fallback de temperatura e anti-alucinação (ótimo equilíbrio velocidade/fidelidade);
            # Na CPU evitamos beam excessivo que multiplicaria o tempo de processamento.
            if used_device == "cuda":
                beam = 5
                best_of = 5
                temps = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if is_high_quality else 0.0
            else:
                beam = 5 if is_high_quality else 2
                best_of = 3 if is_high_quality else 1
                temps = (0.0, 0.2, 0.4) if is_high_quality else 0.0

            kwargs = {
                "language": lang_arg,
                "task": "transcribe",
                "beam_size": beam,
                "best_of": best_of,
                "temperature": temps,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "repetition_penalty": 1.05 if is_high_quality else 1.0,
                "hallucination_silence_threshold": 1.5 if is_high_quality else None,
                "vad_filter": True,
                "vad_parameters": {
                    "min_silence_duration_ms": 380,
                    "speech_pad_ms": 280,
                },
                "word_timestamps": True,
                "condition_on_previous_text": True,
                "initial_prompt": prompt_arg,
            }
            return active_model.transcribe(audio_array, **kwargs)

        try:
            segments_iter, info = _run_transcribe(model)
            total_duration = float(getattr(info, "duration", 0.0) or 0.0)
            segments: List[TranscriptSegment] = []

            for seg in segments_iter:
                words: List[WordTiming] = []
                if seg.words:
                    for w in seg.words:
                        w_text = (w.word or "").strip()
                        if not w_text:
                            continue
                        w_start = float(w.start if w.start is not None else seg.start)
                        w_end = float(w.end if w.end is not None else max(w_start, seg.end))
                        words.append(
                            WordTiming(
                                word=w_text,
                                start=w_start,
                                end=max(w_start, w_end),
                                probability=float(getattr(w, "probability", 1.0) or 1.0),
                            )
                        )

                seg_text = (seg.text or "").strip()
                if not seg_text and not words:
                    continue

                # Caso algum segmento não traga palavras individuais, cria fallback proporcional
                if not words and seg_text:
                    tokens = seg_text.split()
                    seg_dur = max(0.05, float(seg.end) - float(seg.start))
                    step = seg_dur / max(1, len(tokens))
                    for idx, tok in enumerate(tokens):
                        words.append(
                            WordTiming(
                                word=tok,
                                start=float(seg.start) + idx * step,
                                end=float(seg.start) + (idx + 1) * step,
                            )
                        )

                segments.append(
                    TranscriptSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=seg_text,
                        words=words,
                    )
                )

                if progress_callback and total_duration > 0:
                    ratio = min(0.99, max(0.1, float(seg.end) / total_duration))
                    pct_display = int(ratio * 100)
                    progress_callback(ratio, f"Transcrevendo fala... ({pct_display}%)")

        except Exception as exc:
            # Libera o modelo GPU antes de qualquer fallback para evitar retenção dupla de VRAM
            err_lower = str(exc).lower()
            is_cuda_dll_err = any(k in err_lower for k in ("cublas", "cudnn", "not found", "cannot be loaded"))
            is_cuda_mem_err = any(k in err_lower for k in ("out of memory", "cuda", "alloc"))

            if used_device == "cuda" and (is_cuda_dll_err or is_cuda_mem_err):
                del model
                with self._lock:
                    self._clear_cached_model_unlocked()

                # Se o erro for falta de DLL CUDA/cuBLAS/cuDNN, pula direto para CPU (não tenta outra precisão CUDA!)
                if is_cuda_dll_err:
                    logger.warning("Biblioteca CUDA ausente (%s). Alternando diretamente para CPU...", exc)
                    if progress_callback:
                        progress_callback(0.1, "Bibliotecas CUDA ausentes neste pacote; usando CPU automaticamente...")
                    return self.transcribe(
                        wav_path=wav_path,
                        model_name=model_name,
                        language=language,
                        device="cpu",
                        quality_preset=quality_preset,
                        initial_prompt=initial_prompt,
                        progress_callback=progress_callback,
                    )

                if used_compute == "float16":
                    logger.warning("VRAM insuficiente em CUDA float16 (%s). Repetindo em CUDA int8_float16...", exc)
                    if progress_callback:
                        progress_callback(0.1, "Otimizando memória da GPU (int8_float16)...")
                    return self.transcribe(
                        wav_path=wav_path,
                        model_name=model_name,
                        language=language,
                        device="cuda",
                        quality_preset=quality_preset,
                        initial_prompt=initial_prompt,
                        progress_callback=progress_callback,
                        _compute_override="int8_float16",
                    )

                logger.warning("Erro CUDA durante transcrição (%s). Repetindo em CPU...", exc)
                if progress_callback:
                    progress_callback(0.1, "Ajustando para CPU devido à memória/runtime da GPU...")
                return self.transcribe(
                    wav_path=wav_path,
                    model_name=model_name,
                    language=language,
                    device="cpu",
                    quality_preset=quality_preset,
                    initial_prompt=initial_prompt,
                    progress_callback=progress_callback,
                )
            raise

        if progress_callback:
            progress_callback(1.0, "Transcrição concluída.")

        return TranscriptionResult(
            language=str(getattr(info, "language", lang_arg or "unknown")),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            model_name=model_name,
            device_used=used_device,
            compute_type=used_compute,
            segments=segments,
        )


# Instância singleton reutilizável entre requisições
transcriber_engine = WhisperTranscriber()
