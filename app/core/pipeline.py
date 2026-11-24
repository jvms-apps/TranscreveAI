"""Orquestrador completo do pipeline: extração -> transcrição -> diarização -> alinhamento."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.config import DEFAULT_QUALITY_PRESET, DEFAULT_WHISPER_MODEL, TEMP_DIR
from app.core.aligner import align_transcript_and_speakers
from app.core.audio import extract_audio_to_wav
from app.core.diarizer import diarizer_engine
from app.core.transcriber import transcriber_engine

logger = logging.getLogger(__name__)

# (percentual 0..100, etapa, mensagem detalhada)
PipelineProgressCallback = Callable[[int, str, str], None]


def run_transcription_pipeline(
    input_path: Path | str,
    original_filename: Optional[str] = None,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str = "auto",
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    enhance_audio: bool = False,
    enable_diarization: bool = True,
    num_speakers: Optional[int] = None,
    diarization_threshold: float = 0.65,
    diarization_quality: str = "standard",
    initial_prompt: Optional[str] = None,
    device: str = "auto",
    media_stored_filename: Optional[str] = None,
    record_id: Optional[str] = None,
    progress_callback: Optional[PipelineProgressCallback] = None,
) -> Dict[str, Any]:
    """Executa todas as etapas e devolve o documento estruturado da transcrição."""
    started_at = time.perf_counter()
    src_path = Path(input_path)
    job_id = record_id or uuid.uuid4().hex[:12]
    display_filename = original_filename or src_path.name
    temp_wav = TEMP_DIR / f"{job_id}_16k.wav"
    temp_wav_enhanced = TEMP_DIR / f"{job_id}_16k_enhanced.wav"

    def _report(percent: int, stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(max(0, min(100, int(percent))), stage, message)

    try:
        # Etapa 1: Extração de áudio (0% -> 10%)
        _report(2, "extracting", f"Extraindo faixa de áudio de '{display_filename}'...")
        audio_meta = extract_audio_to_wav(src_path, temp_wav, enhance_audio=False)
        whisper_wav = temp_wav
        if enhance_audio:
            _report(6, "extracting", "Aplicando filtro de voz e normalização (mantendo áudio original para locutores)...")
            extract_audio_to_wav(src_path, temp_wav_enhanced, enhance_audio=True)
            whisper_wav = temp_wav_enhanced
        _report(10, "extracting", f"Áudio preparado ({audio_meta.duration:.1f}s em WAV 16kHz mono).")

        # Pesos das etapas conforme diarização ativada ou não
        transcribe_start, transcribe_end = (10, 62) if enable_diarization else (10, 92)
        diarize_start, diarize_end = (62, 94)

        # Etapa 2: Transcrição com Faster-Whisper
        def _on_transcribe(frac: float, msg: str) -> None:
            pct = transcribe_start + int(frac * (transcribe_end - transcribe_start))
            _report(pct, "transcribing", msg)

        tr_result = transcriber_engine.transcribe(
            wav_path=whisper_wav,
            model_name=model_name,
            language=language,
            device=device,
            quality_preset=quality_preset,
            initial_prompt=initial_prompt,
            progress_callback=_on_transcribe,
        )

        # Etapa 3: Diarização de locutores (opcional)
        speaker_turns = None
        diarization_warning: Optional[str] = None
        diarization_applied = False

        if enable_diarization:
            def _on_diarize(frac: float, msg: str) -> None:
                pct = diarize_start + int(frac * (diarize_end - diarize_start))
                _report(pct, "diarizing", msg)

            try:
                speaker_turns = diarizer_engine.diarize(
                    wav_path=temp_wav,
                    num_speakers=num_speakers,
                    threshold=diarization_threshold,
                    embedding_profile=diarization_quality,
                    progress_callback=_on_diarize,
                )
                diarization_applied = True
            except Exception as exc:
                logger.exception("Falha na diarização; continuando com transcrição sem separação de locutores.")
                diarization_warning = (
                    f"A transcrição foi concluída, mas a identificação de locutores falhou: {exc}"
                )
                speaker_turns = None
                diarization_applied = False

        # Etapa 4: Alinhamento e criação de parágrafos
        _report(95, "aligning", "Organizando falas e dividindo em parágrafos por locutor...")
        paragraphs, speakers_meta = align_transcript_and_speakers(
            segments=tr_result.segments,
            speaker_turns=speaker_turns,
            enable_diarization=diarization_applied,
        )

        elapsed = round(time.perf_counter() - started_at, 2)
        total_words = sum(len(p.text.split()) for p in paragraphs)

        document: Dict[str, Any] = {
            "id": job_id,
            "title": Path(display_filename).stem,
            "filename": display_filename,
            "media_file": media_stored_filename,
            "is_video": audio_meta.is_video,
            "file_size_bytes": audio_meta.file_size_bytes,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration": round(audio_meta.duration, 2),
            "processing_seconds": elapsed,
            "language": tr_result.language,
            "language_probability": round(tr_result.language_probability, 3),
            "model_name": tr_result.model_name,
            "device_used": tr_result.device_used,
            "compute_type": tr_result.compute_type,
            "diarization_enabled": diarization_applied,
            "diarization_warning": diarization_warning,
            "total_words": total_words,
            "total_paragraphs": len(paragraphs),
            "speakers": {spk_id: info.to_dict() for spk_id, info in speakers_meta.items()},
            "paragraphs": [p.to_dict() for p in paragraphs],
        }

        _report(100, "completed", f"Concluído em {elapsed:.1f}s ({len(paragraphs)} parágrafos, {len(speakers_meta)} locutor(es)).")
        return document

    finally:
        # Remove WAVs temporários para economizar disco
        for f in (temp_wav, temp_wav_enhanced):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
