"""Aplicação Web FastAPI para transcrição de áudios/vídeos com diarização e exportação."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configura DLLs CUDA e caminhos antes de carregar o pipeline
from app import __version__
from app.config import (
    DEFAULT_QUALITY_PRESET,
    DEFAULT_WHISPER_MODEL,
    DIARIZATION_EMBEDDING_MODELS,
    EXPORTS_DIR,
    QUALITY_PRESETS,
    STATIC_DIR,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    TEMP_DIR,
    UPLOADS_DIR,
    WHISPER_MODELS,
    detect_hardware,
)
from app.core.audio import is_supported_media
from app.core.diarizer import diarizer_engine
from app.core.pipeline import run_transcription_pipeline
from app.exporters import export_to_docx, export_to_pdf, export_to_txt
from app.storage import (
    delete_transcription,
    list_transcriptions,
    load_transcription,
    recalculate_stats,
    save_transcription,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("transcreveai")

app = FastAPI(
    title="TranscreveAI",
    description="Transcrição local de áudios e vídeos com identificação de locutores e exportação em TXT, DOCX e PDF.",
    version=__version__,
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Executor com 1 worker para evitar disputas de VRAM na GPU em transcrições simultâneas
_executor = ThreadPoolExecutor(max_workers=1)
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


class SpeakerUpdate(BaseModel):
    name: str
    color: Optional[str] = None


class ParagraphUpdate(BaseModel):
    id: Optional[int] = None
    speaker_id: str
    start: float
    end: float
    text: str


class TranscriptionUpdatePayload(BaseModel):
    title: Optional[str] = None
    speakers: Optional[Dict[str, SpeakerUpdate]] = None
    paragraphs: Optional[List[ParagraphUpdate]] = None


def _sanitize_filename(name: str) -> str:
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return (base[:90] or "transcricao").strip()


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_job_worker(
    job_id: str,
    input_path: Path,
    original_filename: str,
    media_stored_filename: str,
    model_name: str,
    language: str,
    quality_preset: str,
    enhance_audio: bool,
    enable_diarization: bool,
    num_speakers: Optional[int],
    diarization_threshold: float,
    diarization_quality: str,
    initial_prompt: Optional[str],
) -> None:
    _update_job(job_id, status="processing", stage="extracting", progress=1, message="Iniciando processamento...")

    def _on_progress(percent: int, stage: str, message: str) -> None:
        _update_job(
            job_id,
            status="processing",
            stage=stage,
            progress=percent,
            message=message,
        )

    try:
        document = run_transcription_pipeline(
            input_path=input_path,
            original_filename=original_filename,
            model_name=model_name,
            language=language,
            quality_preset=quality_preset,
            enhance_audio=enhance_audio,
            enable_diarization=enable_diarization,
            num_speakers=num_speakers,
            diarization_threshold=diarization_threshold,
            diarization_quality=diarization_quality,
            initial_prompt=initial_prompt,
            media_stored_filename=media_stored_filename,
            record_id=job_id,
            progress_callback=_on_progress,
        )
        save_transcription(document)
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Transcrição pronta!",
            transcription_id=document["id"],
        )
    except Exception as exc:
        logger.exception("Erro ao processar job %s", job_id)
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            message=f"Erro durante o processamento: {exc}",
            error=str(exc),
        )


@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/system-info")
def get_system_info() -> Dict[str, Any]:
    hw = detect_hardware()
    return {
        "version": __version__,
        "hardware": hw,
        "default_model": DEFAULT_WHISPER_MODEL,
        "models": WHISPER_MODELS,
        "default_quality_preset": DEFAULT_QUALITY_PRESET,
        "quality_presets": QUALITY_PRESETS,
        "diarization_models": {k: v["label"] for k, v in DIARIZATION_EMBEDDING_MODELS.items()},
        "languages": SUPPORTED_LANGUAGES,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "diarization_models_ready": diarizer_engine.models_ready(),
        "exports_dir": str(EXPORTS_DIR),
    }


@app.post("/api/transcribe")
async def create_transcription_job(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_WHISPER_MODEL),
    language: str = Form("auto"),
    quality_preset: str = Form(DEFAULT_QUALITY_PRESET),
    enhance_audio: bool = Form(False),
    enable_diarization: bool = Form(True),
    num_speakers: Optional[str] = Form(None),
    diarization_threshold: float = Form(0.65),
    diarization_quality: str = Form("standard"),
    initial_prompt: Optional[str] = Form(None),
) -> JSONResponse:
    original_name = Path(file.filename or "arquivo.wav").name
    ext = Path(original_name).suffix.lower()

    if not is_supported_media(Path(original_name)):
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{ext or 'desconhecido'}' não suportado. Envie um arquivo de áudio ou vídeo válido.",
        )

    if model_name not in WHISPER_MODELS:
        model_name = DEFAULT_WHISPER_MODEL
    if quality_preset not in QUALITY_PRESETS:
        quality_preset = DEFAULT_QUALITY_PRESET
    if diarization_quality not in DIARIZATION_EMBEDDING_MODELS:
        diarization_quality = "standard"

    parsed_num_speakers: Optional[int] = None
    if num_speakers and str(num_speakers).strip() not in ("", "auto", "0", "-1"):
        try:
            n_val = int(str(num_speakers).strip())
            if 1 <= n_val <= 20:
                parsed_num_speakers = n_val
        except ValueError:
            parsed_num_speakers = None

    clamped_threshold = max(0.30, min(0.95, float(diarization_threshold)))
    job_id = uuid.uuid4().hex[:12]
    stored_media_name = f"{job_id}{ext}"
    stored_media_path = UPLOADS_DIR / stored_media_name

    # Salva upload em streaming (suporta arquivos grandes de vídeo sem consumir RAM excessiva)
    try:
        with open(stored_media_path, "wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
    except Exception as exc:
        stored_media_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivo enviado: {exc}")
    finally:
        await file.close()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": original_name,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "Na fila para processamento...",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "transcription_id": None,
            "error": None,
        }

    _executor.submit(
        _run_job_worker,
        job_id,
        stored_media_path,
        original_name,
        stored_media_name,
        model_name,
        language,
        quality_preset,
        enhance_audio,
        enable_diarization,
        parsed_num_speakers,
        clamped_threshold,
        diarization_quality,
        initial_prompt,
    )

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada.")
        return dict(job)


@app.get("/api/transcriptions")
def api_list_transcriptions() -> Dict[str, Any]:
    return {"items": list_transcriptions()}


@app.get("/api/transcriptions/{record_id}")
def api_get_transcription(record_id: str) -> Dict[str, Any]:
    doc = load_transcription(record_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")
    media_file = doc.get("media_file")
    doc["has_media"] = bool(media_file and (UPLOADS_DIR / Path(str(media_file)).name).is_file())
    return doc


@app.put("/api/transcriptions/{record_id}")
def api_update_transcription(record_id: str, payload: TranscriptionUpdatePayload) -> Dict[str, Any]:
    doc = load_transcription(record_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")

    if payload.title is not None and payload.title.strip():
        doc["title"] = payload.title.strip()

    if payload.speakers is not None:
        existing_speakers = doc.get("speakers") or {}
        for spk_id, spk_upd in payload.speakers.items():
            if spk_id in existing_speakers:
                if spk_upd.name.strip():
                    existing_speakers[spk_id]["name"] = spk_upd.name.strip()
                if spk_upd.color:
                    existing_speakers[spk_id]["color"] = spk_upd.color
            else:
                existing_speakers[spk_id] = {
                    "id": spk_id,
                    "name": spk_upd.name.strip() or spk_id,
                    "color": spk_upd.color or "#1D4ED8",
                    "word_count": 0,
                    "paragraph_count": 0,
                    "duration": 0.0,
                }
        doc["speakers"] = existing_speakers

    if payload.paragraphs is not None:
        doc["paragraphs"] = [
            {
                "id": idx,
                "speaker_id": p.speaker_id,
                "start": round(float(p.start), 2),
                "end": round(float(p.end), 2),
                "text": p.text.strip(),
            }
            for idx, p in enumerate(payload.paragraphs, start=1)
            if p.text.strip()
        ]

    recalculate_stats(doc)
    save_transcription(doc)
    media_file = doc.get("media_file")
    doc["has_media"] = bool(media_file and (UPLOADS_DIR / Path(str(media_file)).name).is_file())
    return doc


@app.delete("/api/transcriptions/{record_id}")
def api_delete_transcription(record_id: str) -> Dict[str, Any]:
    deleted = delete_transcription(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")
    return {"deleted": True, "id": record_id}


@app.get("/api/media/{record_id}")
def api_get_media(record_id: str) -> FileResponse:
    doc = load_transcription(record_id)
    if not doc or not doc.get("media_file"):
        raise HTTPException(status_code=404, detail="Mídia não disponível para esta transcrição.")
    media_path = UPLOADS_DIR / Path(str(doc["media_file"])).name
    if not media_path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo de mídia não encontrado no cache local.")
    return FileResponse(path=str(media_path), filename=str(doc.get("filename") or media_path.name))


@app.get("/api/transcriptions/{record_id}/export/{fmt}")
def api_export_transcription(
    record_id: str,
    fmt: str,
    include_timestamps: bool = Query(True),
    include_metadata: bool = Query(True),
    include_speakers: bool = Query(True),
    save_copy: bool = Query(True),
) -> FileResponse:
    fmt_clean = fmt.lower().strip()
    if fmt_clean not in ("txt", "docx", "pdf"):
        raise HTTPException(status_code=400, detail="Formato inválido. Use 'txt', 'docx' ou 'pdf'.")

    doc = load_transcription(record_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")

    base_name = _sanitize_filename(str(doc.get("title") or doc.get("filename") or record_id))
    download_filename = f"{base_name}.{fmt_clean}"

    target_dir = EXPORTS_DIR if save_copy else TEMP_DIR
    target_path = target_dir / download_filename

    if fmt_clean == "txt":
        export_to_txt(
            doc,
            target_path,
            include_timestamps=include_timestamps,
            include_metadata=include_metadata,
            include_speakers=include_speakers,
        )
        media_type = "text/plain; charset=utf-8"
    elif fmt_clean == "docx":
        export_to_docx(
            doc,
            target_path,
            include_timestamps=include_timestamps,
            include_metadata=include_metadata,
            include_speakers=include_speakers,
        )
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        export_to_pdf(
            doc,
            target_path,
            include_timestamps=include_timestamps,
            include_metadata=include_metadata,
            include_speakers=include_speakers,
        )
        media_type = "application/pdf"

    return FileResponse(
        path=str(target_path),
        media_type=media_type,
        filename=download_filename,
    )


@app.post("/api/open-exports-folder")
def api_open_exports_folder() -> Dict[str, Any]:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        try:
            os.startfile(str(EXPORTS_DIR))
        except Exception:
            subprocess.Popen(["explorer", str(EXPORTS_DIR)])
    return {"opened": True, "path": str(EXPORTS_DIR)}
