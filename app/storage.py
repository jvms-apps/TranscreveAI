"""Persistência em JSON do histórico de transcrições na pasta do projeto."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import HISTORY_DIR, UPLOADS_DIR


def _safe_id(record_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", str(record_id))
    if not cleaned:
        raise ValueError("ID de transcrição inválido.")
    return cleaned


def _record_path(record_id: str) -> Path:
    return HISTORY_DIR / f"{_safe_id(record_id)}.json"


def save_transcription(document: Dict[str, Any]) -> Dict[str, Any]:
    """Salva ou atualiza uma transcrição no histórico local."""
    rec_id = _safe_id(str(document["id"]))
    path = _record_path(rec_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return document


def load_transcription(record_id: str) -> Optional[Dict[str, Any]]:
    path = _record_path(record_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_transcriptions() -> List[Dict[str, Any]]:
    """Retorna lista resumida do histórico, ordenada da mais recente para a mais antiga."""
    items: List[Dict[str, Any]] = []
    if not HISTORY_DIR.is_dir():
        return items

    for path in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data.get("id", path.stem),
                    "title": data.get("title") or data.get("filename") or path.stem,
                    "filename": data.get("filename", ""),
                    "created_at": data.get("created_at", ""),
                    "duration": data.get("duration", 0),
                    "language": data.get("language", ""),
                    "model_name": data.get("model_name", ""),
                    "diarization_enabled": data.get("diarization_enabled", False),
                    "speaker_count": len(data.get("speakers") or {}),
                    "total_words": data.get("total_words", 0),
                    "total_paragraphs": len(data.get("paragraphs") or []),
                }
            )
        except Exception:
            continue

    items.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    return items


def delete_transcription(record_id: str) -> bool:
    """Remove a transcrição do histórico e o respectivo arquivo de mídia em cache local (se existir)."""
    path = _record_path(record_id)
    if not path.is_file():
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        media_file = data.get("media_file")
        if media_file:
            media_path = UPLOADS_DIR / Path(str(media_file)).name
            media_path.unlink(missing_ok=True)
    except Exception:
        pass

    path.unlink(missing_ok=True)
    return True


def recalculate_stats(document: Dict[str, Any]) -> Dict[str, Any]:
    """Recalcula contagem de palavras, parágrafos e tempo por locutor após edições do usuário."""
    speakers = document.get("speakers") or {}
    paragraphs = document.get("paragraphs") or []

    for spk in speakers.values():
        spk["word_count"] = 0
        spk["paragraph_count"] = 0
        spk["duration"] = 0.0

    total_words = 0
    for idx, p in enumerate(paragraphs, start=1):
        p["id"] = idx
        text = str(p.get("text") or "").strip()
        wc = len(text.split()) if text else 0
        total_words += wc
        spk_id = str(p.get("speaker_id") or "")
        if spk_id in speakers:
            speakers[spk_id]["word_count"] += wc
            speakers[spk_id]["paragraph_count"] += 1
            dur = max(0.0, float(p.get("end") or 0.0) - float(p.get("start") or 0.0))
            speakers[spk_id]["duration"] = round(float(speakers[spk_id]["duration"]) + dur, 2)

    document["total_words"] = total_words
    document["total_paragraphs"] = len(paragraphs)
    return document
