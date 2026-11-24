"""Exportador de transcrição para arquivo de texto puro (.txt) em UTF-8."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from app.core.audio import format_duration_human, format_timestamp

LANGUAGE_LABELS = {
    "pt": "Português",
    "en": "Inglês",
    "es": "Espanhol",
}


def _format_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y às %H:%M")
    except Exception:
        return iso_str


def render_txt_content(
    document: Dict[str, Any],
    include_timestamps: bool = True,
    include_metadata: bool = True,
    include_speakers: bool = True,
) -> str:
    """Gera o conteúdo formatado em texto puro (UTF-8) com divisão em parágrafos por locutor."""
    lines = []
    title = document.get("title") or document.get("filename") or "Transcrição"
    speakers: Dict[str, Dict[str, Any]] = document.get("speakers") or {}
    paragraphs = document.get("paragraphs") or []
    duration = float(document.get("duration") or 0.0)
    force_hours = duration >= 3600

    if include_metadata:
        lang_code = str(document.get("language") or "auto").lower()
        lang_label = LANGUAGE_LABELS.get(lang_code, lang_code.upper())
        lines.append("=" * 72)
        lines.append(f"TRANSCRIÇÃO: {title}")
        lines.append("=" * 72)
        lines.append(f"Arquivo original : {document.get('filename', '-')}")
        lines.append(f"Data             : {_format_date(str(document.get('created_at', '')))}")
        lines.append(f"Duração          : {format_duration_human(duration)}")
        lines.append(f"Idioma detectado : {lang_label}")
        lines.append(f"Modelo           : Whisper {document.get('model_name', '-')}")
        if speakers and document.get("diarization_enabled", False):
            spk_names = ", ".join(info.get("name", sid) for sid, info in speakers.items())
            lines.append(f"Locutores ({len(speakers)})    : {spk_names}")
        lines.append(f"Total de palavras: {document.get('total_words', 0)}")
        lines.append("=" * 72)
        lines.append("")

    show_speaker_header = include_speakers and bool(document.get("diarization_enabled", False))

    for p in paragraphs:
        text = str(p.get("text") or "").strip()
        if not text:
            continue

        spk_id = str(p.get("speaker_id") or "")
        spk_name = (speakers.get(spk_id) or {}).get("name") or spk_id or "Locutor"
        t_start = format_timestamp(float(p.get("start") or 0.0), force_hours=force_hours)
        t_end = format_timestamp(float(p.get("end") or 0.0), force_hours=force_hours)

        header_parts = []
        if include_timestamps:
            header_parts.append(f"[{t_start} - {t_end}]")
        if show_speaker_header:
            header_parts.append(f"{spk_name}:")

        if header_parts:
            lines.append(" ".join(header_parts))
        lines.append(text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_to_txt(
    document: Dict[str, Any],
    output_path: Path | str,
    include_timestamps: bool = True,
    include_metadata: bool = True,
    include_speakers: bool = True,
) -> Path:
    """Salva a transcrição em um arquivo .txt UTF-8."""
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = render_txt_content(
        document,
        include_timestamps=include_timestamps,
        include_metadata=include_metadata,
        include_speakers=include_speakers,
    )
    dst.write_text(content, encoding="utf-8")
    return dst
