"""Exportador profissional de transcrição para PDF vetorial com suporte completo a acentuação PT-BR."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.audio import format_duration_human, format_timestamp

LANGUAGE_LABELS = {
    "pt": "Português",
    "en": "Inglês",
    "es": "Espanhol",
}

_FONT_REGISTERED = False
_FONT_REGULAR = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def _ensure_unicode_fonts() -> Tuple[str, str]:
    """Registra fonte TrueType do sistema (Segoe UI / Arial / Calibri) para suportar acentos PT-BR perfeitamente."""
    global _FONT_REGISTERED, _FONT_REGULAR, _FONT_BOLD
    if _FONT_REGISTERED:
        return _FONT_REGULAR, _FONT_BOLD

    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = [
        ("TranscreveUI", "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
        ("TranscreveArial", "arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
        ("TranscreveCalibri", "calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
    ]

    for family, reg_f, bold_f, ital_f, bi_f in candidates:
        reg_path = fonts_dir / reg_f
        bold_path = fonts_dir / bold_f
        if reg_path.is_file() and bold_path.is_file():
            try:
                reg_name = f"{family}-Regular"
                bold_name = f"{family}-Bold"
                pdfmetrics.registerFont(TTFont(reg_name, str(reg_path)))
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))

                ital_name = reg_name
                bi_name = bold_name
                if (fonts_dir / ital_f).is_file():
                    ital_name = f"{family}-Italic"
                    pdfmetrics.registerFont(TTFont(ital_name, str(fonts_dir / ital_f)))
                if (fonts_dir / bi_f).is_file():
                    bi_name = f"{family}-BoldItalic"
                    pdfmetrics.registerFont(TTFont(bi_name, str(fonts_dir / bi_f)))

                registerFontFamily(
                    family,
                    normal=reg_name,
                    bold=bold_name,
                    italic=ital_name,
                    boldItalic=bi_name,
                )
                _FONT_REGULAR = reg_name
                _FONT_BOLD = bold_name
                _FONT_REGISTERED = True
                return _FONT_REGULAR, _FONT_BOLD
            except Exception:
                continue

    _FONT_REGISTERED = True
    return _FONT_REGULAR, _FONT_BOLD


def _format_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y às %H:%M")
    except Exception:
        return iso_str


def _safe_hex_color(hex_str: str, fallback: str = "#1D4ED8") -> colors.Color:
    try:
        return colors.HexColor(hex_str)
    except Exception:
        return colors.HexColor(fallback)


def export_to_pdf(
    document: Dict[str, Any],
    output_path: Path | str,
    include_timestamps: bool = True,
    include_metadata: bool = True,
    include_speakers: bool = True,
) -> Path:
    """Gera documento PDF A4 elegante com cabeçalho, metadados e parágrafos por locutor."""
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    font_regular, font_bold = _ensure_unicode_fonts()
    title = str(document.get("title") or document.get("filename") or "Transcrição")
    speakers: Dict[str, Dict[str, Any]] = document.get("speakers") or {}
    paragraphs = document.get("paragraphs") or []
    duration = float(document.get("duration") or 0.0)
    force_hours = duration >= 3600

    doc = SimpleDocTemplate(
        str(dst),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=title,
        author="TranscreveAI",
    )

    base = getSampleStyleSheet()
    kicker_style = ParagraphStyle(
        "Kicker",
        parent=base["Normal"],
        fontName=font_bold,
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=2,
    )
    title_style = ParagraphStyle(
        "DocTitle",
        parent=base["Title"],
        fontName=font_bold,
        fontSize=18,
        leading=23,
        textColor=colors.HexColor("#0F172A"),
        alignment=0,
        spaceAfter=8,
    )
    meta_label_style = ParagraphStyle(
        "MetaLabel",
        parent=base["Normal"],
        fontName=font_bold,
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#475569"),
    )
    meta_value_style = ParagraphStyle(
        "MetaValue",
        parent=base["Normal"],
        fontName=font_regular,
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#0F172A"),
    )
    body_style = ParagraphStyle(
        "TranscriptBody",
        parent=base["Normal"],
        fontName=font_regular,
        fontSize=10.5,
        leading=15.5,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=10,
        spaceAfter=10,
    )
    body_plain_style = ParagraphStyle(
        "TranscriptBodyPlain",
        parent=body_style,
        leftIndent=0,
    )

    story = []
    story.append(Paragraph("TRANSCRIÇÃO DE ÁUDIO / VÍDEO", kicker_style))
    story.append(Paragraph(escape(title), title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

    if include_metadata:
        lang_code = str(document.get("language") or "auto").lower()
        lang_label = LANGUAGE_LABELS.get(lang_code, lang_code.upper())
        meta_rows = [
            ("Arquivo original", str(document.get("filename") or "-")),
            ("Data da transcrição", _format_date(str(document.get("created_at") or ""))),
            ("Duração", format_duration_human(duration)),
            ("Idioma detectado", lang_label),
            ("Modelo utilizado", f"Whisper {document.get('model_name', '-')}"),
            ("Resumo", f"{document.get('total_words', 0)} palavras em {len(paragraphs)} parágrafo(s)"),
        ]
        if speakers and document.get("diarization_enabled", False):
            spk_summary = ", ".join(
                f"{info.get('name', sid)} ({format_duration_human(float(info.get('duration', 0)))})"
                for sid, info in speakers.items()
            )
            meta_rows.append((f"Locutores ({len(speakers)})", spk_summary))

        table_data = [
            [Paragraph(escape(k), meta_label_style), Paragraph(escape(v), meta_value_style)]
            for k, v in meta_rows
        ]
        meta_table = Table(table_data, colWidths=[4.2 * cm, 12.8 * cm])
        meta_table.setStyle(
            TableStyle(
                [
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#CBD5E1")),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(meta_table)
        story.append(Spacer(1, 14))

    show_speaker_header = include_speakers and bool(document.get("diarization_enabled", False))

    for idx, p in enumerate(paragraphs):
        text = str(p.get("text") or "").strip()
        if not text:
            continue

        spk_id = str(p.get("speaker_id") or "")
        spk_info = speakers.get(spk_id) or {}
        spk_name = str(spk_info.get("name") or spk_id or "Locutor")
        spk_hex = str(spk_info.get("color") or "#1D4ED8")
        spk_color = _safe_hex_color(spk_hex)

        t_start = format_timestamp(float(p.get("start") or 0.0), force_hours=force_hours)
        t_end = format_timestamp(float(p.get("end") or 0.0), force_hours=force_hours)

        if show_speaker_header or include_timestamps:
            header_style = ParagraphStyle(
                f"SpeakerHdr_{idx}",
                parent=base["Normal"],
                fontName=font_bold,
                fontSize=10,
                leading=14,
                textColor=spk_color if show_speaker_header else colors.HexColor("#64748B"),
                spaceBefore=4,
                spaceAfter=2,
                keepWithNext=True,
            )
            parts = []
            if show_speaker_header:
                parts.append(f'<font color="{escape(spk_hex)}">■</font>  <b>{escape(spk_name)}</b>')
            if include_timestamps:
                sep = "   " if parts else ""
                parts.append(f'{sep}<font color="#64748B" size="8.5">[{t_start} – {t_end}]</font>')
            story.append(Paragraph("".join(parts), header_style))

        safe_text = escape(text).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, body_style if show_speaker_header else body_plain_style))

    def _draw_page_chrome(canvas, pdf_doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
        canvas.setLineWidth(0.5)
        canvas.line(2.0 * cm, 1.45 * cm, A4[0] - 2.0 * cm, 1.45 * cm)

        canvas.setFont(font_regular, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(2.0 * cm, 1.05 * cm, f"TranscreveAI — {title[:70]}")
        canvas.drawRightString(A4[0] - 2.0 * cm, 1.05 * cm, f"Página {pdf_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_page_chrome, onLaterPages=_draw_page_chrome)
    return dst
