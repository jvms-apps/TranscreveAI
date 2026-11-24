"""Exportador profissional de transcrição para Microsoft Word (.docx)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

from app.core.audio import format_duration_human, format_timestamp

LANGUAGE_LABELS = {
    "pt": "Português",
    "en": "Inglês",
    "es": "Espanhol",
}


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    cleaned = (hex_color or "#1D4ED8").strip().lstrip("#")
    if len(cleaned) != 6:
        return (29, 78, 216)
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _format_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y às %H:%M")
    except Exception:
        return iso_str


def _set_cell_shading(cell, fill_hex: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex.lstrip("#")}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _set_table_borders_subtle(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        '<w:left w:val="none"/>'
        '<w:right w:val="none"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        '<w:insideV w:val="none"/>'
        "</w:tblBorders>"
    )
    tbl_pr.append(borders)


def _add_left_border_to_paragraph(paragraph, color_hex: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    border = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="18" w:space="10" w:color="{color_hex.lstrip("#")}"/>'
        "</w:pBdr>"
    )
    p_pr.append(border)


def _add_page_number_field(run) -> None:
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def _add_numpages_field(run) -> None:
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "NUMPAGES"
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def export_to_docx(
    document: Dict[str, Any],
    output_path: Path | str,
    include_timestamps: bool = True,
    include_metadata: bool = True,
    include_speakers: bool = True,
) -> Path:
    """Gera documento Word (.docx) estruturado com divisão por locutor e estilos profissionais."""
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Configuração da página (A4 com margens equilibradas)
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

        # Rodapé com paginação
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_left = fp.add_run("Gerado por TranscreveAI   |   Página ")
        r_left.font.name = "Calibri"
        r_left.font.size = Pt(8.5)
        r_left.font.color.rgb = RGBColor(100, 116, 139)

        r_page = fp.add_run()
        r_page.font.name = "Calibri"
        r_page.font.size = Pt(8.5)
        r_page.font.color.rgb = RGBColor(100, 116, 139)
        _add_page_number_field(r_page)

        r_mid = fp.add_run(" de ")
        r_mid.font.name = "Calibri"
        r_mid.font.size = Pt(8.5)
        r_mid.font.color.rgb = RGBColor(100, 116, 139)

        r_total = fp.add_run()
        r_total.font.name = "Calibri"
        r_total.font.size = Pt(8.5)
        r_total.font.color.rgb = RGBColor(100, 116, 139)
        _add_numpages_field(r_total)

    # Estilo base Normal
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)

    title = str(document.get("title") or document.get("filename") or "Transcrição")
    speakers: Dict[str, Dict[str, Any]] = document.get("speakers") or {}
    paragraphs = document.get("paragraphs") or []
    duration = float(document.get("duration") or 0.0)
    force_hours = duration >= 3600

    # Cabeçalho do documento
    kicker = doc.add_paragraph()
    kicker.paragraph_format.space_before = Pt(0)
    kicker.paragraph_format.space_after = Pt(2)
    rk = kicker.add_run("TRANSCRIÇÃO DE ÁUDIO / VÍDEO")
    rk.font.name = "Calibri"
    rk.font.size = Pt(9)
    rk.font.bold = True
    rk.font.color.rgb = RGBColor(37, 99, 235)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(10)
    rt = p_title.add_run(title)
    rt.font.name = "Calibri"
    rt.font.size = Pt(20)
    rt.font.bold = True
    rt.font.color.rgb = RGBColor(15, 23, 42)

    if include_metadata:
        lang_code = str(document.get("language") or "auto").lower()
        lang_label = LANGUAGE_LABELS.get(lang_code, lang_code.upper())
        meta_rows = [
            ("Arquivo original", str(document.get("filename") or "-")),
            ("Data da transcrição", _format_date(str(document.get("created_at") or ""))),
            ("Duração", format_duration_human(duration)),
            ("Idioma detectado", lang_label),
            ("Modelo utilizado", f"Whisper {document.get('model_name', '-')}"),
            ("Estatísticas", f"{document.get('total_words', 0)} palavras em {len(paragraphs)} parágrafo(s)"),
        ]
        if speakers and document.get("diarization_enabled", False):
            spk_summary = ", ".join(
                f"{info.get('name', sid)} ({format_duration_human(float(info.get('duration', 0)))})"
                for sid, info in speakers.items()
            )
            meta_rows.append((f"Locutores ({len(speakers)})", spk_summary))

        table = doc.add_table(rows=len(meta_rows), cols=2)
        table.autofit = False
        _set_table_borders_subtle(table)

        for idx, (label, value) in enumerate(meta_rows):
            row = table.rows[idx]
            c0, c1 = row.cells[0], row.cells[1]
            c0.width = Cm(4.6)
            c1.width = Cm(12.0)
            fill = "F8FAFC" if idx % 2 == 0 else "FFFFFF"
            _set_cell_shading(c0, fill)
            _set_cell_shading(c1, fill)

            p0 = c0.paragraphs[0]
            p0.paragraph_format.space_before = Pt(3)
            p0.paragraph_format.space_after = Pt(3)
            r0 = p0.add_run(label)
            r0.font.name = "Calibri"
            r0.font.size = Pt(9.5)
            r0.font.bold = True
            r0.font.color.rgb = RGBColor(71, 85, 105)

            p1 = c1.paragraphs[0]
            p1.paragraph_format.space_before = Pt(3)
            p1.paragraph_format.space_after = Pt(3)
            r1 = p1.add_run(value)
            r1.font.name = "Calibri"
            r1.font.size = Pt(9.5)
            r1.font.color.rgb = RGBColor(15, 23, 42)

        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_before = Pt(6)
        spacer.paragraph_format.space_after = Pt(6)

    # Corpo da transcrição dividido em parágrafos por locutor
    show_speaker_header = include_speakers and bool(document.get("diarization_enabled", False))

    for p in paragraphs:
        text = str(p.get("text") or "").strip()
        if not text:
            continue

        spk_id = str(p.get("speaker_id") or "")
        spk_info = speakers.get(spk_id) or {}
        spk_name = str(spk_info.get("name") or spk_id or "Locutor")
        spk_hex = str(spk_info.get("color") or "#1D4ED8")
        r_c, g_c, b_c = _hex_to_rgb(spk_hex)

        t_start = format_timestamp(float(p.get("start") or 0.0), force_hours=force_hours)
        t_end = format_timestamp(float(p.get("end") or 0.0), force_hours=force_hours)

        if show_speaker_header or include_timestamps:
            hdr_p = doc.add_paragraph()
            hdr_p.paragraph_format.space_before = Pt(10)
            hdr_p.paragraph_format.space_after = Pt(2)
            hdr_p.paragraph_format.keep_with_next = True
            if show_speaker_header:
                hdr_p.paragraph_format.left_indent = Cm(0.35)
                _add_left_border_to_paragraph(hdr_p, spk_hex)

                r_spk = hdr_p.add_run(spk_name)
                r_spk.font.name = "Calibri"
                r_spk.font.size = Pt(10.5)
                r_spk.font.bold = True
                r_spk.font.color.rgb = RGBColor(r_c, g_c, b_c)

            if include_timestamps:
                prefix = "   " if show_speaker_header else ""
                r_ts = hdr_p.add_run(f"{prefix}[{t_start} – {t_end}]")
                r_ts.font.name = "Calibri"
                r_ts.font.size = Pt(9)
                r_ts.font.color.rgb = RGBColor(100, 116, 139)

        body_p = doc.add_paragraph()
        body_p.paragraph_format.space_before = Pt(1)
        body_p.paragraph_format.space_after = Pt(8)
        body_p.paragraph_format.line_spacing = 1.25
        if show_speaker_header:
            body_p.paragraph_format.left_indent = Cm(0.35)
            _add_left_border_to_paragraph(body_p, spk_hex)

        r_body = body_p.add_run(text)
        r_body.font.name = "Calibri"
        r_body.font.size = Pt(11)
        r_body.font.color.rgb = RGBColor(30, 41, 59)

    doc.save(str(dst))
    return dst
