"""Suite de validação ponta a ponta do TranscreveAI.

Testa:
1. Alinhamento palavra-locutor e suavização de parágrafos (aligner.py)
2. Exportação em TXT, DOCX e PDF com acentuação PT-BR e EN
3. Pipeline real completo com vídeo MP4 multi-locutor (FFmpeg + Faster-Whisper GPU/CPU + Sherpa-ONNX)
4. API Web FastAPI completa (upload, progresso, edição/renomeação de locutores, exportação, limpeza)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from app.config import TEMP_DIR, detect_hardware
from app.core.aligner import align_transcript_and_speakers
from app.core.audio import get_ffmpeg_executable
from app.core.diarizer import SpeakerTurn
from app.core.pipeline import run_transcription_pipeline
from app.core.transcriber import TranscriptSegment, WordTiming
from app.exporters import export_to_docx, export_to_pdf, export_to_txt
from app.main import app


def test_aligner_unit() -> None:
    print("\n[1/4] Testando alinhamento palavra-locutor e suavização...")
    segments = [
        TranscriptSegment(
            start=0.0,
            end=5.0,
            text="Olá pessoal, vamos começar nossa reunião agora.",
            words=[
                WordTiming("Olá", 0.0, 0.5),
                WordTiming("pessoal,", 0.5, 1.1),
                WordTiming("vamos", 1.2, 1.8),
                WordTiming("começar", 1.9, 2.6),
                WordTiming("nossa", 2.7, 3.2),
                WordTiming("reunião", 3.3, 4.1),
                WordTiming("agora.", 4.2, 5.0),
            ],
        ),
        TranscriptSegment(
            start=5.6,
            end=10.5,
            text="Perfeito! Let's review the quarterly metrics today.",
            words=[
                WordTiming("Perfeito!", 5.6, 6.4),
                WordTiming("Let's", 6.6, 7.1),
                WordTiming("review", 7.2, 7.8),
                WordTiming("the", 7.9, 8.2),
                WordTiming("quarterly", 8.3, 9.2),
                WordTiming("metrics", 9.3, 9.9),
                WordTiming("today.", 10.0, 10.5),
            ],
        ),
    ]
    # Inclui uma micro-oscilação espúria no meio da 1ª frase (speaker 1 por 0.3s)
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker=5),
        SpeakerTurn(start=2.0, end=2.5, speaker=2),  # micro-oscilação de 1 palavra
        SpeakerTurn(start=2.5, end=5.1, speaker=5),
        SpeakerTurn(start=5.5, end=10.6, speaker=2),
    ]

    paragraphs, speakers = align_transcript_and_speakers(segments, turns, enable_diarization=True)
    assert len(paragraphs) == 2, f"Esperado 2 parágrafos, obtido {len(paragraphs)}: {paragraphs}"
    assert len(speakers) == 2, f"Esperado 2 locutores, obtido {len(speakers)}"
    assert paragraphs[0].speaker_id == "SPEAKER_01"
    assert paragraphs[1].speaker_id == "SPEAKER_02"
    assert "começar" in paragraphs[0].text
    assert "quarterly" in paragraphs[1].text
    print("  OK — Alinhamento e suavização funcionando perfeitamente!")


def test_exporters_unit() -> None:
    print("\n[2/4] Testando exportadores TXT, DOCX e PDF com acentuação PT-BR e EN...")
    sample_doc = {
        "id": "unit_sample",
        "title": "Reunião Estratégica — Português & English",
        "filename": "reuniao_teste.mp4",
        "created_at": "2026-11-24T14:00:00",
        "duration": 75.4,
        "language": "pt",
        "model_name": "large-v3-turbo",
        "device_used": "cuda",
        "diarization_enabled": True,
        "total_words": 26,
        "total_paragraphs": 2,
        "speakers": {
            "SPEAKER_01": {
                "id": "SPEAKER_01",
                "name": "Ana Cláudia",
                "color": "#1D4ED8",
                "word_count": 13,
                "paragraph_count": 1,
                "duration": 35.0,
            },
            "SPEAKER_02": {
                "id": "SPEAKER_02",
                "name": "João Pedro",
                "color": "#047857",
                "word_count": 13,
                "paragraph_count": 1,
                "duration": 40.0,
            },
        },
        "paragraphs": [
            {
                "id": 1,
                "speaker_id": "SPEAKER_01",
                "start": 0.0,
                "end": 35.0,
                "text": "Atenção à programação: validação de acentuação em português (ç, ã, õ, é, ê, ú).",
            },
            {
                "id": 2,
                "speaker_id": "SPEAKER_02",
                "start": 35.5,
                "end": 75.4,
                "text": "Sounds great! We are ready to deploy the transcription service to production.",
            },
        ],
    }

    out_dir = TEMP_DIR / "test_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = export_to_txt(sample_doc, out_dir / "teste.txt")
    docx_path = export_to_docx(sample_doc, out_dir / "teste.docx")
    pdf_path = export_to_pdf(sample_doc, out_dir / "teste.pdf")

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "Ana Cláudia:" in txt_content
    assert "João Pedro:" in txt_content
    assert "acentuação em português (ç, ã, õ, é, ê, ú)" in txt_content

    docx_loaded = DocxDocument(str(docx_path))
    docx_text = "\n".join(p.text for p in docx_loaded.paragraphs)
    assert "Ana Cláudia" in docx_text
    assert "João Pedro" in docx_text
    assert "acentuação em português" in docx_text

    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-"), "PDF gerado não tem assinatura %PDF-"
    assert len(pdf_bytes) > 5000, f"PDF parece pequeno demais ({len(pdf_bytes)} bytes)"
    print(f"  OK — TXT ({txt_path.stat().st_size} B), DOCX ({docx_path.stat().st_size} B), PDF ({len(pdf_bytes)} B)")


def _create_synthetic_multispk_mp4(output_mp4: Path) -> Path:
    """Cria um vídeo MP4 real com 2 locutores de timbres distintos usando Windows SAPI + FFmpeg."""
    ffmpeg = get_ffmpeg_executable()
    work = TEMP_DIR / "synth_media"
    work.mkdir(parents=True, exist_ok=True)

    seg1_raw = work / "seg1_raw.wav"
    seg2_raw = work / "seg2_raw.wav"
    seg3_raw = work / "seg3_raw.wav"
    seg4_raw = work / "seg4_raw.wav"

    # Gera falas reais em áudio via System.Speech (Windows)
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | Where-Object {{ $_.Enabled }}
$ptVoice = ($voices | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'pt*' }} | Select-Object -First 1).VoiceInfo.Name
$enVoice = ($voices | Where-Object {{ $_.VoiceInfo.Culture.Name -like 'en*' }} | Select-Object -First 1).VoiceInfo.Name
$anyVoice = ($voices | Select-Object -First 1).VoiceInfo.Name
if (-not $ptVoice) {{ $ptVoice = $anyVoice }}
if (-not $enVoice) {{ $enVoice = $anyVoice }}
Write-Output "VOICES: PT=$ptVoice | EN=$enVoice"

function Speak-ToWav($voiceName, $rate, $text, $path) {{
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    if ($voiceName) {{ $s.SelectVoice($voiceName) }}
    $s.Rate = $rate
    $s.SetOutputToWaveFile($path)
    $s.Speak($text)
    $s.Dispose()
}}

Speak-ToWav $ptVoice 0 "Bom dia pessoal, obrigado por participarem desta reuniao sobre o novo sistema de transcricao automatica de videos." "{seg1_raw}"
Speak-ToWav $enVoice 0 "Good morning everyone! Thank you for inviting me. The automatic transcription and speaker diarization work really well." "{seg2_raw}"
Speak-ToWav $ptVoice 0 "Excelente! Agora vamos testar a exportacao dos arquivos em texto, documento Word e tambem em formato PDF." "{seg3_raw}"
Speak-ToWav $enVoice 0 "That sounds fantastic. Let us verify that each speaker is placed into a separate clean paragraph." "{seg4_raw}"
"""
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        raise RuntimeError(f"Falha ao gerar áudio sintético SAPI: {res.stderr}")
    print(" ", res.stdout.strip())

    # Ajusta pitch/timbre distinto no locutor 2 (voz mais grave) e concatena com silêncio de 0.8s entre falas
    # e empacota em vídeo MP4 real (H.264 + AAC)
    filter_complex = (
        "[0:a]aresample=16000,aformat=channel_layouts=mono[a0];"
        "[1:a]asetrate=16000*0.78,aresample=16000,atempo=1.22,aformat=channel_layouts=mono[a1];"
        "[2:a]aresample=16000,aformat=channel_layouts=mono[a2];"
        "[3:a]asetrate=16000*0.78,aresample=16000,atempo=1.22,aformat=channel_layouts=mono[a3];"
        "anullsrc=r=16000:cl=mono:d=0.8[sil1];"
        "anullsrc=r=16000:cl=mono:d=0.8[sil2];"
        "anullsrc=r=16000:cl=mono:d=0.8[sil3];"
        "[a0][sil1][a1][sil2][a2][sil3][a3]concat=n=7:v=0:a=1[aout]"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(seg1_raw),
        "-i",
        str(seg2_raw),
        "-i",
        str(seg3_raw),
        "-i",
        str(seg4_raw),
        "-f",
        "lavfi",
        "-i",
        "color=c=#1e293b:s=320x180:r=15",
        "-filter_complex",
        filter_complex,
        "-map",
        "4:v",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(output_mp4),
    ]
    r2 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r2.returncode != 0 or not output_mp4.is_file():
        raise RuntimeError(f"Falha ao empacotar MP4 de teste: {r2.stderr}")
    return output_mp4


def test_real_end_to_end_pipeline_and_api() -> None:
    print("\n[3/4] Gerando vídeo MP4 real multi-locutor (PT + EN) e executando pipeline completo...")
    hw = detect_hardware()
    print(f"  Hardware detectado: {hw}")

    sample_mp4 = TEMP_DIR / "teste_multilocutor_pt_en.mp4"
    _create_synthetic_multispk_mp4(sample_mp4)
    print(f"  Vídeo MP4 gerado: {sample_mp4} ({sample_mp4.stat().st_size} bytes)")

    def _log_progress(pct: int, stage: str, msg: str) -> None:
        print(f"    [{pct:3d}%] {stage}: {msg}")

    # Usa modelo 'small' em modo Qualidade Máxima (float16 na GPU + aprimoramento de áudio)
    doc = run_transcription_pipeline(
        input_path=sample_mp4,
        original_filename="reuniao_bilingue_teste.mp4",
        model_name="small",
        language="auto",
        quality_preset="high",
        enhance_audio=True,
        enable_diarization=True,
        num_speakers=None,
        diarization_threshold=0.65,
        diarization_quality="standard",
        progress_callback=_log_progress,
    )

    print(
        f"  Resultado pipeline: device={doc['device_used']} ({doc['compute_type']}), "
        f"idioma={doc['language']}, duracao={doc['duration']}s, "
        f"locutores={len(doc['speakers'])}, paragrafos={len(doc['paragraphs'])}"
    )
    if doc.get("diarization_warning"):
        raise AssertionError(f"Diarização retornou aviso/erro: {doc['diarization_warning']}")
    assert doc["diarization_enabled"] is True, "Diarização deveria estar habilitada"
    assert len(doc["paragraphs"]) >= 2, f"Esperado ao menos 2 parágrafos, obtido {len(doc['paragraphs'])}"
    assert len(doc["speakers"]) >= 2, f"Esperado ao menos 2 locutores distintos, obtido {doc['speakers']}"

    for p in doc["paragraphs"]:
        spk_name = doc["speakers"][p["speaker_id"]]["name"]
        print(f"    [{p['start']:5.1f}s - {p['end']:5.1f}s] {spk_name}: {p['text']}")

    print("\n[4/4] Testando endpoints HTTP do servidor FastAPI (upload, polling, edição, exportação TXT/DOCX/PDF)...")
    client = TestClient(app)

    r_home = client.get("/")
    assert r_home.status_code == 200 and "TranscreveAI" in r_home.text

    r_sys = client.get("/api/system-info")
    assert r_sys.status_code == 200
    sys_info = r_sys.json()
    assert sys_info["diarization_models_ready"] is True

    with open(sample_mp4, "rb") as f:
        r_up = client.post(
            "/api/transcribe",
            files={"file": ("demo_video.mp4", f, "video/mp4")},
            data={
                "model_name": "small",
                "language": "auto",
                "enable_diarization": "true",
                "num_speakers": "auto",
                "diarization_threshold": "0.65",
            },
        )
    assert r_up.status_code == 200, r_up.text
    job_id = r_up.json()["job_id"]

    # Aguarda conclusão do job
    rec_id = None
    for _ in range(180):
        rj = client.get(f"/api/jobs/{job_id}").json()
        if rj["status"] == "completed":
            rec_id = rj["transcription_id"]
            break
        if rj["status"] == "failed":
            raise AssertionError(f"Job falhou na API: {rj}")
        time.sleep(0.5)
    assert rec_id is not None, "Timeout aguardando conclusão do job na API"

    # Carrega transcrição e renomeia locutores
    r_doc = client.get(f"/api/transcriptions/{rec_id}")
    assert r_doc.status_code == 200
    loaded = r_doc.json()
    assert loaded["has_media"] is True

    renamed_speakers = {}
    for idx, (sid, info) in enumerate(loaded["speakers"].items(), start=1):
        renamed_speakers[sid] = {"name": f"Participante {idx}", "color": info["color"]}

    r_upd = client.put(
        f"/api/transcriptions/{rec_id}",
        json={"title": "Demo Validada", "speakers": renamed_speakers},
    )
    assert r_upd.status_code == 200
    assert r_upd.json()["title"] == "Demo Validada"

    # Testa streaming de mídia e exportações nos 3 formatos
    r_media = client.get(f"/api/media/{rec_id}")
    assert r_media.status_code == 200 and len(r_media.content) > 1000

    for fmt, expected_prefix in [("txt", None), ("docx", b"PK"), ("pdf", b"%PDF-")]:
        r_exp = client.get(f"/api/transcriptions/{rec_id}/export/{fmt}?save_copy=false")
        assert r_exp.status_code == 200, f"Erro ao exportar {fmt}: {r_exp.text}"
        if expected_prefix:
            assert r_exp.content.startswith(expected_prefix), f"Assinatura inválida em {fmt}"
        else:
            txt_out = r_exp.content.decode("utf-8")
            assert "Participante 1:" in txt_out
        print(f"  OK — API /export/{fmt} ({len(r_exp.content)} bytes)")

    # Limpa o registro de teste do histórico para deixar o app limpo para o usuário
    r_del = client.delete(f"/api/transcriptions/{rec_id}")
    assert r_del.status_code == 200
    print("  OK — Registro de teste limpo do histórico!")


if __name__ == "__main__":
    test_aligner_unit()
    test_exporters_unit()
    test_real_end_to_end_pipeline_and_api()
    print("\n====================================================")
    print("  TODOS OS TESTES PASSARAM COM SUCESSO!")
    print("====================================================")
