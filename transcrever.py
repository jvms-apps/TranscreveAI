"""Interface de linha de comando (CLI) do TranscreveAI.

Exemplos de uso:
  python transcrever.py "reuniao.mp4" --idioma pt --formato all
  python transcrever.py "podcast.mp3" --idioma en --locutores 2 --formato docx
  python transcrever.py "aula.mkv" --sem-locutores --formato pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import DEFAULT_QUALITY_PRESET, DEFAULT_WHISPER_MODEL, EXPORTS_DIR, WHISPER_MODELS
from app.core.pipeline import run_transcription_pipeline
from app.exporters import export_to_docx, export_to_pdf, export_to_txt
from app.storage import save_transcription


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcrever.py",
        description="Transcreve áudios e vídeos (PT/EN) com identificação de locutores e exportação em TXT, DOCX e PDF.",
    )
    parser.add_argument("arquivo", type=Path, help="Caminho do arquivo de áudio ou vídeo.")
    parser.add_argument(
        "--modelo",
        default=DEFAULT_WHISPER_MODEL,
        choices=list(WHISPER_MODELS.keys()),
        help=f"Modelo Whisper a usar (padrão: {DEFAULT_WHISPER_MODEL}).",
    )
    parser.add_argument(
        "--qualidade",
        default=DEFAULT_QUALITY_PRESET,
        choices=["high", "balanced"],
        help="Perfil de precisão do Whisper: 'high' (FP16/Beam 8) ou 'balanced' (padrão: high).",
    )
    parser.add_argument(
        "--melhorar-audio",
        action="store_true",
        help="Normaliza volume (EBU R128) e filtra ruídos antes da transcrição.",
    )
    parser.add_argument(
        "--idioma",
        default="auto",
        help="Idioma da fala: auto, pt, en, es (padrão: auto).",
    )
    parser.add_argument(
        "--formato",
        default="all",
        choices=["txt", "docx", "pdf", "all"],
        help="Formato de exportação: txt, docx, pdf ou all (padrão: all).",
    )
    parser.add_argument(
        "--locutores",
        type=int,
        default=None,
        help="Número exato de locutores (opcional; se omitido, detecta automaticamente).",
    )
    parser.add_argument(
        "--diarizacao-qualidade",
        default="standard",
        choices=["standard", "high"],
        help="Modelo de voz na diarização: 'standard' (ResNet34) ou 'high' (ResNet293).",
    )
    parser.add_argument(
        "--sensibilidade",
        type=float,
        default=0.65,
        help="Threshold de agrupamento de locutores entre 0.35 e 0.90 (padrão: 0.65).",
    )
    parser.add_argument(
        "--sem-locutores",
        action="store_true",
        help="Desativa a diarização (identificação de locutores).",
    )
    parser.add_argument(
        "--sem-timestamps",
        action="store_true",
        help="Oculta horários [MM:SS - MM:SS] nos arquivos exportados.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Texto opcional com nomes próprios ou vocabulário técnico para auxiliar o Whisper.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=EXPORTS_DIR,
        help=f"Pasta de saída dos arquivos exportados (padrão: {EXPORTS_DIR}).",
    )
    parser.add_argument(
        "--dispositivo",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo de inferência do Whisper (padrão: auto).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    src: Path = args.arquivo.resolve()
    if not src.is_file():
        print(f"Erro: arquivo não encontrado: {src}", file=sys.stderr)
        return 1

    last_msg = ""

    def _progress(percent: int, stage: str, msg: str) -> None:
        nonlocal last_msg
        line = f"[{percent:3d}%] ({stage}) {msg}"
        if line != last_msg:
            print(line, flush=True)
            last_msg = line

    doc = run_transcription_pipeline(
        input_path=src,
        original_filename=src.name,
        model_name=args.modelo,
        language=args.idioma,
        quality_preset=args.qualidade,
        enhance_audio=args.melhorar_audio,
        enable_diarization=not args.sem_locutores,
        num_speakers=args.locutores,
        diarization_threshold=args.sensibilidade,
        diarization_quality=args.diarizacao_qualidade,
        initial_prompt=args.prompt,
        device=args.dispositivo,
        progress_callback=_progress,
    )

    save_transcription(doc)

    formats = ["txt", "docx", "pdf"] if args.formato == "all" else [args.formato]
    out_dir: Path = args.saida
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    include_ts = not args.sem_timestamps
    exported = []
    for fmt in formats:
        out_file = out_dir / f"{stem}.{fmt}"
        if fmt == "txt":
            export_to_txt(doc, out_file, include_timestamps=include_ts)
        elif fmt == "docx":
            export_to_docx(doc, out_file, include_timestamps=include_ts)
        elif fmt == "pdf":
            export_to_pdf(doc, out_file, include_timestamps=include_ts)
        exported.append(out_file)

    print("\nArquivos gerados:")
    for path in exported:
        print(f"  - {path}")

    if doc.get("diarization_warning"):
        print(f"\nAviso: {doc['diarization_warning']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
