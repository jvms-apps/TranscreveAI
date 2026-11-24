"""Entrypoint standalone do TranscreveAI (usado no executável .exe empacotado via PyInstaller)."""

from __future__ import annotations

import multiprocessing
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from app import __version__
from app.config import EXPORTS_DIR, detect_hardware
from app.main import app


def find_free_port(preferred_port: int = 8765) -> int:
    """Usa a porta preferida se livre; caso contrário, seleciona uma porta livre automaticamente."""
    for port in (preferred_port, preferred_port + 1, preferred_port + 2, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return int(s.getsockname()[1])
            except OSError:
                continue
    return preferred_port


def open_browser_when_ready(url: str) -> None:
    time.sleep(1.6)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    multiprocessing.freeze_support()

    port = find_free_port(8765)
    url = f"http://127.0.0.1:{port}"
    hw = detect_hardware()
    hw_desc = (
        f"GPU NVIDIA CUDA ativa ({hw['cuda_device_count']} GPU)"
        if hw.get("cuda_available")
        else f"CPU ({hw.get('cpu_threads', 4)} threads)"
    )

    print("=" * 64)
    print(f"  TranscreveAI v{__version__} — Transcrição de Áudios e Vídeos")
    print("=" * 64)
    print(f"  Hardware   : {hw_desc}")
    print(f"  Exportações: {EXPORTS_DIR}")
    print(f"  Endereço   : {url}")
    print("-" * 64)
    print("  O navegador será aberto automaticamente.")
    print("  Mantenha esta janela aberta enquanto utiliza a aplicação.")
    print("  Pressione CTRL+C ou feche esta janela para encerrar.")
    print("=" * 64)

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        print("\nEncerrando TranscreveAI...")
    except Exception as exc:
        print(f"\nErro inesperado: {exc}", file=sys.stderr)
        if getattr(sys, "frozen", False):
            input("\nPressione ENTER para fechar...")
        raise


if __name__ == "__main__":
    main()
