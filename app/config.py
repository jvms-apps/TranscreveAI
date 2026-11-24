"""Configurações globais, diretórios e inicialização de DLLs CUDA no Windows."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _add_dll_dir_safe(dir_path: Path, added: List[str]) -> None:
    if not dir_path.is_dir():
        return
    bin_str = str(dir_path)
    try:
        os.add_dll_directory(bin_str)
    except (AttributeError, OSError):
        pass
    current_path = os.environ.get("PATH", "")
    if bin_str.lower() not in current_path.lower():
        os.environ["PATH"] = bin_str + os.pathsep + current_path
    if bin_str not in added:
        added.append(bin_str)


def setup_cuda_dlls() -> List[str]:
    """Registra diretórios de DLLs NVIDIA (cuBLAS / cuDNN) no Windows, inclusive no modo empacotado."""
    added: List[str] = []
    if sys.platform != "win32":
        return added

    # 1. Pacotes nvidia.cublas / nvidia.cudnn instalados no ambiente atual ou empacotados
    for pkg_name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            mod = __import__(pkg_name, fromlist=["__file__"])
            pkg_file = getattr(mod, "__file__", None)
            if pkg_file:
                pkg_root = Path(pkg_file).parent
            else:
                pkg_paths = list(getattr(mod, "__path__", []))
                if not pkg_paths:
                    continue
                pkg_root = Path(pkg_paths[0])
            _add_dll_dir_safe(pkg_root / "bin", added)
        except Exception:
            continue

    # 2. Se rodando como EXE Universal, aproveita DLLs CUDA 12 caso já existam no CUDA Toolkit do sistema
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_base.is_dir():
        for vdir in sorted(cuda_base.glob("v12.*"), reverse=True):
            _add_dll_dir_safe(vdir / "bin", added)

    return added


# Configura DLLs CUDA imediatamente ao importar configurações
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
CUDA_DLL_DIRS = setup_cuda_dlls()

# Diretórios base — compatível com execução normal e executável empacotado (PyInstaller)
IS_FROZEN = bool(getattr(sys, "frozen", False))
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def _resolve_project_dir() -> Path:
    if not IS_FROZEN:
        return Path(__file__).resolve().parent.parent
    exe_dir = Path(sys.executable).resolve().parent
    # Verifica se a pasta do executável permite escrita (ex: fora de Program Files)
    try:
        test_file = exe_dir / ".transcreveai_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return exe_dir
    except OSError:
        return Path.home() / "Documents" / "TranscreveAI"


PROJECT_DIR = _resolve_project_dir()
STATIC_DIR = BUNDLE_DIR / "app" / "static"
DATA_DIR = PROJECT_DIR / "dados"
HISTORY_DIR = DATA_DIR / "historico"
EXPORTS_DIR = PROJECT_DIR / "exportacoes"

# Diretórios locais rápidos (%LOCALAPPDATA%) — modelos pesados, uploads e temporários
_local_app_data = os.environ.get("LOCALAPPDATA")
if _local_app_data:
    LOCAL_DATA_DIR = Path(_local_app_data) / "TranscricaoAudios"
else:
    LOCAL_DATA_DIR = Path.home() / ".transcricao_audios"

MODELS_DIR = LOCAL_DATA_DIR / "models"
WHISPER_MODELS_DIR = MODELS_DIR / "whisper"
DIARIZATION_MODELS_DIR = MODELS_DIR / "diarization"
UPLOADS_DIR = LOCAL_DATA_DIR / "uploads"
TEMP_DIR = LOCAL_DATA_DIR / "temp"

for _d in (
    DATA_DIR,
    HISTORY_DIR,
    EXPORTS_DIR,
    LOCAL_DATA_DIR,
    MODELS_DIR,
    WHISPER_MODELS_DIR,
    DIARIZATION_MODELS_DIR,
    UPLOADS_DIR,
    TEMP_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# Formatos suportados
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv",
    ".flv", ".m4v", ".mpeg", ".mpg", ".3gp", ".ts",
}
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
    ".wma", ".opus", ".amr", ".aiff", ".aif",
}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Catálogo de modelos Whisper disponíveis
WHISPER_MODELS: Dict[str, Dict[str, str]] = {
    "large-v3-turbo": {
        "label": "Large v3 Turbo (Recomendado — Rápido e Muito Preciso)",
        "description": "Altíssima precisão em Português e Inglês com excelente velocidade na GPU.",
        "vram": "~3.0 GB (FP16)",
    },
    "large-v3": {
        "label": "Large v3 (Precisão Máxima)",
        "description": "Maior fidelidade da família Whisper para áudios complexos, técnicos ou com ruído.",
        "vram": "~4.2 GB (FP16)",
    },
    "large-v2": {
        "label": "Large v2 (Alta Estabilidade PT-BR)",
        "description": "Excelente estabilidade em português e inglês, muito resistente a repetições.",
        "vram": "~4.0 GB (FP16)",
    },
    "medium": {
        "label": "Medium (Equilibrado)",
        "description": "Boa precisão em PT/EN com menor consumo de memória.",
        "vram": "~2.0 GB",
    },
    "small": {
        "label": "Small (Rápido / Ideal para CPU)",
        "description": "Muito rápido, ótimo para computadores sem placa NVIDIA ou áudios limpos.",
        "vram": "~1.0 GB",
    },
    "base": {
        "label": "Base (Ultrarrápido)",
        "description": "Ideal para testes rápidos e pré-visualizações.",
        "vram": "~0.5 GB",
    },
}

DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# Perfis de qualidade de decodificação Whisper
QUALITY_PRESETS: Dict[str, Dict[str, str]] = {
    "high": {
        "label": "Qualidade Máxima (FP16 + Anti-alucinação)",
        "description": "Precisão nativa na GPU (ajuste automático de VRAM), fallback de temperatura e proteção anti-alucinação.",
    },
    "balanced": {
        "label": "Equilibrado (Mais Rápido)",
        "description": "Ótimo equilíbrio entre velocidade e fidelidade (menor uso de memória).",
    },
}

DEFAULT_QUALITY_PRESET = "high"

# Modelos de embeddings de voz para diarização
DIARIZATION_EMBEDDING_MODELS: Dict[str, Dict[str, str]] = {
    "standard": {
        "label": "Padrão — WeSpeaker ResNet34-LM (26 MB, rápido)",
        "filename": "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
        ),
    },
    "high": {
        "label": "Alta Precisão — WeSpeaker ResNet293-LM (105 MB, melhor p/ vozes parecidas)",
        "filename": "wespeaker_en_voxceleb_resnet293_LM.onnx",
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/wespeaker_en_voxceleb_resnet293_LM.onnx"
        ),
    },
}

# Idiomas prioritários
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "auto": "Detecção Automática (PT / EN / Outros)",
    "pt": "Português",
    "en": "Inglês",
    "es": "Espanhol",
}

# Paleta de cores elegante e legível para diferenciar locutores (UI, DOCX e PDF)
SPEAKER_COLORS: List[str] = [
    "#1D4ED8",  # Azul royal
    "#047857",  # Verde esmeralda
    "#B45309",  # Âmbar escuro
    "#7C3AED",  # Violeta
    "#BE123C",  # Carmesim
    "#0E7490",  # Ciano petróleo
    "#C2410C",  # Terracota
    "#4338CA",  # Índigo
    "#15803D",  # Verde floresta
    "#A21CAF",  # Magenta profundo
]

# Modelos Sherpa-ONNX (100% open-source, sem token do HuggingFace)
DIARIZATION_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
DIARIZATION_SEGMENTATION_DIRNAME = "sherpa-onnx-pyannote-segmentation-3-0"

DIARIZATION_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
DIARIZATION_EMBEDDING_FILENAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"


def _check_cuda_runtime_dlls() -> tuple[bool, Optional[str]]:
    """Confirma que cuBLAS 12 e cuDNN (ops) realmente podem ser carregados no Windows."""
    if sys.platform != "win32":
        return True, None

    try:
        ctypes.WinDLL("cublas64_12.dll")
    except OSError:
        return False, "Biblioteca cublas64_12.dll ausente (usando modo CPU compatível)."

    cudnn_ok = False
    for dll_name in ("cudnn_ops64_9.dll", "cudnn_ops_infer64_8.dll"):
        try:
            ctypes.WinDLL(dll_name)
            cudnn_ok = True
            break
        except OSError:
            continue

    if not cudnn_ok:
        return False, "Biblioteca cuDNN ausente (usando modo CPU compatível)."

    return True, None


def get_gpu_free_vram_mb() -> Optional[int]:
    """Consulta rapidamente a VRAM livre (em MiB) da GPU NVIDIA via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode == 0 and res.stdout.strip():
            first_line = res.stdout.strip().splitlines()[0].strip()
            return int(float(first_line))
    except Exception:
        pass
    return None


def detect_hardware() -> Dict[str, object]:
    """Detecta disponibilidade REAL de GPU CUDA (driver + DLLs cuBLAS/cuDNN) e CPU."""
    cuda_count = 0
    cuda_error = None
    cuda_runtime_ok = False
    free_vram_mb: Optional[int] = None

    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        cuda_error = str(exc)

    if cuda_count > 0:
        cuda_runtime_ok, runtime_err = _check_cuda_runtime_dlls()
        if not cuda_runtime_ok:
            cuda_error = runtime_err
        else:
            free_vram_mb = get_gpu_free_vram_mb()

    cuda_ready = cuda_count > 0 and cuda_runtime_ok

    return {
        "cuda_available": cuda_ready,
        "cuda_device_count": cuda_count if cuda_ready else 0,
        "cuda_gpu_detected_without_runtime": (cuda_count > 0 and not cuda_runtime_ok),
        "cuda_free_vram_mb": free_vram_mb,
        "cuda_error": cuda_error,
        "cpu_threads": os.cpu_count() or 4,
        "recommended_device": "cuda" if cuda_ready else "cpu",
    }
