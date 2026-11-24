"""Diarização de locutores 100% local e sem token usando Sherpa-ONNX (Pyannote 3.0 + WeSpeaker)."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.config import (
    DIARIZATION_EMBEDDING_MODELS,
    DIARIZATION_MODELS_DIR,
    DIARIZATION_SEGMENTATION_DIRNAME,
    DIARIZATION_SEGMENTATION_URL,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int


class SpeakerDiarizer:
    """Identifica quem falou em cada trecho do áudio usando modelos ONNX abertos."""

    def __init__(self) -> None:
        self._download_lock = threading.Lock()

    @property
    def segmentation_model_path(self) -> Path:
        return DIARIZATION_MODELS_DIR / DIARIZATION_SEGMENTATION_DIRNAME / "model.onnx"

    def get_embedding_model_info(self, profile: str = "standard") -> Tuple[Path, str, str]:
        info = DIARIZATION_EMBEDDING_MODELS.get(profile) or DIARIZATION_EMBEDDING_MODELS["standard"]
        return DIARIZATION_MODELS_DIR / info["filename"], info["url"], info["label"]

    @property
    def embedding_model_path(self) -> Path:
        path, _, _ = self.get_embedding_model_info("standard")
        return path

    def models_ready(self, profile: str = "standard") -> bool:
        emb_path, _, _ = self.get_embedding_model_info(profile)
        return self.segmentation_model_path.is_file() and emb_path.is_file()

    def _download_file(
        self,
        url: str,
        destination: Path,
        label: str,
        progress_callback: Optional[ProgressCallback] = None,
        progress_range: Tuple[float, float] = (0.0, 0.25),
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".downloading")

        req = urllib.request.Request(url, headers={"User-Agent": "TranscreveAI/1.0"})
        with urllib.request.urlopen(req, timeout=180) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            chunk_size = 256 * 1024

            with open(tmp_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        frac = min(1.0, downloaded / total)
                        p = progress_range[0] + frac * (progress_range[1] - progress_range[0])
                        mb_done = downloaded / (1024 * 1024)
                        mb_tot = total / (1024 * 1024)
                        progress_callback(p, f"Baixando {label}... ({mb_done:.1f}/{mb_tot:.1f} MB)")

        shutil.move(str(tmp_path), str(destination))

    def ensure_models(
        self,
        embedding_profile: str = "standard",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[Path, Path]:
        """Garante que os modelos ONNX de segmentação e embeddings estejam baixados localmente."""
        emb_path, emb_url, emb_label = self.get_embedding_model_info(embedding_profile)

        with self._download_lock:
            DIARIZATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

            if not self.segmentation_model_path.is_file():
                archive_path = DIARIZATION_MODELS_DIR / f"{DIARIZATION_SEGMENTATION_DIRNAME}.tar.bz2"
                if progress_callback:
                    progress_callback(0.02, "Baixando modelo de segmentação de voz (Pyannote 3.0 ONNX)...")
                self._download_file(
                    DIARIZATION_SEGMENTATION_URL,
                    archive_path,
                    "segmentação de voz (Pyannote 3.0)",
                    progress_callback=progress_callback,
                    progress_range=(0.02, 0.12),
                )
                if progress_callback:
                    progress_callback(0.13, "Extraindo modelo de segmentação...")
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(path=DIARIZATION_MODELS_DIR)
                try:
                    archive_path.unlink(missing_ok=True)
                except OSError:
                    pass

            if not emb_path.is_file():
                if progress_callback:
                    progress_callback(0.15, f"Baixando modelo de identificação de voz ({emb_label})...")
                self._download_file(
                    emb_url,
                    emb_path,
                    emb_label,
                    progress_callback=progress_callback,
                    progress_range=(0.15, 0.30),
                )

            if not (self.segmentation_model_path.is_file() and emb_path.is_file()):
                raise RuntimeError("Os modelos de diarização não foram encontrados após o download.")

            return self.segmentation_model_path, emb_path

    def diarize(
        self,
        wav_path: Path | str,
        num_speakers: Optional[int] = None,
        threshold: float = 0.65,
        embedding_profile: str = "standard",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[SpeakerTurn]:
        """Executa diarização offline sobre arquivo WAV 16kHz mono."""
        import sherpa_onnx

        seg_model, emb_model = self.ensure_models(
            embedding_profile=embedding_profile,
            progress_callback=progress_callback,
        )

        samples, sample_rate = sf.read(str(wav_path), dtype="float32")
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1).astype("float32")

        duration = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
        if duration < 1.0:
            return [SpeakerTurn(start=0.0, end=max(duration, 0.1), speaker=0)]

        if progress_callback:
            progress_callback(0.32, "Analisando vozes e identificando locutores...")

        num_threads = min(8, max(2, (os.cpu_count() or 4) // 2))
        num_clusters = int(num_speakers) if (num_speakers is not None and int(num_speakers) > 0) else -1

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model),
                ),
                num_threads=num_threads,
                debug=False,
                provider="cpu",
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model),
                num_threads=num_threads,
                debug=False,
                provider="cpu",
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=float(threshold),
            ),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )

        if not config.validate():
            raise RuntimeError("Configuração inválida para o motor de diarização Sherpa-ONNX.")

        diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        if sample_rate != diarizer.sample_rate:
            raise ValueError(
                f"Taxa de amostragem inesperada para diarização: {sample_rate} Hz "
                f"(esperado {diarizer.sample_rate} Hz)."
            )

        def _on_chunk(num_processed: int, num_total: int) -> int:
            if progress_callback and num_total > 0:
                frac = min(1.0, num_processed / float(num_total))
                p = 0.35 + frac * 0.62
                progress_callback(p, f"Identificando locutores... ({int(frac * 100)}%)")
            return 0

        # Compatibilidade com diferentes assinaturas da API python do sherpa-onnx
        try:
            raw_result = diarizer.process(samples, callback=_on_chunk)
        except TypeError:
            try:
                raw_result = diarizer.process(samples, progress_callback=_on_chunk)
            except TypeError:
                raw_result = diarizer.process(samples)

        if hasattr(raw_result, "sort_by_start_time"):
            items = raw_result.sort_by_start_time()
        else:
            items = sorted(list(raw_result), key=lambda x: float(x.start))

        turns: List[SpeakerTurn] = []
        for seg in items:
            s_start = float(seg.start)
            s_end = float(seg.end)
            if s_end <= s_start:
                continue
            turns.append(SpeakerTurn(start=s_start, end=s_end, speaker=int(seg.speaker)))

        if not turns:
            turns.append(SpeakerTurn(start=0.0, end=duration, speaker=0))

        if progress_callback:
            progress_callback(1.0, f"Diarização concluída ({len({t.speaker for t in turns})} locutor(es) detectado(s)).")

        return turns


diarizer_engine = SpeakerDiarizer()
