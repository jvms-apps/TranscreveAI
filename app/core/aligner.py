"""Alinhamento inteligente entre transcrição (palavras) e diarização (locutores) em parágrafos."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.config import SPEAKER_COLORS
from app.core.diarizer import SpeakerTurn
from app.core.transcriber import TranscriptSegment, WordTiming

SENTENCE_END_RE = re.compile(r"[.!?…]+[\"”’')\]]*$")


@dataclass
class SpeakerInfo:
    id: str
    name: str
    color: str
    word_count: int = 0
    paragraph_count: int = 0
    duration: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "word_count": self.word_count,
            "paragraph_count": self.paragraph_count,
            "duration": round(self.duration, 2),
        }


@dataclass
class ParagraphItem:
    id: int
    speaker_id: str
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
        }


@dataclass
class _AssignedWord:
    word: str
    start: float
    end: float
    speaker_id: str


def _clean_text(words: List[str]) -> str:
    """Junta palavras e normaliza espaços antes de sinais de pontuação."""
    text = " ".join(w.strip() for w in words if w.strip())
    text = re.sub(r"\s+([,.;:!?%…\)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _flatten_words(segments: List[TranscriptSegment]) -> List[WordTiming]:
    words: List[WordTiming] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
        elif seg.text.strip():
            tokens = seg.text.strip().split()
            dur = max(0.05, seg.end - seg.start)
            step = dur / max(1, len(tokens))
            for i, tok in enumerate(tokens):
                words.append(WordTiming(word=tok, start=seg.start + i * step, end=seg.start + (i + 1) * step))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def _remap_turns_by_appearance(turns: List[SpeakerTurn]) -> List[Tuple[float, float, str]]:
    """Renomeia IDs numéricos dos locutores por ordem cronológica de aparição (SPEAKER_01, SPEAKER_02...)."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    mapping: Dict[int, str] = {}
    remapped: List[Tuple[float, float, str]] = []

    for t in ordered:
        if t.speaker not in mapping:
            idx = len(mapping) + 1
            mapping[t.speaker] = f"SPEAKER_{idx:02d}"
        remapped.append((t.start, t.end, mapping[t.speaker]))

    return remapped


def _find_speaker_for_word(
    w_start: float,
    w_end: float,
    turns: List[Tuple[float, float, str]],
    fallback_speaker: str,
) -> str:
    """Atribui o locutor com maior sobreposição temporal para uma palavra; fallback por proximidade."""
    best_speaker: Optional[str] = None
    best_overlap = 0.0

    # Soma sobreposições caso múltiplos trechos curtos do mesmo locutor toquem a palavra
    overlap_by_speaker: Dict[str, float] = {}
    for t_start, t_end, spk in turns:
        if t_end < w_start:
            continue
        if t_start > w_end:
            break
        ov = min(w_end, t_end) - max(w_start, t_start)
        if ov > 0:
            overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + ov

    if overlap_by_speaker:
        best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])
        if best_overlap > 0:
            return best_speaker

    # Se não houve sobreposição direta, encontra o turno mais próximo
    w_mid = 0.5 * (w_start + w_end)
    nearest_spk: Optional[str] = None
    nearest_dist = float("inf")
    for t_start, t_end, spk in turns:
        if w_mid < t_start:
            dist = t_start - w_mid
        elif w_mid > t_end:
            dist = w_mid - t_end
        else:
            dist = 0.0
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_spk = spk

    if nearest_spk is not None and nearest_dist <= 3.0:
        return nearest_spk

    return fallback_speaker


def _smooth_speaker_assignments(assigned: List[_AssignedWord]) -> List[_AssignedWord]:
    """Remove oscilações espúrias (1-2 palavras no meio da frase) e alinha trocas a fronteiras de frase."""
    if len(assigned) < 4:
        return assigned

    # Passo 1: absorver "ilhas" minúsculas de 1 a 2 palavras cercadas pelo mesmo locutor sem pontuação
    runs: List[List[int]] = []
    cur_run = [0]
    for i in range(1, len(assigned)):
        if assigned[i].speaker_id == assigned[cur_run[0]].speaker_id:
            cur_run.append(i)
        else:
            runs.append(cur_run)
            cur_run = [i]
    runs.append(cur_run)

    for r_idx in range(1, len(runs) - 1):
        prev_run = runs[r_idx - 1]
        mid_run = runs[r_idx]
        next_run = runs[r_idx + 1]

        prev_spk = assigned[prev_run[0]].speaker_id
        next_spk = assigned[next_run[0]].speaker_id
        mid_duration = assigned[mid_run[-1]].end - assigned[mid_run[0]].start

        if prev_spk == next_spk and len(mid_run) <= 2 and mid_duration < 1.2:
            prev_ends_sentence = bool(SENTENCE_END_RE.search(assigned[prev_run[-1]].word))
            mid_ends_sentence = bool(SENTENCE_END_RE.search(assigned[mid_run[-1]].word))
            # Se não é uma interjeição clara isolada entre sentenças, absorve
            if not (prev_ends_sentence and mid_ends_sentence):
                for idx in mid_run:
                    assigned[idx].speaker_id = prev_spk

    # Passo 2: alinhar trocas de locutor a 1 palavra de distância com fronteiras claras de frase
    for i in range(1, len(assigned) - 1):
        # Caso A: última palavra da frase anterior caiu no próximo locutor
        if assigned[i].speaker_id != assigned[i - 1].speaker_id:
            prev_ends = bool(SENTENCE_END_RE.search(assigned[i - 1].word))
            curr_ends = bool(SENTENCE_END_RE.search(assigned[i].word))
            gap_before = assigned[i].start - assigned[i - 1].end
            gap_after = assigned[i + 1].start - assigned[i].end

            if (not prev_ends) and curr_ends and gap_before < 0.35 and gap_after >= gap_before:
                assigned[i].speaker_id = assigned[i - 1].speaker_id

        # Caso B: primeira palavra da frase seguinte caiu no locutor anterior (ex: "... PDF. That" | "sounds...")
        if assigned[i].speaker_id != assigned[i + 1].speaker_id and assigned[i].speaker_id == assigned[i - 1].speaker_id:
            prev_ends = bool(SENTENCE_END_RE.search(assigned[i - 1].word))
            curr_ends = bool(SENTENCE_END_RE.search(assigned[i].word))
            gap_before = assigned[i].start - assigned[i - 1].end
            gap_after = assigned[i + 1].start - assigned[i].end

            if prev_ends and (not curr_ends) and gap_after < 0.35 and gap_before >= max(0.30, gap_after * 1.5):
                assigned[i].speaker_id = assigned[i + 1].speaker_id

    return assigned


def _split_long_speaker_block(
    words: List[_AssignedWord],
    max_chars_before_pause_split: int = 520,
    pause_threshold_sec: float = 1.6,
) -> List[List[_AssignedWord]]:
    """Quebra monólogos muito longos do mesmo locutor em parágrafos legíveis em pausas naturais."""
    if not words:
        return []

    chunks: List[List[_AssignedWord]] = []
    current: List[_AssignedWord] = [words[0]]
    char_count = len(words[0].word)

    for idx in range(1, len(words)):
        w = words[idx]
        prev = words[idx - 1]
        gap = w.start - prev.end
        prev_ends_sentence = bool(SENTENCE_END_RE.search(prev.word))

        should_break = (
            (char_count >= max_chars_before_pause_split and prev_ends_sentence and gap >= 0.45)
            or (char_count >= 240 and prev_ends_sentence and gap >= pause_threshold_sec)
            or (char_count >= 950 and prev_ends_sentence)
        )

        if should_break:
            chunks.append(current)
            current = [w]
            char_count = len(w.word)
        else:
            current.append(w)
            char_count += len(w.word) + 1

    if current:
        chunks.append(current)
    return chunks


def align_transcript_and_speakers(
    segments: List[TranscriptSegment],
    speaker_turns: Optional[List[SpeakerTurn]] = None,
    enable_diarization: bool = True,
) -> Tuple[List[ParagraphItem], Dict[str, SpeakerInfo]]:
    """Combina transcrição e diarização gerando parágrafos estruturados e estatísticas por locutor."""
    words = _flatten_words(segments)
    if not words:
        return [], {}

    assigned: List[_AssignedWord] = []

    if enable_diarization and speaker_turns:
        remapped_turns = _remap_turns_by_appearance(speaker_turns)
        last_spk = remapped_turns[0][2] if remapped_turns else "SPEAKER_01"

        for w in words:
            spk = _find_speaker_for_word(w.start, w.end, remapped_turns, fallback_speaker=last_spk)
            last_spk = spk
            assigned.append(_AssignedWord(word=w.word, start=w.start, end=w.end, speaker_id=spk))

        assigned = _smooth_speaker_assignments(assigned)

        # Reordena IDs novamente para garantir que o 1º parágrafo sempre comece em Locutor 1
        final_map: Dict[str, str] = {}
        for aw in assigned:
            if aw.speaker_id not in final_map:
                final_map[aw.speaker_id] = f"SPEAKER_{len(final_map) + 1:02d}"
            aw.speaker_id = final_map[aw.speaker_id]
    else:
        for w in words:
            assigned.append(_AssignedWord(word=w.word, start=w.start, end=w.end, speaker_id="SPEAKER_01"))

    # Agrupa palavras consecutivas do mesmo locutor
    speaker_blocks: List[List[_AssignedWord]] = []
    cur_block: List[_AssignedWord] = [assigned[0]]
    for aw in assigned[1:]:
        if aw.speaker_id == cur_block[0].speaker_id:
            cur_block.append(aw)
        else:
            speaker_blocks.append(cur_block)
            cur_block = [aw]
    speaker_blocks.append(cur_block)

    # Constrói parágrafos finais (quebrando monólogos longos em pausas naturais)
    paragraphs: List[ParagraphItem] = []
    speakers_meta: Dict[str, SpeakerInfo] = {}

    for block in speaker_blocks:
        sub_blocks = _split_long_speaker_block(block)
        for sub in sub_blocks:
            spk_id = sub[0].speaker_id
            text = _clean_text([w.word for w in sub])
            if not text:
                continue

            if spk_id not in speakers_meta:
                spk_index = len(speakers_meta)
                default_name = f"Locutor {spk_index + 1}" if enable_diarization else "Transcrição"
                color = SPEAKER_COLORS[spk_index % len(SPEAKER_COLORS)]
                speakers_meta[spk_id] = SpeakerInfo(
                    id=spk_id,
                    name=default_name,
                    color=color,
                )

            p_start = sub[0].start
            p_end = max(sub[-1].end, p_start)
            p_dur = max(0.0, p_end - p_start)

            paragraphs.append(
                ParagraphItem(
                    id=len(paragraphs) + 1,
                    speaker_id=spk_id,
                    start=p_start,
                    end=p_end,
                    text=text,
                )
            )

            info = speakers_meta[spk_id]
            info.word_count += len(text.split())
            info.paragraph_count += 1
            info.duration += p_dur

    return paragraphs, speakers_meta
