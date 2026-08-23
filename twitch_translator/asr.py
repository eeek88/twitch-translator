"""Speech-to-text on individual VAD utterances via faster-whisper."""
from __future__ import annotations

import difflib
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from .text_repetition import has_repetition_loop

CONTEXT_UTTERANCES = 3           # recent transcripts fed back as context
MAX_PROMPT_CHARS = 600           # whisper's prompt window is ~224 tokens; stay well under
HALLUCINATION_SIMILARITY = 0.75  # text this close to the glossary is a prompt echo, not speech


@dataclass
class ASRResult:
    text: str
    language: str
    # Average per-segment avg_logprob (faster-whisper's own confidence signal)
    # — 0 is maximally confident, more negative is less so. 0.0 for empty text.
    confidence: float


class ASR:
    """Each utterance is transcribed with an initial_prompt built from the
    glossary plus the last few transcripts. Whisper uses the prompt as prior
    context, which makes it far more consistent on recurring names/jargon
    (game terms, streamer names) that it otherwise garbles differently each
    time. The rolling context resets whenever the detected language changes,
    so a stray English clip doesn't pollute Japanese recognition."""

    def __init__(self, model_size: str = "medium", device: str = "cuda",
                 compute_type: str = "float16", glossary: str = ""):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.glossary = glossary.strip()
        self._recent: deque[str] = deque(maxlen=CONTEXT_UTTERANCES)
        self._recent_lang: str | None = None

    def _build_prompt(self) -> str | None:
        parts = []
        if self.glossary:
            parts.append(self.glossary)
        parts.extend(self._recent)
        prompt = " ".join(parts).strip()
        return prompt[-MAX_PROMPT_CHARS:] if prompt else None

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """audio: float32 mono 16kHz in [-1, 1]. Returns (text, detected_language_code)."""
        result = self.transcribe_scored(audio)
        return result.text, result.language

    def transcribe_scored(self, audio: np.ndarray) -> ASRResult:
        """Same as transcribe(), but also returns Whisper's own confidence —
        lets the caller flag a transcription that was itself shaky, not just
        a translation of text Whisper heard confidently but wrong."""
        segments, info = self.model.transcribe(
            audio, beam_size=5, initial_prompt=self._build_prompt(),
        )
        # The generator is consumed once; materialize it so both the text
        # join and the confidence average can walk it.
        segments = list(segments)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        confidence = sum(seg.avg_logprob for seg in segments) / len(segments) if segments else 0.0
        # Feeding a prompt can bias Whisper to hallucinate a paraphrased echo of it
        # on near-silent/ambiguous audio (observed: prompt "Joy-Con, ProCon
        # controller, Splatoon" -> transcribed "ProCon, ProCon controller,
        # Splatoon" from pure silence). That's not real speech — drop it. A fuzzy
        # ratio (not exact match) is needed since the echo isn't always verbatim.
        if self.glossary and text and difflib.SequenceMatcher(
            None, text.lower(), self.glossary.lower()
        ).ratio() > HALLUCINATION_SIMILARITY:
            return ASRResult("", info.language, confidence)
        # Discard the whole utterance rather than trimming to one repeat: a
        # repetition loop is a strong signal the model was never confident
        # about this audio in the first place, not just padding a real answer.
        if text and has_repetition_loop(text):
            print(f"[repetition loop dropped] {text[:120]!r}...", file=sys.stderr)
            return ASRResult("", info.language, confidence)
        if text:
            if info.language != self._recent_lang:
                self._recent.clear()
                self._recent_lang = info.language
            self._recent.append(text)
        return ASRResult(text, info.language, confidence)
