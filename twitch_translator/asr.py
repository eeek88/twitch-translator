"""Speech-to-text on individual VAD utterances via faster-whisper."""
from __future__ import annotations

import difflib
import sys
from collections import deque

import numpy as np
from faster_whisper import WhisperModel

from .text_repetition import has_repetition_loop

CONTEXT_UTTERANCES = 3           # recent transcripts fed back as context
MAX_PROMPT_CHARS = 600           # whisper's prompt window is ~224 tokens; stay well under
HALLUCINATION_SIMILARITY = 0.75  # text this close to the glossary is a prompt echo, not speech


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
        segments, info = self.model.transcribe(
            audio, beam_size=5, initial_prompt=self._build_prompt(),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        # Feeding a prompt can bias Whisper to hallucinate a paraphrased echo of it
        # on near-silent/ambiguous audio (observed: prompt "Joy-Con, ProCon
        # controller, Splatoon" -> transcribed "ProCon, ProCon controller,
        # Splatoon" from pure silence). That's not real speech — drop it. A fuzzy
        # ratio (not exact match) is needed since the echo isn't always verbatim.
        if self.glossary and text and difflib.SequenceMatcher(
            None, text.lower(), self.glossary.lower()
        ).ratio() > HALLUCINATION_SIMILARITY:
            return "", info.language
        # Discard the whole utterance rather than trimming to one repeat: a
        # repetition loop is a strong signal the model was never confident
        # about this audio in the first place, not just padding a real answer.
        if text and has_repetition_loop(text):
            print(f"[repetition loop dropped] {text[:120]!r}...", file=sys.stderr)
            return "", info.language
        if text:
            if info.language != self._recent_lang:
                self._recent.clear()
                self._recent_lang = info.language
            self._recent.append(text)
        return text, info.language
