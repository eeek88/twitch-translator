"""Speech-to-text on individual VAD utterances via faster-whisper."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class ASR:
    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """audio: float32 mono 16kHz in [-1, 1]. Returns (text, detected_language_code)."""
        segments, info = self.model.transcribe(audio, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info.language
