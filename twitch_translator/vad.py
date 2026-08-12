"""Groups a raw PCM stream into speech utterances using Silero VAD, run frame-by-frame
in streaming mode so we don't need the whole audio in memory."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # required chunk size for silero's streaming JIT model at 16kHz
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 s16le


class VADSegmenter:
    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 300,
        max_speech_ms: int = 12000,
    ):
        self.model = load_silero_vad()
        self.threshold = threshold
        frame_ms = FRAME_SAMPLES / SAMPLE_RATE * 1000
        self.min_silence_frames = max(1, int(min_silence_ms / frame_ms))
        self.min_speech_samples = int(SAMPLE_RATE * min_speech_ms / 1000)
        self.max_speech_samples = int(SAMPLE_RATE * max_speech_ms / 1000)

    def segments(self, pcm_chunks: Iterator[bytes]) -> Iterator[np.ndarray]:
        """pcm_chunks: raw s16le mono 16kHz bytes. Yields float32 audio in [-1, 1] per utterance."""
        buf = bytearray()
        in_speech = False
        speech_frames: list[np.ndarray] = []
        silence_run = 0
        self.model.reset_states()

        for chunk in pcm_chunks:
            buf.extend(chunk)
            while len(buf) >= FRAME_BYTES:
                frame_bytes = bytes(buf[:FRAME_BYTES])
                del buf[:FRAME_BYTES]
                frame_i16 = np.frombuffer(frame_bytes, dtype=np.int16)
                frame_f32 = frame_i16.astype(np.float32) / 32768.0
                prob = self.model(torch.from_numpy(frame_f32), SAMPLE_RATE).item()

                if prob >= self.threshold:
                    in_speech = True
                    silence_run = 0
                    speech_frames.append(frame_f32)
                    if sum(len(a) for a in speech_frames) >= self.max_speech_samples:
                        yield np.concatenate(speech_frames)
                        speech_frames = []
                        in_speech = False
                elif in_speech:
                    silence_run += 1
                    speech_frames.append(frame_f32)  # keep trailing silence for a natural boundary
                    if silence_run >= self.min_silence_frames:
                        if sum(len(a) for a in speech_frames) >= self.min_speech_samples:
                            yield np.concatenate(speech_frames)
                        speech_frames = []
                        in_speech = False
                        silence_run = 0

        if speech_frames and sum(len(a) for a in speech_frames) >= self.min_speech_samples:
            yield np.concatenate(speech_frames)
