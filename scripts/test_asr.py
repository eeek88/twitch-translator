"""Smoke test: transcribe a WAV file with faster-whisper."""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.asr import ASR


def load_wav_f32(path):
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "tts_test_16k.wav"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    audio = load_wav_f32(wav_path)
    asr = ASR(model_size=model_size)
    text, lang = asr.transcribe(audio)
    print(f"lang={lang}")
    print(f"text={text!r}")


if __name__ == "__main__":
    main()
