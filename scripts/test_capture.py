"""Smoke test: capture N seconds of audio and write it to a WAV file for manual listening."""
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator import capture

SAMPLE_RATE = 16000


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "browser"
    target = sys.argv[2] if len(sys.argv) > 2 else "firefox"
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    out_path = sys.argv[4] if len(sys.argv) > 4 else "capture_test.wav"

    print(f"Capturing {seconds}s from source={source} target={target} -> {out_path}", file=sys.stderr)
    n_bytes_needed = int(SAMPLE_RATE * 2 * seconds)
    collected = bytearray()
    start = time.time()
    for chunk in capture.audio_stream(source, target):
        collected.extend(chunk)
        if len(collected) >= n_bytes_needed or (time.time() - start) > seconds + 10:
            break

    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(collected[:n_bytes_needed]))

    print(f"Wrote {len(collected[:n_bytes_needed])} bytes to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
