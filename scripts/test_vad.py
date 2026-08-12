"""Smoke test: run VAD segmentation over a WAV file, print detected utterance boundaries."""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.vad import VADSegmenter, SAMPLE_RATE


def chunks_from_wav(path, chunk_bytes=3200):
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, wf.getframerate()
        assert wf.getnchannels() == 1
        data = wf.readframes(wf.getnframes())
    for i in range(0, len(data), chunk_bytes):
        yield data[i:i + chunk_bytes]


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "scripts_test_browser.wav"
    seg = VADSegmenter()
    count = 0
    for utt in seg.segments(chunks_from_wav(wav_path)):
        count += 1
        print(f"utterance {count}: {len(utt)/SAMPLE_RATE:.2f}s")
    print(f"total utterances: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
