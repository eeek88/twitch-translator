"""Smoke test: ask the context helper to explain a deliberately shaky translation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.context_helper import ContextHelper

SAMPLES = [
    ("ja", "リュウカごとく", "It's like a lyre."),
    ("ja", "冷えピタ全身に貼ろう", "Let's get some cold pie on you."),
]


def main():
    helper = ContextHelper()
    for lang, original, translated in SAMPLES:
        note = helper.explain(original, translated, lang)
        line = f"[{lang}] {original!r} -> {translated!r}\n  note: {note!r}"
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
