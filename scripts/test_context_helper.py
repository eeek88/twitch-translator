"""Smoke test: ask the context helper to explain a deliberately shaky translation,
with and without recent-chat context, to compare the two."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.context_helper import ContextHelper

SAMPLES = [
    ("ja", "リュウカごとく", "It's like a lyre.", None),
    ("ja", "冷えピタ全身に貼ろう", "Let's get some cold pie on you.", None),
    # Same shaky line as the first sample, but this time with recent chat that
    # names the game — checks whether that context actually helps the note.
    ("ja", "リュウカごとく", "It's like a lyre.",
     ["viewer1: is this the new Like a Dragon game?", "viewer2: yakuza spinoff lol"]),
]


def main():
    helper = ContextHelper()
    for lang, original, translated, recent_chat in SAMPLES:
        note = helper.explain(original, translated, lang, recent_chat=recent_chat)
        tag = " (with chat context)" if recent_chat else ""
        line = f"[{lang}]{tag} {original!r} -> {translated!r}\n  note: {note!r}"
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
