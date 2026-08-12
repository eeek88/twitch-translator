"""Smoke test: run the overlay with fake captions fed in from a background thread."""
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay
from twitch_translator.pipeline import Caption

FAKE_CAPTIONS = [
    Caption("es", "Hola a todos", "Hello everyone, welcome to the stream!"),
    Caption("es", "Este juego es dificil", "This game is really difficult but fun."),
    Caption("ja", "new game", "Today we're playing something new."),
]


def feeder(q):
    for cap in FAKE_CAPTIONS:
        time.sleep(1.5)
        q.put(cap)


def main():
    q = queue.Queue()
    threading.Thread(target=feeder, args=(q,), daemon=True).start()
    overlay = CaptionOverlay(q)
    overlay.root.after(30000, overlay.root.destroy)
    overlay.run()


if __name__ == "__main__":
    main()
