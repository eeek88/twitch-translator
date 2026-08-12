"""Smoke test: run both overlay windows with fake captions and chat lines."""
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay
from twitch_translator.pipeline import Caption, ChatLine

FAKE_CAPTIONS = [
    Caption("es", "Hola a todos", "Hello everyone, welcome to the stream!"),
    Caption("es", "Este juego es dificil", "This game is really difficult but fun."),
    Caption("ja", "new game", "Today we're playing something new."),
    Caption("ja", "scroll test", "Line four — try scrolling up now."),
    Caption("ja", "scroll test", "Line five, still arriving while you read."),
]

FAKE_CHAT = [
    ChatLine("viewer1", "ja", "かっこいい", "That's cool!"),
    ChatLine("viewer2", "es", "que bueno", "How good."),
    ChatLine("viewer3", "ja", "草", "lol"),
]


def feeder(cap_q, chat_q):
    for i, cap in enumerate(FAKE_CAPTIONS):
        time.sleep(1.2)
        cap_q.put(cap)
        if i < len(FAKE_CHAT):
            chat_q.put(FAKE_CHAT[i])


def main():
    cap_q: queue.Queue = queue.Queue()
    chat_q: queue.Queue = queue.Queue()
    threading.Thread(target=feeder, args=(cap_q, chat_q), daemon=True).start()
    overlay = CaptionOverlay(cap_q)
    overlay.attach_chat(chat_q)
    overlay.root.after(30000, overlay.root.destroy)
    overlay.run()


if __name__ == "__main__":
    main()
