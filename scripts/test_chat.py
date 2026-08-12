"""Smoke test: print raw chat messages from a channel for ~15 seconds."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.chat import chat_messages


def main():
    channel = sys.argv[1] if len(sys.argv) > 1 else "gory_smg2"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    count = 0

    def reader():
        nonlocal count
        for username, message in chat_messages(channel):
            line = f"{username}: {message}"
            sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()
            count += 1

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(seconds)
    print(f"--- {count} messages in {seconds:.0f}s ---", file=sys.stderr)


if __name__ == "__main__":
    main()
