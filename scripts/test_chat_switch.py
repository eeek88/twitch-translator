"""Smoke test: connect to one channel, then switch to another on the same
live connection, and time how long it takes for the new channel's messages
to start arriving — verifies switch_channel() actually works against real
Twitch IRC, not just that it doesn't crash."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.chat import ChatReader


def main():
    first_channel = sys.argv[1] if len(sys.argv) > 1 else "gory_smg2"
    second_channel = sys.argv[2] if len(sys.argv) > 2 else "xiah7s"
    settle_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    wait_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0

    reader = ChatReader(first_channel)
    messages: list[tuple[float, str, str]] = []
    switch_time = {"t": None}

    def consume():
        for username, message in reader.messages():
            messages.append((time.perf_counter(), username, message))

    t = threading.Thread(target=consume, daemon=True)
    t.start()

    print(f"connected, joined #{first_channel}; settling {settle_seconds:.0f}s...")
    time.sleep(settle_seconds)
    before_switch_count = len(messages)
    print(f"messages from #{first_channel} before switch: {before_switch_count}")

    print(f"switching to #{second_channel}...")
    t0 = time.perf_counter()
    reader.switch_channel(second_channel)
    switch_call_elapsed = time.perf_counter() - t0
    switch_time["t"] = t0
    print(f"switch_channel() call itself took {switch_call_elapsed*1000:.1f}ms "
          f"(this is just the socket write — not how long until new messages arrive)")

    time.sleep(wait_seconds)
    after = [m for m in messages if m[0] >= switch_time["t"]]
    print(f"messages arrived in the {wait_seconds:.0f}s after switching: {len(after)}")
    if after:
        first_after_delay = after[0][0] - switch_time["t"]
        print(f"first post-switch message arrived {first_after_delay:.2f}s after switch_channel()")
        for _, username, message in after[:5]:
            line = f"  {username}: {message}"
            sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
    else:
        print("  (no messages arrived in the window — inconclusive if #{} is quiet "
              "right now, not necessarily a failure)".format(second_channel))

    print(f"reader.channel after switch: {reader.channel!r} (want {second_channel.lower()!r})")
    reader.stop()
    t.join(timeout=2.0)  # let the reader thread actually unwind before the
                         # interpreter starts finalizing, or its daemon
                         # thread can race stdout teardown and crash noisily


if __name__ == "__main__":
    main()
