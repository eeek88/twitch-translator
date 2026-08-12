"""Test: chat_btn colors blue when chat is visible, default gray when hidden;
combine_btn is mapped (visible) only while chat is visible. Drives _poll()
directly rather than waiting on the real timer."""
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import ACTIVE_COLOR, CONTROL_COLOR, CaptionOverlay


def main():
    overlay = CaptionOverlay(queue.Queue())
    overlay.attach_chat(queue.Queue())
    overlay.root.update()

    def check(label, expect_chat_color, expect_combine_mapped):
        overlay._poll()  # does its work synchronously, then re-schedules itself (harmless, never fires — we destroy root before mainloop)
        overlay.root.update()
        chat_color = overlay.chat_btn.cget("fg")
        combine_mapped = bool(overlay.combine_btn.winfo_ismapped())
        ok = chat_color == expect_chat_color and combine_mapped == expect_combine_mapped
        print(f"{'PASS' if ok else 'FAIL'} [{label}] "
              f"chat_btn fg={chat_color} (want {expect_chat_color}), "
              f"combine_btn mapped={combine_mapped} (want {expect_combine_mapped})")

    check("chat visible (default)", ACTIVE_COLOR, True)

    overlay.chat_overlay.toggle()  # hide
    check("chat hidden", CONTROL_COLOR, False)

    overlay.chat_overlay.toggle()  # show again
    check("chat shown again", ACTIVE_COLOR, True)

    overlay.root.destroy()


if __name__ == "__main__":
    main()
