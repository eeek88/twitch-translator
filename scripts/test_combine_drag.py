"""Test: combine windows, then simulate a drag via Tkinter's own synthetic
event system (widget-relative coords, resolved internally to screen coords)
rather than OS-level pixel clicking, which proved unreliable for small
targets. Verifies tandem dragging while combined, and independent dragging
after separating."""
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay


def rect(win):
    win.update_idletasks()
    return (win.winfo_x(), win.winfo_y(), win.winfo_width(), win.winfo_height())


def main():
    overlay = CaptionOverlay(queue.Queue())
    overlay.root.geometry("900x160+100+100")
    overlay.attach_chat(queue.Queue(), geometry="380x300+1200+400")
    overlay.root.update_idletasks()

    print("before combine:")
    print("  caption:", rect(overlay.root))
    print("  chat:   ", rect(overlay.chat_overlay.win))

    overlay._combine()
    overlay.root.update_idletasks()
    print("after combine:")
    cap_before = rect(overlay.root)
    chat_before = rect(overlay.chat_overlay.win)
    print("  caption:", cap_before)
    print("  chat:   ", chat_before)
    print("  is_combined:", overlay.is_combined)

    # Simulate dragging the caption window by (+150, +80) via its own text widget,
    # exactly the path a real mouse drag takes (ButtonPress-1 then B1-Motion).
    text = overlay.scrollback.text
    text.event_generate("<ButtonPress-1>", x=50, y=20)
    overlay.root.update()
    text.event_generate("<B1-Motion>", x=200, y=100)
    overlay.root.update()

    cap_after = rect(overlay.root)
    chat_after = rect(overlay.chat_overlay.win)
    print("after dragging caption by simulated (+150,+80):")
    print("  caption:", cap_after)
    print("  chat:   ", chat_after)

    cap_dx, cap_dy = cap_after[0] - cap_before[0], cap_after[1] - cap_before[1]
    chat_dx, chat_dy = chat_after[0] - chat_before[0], chat_after[1] - chat_before[1]
    print(f"  caption moved by: ({cap_dx},{cap_dy})")
    print(f"  chat moved by:    ({chat_dx},{chat_dy})")
    tandem_ok = (cap_dx, cap_dy) == (chat_dx, chat_dy) and cap_dx != 0
    print("  TANDEM DRAG WHILE COMBINED:", "PASS" if tandem_ok else "FAIL")

    # Now separate, then drag caption again — chat should NOT move this time.
    overlay._separate()
    overlay.root.update_idletasks()
    chat_before2 = rect(overlay.chat_overlay.win)
    text.event_generate("<ButtonPress-1>", x=50, y=20)
    overlay.root.update()
    text.event_generate("<B1-Motion>", x=150, y=70)
    overlay.root.update()
    chat_after2 = rect(overlay.chat_overlay.win)
    independent_ok = chat_after2 == chat_before2
    print("after separating and dragging caption again:")
    print("  chat unchanged:", "PASS" if independent_ok else "FAIL",
          chat_before2, "->", chat_after2)

    overlay.root.destroy()


if __name__ == "__main__":
    main()
