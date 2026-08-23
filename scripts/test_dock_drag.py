"""Test: chat auto-docks against the caption window on attach_chat(), tandem
dragging moves both, and resizing either syncs the other's height. Uses
Tkinter's own synthetic event system (widget-relative coords, resolved
internally to screen coords) rather than OS-level pixel clicking, which
proved unreliable for small targets."""
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

    cap = rect(overlay.root)
    chat = rect(overlay.chat_overlay.win)
    print("after attach (auto-docked):")
    print("  caption:", cap)
    print("  chat:   ", chat)
    docked_ok = chat[1] == cap[1] and chat[3] == cap[3] and (
        chat[0] == cap[0] + cap[2] or chat[0] + chat[2] == cap[0])
    print("  DOCKED ON ATTACH:", "PASS" if docked_ok else "FAIL")

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

    cap_dx, cap_dy = cap_after[0] - cap[0], cap_after[1] - cap[1]
    chat_dx, chat_dy = chat_after[0] - chat[0], chat_after[1] - chat[1]
    print(f"  caption moved by: ({cap_dx},{cap_dy})")
    print(f"  chat moved by:    ({chat_dx},{chat_dy})")
    tandem_ok = (cap_dx, cap_dy) == (chat_dx, chat_dy) and cap_dx != 0
    print("  TANDEM DRAG:", "PASS" if tandem_ok else "FAIL")

    # Resizing the caption window's height (via its grip callback directly,
    # since simulating a corner-drag reliably needs real widget geometry) should
    # carry over to the chat window's height too.
    overlay._sync_chat_height(cap_after[2], cap_after[3] + 40)
    overlay.root.update_idletasks()
    chat_resized = rect(overlay.chat_overlay.win)
    print("after growing caption's height by 40 (via sync callback):")
    print("  chat:", chat_resized)
    height_sync_ok = chat_resized[3] == cap_after[3] + 40
    print("  HEIGHT SYNCED:", "PASS" if height_sync_ok else "FAIL")

    overlay.root.destroy()


if __name__ == "__main__":
    main()
