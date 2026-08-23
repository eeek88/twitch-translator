"""Integration test for stream-docked mode, against the real live Firefox +
Twitch setup (requires Firefox running with -marionette and a xiah7s tab
open — same preconditions validated manually during development). Uses a
lightweight fake pipeline (not a real Pipeline) since docking only reads
chat_channel/context_helper_enabled/etc. as plain attributes — no need to
pay the real model-loading cost for a UI-focused test."""
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay


def check(label, ok, detail=""):
    line = f"{'PASS' if ok else 'FAIL'} [{label}] {detail}"
    sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()


def main():
    fake_pipeline = SimpleNamespace(
        chat_channel="xiah7s",
        context_helper_enabled=False,  # keep the menu/context_dot paths simple for this test
        context_helper_busy=False,
        is_recording_transcript=False,
        vad=SimpleNamespace(is_speaking=False),
    )

    overlay = CaptionOverlay(queue.Queue(), pipeline=fake_pipeline)
    overlay.root.update_idletasks()

    overlay._dock_to_stream()
    check("dock_tracker started", overlay.dock_tracker is not None)
    check("_stream_docked flag set", overlay._stream_docked is True)

    print("waiting up to 8s for the tracker to find the tab and report a rect "
          "(the tracked tab must actually be Firefox's visible/selected tab — "
          "document.visibilityState reports hidden otherwise, by design)...")
    update = None
    deadline = time.time() + 8
    while time.time() < deadline:
        overlay._poll()
        overlay.root.update()
        if overlay._last_dock_update is not None and overlay._last_dock_update.visible:
            update = overlay._last_dock_update
            break
        time.sleep(0.2)

    check("got a visible DockUpdate", update is not None,
          repr(overlay._last_dock_update))
    if update is not None:
        check("update has plausible video-sized width", update.width and update.width > 100,
              f"width={update.width}")
        overlay.root.update_idletasks()
        actual_w = overlay.root.winfo_width()
        check("window width matches tracked video width", actual_w == update.width,
              f"window={actual_w} tracked={update.width}")

    # --- offset-on-drag ---
    if update is not None:
        before_x, before_y = overlay._dock_offset_x, overlay._dock_offset_y
        target_x, target_y = update.left + 20, (update.bottom - overlay.root.winfo_height()) - 10
        overlay._on_dragged(target_x, target_y)
        check("drag recorded offset_x", overlay._dock_offset_x == 20, f"got {overlay._dock_offset_x}")
        check("drag recorded offset_y", overlay._dock_offset_y == -10, f"got {overlay._dock_offset_y}")
        # Re-applying the same update should now land exactly at the dragged position.
        overlay._apply_dock_update(update)
        overlay.root.update_idletasks()
        check("re-applied update honors the offset",
              overlay.root.winfo_x() == target_x and overlay.root.winfo_y() == target_y,
              f"got ({overlay.root.winfo_x()},{overlay.root.winfo_y()}) want ({target_x},{target_y})")

    # --- undock ---
    overlay._undock_from_stream()
    check("dock_tracker cleared", overlay.dock_tracker is None)
    check("_stream_docked flag cleared", overlay._stream_docked is False)

    overlay.root.destroy()


if __name__ == "__main__":
    main()
