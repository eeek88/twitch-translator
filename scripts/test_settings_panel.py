"""Test: SettingsPanel opens instantly, docks below the caption window (or
above if there's no room), shows only basic fields by default, and Advanced
reveals the rest."""
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay, SettingsPanel
from twitch_translator.settings import SETTING_SPECS

BASIC_COUNT = sum(1 for s in SETTING_SPECS if not s.advanced)
ADVANCED_COUNT = sum(1 for s in SETTING_SPECS if s.advanced)


def main():
    overlay = CaptionOverlay(queue.Queue())
    overlay.root.geometry("900x160+100+100")
    overlay.root.update_idletasks()

    t0 = time.perf_counter()
    panel = SettingsPanel(overlay.root, overlay)
    overlay.root.update()
    elapsed = time.perf_counter() - t0
    print(f"open time: {elapsed*1000:.1f} ms", "PASS" if elapsed < 0.5 else "FAIL (slow)")

    shown = len(panel.vars)
    print(f"basic fields shown: {shown} (want {BASIC_COUNT})",
          "PASS" if shown == BASIC_COUNT else "FAIL")

    px, py = overlay.root.winfo_x(), overlay.root.winfo_y()
    ph = overlay.root.winfo_height()
    sx, sy = panel.win.winfo_x(), panel.win.winfo_y()
    docked_below = sx == px and sy == py + ph
    print(f"docked below caption: panel=({sx},{sy}) caption_bottom=({px},{py+ph})",
          "PASS" if docked_below else "FAIL")

    panel._toggle_advanced()
    overlay.root.update()
    shown_adv = len(panel.vars)
    print(f"advanced toggle -> fields shown: {shown_adv} (want {BASIC_COUNT + ADVANCED_COUNT})",
          "PASS" if shown_adv == BASIC_COUNT + ADVANCED_COUNT else "FAIL")

    panel._toggle_advanced()
    overlay.root.update()
    shown_back = len(panel.vars)
    print(f"advanced toggle again -> fields shown: {shown_back} (want {BASIC_COUNT})",
          "PASS" if shown_back == BASIC_COUNT else "FAIL")

    panel.win.destroy()

    # Now test the above-fallback: put the caption window near the bottom
    # of the (primary) screen so there's no room below it.
    import ctypes
    user32 = ctypes.windll.user32
    sh = user32.GetSystemMetrics(1)  # SM_CYSCREEN
    overlay.root.geometry(f"900x160+100+{sh - 170}")
    overlay.root.update_idletasks()
    py2 = overlay.root.winfo_y()
    ph2 = overlay.root.winfo_height()

    panel2 = SettingsPanel(overlay.root, overlay)
    overlay.root.update()
    sy2 = panel2.win.winfo_y()
    docked_above = sy2 <= py2
    print(f"docked above when no room below: panel_y={sy2} caption_y={py2}",
          "PASS" if docked_above else "FAIL")
    panel2.win.destroy()

    overlay.root.destroy()


if __name__ == "__main__":
    main()
