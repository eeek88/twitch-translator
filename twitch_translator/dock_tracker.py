"""Background thread that polls a Twitch tab's video element position via
BrowserBridge, so the caption window can dock to and follow it (including
through scrolling — see browser_bridge.get_element_rect's docstring for why
that needs no special handling) instead of sitting as a free-floating window.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .browser_bridge import BrowserBridge, MarionetteError

POLL_INTERVAL = 1.0 / 15  # ~15Hz: smooth enough to track scrolling, cheap enough to run continuously
RECONNECT_BACKOFF_MAX = 5.0
VIDEO_SELECTOR = "[data-a-target='video-player']"


@dataclass
class DockUpdate:
    """One polled snapshot. visible=False means "hide the window" — the
    video wasn't found, the tab isn't the visible one, or the connection to
    Firefox is down. left/bottom/width are in the same coordinate space as
    Tk's own .geometry() calls: confirmed empirically that this app's Tk
    windows run DPI-unaware, and Firefox's window.screenX/outerHeight/
    getBoundingClientRect() etc. report in that same virtualized-pixel space
    on this kind of setup — so no devicePixelRatio scaling is applied. (If
    the app is ever made DPI-aware, this direct mapping needs revisiting.)
    """
    visible: bool
    left: Optional[int] = None
    bottom: Optional[int] = None  # video's bottom edge — captions dock flush with this
    width: Optional[int] = None


class DockTracker:
    """Owns one BrowserBridge connection on its own thread. start() once;
    updates arrive on out_queue as DockUpdate objects, meant to be drained
    from the Tk polling loop (never touch Tk objects from this thread).
    stop() ends the thread and closes the connection."""

    def __init__(self, channel: str, out_queue: "queue.Queue[DockUpdate]"):
        self.channel = channel
        self.out_queue = out_queue
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._track_until_error()
                backoff = 1.0
            except Exception:
                self.out_queue.put(DockUpdate(visible=False))
                time.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)

    def _track_until_error(self) -> None:
        bridge = BrowserBridge()
        try:
            handle = bridge.find_tab_by_channel(self.channel)
            if handle is None:
                raise MarionetteError(f"no open tab found for channel {self.channel!r}")
            while not self._stop.is_set():
                self._poll_once(bridge)
                time.sleep(POLL_INTERVAL)
        finally:
            bridge.close()

    def _poll_once(self, bridge: BrowserBridge) -> None:
        info = bridge.get_window_info()
        if info.get("visibilityState") != "visible":
            self.out_queue.put(DockUpdate(visible=False))
            return
        rect = bridge.get_element_rect(VIDEO_SELECTOR)
        if rect is None:
            self.out_queue.put(DockUpdate(visible=False))
            return
        chrome_h = info["outerHeight"] - info["innerHeight"]
        left = round(info["screenX"] + rect["left"])
        bottom = round(info["screenY"] + chrome_h + rect["top"] + rect["height"])
        width = round(rect["width"])
        self.out_queue.put(DockUpdate(visible=True, left=left, bottom=bottom, width=width))
