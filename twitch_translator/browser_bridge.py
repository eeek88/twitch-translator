"""Minimal hand-rolled client for Firefox's Marionette remote-automation
protocol — used to read the live DOM position of the Twitch video/chat
elements so overlay windows can dock to them. Deliberately not the official
marionette_driver package: that pulls in the whole mozbase test-automation
dependency tree (mozrunner, mozprofile, mozdevice, ...), meant for running
Firefox's own test suite, not for a small runtime query like this. The wire
protocol itself is simple enough to implement directly (see
https://firefox-source-docs.mozilla.org/testing/marionette/Protocol.html).

Requires Firefox to have been launched with the -marionette flag (a
persistent about:config pref alone was tried and confirmed NOT sufficient —
the flag is what actually opens the local automation port).
"""
from __future__ import annotations

import json
import socket
from typing import Any, Optional

HOST = "127.0.0.1"
PORT = 2828


class MarionetteError(Exception):
    pass


class BrowserBridge:
    """One connection to Firefox's Marionette server. Not thread-safe —
    callers running a polling loop on a background thread should own one
    instance exclusively."""

    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = 3.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        self._buf = b""
        self._next_id = 1
        self._read_message()  # initial unsolicited handshake from the server
        self._session_id = self._command("WebDriver:NewSession", {})["sessionId"]

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    # --- wire protocol ----------------------------------------------------

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise MarionetteError("connection closed by Firefox")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _read_message(self) -> Any:
        # Wire format: "<byte length>:<json>", length counts the JSON bytes only.
        length_digits = b""
        while True:
            b = self._read_exact(1)
            if b == b":":
                break
            length_digits += b
        length = int(length_digits)
        payload = self._read_exact(length)
        return json.loads(payload.decode("utf-8"))

    def _command(self, name: str, params: dict) -> dict:
        msg_id = self._next_id
        self._next_id += 1
        packet = [0, msg_id, name, params]  # 0 = command message type
        body = json.dumps(packet).encode("utf-8")
        self._sock.sendall(f"{len(body)}:".encode("ascii") + body)
        response = self._read_message()
        # Response: [1, msg_id, error, result] — 1 = response message type
        _type, _resp_id, error, result = response
        if error:
            raise MarionetteError(f"{name} failed: {error}")
        return result or {}

    # --- high-level API -----------------------------------------------------

    def window_handles(self) -> list[str]:
        return self._command("WebDriver:GetWindowHandles", {})

    def switch_to_window(self, handle: str) -> None:
        self._command("WebDriver:SwitchToWindow", {"handle": handle, "focus": False})

    def get_url(self) -> str:
        return self._command("WebDriver:GetCurrentURL", {}).get("value", "")

    def execute_script(self, script: str, args: Optional[list] = None) -> Any:
        """script is a full function body — return a value explicitly, same
        convention as Selenium/marionette_driver's execute_script."""
        result = self._command("WebDriver:ExecuteScript", {
            "script": script, "args": args or [], "sandbox": None,
        })
        return result.get("value")

    def find_tab_by_channel(self, channel: str) -> Optional[str]:
        """Returns the window handle of the first tab whose URL is
        twitch.tv/<channel> (case-insensitive), or None if not found."""
        channel = channel.lower().lstrip("#")
        for handle in self.window_handles():
            self.switch_to_window(handle)
            url = self.get_url().lower()
            if f"twitch.tv/{channel}" in url:
                return handle
        return None

    def get_element_rect(self, selector: str) -> Optional[dict]:
        """CSS-pixel viewport-relative rect of the first element matching
        selector, or None if not present. Works regardless of what caused
        the element to move (window scroll or an inner scroll container) —
        getBoundingClientRect() always reflects the current viewport."""
        return self.execute_script(f"""
            const el = document.querySelector({selector!r});
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {{left: r.left, top: r.top, width: r.width, height: r.height}};
        """)

    def get_window_info(self) -> dict:
        """Chrome-offset and visibility info needed to convert the CSS-pixel
        rect above into physical screen pixels, and to know whether the tab
        is actually the one currently showing."""
        return self.execute_script("""
            return {
                screenX: window.screenX, screenY: window.screenY,
                outerWidth: window.outerWidth, outerHeight: window.outerHeight,
                innerWidth: window.innerWidth, innerHeight: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio,
                visibilityState: document.visibilityState,
            };
        """)
