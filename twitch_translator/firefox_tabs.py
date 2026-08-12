"""Detects Twitch channels open in Firefox tabs, without any browser extension.

Firefox persists its open tabs to sessionstore-backups/recovery.jsonlz4 in the
profile directory (rewritten every ~15s), compressed in Mozilla's lz4 framing:
the magic b"mozLz40\\0" followed by a standard lz4 block (which itself starts
with a 4-byte decompressed-size prefix, exactly what lz4.block expects).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import lz4.block

MOZ_MAGIC = b"mozLz40\0"

# twitch.tv paths that are site pages, not channel names
_NON_CHANNELS = {
    "directory", "videos", "settings", "subscriptions", "wallet", "drops",
    "search", "following", "downloads", "jobs", "turbo", "store", "p",
    "popout", "moderator", "u", "collections", "team",
}
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_]{3,25}$")


def _newest_recovery_file() -> Optional[Path]:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    profiles = Path(appdata) / "Mozilla" / "Firefox" / "Profiles"
    candidates = list(profiles.glob("*/sessionstore-backups/recovery.jsonlz4"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _channel_from_url(url: str) -> Optional[str]:
    m = re.match(r"https?://(?:www\.|m\.)?twitch\.tv/([^/?#]+)(?:/([^/?#]+))?", url)
    if not m:
        return None
    first, second = m.group(1), m.group(2)
    # popout chat windows look like twitch.tv/popout/<channel>/chat
    if first == "popout" and second:
        first = second
    if first.lower() in _NON_CHANNELS or not _CHANNEL_RE.match(first):
        return None
    return first.lower()


def list_twitch_channels() -> list[str]:
    """Channels open in Firefox tabs right now (best-effort; [] on any failure)."""
    try:
        recovery = _newest_recovery_file()
        if recovery is None:
            return []
        raw = recovery.read_bytes()
        if not raw.startswith(MOZ_MAGIC):
            return []
        session = json.loads(lz4.block.decompress(raw[len(MOZ_MAGIC):]))
        channels: list[str] = []
        for window in session.get("windows", []):
            for tab in window.get("tabs", []):
                entries = tab.get("entries", [])
                index = tab.get("index", len(entries))
                if not entries:
                    continue
                url = entries[min(index, len(entries)) - 1].get("url", "")
                channel = _channel_from_url(url)
                if channel and channel not in channels:
                    channels.append(channel)
        return channels
    except Exception:
        return []  # tab detection is a convenience; never let it break the app
