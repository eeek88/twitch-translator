"""Audio capture backends. Both backends yield the same thing: raw PCM chunks,
16kHz mono s16le, regardless of where the audio actually came from.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

CHUNK_BYTES = 3200  # 100ms @ 16kHz mono s16le (16000 * 2 * 0.1)

HELPER_EXE = (
    Path(__file__).resolve().parent.parent
    / "helper"
    / "bin"
    / "Release"
    / "net8.0-windows10.0.19041.0"
    / "FirefoxLoopbackCapture.exe"
)


def _require(binary: str) -> str:
    # A shell opened before an install won't have the new PATH entries, so don't
    # rely on PATH alone — also check where our tools actually get installed:
    # the venv's own Scripts dir (streamlink) and winget's package/link dirs (ffmpeg).
    candidates = [
        Path(sys.executable).parent,  # .venv/Scripts
    ]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(Path(local_appdata) / "Microsoft" / "WinGet" / "Links")
        packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            candidates.extend(sorted(packages.glob(f"*/**/bin"))[:20])

    path = shutil.which(binary)
    if path is None:
        for cand in candidates:
            exe = cand / f"{binary}.exe"
            if exe.is_file():
                return str(exe)
        raise RuntimeError(
            f"'{binary}' not found on PATH. If you just installed it, restart your shell."
        )
    return path


def _read_ffmpeg_pcm(ffmpeg_proc: subprocess.Popen, upstream_proc: subprocess.Popen) -> Iterator[bytes]:
    """upstream_proc is the helper.exe/streamlink process piping into ffmpeg's
    stdin. It must be terminated here too: on generator exit only ffmpeg_proc
    was being killed, leaving upstream_proc orphaned (confirmed live — multiple
    stray FirefoxLoopbackCapture.exe processes accumulated across reconnects)."""
    assert ffmpeg_proc.stdout is not None
    try:
        while True:
            chunk = ffmpeg_proc.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
    finally:
        ffmpeg_proc.terminate()
        upstream_proc.terminate()


def browser_audio_stream(process_name: str = "firefox") -> Iterator[bytes]:
    """Capture audio from a specific browser process via Windows process-loopback
    (the C# helper), then normalize to 16kHz mono s16le via ffmpeg."""
    if not HELPER_EXE.exists():
        raise RuntimeError(
            f"Capture helper not built: {HELPER_EXE}. "
            f"Run 'dotnet build -c Release' in the helper/ directory first."
        )
    ffmpeg = _require("ffmpeg")

    # stderr=None (inherit) rather than PIPE: these processes only print a few
    # diagnostic lines, but an unread PIPE fills its OS buffer and then blocks the
    # writer forever if we never drain it. Inheriting lets it flow straight to our
    # own stderr instead.
    helper_proc = subprocess.Popen(
        [str(HELPER_EXE), process_name],
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert helper_proc.stdout is not None

    ffmpeg_proc = subprocess.Popen(
        [
            ffmpeg,
            "-loglevel", "error",
            "-f", "f32le", "-ar", "48000", "-ac", "2",
            "-i", "pipe:0",
            "-vn", "-f", "s16le", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ],
        stdin=helper_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    helper_proc.stdout.close()  # let ffmpeg own the read end

    yield from _read_ffmpeg_pcm(ffmpeg_proc, helper_proc)


def twitch_stream_audio(channel: str) -> Iterator[bytes]:
    """Pull audio directly from Twitch's servers via streamlink, bypassing the
    browser entirely. Fallback backend when browser capture isn't viable."""
    streamlink = _require("streamlink")
    ffmpeg = _require("ffmpeg")

    url = f"https://twitch.tv/{channel}"
    streamlink_proc = subprocess.Popen(
        [streamlink, "--stdout", url, "worst"],
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert streamlink_proc.stdout is not None

    ffmpeg_proc = subprocess.Popen(
        [
            ffmpeg,
            "-loglevel", "error",
            "-i", "pipe:0",
            "-vn", "-f", "s16le", "-ar", "16000", "-ac", "1",
            "pipe:1",
        ],
        stdin=streamlink_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    streamlink_proc.stdout.close()

    yield from _read_ffmpeg_pcm(ffmpeg_proc, streamlink_proc)


def audio_stream(source: str, target: str) -> Iterator[bytes]:
    """source: 'browser' or 'stream'. target: process name (browser) or channel (stream)."""
    if source == "browser":
        return browser_audio_stream(target)
    elif source == "stream":
        return twitch_stream_audio(target)
    raise ValueError(f"unknown audio source: {source}")
