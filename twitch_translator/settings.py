"""Loads settings.json (project root) merged over built-in defaults.
Any field can also be overridden per-run via the matching CLI flag in main.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

DEFAULTS: dict[str, Any] = {
    "audio_source": "browser",       # "browser" or "stream"
    "target": None,                  # browser: process name (default "firefox"); stream: channel name
    "model_size": "large-v3",        # faster-whisper model size
    "translation_model": "facebook/nllb-200-distilled-1.3B",
    "device": "cuda",                # "cuda" or "cpu"
    "target_lang": "eng_Latn",       # NLLB target language code
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 800,       # silence gap that ends an utterance
    "vad_min_speech_ms": 300,        # utterances shorter than this are discarded as noise
    "vad_max_speech_ms": 12000,      # utterances longer than this are cut off, to bound ASR latency
    "enable_chat": True,             # show the translated-chat panel
    "chat_channel": None,            # Twitch channel whose chat to read; in stream mode defaults to target
    "chat_max_messages": 15,         # lines kept visible in the chat panel
    "chat_queue_maxsize": 50,        # pending chat translations before new ones are dropped
}


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    return settings
