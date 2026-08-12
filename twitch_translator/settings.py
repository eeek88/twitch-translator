"""Loads settings.json (project root) merged over built-in defaults.
Any field can also be overridden per-run via the matching CLI flag in main.py,
or edited from the in-app settings dialog (which writes settings.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

DEFAULTS: dict[str, Any] = {
    "audio_source": "browser",       # "browser" or "stream"
    "target": None,                  # browser: process name (default "firefox"); stream: channel name
    "model_size": "large-v3",        # faster-whisper model size
    "translation_model": "facebook/nllb-200-distilled-1.3B",
    "device": "cuda",                # "cuda" or "cpu"
    "target_lang": "eng_Latn",       # NLLB target language code
    "glossary": "",                  # names/jargon fed to Whisper as context, e.g. "Joy-Con, ProCon, Splatoon"
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 800,       # silence gap that ends an utterance
    "vad_min_speech_ms": 300,        # utterances shorter than this are discarded as noise
    "vad_max_speech_ms": 12000,      # utterances longer than this are cut off, to bound ASR latency
    "enable_chat": True,             # show the translated-chat panel
    "chat_channel": None,            # Twitch channel whose chat to read; in stream mode defaults to target
    "chat_queue_maxsize": 50,        # pending chat translations before new ones are dropped
    "caption_geometry": None,        # last caption-window geometry ("WxH+X+Y"), saved on exit
    "chat_geometry": None,           # last chat-window geometry, saved on exit
    "caption_font_size": 18,         # also adjustable with Ctrl+scroll on the caption window
    "chat_font_size": 11,            # also adjustable with Ctrl+scroll on the chat window
}


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    kind: str                        # "choice" | "bool" | "int" | "float" | "text"
    choices: Optional[list] = None
    live: bool = False               # can apply without restarting the app
    help: str = ""


# Drives the in-app settings dialog: one row per user-facing setting.
# Geometry keys are managed automatically and intentionally not listed.
SETTING_SPECS: list[SettingSpec] = [
    SettingSpec("audio_source", "Audio source", "choice", ["browser", "stream"],
                help="browser: capture a browser process's audio; stream: pull from Twitch servers"),
    SettingSpec("target", "Capture target", "text",
                help="browser mode: process name (blank = firefox); stream mode: channel name"),
    SettingSpec("model_size", "Whisper model", "choice",
                ["tiny", "base", "small", "medium", "large-v3"],
                help="bigger = more accurate, slower; large-v3 recommended for non-English"),
    SettingSpec("translation_model", "Translation model", "choice",
                ["facebook/nllb-200-distilled-600M",
                 "facebook/nllb-200-distilled-1.3B",
                 "facebook/nllb-200-3.3B"],
                help="bigger = better translations, more VRAM"),
    SettingSpec("device", "Inference device", "choice", ["cuda", "cpu"],
                help="cuda needs an NVIDIA GPU"),
    SettingSpec("target_lang", "Target language", "text",
                help="NLLB code, e.g. eng_Latn, spa_Latn, jpn_Jpan"),
    SettingSpec("glossary", "Glossary", "text",
                help="comma-separated names/jargon the streamer says often — helps Whisper hear them right"),
    SettingSpec("vad_threshold", "VAD speech threshold", "float",
                help="0-1; higher = stricter about what counts as speech"),
    SettingSpec("vad_min_silence_ms", "VAD silence gap (ms)", "int",
                help="pause length that ends a sentence; higher = fewer mid-sentence cuts, more lag"),
    SettingSpec("vad_min_speech_ms", "VAD min speech (ms)", "int",
                help="shorter utterances are ignored as noise"),
    SettingSpec("vad_max_speech_ms", "VAD max speech (ms)", "int",
                help="utterances are cut at this length to bound latency"),
    SettingSpec("enable_chat", "Chat panel", "bool", live=True,
                help="show translated chat messages in a second window"),
    SettingSpec("chat_channel", "Chat channel", "text", live=True,
                help="Twitch channel whose chat to translate (also editable directly on the chat panel)"),
    SettingSpec("chat_queue_maxsize", "Chat queue size", "int",
                help="pending chat translations before new ones are dropped"),
    SettingSpec("caption_font_size", "Caption font size", "int", live=True,
                help="also adjustable with Ctrl+scroll directly on the caption window"),
    SettingSpec("chat_font_size", "Chat font size", "int", live=True,
                help="also adjustable with Ctrl+scroll directly on the chat window"),
]


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings.update(json.load(f))
    return settings


def save_settings(updates: dict[str, Any]) -> None:
    """Merge updates over whatever is currently in settings.json and write it back."""
    current: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            current = json.load(f)
    current.update(updates)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
        f.write("\n")
