"""Twitch Live Translator: captures audio from a Twitch stream (via a specific
browser process or directly from Twitch's servers), transcribes it, translates
it, and shows the result as a floating always-on-top caption bar.

Defaults come from settings.json (edit it directly to change your usual setup);
any field can be overridden per-run with the matching CLI flag below.
"""
from __future__ import annotations

import argparse
import sys

from twitch_translator.overlay import CaptionOverlay
from twitch_translator.pipeline import Pipeline
from twitch_translator.settings import load_settings


def parse_args():
    s = load_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--audio-source",
        choices=["browser", "stream"],
        default=s["audio_source"],
        help="'browser': capture a specific browser process's audio (synced to what you see). "
             "'stream': pull audio directly from Twitch's servers via streamlink (works headless, "
             f"may drift out of sync with the browser). (default: {s['audio_source']})",
    )
    p.add_argument(
        "--target",
        default=s["target"],
        help="For --audio-source browser: the process name to capture (default: firefox). "
             "For --audio-source stream: the Twitch channel name (required).",
    )
    p.add_argument("--model-size", default=s["model_size"],
                    help=f"faster-whisper model size (default: {s['model_size']})")
    p.add_argument("--translation-model", default=s["translation_model"],
                    help=f"NLLB-200 checkpoint (default: {s['translation_model']})")
    p.add_argument("--device", default=s["device"], choices=["cuda", "cpu"],
                    help=f"inference device (default: {s['device']})")
    p.add_argument("--target-lang", default=s["target_lang"],
                    help=f"NLLB target language code (default: {s['target_lang']})")
    p.add_argument("--glossary", default=s["glossary"],
                    help="comma-separated names/jargon the streamer says often, fed to Whisper as context")
    p.add_argument("--vad-threshold", type=float, default=s["vad_threshold"],
                    help=f"speech probability threshold, 0-1 (default: {s['vad_threshold']})")
    p.add_argument("--vad-min-silence-ms", type=int, default=s["vad_min_silence_ms"],
                    help="silence gap that ends an utterance; higher reduces mid-sentence cutoffs "
                         f"at the cost of latency (default: {s['vad_min_silence_ms']})")
    p.add_argument("--vad-min-speech-ms", type=int, default=s["vad_min_speech_ms"],
                    help=f"utterances shorter than this are discarded as noise (default: {s['vad_min_speech_ms']})")
    p.add_argument("--vad-max-speech-ms", type=int, default=s["vad_max_speech_ms"],
                    help=f"utterances longer than this are cut off (default: {s['vad_max_speech_ms']})")
    p.add_argument("--no-chat", action="store_true", default=not s["enable_chat"],
                    help="disable the translated-chat panel")
    p.add_argument("--chat-channel", default=s["chat_channel"],
                    help="Twitch channel whose chat to translate. Defaults to --target in stream mode; "
                         "required for chat in browser mode (the process name isn't a channel).")
    p.add_argument("--chat-queue-maxsize", type=int, default=s["chat_queue_maxsize"],
                    help=f"pending chat translations before new ones are dropped (default: {s['chat_queue_maxsize']})")
    return p.parse_args()


def main():
    args = parse_args()

    target = args.target
    if target is None:
        if args.audio_source == "browser":
            target = "firefox"
        else:
            print("error: --target <channel name> is required for --audio-source stream", file=sys.stderr)
            sys.exit(1)

    chat_channel = None
    if not args.no_chat:
        chat_channel = args.chat_channel
        if chat_channel is None and args.audio_source == "stream":
            chat_channel = target  # in stream mode the target IS the channel name
        if chat_channel is None:
            from twitch_translator.firefox_tabs import list_twitch_channels
            tabs = list_twitch_channels()
            if len(tabs) == 1:
                chat_channel = tabs[0]
                print(f"note: chat channel auto-detected from Firefox tab: {chat_channel}",
                      file=sys.stderr)
        if chat_channel is None:
            print(
                "note: chat panel disabled — set the chat channel in Settings (… menu) "
                "or pass --chat-channel <channel>",
                file=sys.stderr,
            )

    pipeline = Pipeline(
        audio_source=args.audio_source,
        target=target,
        model_size=args.model_size,
        translation_model=args.translation_model,
        device=args.device,
        target_lang=args.target_lang,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_max_speech_ms=args.vad_max_speech_ms,
        chat_channel=chat_channel,
        chat_queue_maxsize=args.chat_queue_maxsize,
        glossary=args.glossary or "",
    )
    pipeline.start()

    s = load_settings()
    overlay = CaptionOverlay(pipeline.caption_queue, pipeline=pipeline,
                             geometry=s["caption_geometry"])
    if chat_channel:
        overlay.attach_chat(pipeline.chat_out_queue, geometry=s["chat_geometry"])
    try:
        overlay.run()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
