"""Twitch Live Translator: captures audio from a Twitch stream (via a specific
browser process or directly from Twitch's servers), transcribes it, translates
it, and shows the result as a floating always-on-top caption bar.

Defaults come from settings.json (edit it directly to change your usual setup);
any field can be overridden per-run with the matching CLI flag below.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

# overlay/pipeline are NOT imported here — they transitively pull in
# torch/transformers/faster-whisper, which take several seconds just to
# import. Importing them at module level means Python spends that whole time
# before main() runs a single line of our own code, so the console sits
# blank with no way to tell it's not just hung. main() prints a startup
# message first, then imports these locally right after.
from twitch_translator.settings import load_settings

LOG_PATH = Path(__file__).resolve().parent / "logs" / "latest.log"


class _Tee:
    """Writes to multiple streams — lets the console show output during
    startup while a log file also keeps it after the console is hidden.
    Falls back to UTF-8 bytes on a stream whose codec can't encode the text
    (Windows consoles often default to a non-UTF-8 codepage, e.g. cp1252/850,
    which chokes on Japanese/etc. — write raw encoded bytes to that stream's
    buffer instead of failing the whole write)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                buf = getattr(s, "buffer", None)
                if buf is not None:
                    buf.write(data.encode("utf-8", errors="replace"))
            # The log file isn't a terminal, so it's block-buffered by default —
            # writes sit invisible until enough accumulate. Flush every write so
            # tailing the log always shows what just happened, not what happened
            # several KB ago.
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _tee_output_to_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def _set_console_visible(visible: bool):
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 5 if visible else 0)  # SW_SHOW / SW_HIDE


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
    p.add_argument("--no-context-helper", action="store_true", default=not s["enable_context_helper"],
                    help="disable flagging shaky translations for a hover explanation")
    p.add_argument("--context-confidence-threshold", type=float, default=s["context_confidence_threshold"],
                    help="avg. log-prob below this gets flagged as shaky "
                         f"(default: {s['context_confidence_threshold']})")
    return p.parse_args()


def main():
    _tee_output_to_log()
    print("Twitch Live Translator starting...", file=sys.stderr)
    print("Importing torch/transformers/faster-whisper (several seconds, one-time per launch)...",
          file=sys.stderr)
    from twitch_translator.overlay import CaptionOverlay
    from twitch_translator.pipeline import Pipeline

    args = parse_args()

    target = args.target
    if target is None:
        if args.audio_source == "browser":
            target = "firefox"
        else:
            print("error: --target <channel name> is required for --audio-source stream", file=sys.stderr)
            sys.exit(1)

    # Chat no longer needs a channel at launch — the reader thread idles until
    # one is set, and the panel's own combobox (or Settings) can set it live.
    # This is just the *initial* value it'll start on, if any is known yet.
    chat_channel = args.chat_channel
    if chat_channel is None and args.audio_source == "stream":
        chat_channel = target  # in stream mode the target IS the channel name
    if chat_channel is None and not args.no_chat:
        from twitch_translator.firefox_tabs import list_twitch_channels
        tabs = list_twitch_channels()
        if len(tabs) == 1:
            chat_channel = tabs[0]
            print(f"note: chat channel auto-detected from Firefox tab: {chat_channel}",
                  file=sys.stderr)

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
        chat_disabled=args.no_chat,
        context_helper_enabled=not args.no_context_helper,
        context_confidence_threshold=args.context_confidence_threshold,
    )
    pipeline.start()

    s = load_settings()
    overlay = CaptionOverlay(pipeline.caption_queue, pipeline=pipeline,
                             geometry=s["caption_geometry"])
    if not args.no_chat:
        overlay.attach_chat(pipeline.chat_out_queue, geometry=s["chat_geometry"])

    # The windows are already showing by this point (Tk maps them on creation,
    # not on mainloop) — safe to hide the console now that we've "booted up".
    # Full output still goes to logs/latest.log if something needs checking later.
    print("Ready — hiding this console. Check logs/latest.log if something looks wrong.",
          file=sys.stderr)
    _set_console_visible(False)
    try:
        overlay.run()
    except Exception:
        _set_console_visible(True)  # surface crashes rather than failing silently hidden
        raise
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
