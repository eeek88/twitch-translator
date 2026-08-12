# Twitch Live Translator

Captures audio from a Twitch stream, transcribes it, translates it to English,
and shows the result as a floating always-on-top caption bar you can drag
anywhere on screen — over Firefox, a game, whatever.

Everything runs locally on your GPU. No cloud APIs, no API keys, no per-minute
cost.

## How it works

```
Twitch channel
   -> audio capture (Firefox's own audio, or a direct pull from Twitch's servers)
   -> Silero VAD (splits speech into utterances)
   -> faster-whisper (transcribes + detects source language)
   -> NLLB-200 (translates to English)
   -> floating caption window
```

Expect roughly 2-5 seconds of lag between speech and caption — this pipeline
processes utterance-by-utterance, not word-by-word (Whisper isn't a true
streaming model).

## One-time setup

Requirements: NVIDIA GPU (recommended), Windows 10 2004+ or Windows 11.

1. **Python 3.11+**, **ffmpeg**, and **.NET 8 SDK** must be on PATH.
2. Create a venv and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   .venv\Scripts\pip install -r requirements.txt
   ```
3. Build the browser-capture helper:
   ```
   cd helper
   dotnet build -c Release
   cd ..
   ```

## Usage

Open the Twitch stream in Firefox, then run:

```
.venv\Scripts\python main.py
```

By default this captures Firefox's own audio (synced to what you see, and
immune to other sounds on your PC — Discord, music, etc. — since it targets
only the Firefox process).

Drag the caption bar wherever you like. Press **Escape** with the caption
window focused to quit.

### Translated chat panel

A second draggable panel shows chat messages translated to your target
language (foreign-language messages only — messages already in the target
language, emote spam, and messages whose language can't be confidently
detected are skipped). Chat is read anonymously over Twitch IRC; no login or
API key needed.

The chat reader needs to know the actual Twitch channel name. In `stream`
mode it reuses `--target` automatically; in `browser` mode (where the target
is a process name, not a channel) pass it explicitly:

```
.venv\Scripts\python main.py --chat-channel <channel_name>
```

Speech captions always take priority for the GPU — chat translations happen
in the gaps between utterances, and during heavy chat bursts excess messages
are dropped rather than queued up stale.

### Settings

Defaults live in `settings.json` at the project root — edit it directly to change
your usual setup. Any field can also be overridden per-run with the matching CLI
flag, which takes precedence for that run only:

```
--audio-source {browser,stream}
--target TEXT                browser: process name (default "firefox")
                              stream: Twitch channel name (required)
--model-size TEXT            faster-whisper model size
--translation-model TEXT     NLLB-200 checkpoint (huggingface repo id)
--device {cuda,cpu}
--target-lang TEXT           NLLB target language code
--vad-threshold FLOAT        speech probability threshold, 0-1
--vad-min-silence-ms INT     silence gap that ends an utterance
--vad-min-speech-ms INT      utterances shorter than this are discarded as noise
--vad-max-speech-ms INT      utterances longer than this are cut off
--no-chat                    disable the translated-chat panel
--chat-channel TEXT          channel whose chat to translate (see chat section)
--chat-max-messages INT      lines kept visible in the chat panel
--chat-queue-maxsize INT     pending chat translations before new ones drop
```

**Quality vs. speed/VRAM tradeoffs** worth knowing about:
- `model_size`: bigger Whisper models are meaningfully more accurate on
  non-English speech, especially languages structurally far from English
  (Japanese, Korean, etc.) — `large-v3` is the default; drop to `medium` or
  `small` if you need faster turnaround or have less VRAM.
- `translation_model`: `facebook/nllb-200-distilled-1.3B` (default) is
  noticeably better than `distilled-600M`, at some added latency/VRAM.
  `facebook/nllb-200-3.3B` is better still if you have the VRAM to spare.
- `vad_min_silence_ms`: raising this (e.g. 800-1200ms) reduces mid-sentence
  cutoffs, which matters most for verb-final/high-context languages like
  Japanese where cutting before the sentence's end can flip the meaning. The
  cost is added latency, since the VAD waits longer to confirm a pause is real.

**Fallback backend** — if browser capture ever proves unreliable (Firefox not
open, capture helper failing), pull audio directly from Twitch's servers
instead. This avoids the browser dependency entirely (works headless) but the
captions may drift a few seconds out of sync with what's on your screen, since
streamlink and the browser player don't necessarily buffer to the same point
in the live stream:

```
.venv\Scripts\python main.py --audio-source stream --target <channel_name>
```

## Project layout

```
helper/                C# console app: Windows process-loopback audio capture (targets firefox.exe)
twitch_translator/
  capture.py            audio capture backends -> raw PCM
  vad.py                 Silero VAD speech segmentation
  asr.py                  faster-whisper transcription
  translate.py             NLLB-200 translation
  langcodes.py              Whisper <-> NLLB language code mapping
  overlay.py                 floating caption window
  pipeline.py                 wires it all together across threads
main.py                CLI entrypoint
scripts/               standalone smoke tests for each module
```

## Known limitations

- Browser capture targets whichever `firefox.exe` process owns a visible main
  window. If you have multiple Firefox windows open, it may not pick the one
  playing the stream.
- `--audio-source stream` may lag behind what's on your browser screen by an
  unpredictable amount (on top of the usual ASR/translation delay), since it's
  an independent pull from Twitch's HLS feed.
- Translation quality depends on NLLB-200-distilled-600M, a relatively small
  model chosen for speed. It's solid for casual understanding, not
  publication-grade.
