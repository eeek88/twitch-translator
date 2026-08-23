"""Wires capture -> VAD -> ASR -> translation together across background threads.

Threads, connected by queues:
  - capture+VAD thread: continuously drains the audio pipe and segments it into
    utterances. Kept together because VAD itself is cheap; the audio pipe must
    never be left unread or the upstream process (ffmpeg/helper.exe) stalls.
  - chat reader thread (optional): reads Twitch IRC, language-detects, and
    enqueues foreign-language messages. The queue is bounded and drops on
    overflow — during a chat burst, stale messages are worthless.
  - GPU worker thread: the slow steps (ASR + translation). One thread owns all
    GPU work because the models aren't safe to call concurrently and loading a
    second copy would waste VRAM. Speech has strict priority; chat messages are
    only translated when no utterance is waiting.
  - context-helper thread (optional): explains translations the translation
    model itself was unconfident about. Deliberately kept off the GPU (a
    separate small CPU-only model) and off the above threads' critical path.
"""
from __future__ import annotations

import itertools
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

from . import capture
from .asr import ASR
from .context_helper import DEFAULT_MODEL_NAME as DEFAULT_CONTEXT_MODEL_NAME
from .langcodes import WHISPER_TO_NLLB, to_nllb
from .translate import DEFAULT_MODEL_NAME, Translator
from .vad import VADSegmenter

CHAT_DETECT_MIN_CONFIDENCE = 0.9  # below this, short-text language detection is guesswork
CONTEXT_QUEUE_MAXSIZE = 20  # pending context-helper requests before new ones drop
CONTEXT_CACHE_MAXSIZE = 500  # cached notes kept before the oldest is evicted
RECENT_CHAT_MAXLEN = 5  # translated chat lines kept as context for the helper's notes

_line_ids = itertools.count()


def _normalize_for_cache(text: str) -> str:
    """Collapses whitespace/case differences so near-identical repeats (a
    recurring phrase, a chat burst of copies with a stray space or two) hit
    the same cache entry instead of each triggering their own LLM call."""
    return " ".join(text.split()).lower()


@dataclass
class Caption:
    detected_lang: str
    original_text: str
    translated_text: str
    line_id: int = field(default_factory=lambda: next(_line_ids))
    needs_context: bool = False

    @classmethod
    def status(cls, message: str) -> "Caption":
        """A non-fatal informational line (e.g. reconnect attempts) rendered like a caption."""
        return cls("system", "", message)


@dataclass
class ChatLine:
    username: str
    detected_lang: str
    original_text: str
    translated_text: str
    line_id: int = field(default_factory=lambda: next(_line_ids))
    needs_context: bool = False


@dataclass
class ContextReady:
    """Posted to the same queue as the Caption/ChatLine it annotates, once the
    context helper (running on CPU, in its own thread) finishes explaining a
    flagged line. The overlay matches it back up by line_id."""
    line_id: int
    context: str


def _log(line: str) -> None:
    # main.py's console/log Tee handles non-UTF-8-console fallback centrally;
    # a plain write here works whether stderr is the real console or that Tee.
    # flush=True matters: the log file is opened in default (block) buffering
    # since it isn't a terminal, so without this, lines sit invisible in the
    # buffer until enough accumulate — exactly when you'd want to tail the
    # log to see what just happened, it wouldn't be there yet.
    print(line, file=sys.stderr, flush=True)


class Pipeline:
    def __init__(
        self,
        audio_source: str,
        target: str,
        model_size: str = "medium",
        translation_model: str = DEFAULT_MODEL_NAME,
        device: str = "cuda",
        target_lang: str = "eng_Latn",
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 500,
        vad_min_speech_ms: int = 300,
        vad_max_speech_ms: int = 12000,
        chat_channel: Optional[str] = None,
        chat_queue_maxsize: int = 50,
        glossary: str = "",
        chat_disabled: bool = False,
        context_helper_enabled: bool = True,
        context_confidence_threshold: float = -0.5,
        context_helper_model: str = DEFAULT_CONTEXT_MODEL_NAME,
    ):
        self.audio_source = audio_source
        self.target = target
        self.target_lang = target_lang
        self.chat_channel = chat_channel
        self.chat_disabled = chat_disabled
        self.context_helper_enabled = context_helper_enabled
        self.context_confidence_threshold = context_confidence_threshold
        self.context_helper_model = context_helper_model
        self._chat_reader = None
        self._chat_reader_lock = threading.Lock()

        _log("Loading voice-activity detector (Silero VAD)...")
        self.vad = VADSegmenter(
            threshold=vad_threshold,
            min_silence_ms=vad_min_silence_ms,
            min_speech_ms=vad_min_speech_ms,
            max_speech_ms=vad_max_speech_ms,
        )
        _log(f"Loading speech recognition model (faster-whisper {model_size})...")
        self.asr = ASR(model_size=model_size, device=device, glossary=glossary)
        _log(f"Loading translation model ({translation_model})...")
        self.translator = Translator(model_name=translation_model, device=device)
        _log("Models loaded.")

        self._utterance_queue: "queue.Queue[object]" = queue.Queue()
        self._chat_in_queue: "queue.Queue[tuple[str, str, str]]" = queue.Queue(maxsize=chat_queue_maxsize)
        self.caption_queue: "queue.Queue[object]" = queue.Queue()
        self.chat_out_queue: "queue.Queue[object]" = queue.Queue()
        # (cache_key, original_text, translated_text, lang) for text awaiting a
        # context-helper explanation. Bounded: under a sustained high flag rate
        # (e.g. an aggressive confidence threshold on a busy chat) the CPU
        # worker can't keep up, so excess requests drop rather than backlog
        # forever (confirmed live: unbounded, this grew ~1GB/36s under stress).
        self._context_queue: "queue.Queue[tuple[str, str, str, str]]" = queue.Queue(maxsize=CONTEXT_QUEUE_MAXSIZE)
        # cache_key -> note, for text seen before (recurring phrases skip the
        # LLM call entirely); cache_key -> [(line_id, is_chat), ...] for text
        # currently being processed (concurrent duplicates piggyback on that
        # one in-flight call instead of each starting their own).
        self._context_cache: dict[str, str] = {}
        self._context_inflight: dict[str, list[tuple[int, bool]]] = {}
        self._context_cache_lock = threading.Lock()
        # Recent translated chat lines, given to the context helper as extra
        # situational context for its notes.
        self._recent_chat: "deque[str]" = deque(maxlen=RECENT_CHAT_MAXLEN)
        self._stop = threading.Event()
        self._chat_enabled = threading.Event()
        self._chat_enabled.set()
        self._threads: list[threading.Thread] = []

    def set_chat_channel(self, channel: Optional[str]) -> None:
        """Switch the chat reader to a different channel. If a connection is
        already live, switches channel on it directly (IRC PART/JOIN) rather
        than tearing down and reconnecting — skips the TCP handshake and NICK
        registration round-trip, which is most of what makes a full reconnect
        feel slow. Falls back to a full reconnect (via stop(), same as
        before) when there's no live connection yet, or the live switch
        itself fails, or the new channel is None (nothing to join)."""
        self.chat_channel = channel
        while True:  # old channel's backlog shouldn't bleed into the new one
            try:
                self._chat_in_queue.get_nowait()
            except queue.Empty:
                break
        with self._chat_reader_lock:
            reader = self._chat_reader
        if reader is not None and channel:
            try:
                reader.switch_channel(channel)
                return
            except Exception:
                pass  # dead connection or similar — fall through to a full reconnect
        with self._chat_reader_lock:
            if self._chat_reader is not None:
                self._chat_reader.stop()

    def set_chat_enabled(self, enabled: bool) -> None:
        """Pause/resume chat translation at runtime (e.g. when the panel is hidden).
        Paused chat consumes no GPU time; pending messages are discarded."""
        if enabled:
            self._chat_enabled.set()
        else:
            self._chat_enabled.clear()
            while True:  # stale messages shouldn't appear when re-enabled later
                try:
                    self._chat_in_queue.get_nowait()
                except queue.Empty:
                    break

    def _capture_and_segment(self):
        # Capture backends fail transiently (stream buffering, Firefox not yet open,
        # brief network hiccups). Retry with backoff instead of killing the pipeline;
        # only Pipeline.stop() should end this thread.
        backoff = 2.0
        try:
            while not self._stop.is_set():
                try:
                    pcm_iter = capture.audio_stream(self.audio_source, self.target)
                    for utterance in self.vad.segments(pcm_iter):
                        if self._stop.is_set():
                            return
                        self._utterance_queue.put(utterance)
                        backoff = 2.0
                    if self._stop.is_set():
                        return
                    self.caption_queue.put(Caption.status("[stream ended, reconnecting...]"))
                except Exception as exc:
                    if self._stop.is_set():
                        return
                    self.caption_queue.put(Caption.status(f"[capture error, retrying: {exc}]"))
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
        finally:
            self._utterance_queue.put(None)  # always unblock the GPU worker on the way out

    def _read_chat(self):
        from py3langid.langid import LanguageIdentifier, MODEL_FILE

        from .chat import ChatReader

        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        identifier.set_languages([l for l in WHISPER_TO_NLLB if l in identifier.nb_classes])

        backoff = 2.0
        while not self._stop.is_set():
            channel = self.chat_channel
            if not channel:
                time.sleep(0.5)  # no channel picked yet — idle until set_chat_channel()
                continue

            target_whisper_langs = {w for w, n in WHISPER_TO_NLLB.items() if n == self.target_lang}
            reader = ChatReader(channel)
            with self._chat_reader_lock:
                self._chat_reader = reader
            try:
                for username, message in reader.messages():
                    # reader.channel (not the local `channel`) reflects a live
                    # switch_channel() call — this connection is meant to keep
                    # running through those, so only a stop() (which leaves
                    # reader.channel behind) or explicit pipeline stop ends it.
                    if self._stop.is_set() or self.chat_channel != reader.channel:
                        break  # explicit stop, or a switch that needed a fresh connection
                    backoff = 2.0
                    if not self._chat_enabled.is_set():
                        continue  # panel hidden: keep the socket alive but do no work
                    # Emote-spam and symbol-only messages translate to garbage.
                    if not any(ch.isalpha() for ch in message):
                        continue
                    lang, prob = identifier.classify(message)
                    if lang in target_whisper_langs:
                        continue  # already in the target language
                    if float(prob) < CHAT_DETECT_MIN_CONFIDENCE:
                        continue  # too short/ambiguous to trust the detection
                    try:
                        self._chat_in_queue.put_nowait((username, lang, message))
                    except queue.Full:
                        pass  # burst overflow: drop rather than backlog stale chat
            except Exception as exc:
                if self._stop.is_set():
                    return
                if self.chat_channel == reader.channel:  # a switch-triggered close isn't an error
                    _log(f"[chat error, retrying: {exc}]")
            finally:
                with self._chat_reader_lock:
                    if self._chat_reader is reader:
                        self._chat_reader = None

            if self._stop.is_set():
                return
            if self.chat_channel != reader.channel:
                continue  # switched to a channel this connection couldn't join live — reconnect now
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _gpu_worker(self):
        """Owns all GPU inference. Speech utterances take strict priority; chat
        is translated only while the utterance queue is empty."""
        while not self._stop.is_set():
            try:
                utterance = self._utterance_queue.get(timeout=0.2)
            except queue.Empty:
                self._translate_pending_chat()
                continue
            if utterance is None:
                self.caption_queue.put(None)
                return
            try:
                text, lang = self.asr.transcribe(utterance)
                if not text:
                    continue
                nllb_lang = to_nllb(lang)
                result = self.translator.translate_scored(text, nllb_lang, self.target_lang)
                if not result.text:
                    _log(f"[{lang}] {text!r} -> (suppressed: repetition loop)")
                    continue
                _log(f"[{lang}] {text!r} -> {result.text!r}")
                caption = Caption(lang, text, result.text)
                self._maybe_flag_for_context(caption, text, result, lang, is_chat=False)
                self.caption_queue.put(caption)
            except Exception as exc:
                self.caption_queue.put(Caption.status(f"[transcription error: {exc}]"))

    def _translate_pending_chat(self):
        if not self._chat_enabled.is_set():
            return
        try:
            username, lang, message = self._chat_in_queue.get_nowait()
        except queue.Empty:
            return
        try:
            result = self.translator.translate_scored(message, to_nllb(lang), self.target_lang)
            if not result.text:
                _log(f"[chat/{lang}] {username}: {message!r} -> (suppressed: repetition loop)")
                return
            _log(f"[chat/{lang}] {username}: {message!r} -> {result.text!r}")
            self._recent_chat.append(f"{username}: {result.text}")
            chat_line = ChatLine(username, lang, message, result.text)
            self._maybe_flag_for_context(chat_line, message, result, lang, is_chat=True)
            self.chat_out_queue.put(chat_line)
        except Exception as exc:
            _log(f"[chat translation error: {exc}]")

    def _maybe_flag_for_context(self, line, original_text: str, result, lang: str, is_chat: bool) -> None:
        """Marks a line for the context helper when the translation model's own
        confidence was low. Text matching something already explained (or
        currently being explained) skips straight to that result instead of
        triggering another LLM call — covers both a recurring phrase (cache)
        and a burst of near-duplicate chat messages arriving together
        (in-flight coalescing)."""
        if not self.context_helper_enabled:
            return
        if result.confidence >= self.context_confidence_threshold:
            return
        key = _normalize_for_cache(original_text)
        with self._context_cache_lock:
            cached = self._context_cache.get(key)
            if cached is not None:
                line.needs_context = True
                target_queue = self.chat_out_queue if is_chat else self.caption_queue
                target_queue.put(ContextReady(line.line_id, cached))
                return
            waiters = self._context_inflight.get(key)
            if waiters is not None:
                waiters.append((line.line_id, is_chat))
                line.needs_context = True
                return
            try:
                self._context_queue.put_nowait((key, original_text, result.text, lang))
            except queue.Full:
                return  # backlog full: skip flagging rather than show a marker that never resolves
            self._context_inflight[key] = [(line.line_id, is_chat)]
        line.needs_context = True

    def _context_worker(self):
        """Explains flagged lines on CPU, off the GPU worker's critical path
        entirely. The model loads lazily on the first flagged line rather than
        at startup, since plenty of sessions may never trigger it."""
        helper = None
        while not self._stop.is_set():
            try:
                key, original_text, translated_text, lang = self._context_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if helper is None:
                _log("Loading context-helper model (first flagged line)...")
                from .context_helper import ContextHelper
                helper = ContextHelper(model_name=self.context_helper_model)
            try:
                note = helper.explain(original_text, translated_text, lang,
                                      recent_chat=list(self._recent_chat))
            except Exception as exc:
                _log(f"[context helper error: {exc}]")
                note = ""
            with self._context_cache_lock:
                waiters = self._context_inflight.pop(key, [])
                if note:
                    self._context_cache[key] = note
                    if len(self._context_cache) > CONTEXT_CACHE_MAXSIZE:
                        del self._context_cache[next(iter(self._context_cache))]
            if not note:
                continue
            for line_id, is_chat in waiters:
                target_queue = self.chat_out_queue if is_chat else self.caption_queue
                target_queue.put(ContextReady(line_id, note))

    def start(self):
        self._threads = [
            threading.Thread(target=self._capture_and_segment, daemon=True),
            threading.Thread(target=self._gpu_worker, daemon=True),
        ]
        if not self.chat_disabled:
            self._threads.append(threading.Thread(target=self._read_chat, daemon=True))
        if self.context_helper_enabled:
            self._threads.append(threading.Thread(target=self._context_worker, daemon=True))
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()

    def captions(self) -> Iterator[Caption]:
        """Blocking iterator over captions (including status/error lines).
        Only ends when Pipeline.stop() causes the capture thread to exit."""
        while True:
            item = self.caption_queue.get()
            if item is None:
                return
            yield item
