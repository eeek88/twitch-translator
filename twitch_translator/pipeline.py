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
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from . import capture
from .asr import ASR
from .langcodes import WHISPER_TO_NLLB, to_nllb
from .translate import DEFAULT_MODEL_NAME, Translator
from .vad import VADSegmenter

CHAT_DETECT_MIN_CONFIDENCE = 0.9  # below this, short-text language detection is guesswork


@dataclass
class Caption:
    detected_lang: str
    original_text: str
    translated_text: str

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


def _log(line: str) -> None:
    sys.stderr.buffer.write((line + "\n").encode("utf-8", errors="replace"))
    sys.stderr.flush()


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
    ):
        self.audio_source = audio_source
        self.target = target
        self.target_lang = target_lang
        self.chat_channel = chat_channel

        self.vad = VADSegmenter(
            threshold=vad_threshold,
            min_silence_ms=vad_min_silence_ms,
            min_speech_ms=vad_min_speech_ms,
            max_speech_ms=vad_max_speech_ms,
        )
        self.asr = ASR(model_size=model_size, device=device)
        self.translator = Translator(model_name=translation_model, device=device)

        self._utterance_queue: "queue.Queue[object]" = queue.Queue()
        self._chat_in_queue: "queue.Queue[tuple[str, str, str]]" = queue.Queue(maxsize=chat_queue_maxsize)
        self.caption_queue: "queue.Queue[object]" = queue.Queue()
        self.chat_out_queue: "queue.Queue[ChatLine]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

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

        from .chat import chat_messages

        identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        identifier.set_languages([l for l in WHISPER_TO_NLLB if l in identifier.nb_classes])
        target_whisper_langs = {w for w, n in WHISPER_TO_NLLB.items() if n == self.target_lang}

        backoff = 2.0
        while not self._stop.is_set():
            try:
                for username, message in chat_messages(self.chat_channel):
                    if self._stop.is_set():
                        return
                    backoff = 2.0
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
                _log(f"[chat error, retrying: {exc}]")
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
                translated = self.translator.translate(text, nllb_lang, self.target_lang)
                _log(f"[{lang}] {text!r} -> {translated!r}")
                self.caption_queue.put(Caption(lang, text, translated))
            except Exception as exc:
                self.caption_queue.put(Caption.status(f"[transcription error: {exc}]"))

    def _translate_pending_chat(self):
        try:
            username, lang, message = self._chat_in_queue.get_nowait()
        except queue.Empty:
            return
        try:
            translated = self.translator.translate(message, to_nllb(lang), self.target_lang)
            _log(f"[chat/{lang}] {username}: {message!r} -> {translated!r}")
            self.chat_out_queue.put(ChatLine(username, lang, message, translated))
        except Exception as exc:
            _log(f"[chat translation error: {exc}]")

    def start(self):
        self._threads = [
            threading.Thread(target=self._capture_and_segment, daemon=True),
            threading.Thread(target=self._gpu_worker, daemon=True),
        ]
        if self.chat_channel:
            self._threads.append(threading.Thread(target=self._read_chat, daemon=True))
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
