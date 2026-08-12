"""Shared repetition-loop detector, used to guard both Whisper's transcription
output (asr.py) and NLLB's translation output (translate.py) — this failure
mode isn't specific to either model, it's a generic autoregressive-decoding
pathology on inputs the model isn't confident about, and it turns out to hit
translation just as much as transcription (confirmed live: a chat message
translated into "really," repeated dozens of times, with no ASR involved at
all — the earlier ASR-only guard was fixing half the problem).
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # alnum runs, punctuation/case-insensitive


def has_repetition_loop(text: str) -> bool:
    """Checked at the word level (case/punctuation stripped), not as an exact
    character-substring match: real repetition loops often drift slightly
    between iterations (a dropped comma, a doubled space), which breaks any
    check requiring byte-for-byte identical repeats.
    """
    words = _WORD_RE.findall(text.lower())
    n_words = len(words)
    # Candidate repeating-phrase lengths, in words. Apostrophes split into
    # separate tokens ("I'm" -> "i", "m"), so a phrase that reads short can
    # still need a longer window than it looks like it should — scale the
    # cap with the text instead of hardcoding one that quietly misses cases.
    max_n = min(24, n_words // 3)
    for n in range(3, max_n + 1):
        for i in range(n_words - n * 3 + 1):
            window = words[i:i + n]
            if words[i + n:i + 2 * n] == window and words[i + 2 * n:i + 3 * n] == window:
                return True
    return False
