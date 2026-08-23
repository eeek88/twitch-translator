"""Translates transcript text to English (or any target) via an NLLB-200 checkpoint."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .text_repetition import has_repetition_loop

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"


@dataclass
class Translation:
    text: str
    # Average per-token log-probability of the generated sequence (from NLLB's
    # own generation scores) — 0 is maximally confident, more negative is less
    # so. Used to flag translations worth a second look from the context
    # helper (see context_helper.py); not meaningful for empty text.
    confidence: float


class Translator:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        # device_map loads weights straight to the target device instead of CPU
        # then a separate .to(device) copy — roughly 2x faster to load (measured
        # 23.5s -> 11s for NLLB-1.3B), since it skips a full host-memory pass.
        if device == "cuda":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, dtype=dtype, use_safetensors=True, device_map=device,
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, dtype=dtype, use_safetensors=True,
            ).to(device)
        self.model.eval()
        # NLLB checkpoints ship max_length=200 in their generation config; we pass
        # max_new_tokens per call, and transformers warns about the redundant pair
        # on every single generate(). Drop the checkpoint's limit so ours is the
        # only one in play.
        self.model.generation_config.max_length = None

    def translate(self, text: str, src_lang: str, tgt_lang: str = "eng_Latn") -> str:
        return self.translate_scored(text, src_lang, tgt_lang).text

    @torch.inference_mode()
    def translate_scored(self, text: str, src_lang: str, tgt_lang: str = "eng_Latn") -> Translation:
        if not text.strip():
            return Translation("", 0.0)
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            # The repetition-loop failure isn't just a Whisper/ASR thing — NLLB
            # hits it too (confirmed live: a short chat message translated into
            # "really," repeated dozens of times). no_repeat_ngram_size blocks
            # any 3-token sequence from being generated twice, stopping the loop
            # at the source instead of only catching it after the fact.
            no_repeat_ngram_size=3,
            output_scores=True,
            return_dict_in_generate=True,
        )
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True,
        )
        # -inf entries pad sequences shorter than the batch max (batch size is
        # always 1 here, but the API always returns a 2D tensor) — exclude them
        # rather than let them wreck the average.
        valid = transition_scores > float("-inf")
        confidence = (transition_scores[valid].mean().item() if valid.any() else 0.0)
        result = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        # Safety net for whatever no_repeat_ngram_size doesn't catch (e.g. a
        # longer phrase repeating with enough varying tokens between repeats).
        if result and has_repetition_loop(result):
            return Translation("", confidence)
        return Translation(result, confidence)
