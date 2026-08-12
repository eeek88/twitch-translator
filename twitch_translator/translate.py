"""Translates transcript text to English (or any target) via an NLLB-200 checkpoint."""
from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"


class Translator:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype, use_safetensors=True).to(device)
        self.model.eval()

    @torch.inference_mode()
    def translate(self, text: str, src_lang: str, tgt_lang: str = "eng_Latn") -> str:
        if not text.strip():
            return ""
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
        )
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
