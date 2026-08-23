"""CPU-side helper that adds a short clarifying note to translations the
translation model produced with low confidence. Deliberately kept off the
GPU and off the hot path: it runs in its own thread (see pipeline.py's
_context_worker), triggered only for flagged lines, so it never competes
with the live speech/chat pipeline for VRAM or GPU scheduling time.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM_PROMPT = (
    "You add a short clarifying note to a live speech/chat translation that "
    "looked shaky. In one short sentence (under 25 words), explain what the "
    "original most likely means, flag slang/wordplay/names that don't "
    "translate cleanly, or say the meaning is genuinely unclear if you can't "
    "tell. Do not just repeat the translation. Reply with only the note, no "
    "preamble."
)


class ContextHelper:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # bfloat16 on CPU: half the download/RAM of fp32, and modern CPUs
        # handle it fine for a model this small — this only ever runs on the
        # occasional flagged line, so raw throughput isn't the bottleneck.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()

    @torch.inference_mode()
    def explain(self, original_text: str, translated_text: str, src_lang: str) -> str:
        user = (
            f"Source language: {src_lang}\n"
            f"Original: {original_text!r}\n"
            f"Translation: {translated_text!r}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        output = self.model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
