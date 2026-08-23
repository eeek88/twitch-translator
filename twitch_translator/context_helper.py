"""CPU-side helper that adds a short clarifying note to translations the
translation model produced with low confidence. Deliberately kept off the
GPU and off the hot path: it runs in its own thread (see pipeline.py's
_context_worker), triggered only for flagged lines, so it never competes
with the live speech/chat pipeline for VRAM or GPU scheduling time.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM_PROMPT = (
    "You add a short clarifying note to a live speech/chat translation that "
    "looked shaky. In one short sentence (under 25 words), explain what the "
    "original most likely means, flag slang/wordplay/names that don't "
    "translate cleanly, or say the meaning is genuinely unclear if you can't "
    "tell. Do not just repeat the translation. Reply with only the note, no "
    "preamble. Recent chat may be included for situational context — use it "
    "only if it actually helps; ignore it otherwise."
)


class ContextHelper:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # float32, not bfloat16: measured live, bf16 was ~13x slower per token
        # on a consumer CPU (2.9s/tok vs 0.22s/tok) — client Intel/AMD chips
        # since ~2022 dropped AVX-512 (E-cores don't support it), so bf16 has
        # no hardware acceleration and PyTorch falls back to slow emulation.
        # fp32 costs ~2x the RAM but runs on the CPU's actual fast path.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()

    @torch.inference_mode()
    def explain(self, original_text: str, translated_text: str, src_lang: str,
               recent_chat: Optional[list[str]] = None) -> str:
        context_block = ""
        if recent_chat:
            context_block = (
                "Recent chat messages (already translated to English), for "
                "context only:\n" + "\n".join(recent_chat) + "\n\n"
            )
        user = (
            f"{context_block}"
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
