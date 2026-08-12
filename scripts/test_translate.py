"""Smoke test: translate a few sample sentences to English."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.translate import Translator
from twitch_translator.langcodes import to_nllb

SAMPLES = [
    ("es", "Hola a todos, bienvenidos a mi transmisión de hoy."),
    ("fr", "Je pense que ce jeu est vraiment difficile mais amusant."),
    ("ja", "今日は新しいゲームをプレイします。"),
    ("de", "Das war ein unglaublicher Sieg für unser Team."),
]


def main():
    t = Translator()
    for whisper_lang, text in SAMPLES:
        nllb_lang = to_nllb(whisper_lang)
        translated = t.translate(text, nllb_lang)
        line = f"[{whisper_lang} -> {nllb_lang}] {text!r} -> {translated!r}"
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
