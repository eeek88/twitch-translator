"""Smoke test for this round's additions: live target-lang switching, the
context-helper busy flag, caption latency tracking, transcript recording,
and the perf/notes overlay toggles. Uses small/fast models and never calls
Pipeline.start(), so no live stream/Firefox connection is needed — capture
threads only spin up on start()."""
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from twitch_translator.overlay import CaptionOverlay, NotesOverlay, PerfOverlay
from twitch_translator.pipeline import NoteEvent, Pipeline


def check(label, ok, detail=""):
    line = f"{'PASS' if ok else 'FAIL'} [{label}] {detail}"
    sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()


def main():
    pipeline = Pipeline(
        audio_source="stream", target="dummy",
        model_size="tiny", translation_model="facebook/nllb-200-distilled-600M",
    )

    # --- live target-lang switching ---
    pipeline.set_target_lang("spa_Latn")
    check("set_target_lang", pipeline.target_lang == "spa_Latn", f"got {pipeline.target_lang!r}")

    # --- context-helper busy flag ---
    check("context_helper_busy initially False", pipeline.context_helper_busy is False)
    pipeline._context_worker_busy.set()
    check("context_helper_busy True after set()", pipeline.context_helper_busy is True)
    pipeline._context_worker_busy.clear()

    # --- caption latency tracking ---
    check("last_caption_latency initially None", pipeline.last_caption_latency is None)
    pipeline._record_caption_latency(1.5)
    pipeline._record_caption_latency(2.5)
    check("last_caption_latency", pipeline.last_caption_latency == 2.5)
    check("avg_caption_latency", pipeline.avg_caption_latency == 2.0)

    # --- transcript recording ---
    path = pipeline.start_transcript_recording()
    check("is_recording_transcript True", pipeline.is_recording_transcript is True)
    pipeline._write_transcript("[test] hello -> world")
    pipeline.stop_transcript_recording()
    check("is_recording_transcript False after stop", pipeline.is_recording_transcript is False)
    content = path.read_text(encoding="utf-8")
    check("transcript file has the written line", "[test] hello -> world" in content, f"path={path}")
    path.unlink()  # clean up the test artifact

    # --- notes_queue + NotesOverlay ---
    overlay = CaptionOverlay(queue.Queue(), pipeline=pipeline)
    overlay.attach_chat(queue.Queue())
    overlay.root.update_idletasks()

    check("context_dot created (context helper enabled by default)", overlay.context_dot is not None)

    overlay._toggle_perf_overlay()
    check("perf_overlay opened", isinstance(overlay.perf_overlay, PerfOverlay))
    overlay._toggle_perf_overlay()
    check("perf_overlay closed", overlay.perf_overlay is None)

    overlay._toggle_notes_overlay()
    check("notes_overlay opened", isinstance(overlay.notes_overlay, NotesOverlay))
    pipeline.notes_queue.put(NoteEvent("原文", "translation", "ja", "a helpful note"))
    overlay.notes_overlay._poll()
    overlay.root.update()
    text_content = overlay.notes_overlay.scrollback.text.get("1.0", "end")
    check("note rendered in notes window", "translation" in text_content and "helpful note" in text_content,
          repr(text_content.strip()))
    overlay._toggle_notes_overlay()
    check("notes_overlay closed", overlay.notes_overlay is None)

    overlay.root.destroy()


if __name__ == "__main__":
    main()
