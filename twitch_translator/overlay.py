"""Always-on-top, draggable caption bar. Uses real per-window alpha blending
(not chroma-key transparency) so it reads as a semi-see-through subtitle bar
rather than a fully invisible background — closer to what a viewer actually wants."""
from __future__ import annotations

import queue
import tkinter as tk

from .pipeline import Caption, ChatLine

BG_COLOR = "#101010"
FG_COLOR = "#ffffff"
CHAT_NAME_COLOR = "#7fb8ff"
WINDOW_ALPHA = 0.82
MAX_LINES = 3


class CaptionOverlay:
    def __init__(self, caption_queue: "queue.Queue[object]", poll_ms: int = 100):
        self.caption_queue = caption_queue
        self.poll_ms = poll_ms
        self.lines: list[str] = []
        self._drag_start = (0, 0)

        self.root = tk.Tk()
        self.root.title("Twitch Live Translator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", WINDOW_ALPHA)
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("900x160+180+760")

        self.label = tk.Label(
            self.root,
            text="Waiting for speech…",
            font=("Segoe UI", 18, "bold"),
            fg=FG_COLOR,
            bg=BG_COLOR,
            justify="center",
            wraplength=860,
            padx=16,
            pady=12,
        )
        self.label.pack(expand=True, fill="both")

        for widget in (self.root, self.label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

        self.root.bind("<Escape>", lambda _e: self.root.destroy())

        self.root.after(self.poll_ms, self._poll)

    def _start_drag(self, event):
        self._drag_start = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _on_drag(self, event):
        x = event.x_root - self._drag_start[0]
        y = event.y_root - self._drag_start[1]
        self.root.geometry(f"+{x}+{y}")

    def _poll(self):
        try:
            while True:
                item = self.caption_queue.get_nowait()
                if item is None:
                    continue
                caption: Caption = item
                self.lines.append(caption.translated_text)
                self.lines = self.lines[-MAX_LINES:]
                self.label.config(text="\n".join(self.lines))
        except queue.Empty:
            pass
        self.root.after(self.poll_ms, self._poll)

    def attach_chat(self, chat_queue: "queue.Queue[ChatLine]", max_messages: int = 15):
        """Open the translated-chat panel as a second draggable window."""
        ChatOverlay(self.root, chat_queue, max_messages, self.poll_ms)

    def run(self):
        self.root.mainloop()


class ChatOverlay:
    """Small scrolling panel of translated chat messages. A Toplevel of the
    caption window so both share one Tk mainloop; closes together with it."""

    def __init__(self, parent: tk.Tk, chat_queue: "queue.Queue[ChatLine]",
                 max_messages: int, poll_ms: int):
        self.chat_queue = chat_queue
        self.max_messages = max_messages
        self.poll_ms = poll_ms
        self._drag_start = (0, 0)

        self.win = tk.Toplevel(parent)
        self.win.title("Translated Chat")
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", WINDOW_ALPHA)
        self.win.configure(bg=BG_COLOR)
        self.win.geometry("380x300+1160+400")

        self.text = tk.Text(
            self.win,
            font=("Segoe UI", 11),
            fg=FG_COLOR,
            bg=BG_COLOR,
            wrap="word",
            state="disabled",
            relief="flat",
            padx=10,
            pady=8,
            cursor="arrow",
        )
        self.text.tag_configure("name", foreground=CHAT_NAME_COLOR, font=("Segoe UI", 11, "bold"))
        self.text.pack(expand=True, fill="both")

        for widget in (self.win, self.text):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

        self.win.after(self.poll_ms, self._poll)

    def _start_drag(self, event):
        self._drag_start = (event.x_root - self.win.winfo_x(), event.y_root - self.win.winfo_y())
        return "break"  # keep the Text widget from starting a selection drag

    def _on_drag(self, event):
        x = event.x_root - self._drag_start[0]
        y = event.y_root - self._drag_start[1]
        self.win.geometry(f"+{x}+{y}")
        return "break"

    def _poll(self):
        try:
            while True:
                line: ChatLine = self.chat_queue.get_nowait()
                self.text.configure(state="normal")
                self.text.insert("end", f"{line.username}: ", "name")
                self.text.insert("end", f"{line.translated_text}\n")
                # trim to the newest max_messages lines
                line_count = int(self.text.index("end-1c").split(".")[0])
                if line_count > self.max_messages:
                    self.text.delete("1.0", f"{line_count - self.max_messages + 1}.0")
                self.text.configure(state="disabled")
                self.text.see("end")
        except queue.Empty:
            pass
        self.win.after(self.poll_ms, self._poll)
