"""Always-on-top, draggable, resizable overlay windows.

Both windows are frameless (overrideredirect), so they provide their own
chrome: drag anywhere to move, drag the ⇲ grip to resize, "…" opens the menu
on the caption bar, ✕ hides the chat panel. Both keep a scrollback history —
scroll up to read, scroll back to the bottom to resume auto-following.
"""
from __future__ import annotations

import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

from .firefox_tabs import list_twitch_channels
from .pipeline import Caption, ChatLine, Pipeline
from .settings import SETTING_SPECS, load_settings, save_settings

BG_COLOR = "#101010"
FG_COLOR = "#ffffff"
CHAT_NAME_COLOR = "#7fb8ff"
CONTROL_COLOR = "#9a9a9a"
TIMESTAMP_COLOR = "#6f6f6f"
SCROLLBAR_IDLE = "#242424"    # barely visible against the background...
SCROLLBAR_HOVER = "#7a7a7a"   # ...until hovered or dragged
WINDOW_ALPHA = 0.82
HISTORY_LINES = 300

CAPTION_MIN_W, CAPTION_MIN_H = 300, 80
CHAT_MIN_W, CHAT_MIN_H = 200, 120


def _make_draggable(win: tk.Misc, *widgets: tk.Misc):
    state = {"dx": 0, "dy": 0}

    def start(event):
        state["dx"] = event.x_root - win.winfo_x()
        state["dy"] = event.y_root - win.winfo_y()
        return "break"

    def move(event):
        win.geometry(f"+{event.x_root - state['dx']}+{event.y_root - state['dy']}")
        return "break"

    for w in widgets:
        w.bind("<ButtonPress-1>", start)
        w.bind("<B1-Motion>", move)


def _make_resize_grip(win: tk.Misc, min_w: int, min_h: int) -> tk.Label:
    grip = tk.Label(win, text="⇲", font=("Segoe UI", 10), fg=CONTROL_COLOR,
                    bg=BG_COLOR, cursor="size_nw_se")
    state = {"w": 0, "h": 0, "x": 0, "y": 0}

    def start(event):
        state.update(w=win.winfo_width(), h=win.winfo_height(),
                     x=event.x_root, y=event.y_root)
        return "break"

    def resize(event):
        w = max(min_w, state["w"] + (event.x_root - state["x"]))
        h = max(min_h, state["h"] + (event.y_root - state["y"]))
        win.geometry(f"{w}x{h}")
        return "break"

    grip.bind("<ButtonPress-1>", start)
    grip.bind("<B1-Motion>", resize)
    return grip


_SCROLLBAR_STYLE_READY = False


def _thin_scrollbar_style() -> str:
    """A ~7px flat scrollbar, near-invisible until hovered/dragged. Tk can't
    make a single widget transparent, so 'transparent' is emulated by keeping
    the idle thumb barely lighter than the window background."""
    global _SCROLLBAR_STYLE_READY
    style = ttk.Style()
    if not _SCROLLBAR_STYLE_READY:
        style.theme_use("clam")  # the Windows-native theme ignores scrollbar colors
        style.layout("Thin.Vertical.TScrollbar",
                     [("Vertical.Scrollbar.trough",
                       {"children": [("Vertical.Scrollbar.thumb",
                                      {"expand": "1", "sticky": "nswe"})],
                        "sticky": "ns"})])  # no arrow buttons
        style.configure("Thin.Vertical.TScrollbar",
                        troughcolor=BG_COLOR, background=SCROLLBAR_IDLE,
                        bordercolor=BG_COLOR, arrowsize=0, borderwidth=0,
                        relief="flat", width=7)
        style.map("Thin.Vertical.TScrollbar",
                  background=[("active", SCROLLBAR_HOVER), ("pressed", SCROLLBAR_HOVER)])
        _SCROLLBAR_STYLE_READY = True
    return "Thin.Vertical.TScrollbar"


class _ScrollbackText:
    """A read-only, wheel-scrollable Text with a thin drag scrollbar and a dim
    timestamp on every line. Auto-follows only while the view is at the
    bottom, and trims history past HISTORY_LINES."""

    def __init__(self, parent: tk.Misc, font: tuple):
        self.frame = tk.Frame(parent, bg=BG_COLOR)
        self.text = tk.Text(
            self.frame, font=font, fg=FG_COLOR, bg=BG_COLOR, wrap="word",
            state="disabled", relief="flat", padx=12, pady=8, cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical",
                                  style=_thin_scrollbar_style(),
                                  command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.text.pack(side="left", expand=True, fill="both")

        time_size = max(8, font[1] - 6)
        self.text.tag_configure("time", foreground=TIMESTAMP_COLOR,
                                font=("Consolas", time_size))

    def append(self, segments: list[tuple[str, Optional[str]]]):
        """segments: list of (text, extra_tag). Appends one timestamped line."""
        at_bottom = self.text.yview()[1] >= 0.999
        self.text.configure(state="normal")
        self.text.insert("end", time.strftime("%H:%M:%S "), ("time",))
        for content, tag in segments:
            self.text.insert("end", content, (tag,) if tag else ())
        self.text.insert("end", "\n")
        line_count = int(self.text.index("end-1c").split(".")[0])
        if line_count > HISTORY_LINES:
            self.text.delete("1.0", f"{line_count - HISTORY_LINES + 1}.0")
        self.text.configure(state="disabled")
        if at_bottom:
            # see("end") under-scrolls when the new text wraps to multiple display
            # lines, because it runs before the widget re-wraps. Scroll to the
            # absolute bottom after the pending redraw instead.
            self.text.after_idle(lambda: self.text.yview_moveto(1.0))


class CaptionOverlay:
    def __init__(self, caption_queue: "queue.Queue[object]", poll_ms: int = 100,
                 pipeline: Optional[Pipeline] = None,
                 geometry: Optional[str] = None):
        self.caption_queue = caption_queue
        self.poll_ms = poll_ms
        self.pipeline = pipeline
        self.chat_overlay: Optional[ChatOverlay] = None

        self.root = tk.Tk()
        self.root.title("Twitch Live Translator")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", WINDOW_ALPHA)
        self.root.configure(bg=BG_COLOR)
        self.root.geometry(geometry or "900x160+180+760")
        self.root.minsize(CAPTION_MIN_W, CAPTION_MIN_H)

        self.scrollback = _ScrollbackText(self.root, ("Segoe UI", 18, "bold"))
        self.scrollback.frame.pack(expand=True, fill="both")

        # Corner controls float over the text so the full window is text area.
        menu_btn = tk.Label(self.root, text="…", font=("Segoe UI", 13, "bold"),
                            fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=6)
        menu_btn.place(relx=1.0, rely=0.0, anchor="ne")
        menu_btn.bind("<ButtonPress-1>", self._open_menu)
        menu_btn.lift()

        grip = _make_resize_grip(self.root, CAPTION_MIN_W, CAPTION_MIN_H)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.lift()

        _make_draggable(self.root, self.root, self.scrollback.text)
        self.root.bind("<Escape>", lambda _e: self.quit())
        self.root.after(self.poll_ms, self._poll)

    # --- menu ---------------------------------------------------------------

    def _open_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Settings…", command=self._open_settings)
        if self.chat_overlay is not None:
            visible = self.chat_overlay.is_visible()
            menu.add_command(
                label="Hide chat panel" if visible else "Show chat panel",
                command=self.chat_overlay.toggle,
            )
        menu.add_separator()
        menu.add_command(label="Quit", command=self.quit)
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _open_settings(self):
        SettingsDialog(self.root, self)

    def quit(self):
        # Remember where the user put the windows for next launch.
        updates = {"caption_geometry": self.root.geometry()}
        if self.chat_overlay is not None:
            updates["chat_geometry"] = self.chat_overlay.win.geometry()
        try:
            save_settings(updates)
        except OSError:
            pass  # a failed geometry save should never block quitting
        self.root.destroy()

    # --- captions -----------------------------------------------------------

    def _poll(self):
        try:
            while True:
                item = self.caption_queue.get_nowait()
                if item is None:
                    continue
                caption: Caption = item
                self.scrollback.append([(caption.translated_text, None)])
        except queue.Empty:
            pass
        self.root.after(self.poll_ms, self._poll)

    def attach_chat(self, chat_queue: "queue.Queue[ChatLine]",
                    geometry: Optional[str] = None):
        self.chat_overlay = ChatOverlay(self.root, chat_queue, self.poll_ms,
                                        self.pipeline, geometry)

    def run(self):
        self.root.mainloop()


class ChatOverlay:
    """Scrolling panel of translated chat messages, hideable via its ✕ or the
    caption bar's menu. Hiding also pauses chat GPU work in the pipeline."""

    def __init__(self, parent: tk.Tk, chat_queue: "queue.Queue[ChatLine]",
                 poll_ms: int, pipeline: Optional[Pipeline],
                 geometry: Optional[str] = None):
        self.chat_queue = chat_queue
        self.poll_ms = poll_ms
        self.pipeline = pipeline

        self.win = tk.Toplevel(parent)
        self.win.title("Translated Chat")
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", WINDOW_ALPHA)
        self.win.configure(bg=BG_COLOR)
        self.win.geometry(geometry or "380x300+1160+400")
        self.win.minsize(CHAT_MIN_W, CHAT_MIN_H)

        self.scrollback = _ScrollbackText(self.win, ("Segoe UI", 11))
        self.scrollback.text.tag_configure("name", foreground=CHAT_NAME_COLOR,
                                           font=("Segoe UI", 11, "bold"))
        self.scrollback.frame.pack(expand=True, fill="both")

        close_btn = tk.Label(self.win, text="✕", font=("Segoe UI", 11),
                             fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=6)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne")
        close_btn.bind("<ButtonPress-1>", lambda _e: self.toggle())
        close_btn.lift()

        grip = _make_resize_grip(self.win, CHAT_MIN_W, CHAT_MIN_H)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.lift()

        _make_draggable(self.win, self.win, self.scrollback.text)
        self.win.after(self.poll_ms, self._poll)

    def is_visible(self) -> bool:
        return self.win.state() != "withdrawn"

    def toggle(self):
        if self.is_visible():
            self.win.withdraw()
            if self.pipeline is not None:
                self.pipeline.set_chat_enabled(False)
        else:
            self.win.deiconify()
            if self.pipeline is not None:
                self.pipeline.set_chat_enabled(True)

    def _poll(self):
        try:
            while True:
                line: ChatLine = self.chat_queue.get_nowait()
                self.scrollback.append([
                    (f"{line.username}: ", "name"),
                    (line.translated_text, None),
                ])
        except queue.Empty:
            pass
        self.win.after(self.poll_ms, self._poll)


class SettingsDialog:
    """All settings from SETTING_SPECS, editable, saved to settings.json.
    Live-safe settings apply immediately; the rest take effect on restart."""

    def __init__(self, parent: tk.Tk, overlay: CaptionOverlay):
        self.overlay = overlay
        self.current = load_settings()
        self.vars: dict[str, tk.Variable] = {}

        self.win = tk.Toplevel(parent)
        self.win.title("Settings — Twitch Live Translator")
        self.win.attributes("-topmost", True)
        self.win.resizable(False, False)

        frame = ttk.Frame(self.win, padding=12)
        frame.grid(sticky="nsew")

        twitch_tabs = list_twitch_channels()  # for the chat-channel dropdown

        for row, spec in enumerate(SETTING_SPECS):
            ttk.Label(frame, text=spec.label).grid(row=row, column=0, sticky="w", pady=2)
            value = self.current.get(spec.key)
            if spec.kind == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(value))
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky="w", padx=8)
            elif spec.kind == "choice":
                var = tk.StringVar(value="" if value is None else str(value))
                ttk.Combobox(frame, textvariable=var, values=spec.choices,
                             state="readonly", width=34).grid(row=row, column=1, sticky="w", padx=8)
            elif spec.key == "chat_channel":
                # Editable dropdown pre-filled with Twitch tabs currently open
                # in Firefox — pick one or type any channel name manually.
                var = tk.StringVar(value="" if value is None else str(value))
                ttk.Combobox(frame, textvariable=var, values=twitch_tabs,
                             width=34).grid(row=row, column=1, sticky="w", padx=8)
            else:
                var = tk.StringVar(value="" if value is None else str(value))
                ttk.Entry(frame, textvariable=var, width=36).grid(row=row, column=1, sticky="w", padx=8)
            self.vars[spec.key] = var
            if spec.help:
                ttk.Label(frame, text=spec.help, foreground="#777777",
                          font=("Segoe UI", 8)).grid(row=row, column=2, sticky="w", padx=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(SETTING_SPECS), column=0, columnspan=3, pady=(12, 0), sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.win.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="Save", command=self._save).pack(side="right")

    def _save(self):
        updates: dict = {}
        for spec in SETTING_SPECS:
            raw = self.vars[spec.key].get()
            if spec.kind == "bool":
                value: object = bool(raw)
            elif spec.kind in ("int", "float"):
                try:
                    value = int(raw) if spec.kind == "int" else float(raw)
                except ValueError:
                    messagebox.showerror(
                        "Invalid value",
                        f"{spec.label} must be a number (got {raw!r}).",
                        parent=self.win,
                    )
                    return
            else:
                value = str(raw).strip() or None
            if value != self.current.get(spec.key):
                updates[spec.key] = value

        if updates:
            save_settings(updates)
            self._apply_live(updates)
            restart_keys = [s.label for s in SETTING_SPECS if s.key in updates and not s.live]
            if restart_keys:
                messagebox.showinfo(
                    "Saved",
                    "Saved. These changes take effect next launch:\n- "
                    + "\n- ".join(restart_keys),
                    parent=self.win,
                )
        self.win.destroy()

    def _apply_live(self, updates: dict):
        if "enable_chat" in updates and self.overlay.chat_overlay is not None:
            if updates["enable_chat"] != self.overlay.chat_overlay.is_visible():
                self.overlay.chat_overlay.toggle()
