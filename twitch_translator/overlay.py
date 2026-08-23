"""Always-on-top, draggable, resizable overlay windows.

Both windows are frameless (overrideredirect), so they provide their own
chrome: drag anywhere to move, drag the ⇲ grip to resize, "…" opens the menu
on the caption bar, ✕ hides the chat panel. Both keep a scrollback history —
scroll up to read, scroll back to the bottom to resume auto-following.
"""
from __future__ import annotations

import ctypes
import queue
import re
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from .firefox_tabs import list_twitch_channels
from .pipeline import Caption, ChatLine, ContextReady, Pipeline
from .settings import SETTING_SPECS, load_settings, save_settings

BG_COLOR = "#101010"
FG_COLOR = "#ffffff"
CHAT_NAME_COLOR = "#7fb8ff"
CONTROL_COLOR = "#9a9a9a"
TIMESTAMP_COLOR = "#6f6f6f"
SCROLLBAR_IDLE = "#242424"    # barely visible against the background...
SCROLLBAR_HOVER = "#7a7a7a"   # ...until hovered or dragged
LISTENING_IDLE_COLOR = "#3a3a3a"   # dim: no speech detected right now
LISTENING_ACTIVE_COLOR = "#5fd68a"  # soft green: VAD currently sees speech
TOOLTIP_BG = "#2a2a2a"
CONTEXT_FLAG_COLOR = "#e8b339"  # marks a line the context helper flagged as shaky
ACTIVE_COLOR = "#7fb8ff"   # same accent as chat usernames: "on/docked" state for toggle buttons
WINDOW_ALPHA = 0.82
HISTORY_LINES = 300
MIN_FONT_SIZE, MAX_FONT_SIZE = 8, 48

CAPTION_MIN_W, CAPTION_MIN_H = 300, 80
CHAT_MIN_W, CHAT_MIN_H = 200, 120


_GEOMETRY_RE = re.compile(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)")

SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79


def _validate_geometry(geometry: Optional[str]) -> Optional[str]:
    """Reject a saved geometry whose position isn't on any currently attached
    monitor (e.g. a monitor got disconnected/rearranged since last save) — a
    window placed there is invisible with no way back to it, so fall back to
    the default instead. Uses the full virtual screen (spans all monitors),
    since Tkinter's own screen-size queries only see the primary one."""
    if not geometry:
        return None
    m = _GEOMETRY_RE.match(geometry)
    if not m:
        return None
    x, y = int(m.group(3)), int(m.group(4))
    user32 = ctypes.windll.user32
    vx, vy = user32.GetSystemMetrics(SM_XVIRTUALSCREEN), user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw, vh = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    if vx <= x < vx + vw and vy <= y < vy + vh:
        return geometry
    return None


def _make_draggable(win: tk.Misc, *widgets: tk.Misc,
                    companion: Optional[Callable[[], Optional[tk.Misc]]] = None):
    """companion, if given, is called on every drag step; when it returns a
    window (rather than None), that window is moved by the same delta as win
    — used so the docked caption/chat windows drag as one unit."""
    state = {"dx": 0, "dy": 0}

    def start(event):
        state["dx"] = event.x_root - win.winfo_x()
        state["dy"] = event.y_root - win.winfo_y()
        return "break"

    def move(event):
        new_x, new_y = event.x_root - state["dx"], event.y_root - state["dy"]
        other = companion() if companion else None
        if other is not None:
            dx, dy = new_x - win.winfo_x(), new_y - win.winfo_y()
            other.geometry(f"+{other.winfo_x() + dx}+{other.winfo_y() + dy}")
        win.geometry(f"+{new_x}+{new_y}")
        return "break"

    for w in widgets:
        w.bind("<ButtonPress-1>", start)
        w.bind("<B1-Motion>", move)


def _make_resize_grip(win: tk.Misc, min_w: int, min_h: int,
                      on_resize: Optional[Callable[[int, int], None]] = None) -> tk.Label:
    """on_resize, if given, fires with the new (w, h) on every resize step —
    used to keep the docked caption/chat windows' heights in sync with
    each other as either is resized."""
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
        if on_resize:
            on_resize(w, h)
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


_DARK_COMBOBOX_STYLE_READY = False


def _dark_combobox_style() -> str:
    """Dark-themed dropdown matching the rest of the UI — ttk widgets ignore
    plain tk bg/fg options, so this needs its own style (same approach as
    the scrollbar above; both rely on the 'clam' theme already activated
    there, since it's the one that actually honors these overrides)."""
    global _DARK_COMBOBOX_STYLE_READY
    style = ttk.Style()
    if not _DARK_COMBOBOX_STYLE_READY:
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=BG_COLOR, background=BG_COLOR,
                        foreground=FG_COLOR, arrowcolor=CONTROL_COLOR,
                        bordercolor=CONTROL_COLOR, lightcolor=BG_COLOR,
                        darkcolor=BG_COLOR, borderwidth=1, relief="flat")
        style.map("Dark.TCombobox",
                  fieldbackground=[("readonly", BG_COLOR)],
                  foreground=[("readonly", FG_COLOR)],
                  bordercolor=[("focus", ACTIVE_COLOR)])
        # The dropdown listbox itself isn't a ttk widget — it's styled via
        # option database entries, the only way Tk exposes it.
        style.master.option_add("*TCombobox*Listbox.background", BG_COLOR)
        style.master.option_add("*TCombobox*Listbox.foreground", FG_COLOR)
        style.master.option_add("*TCombobox*Listbox.selectBackground", ACTIVE_COLOR)
        style.master.option_add("*TCombobox*Listbox.selectForeground", BG_COLOR)
        _DARK_COMBOBOX_STYLE_READY = True
    return "Dark.TCombobox"


class _ScrollbackText:
    """A read-only, wheel-scrollable Text with a thin drag scrollbar and a dim
    timestamp on every line. Auto-follows only while the view is at the
    bottom, and trims history past HISTORY_LINES. Ctrl+wheel resizes the font
    (persisted via on_font_size_change); hovering a line with an original-text
    counterpart shows it in a small tooltip."""

    def __init__(self, parent: tk.Misc, font: tuple,
                 on_font_size_change: Optional[Callable[[int], None]] = None):
        self.font_family = font[0]
        self.font_size = font[1]
        self._font_extra = font[2:]
        self._time_base_size = font[1]
        self.on_font_size_change = on_font_size_change

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

        self.text.tag_configure("time", foreground=TIMESTAMP_COLOR,
                                font=("Consolas", max(MIN_FONT_SIZE, font[1] - 6)))
        self.text.tag_configure("contextflag", foreground=CONTEXT_FLAG_COLOR)

        self._originals: dict[str, str] = {}
        self._context_notes: dict[str, str] = {}
        self._pending_context: set[str] = set()
        self._line_tags: dict[int, str] = {}
        self._next_tag_id = 0
        self._tooltip: Optional[tk.Toplevel] = None
        self._hovered_tag: Optional[str] = None
        self.text.bind("<Motion>", self._on_motion)
        self.text.bind("<Leave>", self._on_leave)
        self.text.bind("<Control-MouseWheel>", self._on_ctrl_wheel)

    def append(self, segments: list[tuple[str, Optional[str]]], original: Optional[str] = None,
              line_id: Optional[int] = None, flagged: bool = False):
        """segments: list of (text, extra_tag). Appends one timestamped line.
        original, if given, becomes a hover tooltip over the whole line.
        flagged marks the line with a visual indicator (a colored dot after
        the text, so the timestamp's position stays fixed either way) and,
        together with line_id, lets a later set_context() call attach a
        context-helper note to this same line's tooltip."""
        line_tag = None
        if original or flagged:
            line_tag = f"orig{self._next_tag_id}"
            self._next_tag_id += 1
            self._originals[line_tag] = original or ""
            if len(self._originals) > HISTORY_LINES:
                oldest = next(iter(self._originals))
                del self._originals[oldest]
                self._context_notes.pop(oldest, None)
                self._pending_context.discard(oldest)

        at_bottom = self.text.yview()[1] >= 0.999
        self.text.configure(state="normal")
        extra = (line_tag,) if line_tag else ()
        if flagged and line_tag is not None:
            self._pending_context.add(line_tag)
            if line_id is not None:
                self._line_tags[line_id] = line_tag
        self.text.insert("end", time.strftime("%H:%M:%S "), ("time",) + extra)
        for content, tag in segments:
            tags = ((tag,) if tag else ()) + extra
            self.text.insert("end", content, tags)
        if flagged and line_tag is not None:
            self.text.insert("end", " ●", ("contextflag",) + extra)
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

    def set_context(self, line_id: int, context: str):
        """Attaches a context-helper note to a previously-appended flagged
        line, found via the line_id passed to append(). Silently a no-op if
        that line has since scrolled out of history."""
        tag = self._line_tags.pop(line_id, None)
        if tag is None:
            return
        self._context_notes[tag] = context
        self._pending_context.discard(tag)

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._originals.clear()
        self._context_notes.clear()
        self._pending_context.clear()
        self._line_tags.clear()
        self._hide_tooltip()

    # --- font size ------------------------------------------------------

    def set_font_size(self, size: int) -> int:
        size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, size))
        self.font_size = size
        self.text.configure(font=(self.font_family, size, *self._font_extra))
        self.text.tag_configure("time", font=("Consolas", max(MIN_FONT_SIZE, size - 6)))
        return size

    def _on_ctrl_wheel(self, event):
        step = 1 if event.delta > 0 else -1
        new_size = self.set_font_size(self.font_size + step)
        if self.on_font_size_change:
            self.on_font_size_change(new_size)
        return "break"

    # --- hover-to-see-original -------------------------------------------

    def _on_motion(self, event):
        try:
            index = self.text.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        tags = self.text.tag_names(index)
        hover_tag = next((t for t in tags if t.startswith("orig")), None)
        if hover_tag == self._hovered_tag:
            if self._tooltip:
                self._move_tooltip(event)
            return
        self._hovered_tag = hover_tag
        self._hide_tooltip()
        if hover_tag and hover_tag in self._originals:
            self._show_tooltip(event, self._tooltip_content(hover_tag))

    def _tooltip_content(self, tag: str) -> str:
        content = self._originals.get(tag, "")
        if tag in self._context_notes:
            content += f"\n\n💡 {self._context_notes[tag]}"
        elif tag in self._pending_context:
            content += "\n\n(analyzing this translation…)"
        return content

    def _on_leave(self, _event=None):
        self._hovered_tag = None
        self._hide_tooltip()

    def _show_tooltip(self, event, content: str):
        tip = tk.Toplevel(self.text)
        tip.overrideredirect(True)
        tip.attributes("-topmost", True)
        tk.Label(tip, text=content, bg=TOOLTIP_BG, fg=FG_COLOR,
                font=("Segoe UI", 9), padx=8, pady=4, wraplength=360,
                justify="left").pack()
        self._tooltip = tip
        self._move_tooltip(event)

    def _move_tooltip(self, event):
        if self._tooltip is None:
            return
        x = self.text.winfo_rootx() + event.x + 16
        y = self.text.winfo_rooty() + event.y + 16
        self._tooltip.geometry(f"+{x}+{y}")

    def _hide_tooltip(self):
        if self._tooltip is not None:
            self._tooltip.destroy()
            self._tooltip = None


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
        self.root.geometry(_validate_geometry(geometry) or "900x160+180+760")
        self.root.minsize(CAPTION_MIN_W, CAPTION_MIN_H)

        font_size = load_settings().get("caption_font_size", 18)
        self.scrollback = _ScrollbackText(
            self.root, ("Segoe UI", font_size, "bold"),
            on_font_size_change=lambda size: save_settings({"caption_font_size": size}),
        )
        self.scrollback.frame.pack(expand=True, fill="both")

        # Corner controls float over the text so the full window is text area.
        # Positioned right-to-left using each widget's own measured width
        # rather than fixed pixel offsets — glyphs (especially color emoji
        # like 💬) can render wider than assumed and silently overlap a
        # fixed-offset neighbor.
        menu_btn = tk.Label(self.root, text="…", font=("Segoe UI", 13, "bold"),
                            fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=6)
        menu_btn.bind("<ButtonPress-1>", self._open_menu)

        self.chat_btn = tk.Label(self.root, text="💬", font=("Segoe UI", 12),
                                 fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=4)
        self.chat_btn.bind("<ButtonPress-1>", self._toggle_chat_button)

        self.root.update_idletasks()
        x = 0
        for widget in (menu_btn, self.chat_btn):
            widget.place(relx=1.0, rely=0.0, anchor="ne", x=-x)
            widget.lift()
            x += widget.winfo_reqwidth() + 4

        # Subtle "listening" indicator: lights up while VAD currently sees
        # speech, so there's feedback in the gap between talking and a
        # caption actually landing (which can be a couple seconds).
        self.listening_dot = tk.Label(self.root, text="●", font=("Segoe UI", 9),
                                      fg=LISTENING_IDLE_COLOR, bg=BG_COLOR, padx=6)
        self.listening_dot.place(relx=0.0, rely=0.0, anchor="nw")
        self.listening_dot.lift()

        grip = _make_resize_grip(self.root, CAPTION_MIN_W, CAPTION_MIN_H,
                                 on_resize=self._sync_chat_height)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.lift()

        _make_draggable(self.root, self.root, self.scrollback.text,
                        companion=self._chat_companion)
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

    def _toggle_chat_button(self, _event=None):
        if self.chat_overlay is not None:
            self.chat_overlay.toggle()
        else:
            # Only reachable with --no-chat: chat panel now starts (idling on
            # no channel) whenever chat isn't explicitly disabled, so there's
            # nothing to toggle here at all — no in-app fix to point at.
            messagebox.showinfo(
                "Chat panel",
                "Chat was disabled with --no-chat for this session.\n\n"
                "Remove that flag (or the equivalent setting) and restart to enable it.",
                parent=self.root,
            )

    def _dock_chat(self):
        """Docks the chat window flush against this one's right edge (or left,
        if there isn't room on the right), matching its height. Caption and
        chat are always connected this way — there's no separate/undocked
        mode — so this runs once at startup, and _sync_chat_height keeps the
        heights matched afterward as either is resized."""
        chat = self.chat_overlay
        if chat is None:
            return
        self.root.update_idletasks()
        cx, cy = self.root.winfo_x(), self.root.winfo_y()
        cw, ch = self.root.winfo_width(), self.root.winfo_height()
        chat_w = max(CHAT_MIN_W, chat.win.winfo_width())

        user32 = ctypes.windll.user32
        vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        new_x = cx + cw if cx + cw + chat_w <= vx + vw else max(vx, cx - chat_w)

        chat.win.geometry(f"{chat_w}x{ch}+{new_x}+{cy}")

    def _sync_chat_height(self, _w: int, h: int):
        """on_resize callback for this window's grip: keeps the docked chat
        window's height matched, without touching its width/position."""
        if self.chat_overlay is not None:
            chat_w = self.chat_overlay.win.winfo_width()
            self.chat_overlay.win.geometry(f"{chat_w}x{h}")

    def _chat_companion(self) -> Optional[tk.Misc]:
        """companion callable for _make_draggable: dragging the caption
        window drags the docked chat window along with it."""
        if self.chat_overlay is not None and self.chat_overlay.is_visible():
            return self.chat_overlay.win
        return None

    def _open_settings(self):
        SettingsPanel(self.root, self)

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
                if isinstance(item, ContextReady):
                    self.scrollback.set_context(item.line_id, item.context)
                    continue
                caption: Caption = item
                self.scrollback.append([(caption.translated_text, None)],
                                       original=caption.original_text,
                                       line_id=caption.line_id, flagged=caption.needs_context)
        except queue.Empty:
            pass
        if self.pipeline is not None:
            speaking = self.pipeline.vad.is_speaking
            self.listening_dot.configure(
                fg=LISTENING_ACTIVE_COLOR if speaking else LISTENING_IDLE_COLOR)
        if self.chat_overlay is not None:
            # Polled rather than hooked into every place chat visibility can
            # change (the 💬 button, the … menu, Settings) — simpler than
            # threading a callback through all of them, and self-correcting
            # if anything else toggles it.
            visible = self.chat_overlay.is_visible()
            self.chat_btn.configure(fg=ACTIVE_COLOR if visible else CONTROL_COLOR)
        self.root.after(self.poll_ms, self._poll)

    def attach_chat(self, chat_queue: "queue.Queue[object]",
                    geometry: Optional[str] = None):
        self.chat_overlay = ChatOverlay(self.root, chat_queue, self.poll_ms,
                                        self.pipeline, geometry, caption_overlay=self)
        self._dock_chat()

    def run(self):
        self.root.mainloop()


class ChatOverlay:
    """Scrolling panel of translated chat messages, hideable via its ✕ or the
    caption bar's menu. Hiding also pauses chat GPU work in the pipeline."""

    def __init__(self, parent: tk.Tk, chat_queue: "queue.Queue[object]",
                 poll_ms: int, pipeline: Optional[Pipeline],
                 geometry: Optional[str] = None,
                 caption_overlay: Optional["CaptionOverlay"] = None):
        self.chat_queue = chat_queue
        self.poll_ms = poll_ms
        self.pipeline = pipeline
        self.caption_overlay = caption_overlay

        self.win = tk.Toplevel(parent)
        self.win.title("Translated Chat")
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", WINDOW_ALPHA)
        self.win.configure(bg=BG_COLOR)
        self.win.geometry(_validate_geometry(geometry) or "380x300+1160+400")
        self.win.minsize(CHAT_MIN_W, CHAT_MIN_H)

        top_bar = tk.Frame(self.win, bg=BG_COLOR)
        top_bar.pack(side="top", fill="x")

        initial_channel = pipeline.chat_channel if pipeline is not None else None
        self.channel_var = tk.StringVar(value=initial_channel or "")
        self.channel_combo = ttk.Combobox(
            top_bar, textvariable=self.channel_var,
            values=list_twitch_channels(), width=18, font=("Segoe UI", 9),
        )
        self.channel_combo.pack(side="left", padx=(6, 0), pady=3)
        self.channel_combo.bind("<Return>", self._on_channel_change)
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_channel_change)

        refresh_btn = tk.Label(top_bar, text="⟳", font=("Segoe UI", 10),
                               fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=4)
        refresh_btn.pack(side="left")
        refresh_btn.bind("<ButtonPress-1>", self._refresh_channels)

        close_btn = tk.Label(top_bar, text="✕", font=("Segoe UI", 11),
                             fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=6)
        close_btn.pack(side="right")
        close_btn.bind("<ButtonPress-1>", lambda _e: self.toggle())

        chat_font_size = load_settings().get("chat_font_size", 11)
        self.scrollback = _ScrollbackText(
            self.win, ("Segoe UI", chat_font_size),
            on_font_size_change=self._on_font_size_change,
        )
        self.scrollback.text.tag_configure("name", foreground=CHAT_NAME_COLOR,
                                           font=("Segoe UI", chat_font_size, "bold"))
        self.scrollback.frame.pack(expand=True, fill="both")

        grip = _make_resize_grip(self.win, CHAT_MIN_W, CHAT_MIN_H,
                                 on_resize=self._sync_caption_height)
        grip.place(relx=1.0, rely=1.0, anchor="se")
        grip.lift()

        _make_draggable(self.win, self.win, top_bar, self.scrollback.text,
                        companion=self._caption_companion)
        self.win.after(self.poll_ms, self._poll)

    def _sync_caption_height(self, _w: int, h: int):
        """on_resize callback for this window's grip: keeps the docked
        caption window's height matched, without touching its width/position."""
        if self.caption_overlay is not None:
            cap_w = self.caption_overlay.root.winfo_width()
            self.caption_overlay.root.geometry(f"{cap_w}x{h}")

    def _caption_companion(self) -> Optional[tk.Misc]:
        """companion callable for _make_draggable: dragging the chat window
        drags the docked caption window along with it."""
        if self.caption_overlay is not None:
            return self.caption_overlay.root
        return None

    def _on_font_size_change(self, size: int):
        # "name" (the bold username prefix) isn't the base font, so it needs
        # its own resize — set_font_size() only touches the base text + timestamp.
        self.scrollback.text.tag_configure("name", font=("Segoe UI", size, "bold"))
        save_settings({"chat_font_size": size})

    def _on_channel_change(self, _event=None):
        channel = self.channel_var.get().strip().lstrip("#")
        if not channel or self.pipeline is None:
            return
        self.pipeline.set_chat_channel(channel)
        self.scrollback.clear()
        # Switching now usually reuses the live connection (IRC PART/JOIN,
        # see Pipeline.set_chat_channel) rather than reconnecting from
        # scratch, so messages typically resume in well under a second — but
        # a blank panel with no feedback still reads as broken in that gap.
        self.scrollback.append([(f"— now watching #{channel} —", "time")])
        try:
            save_settings({"chat_channel": channel})
        except OSError:
            pass
        self.win.focus_set()  # drop focus out of the combobox now that Enter was handled

    def _refresh_channels(self, _event=None):
        self.channel_combo["values"] = list_twitch_channels()

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
                item = self.chat_queue.get_nowait()
                if isinstance(item, ContextReady):
                    self.scrollback.set_context(item.line_id, item.context)
                    continue
                line: ChatLine = item
                self.scrollback.append([
                    (f"{line.username}: ", "name"),
                    (line.translated_text, None),
                ], original=line.original_text,
                   line_id=line.line_id, flagged=line.needs_context)
        except queue.Empty:
            pass
        self.win.after(self.poll_ms, self._poll)


class SettingsPanel:
    """Dark-themed settings panel docked against the caption window — same
    visual language as the rest of the app (CaptionOverlay/ChatOverlay), not
    a generic OS-styled dialog. Shows only the most commonly-changed
    settings by default; "Advanced" reveals the rest. Docks directly below
    the caption window (not left/right) so it doesn't collide with the chat
    window, which is always docked there via CaptionOverlay._dock_chat()."""

    WIDTH = 440

    def __init__(self, parent: tk.Tk, overlay: "CaptionOverlay"):
        self.overlay = overlay
        self.current = load_settings()
        self.vars: dict[str, tk.Variable] = {}
        self.show_advanced = False

        self.win = tk.Toplevel(parent)
        self.win.title("Settings")
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", WINDOW_ALPHA)
        self.win.configure(bg=BG_COLOR)

        header = tk.Frame(self.win, bg=BG_COLOR)
        header.pack(side="top", fill="x")
        tk.Label(header, text="Settings", font=("Segoe UI", 11, "bold"),
                fg=FG_COLOR, bg=BG_COLOR, padx=10, pady=8).pack(side="left")

        close_btn = tk.Label(header, text="✕", font=("Segoe UI", 11),
                             fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=8)
        close_btn.pack(side="right")
        close_btn.bind("<ButtonPress-1>", lambda _e: self.win.destroy())

        self.advanced_btn = tk.Label(header, text="Advanced", font=("Segoe UI", 9),
                                     fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=8)
        self.advanced_btn.pack(side="right")
        self.advanced_btn.bind("<ButtonPress-1>", self._toggle_advanced)

        _make_draggable(self.win, header)

        self.rows_frame = tk.Frame(self.win, bg=BG_COLOR)
        self.rows_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(2, 8))

        footer = tk.Frame(self.win, bg=BG_COLOR)
        footer.pack(side="bottom", fill="x", padx=12, pady=(0, 10))
        save_btn = tk.Label(footer, text="Save", font=("Segoe UI", 9, "bold"),
                            fg=ACTIVE_COLOR, bg=BG_COLOR, cursor="hand2", padx=10, pady=4)
        save_btn.pack(side="right")
        save_btn.bind("<ButtonPress-1>", lambda _e: self._save())
        cancel_btn = tk.Label(footer, text="Cancel", font=("Segoe UI", 9),
                              fg=CONTROL_COLOR, bg=BG_COLOR, cursor="hand2", padx=10, pady=4)
        cancel_btn.pack(side="right")
        cancel_btn.bind("<ButtonPress-1>", lambda _e: self.win.destroy())

        self._build_rows()
        self._position()

    # --- layout ---------------------------------------------------------

    def _visible_specs(self):
        return [s for s in SETTING_SPECS if self.show_advanced or not s.advanced]

    def _build_rows(self):
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self.vars.clear()

        combo_style = _dark_combobox_style()
        for row, spec in enumerate(self._visible_specs()):
            tk.Label(self.rows_frame, text=spec.label, font=("Segoe UI", 9),
                    fg=FG_COLOR, bg=BG_COLOR, anchor="w").grid(
                row=row, column=0, sticky="w", pady=3)
            value = self.current.get(spec.key)

            if spec.kind == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(value))
                tk.Checkbutton(self.rows_frame, variable=var, bg=BG_COLOR,
                              activebackground=BG_COLOR, selectcolor=BG_COLOR,
                              highlightthickness=0, bd=0).grid(
                    row=row, column=1, sticky="w", padx=8)
            elif spec.kind == "choice":
                var = tk.StringVar(value="" if value is None else str(value))
                ttk.Combobox(self.rows_frame, textvariable=var, values=spec.choices,
                            state="readonly", style=combo_style, width=28).grid(
                    row=row, column=1, sticky="w", padx=8)
            else:
                var = tk.StringVar(value="" if value is None else str(value))
                tk.Entry(self.rows_frame, textvariable=var, width=30,
                        bg=BG_COLOR, fg=FG_COLOR, insertbackground=FG_COLOR,
                        relief="flat", highlightthickness=1,
                        highlightbackground=CONTROL_COLOR,
                        highlightcolor=ACTIVE_COLOR).grid(
                    row=row, column=1, sticky="w", padx=8)
            self.vars[spec.key] = var

            if spec.help:
                tk.Label(self.rows_frame, text=spec.help, font=("Segoe UI", 8),
                        fg=TIMESTAMP_COLOR, bg=BG_COLOR, anchor="w",
                        wraplength=140, justify="left").grid(
                    row=row, column=2, sticky="w", padx=4)

        self.advanced_btn.configure(
            fg=ACTIVE_COLOR if self.show_advanced else CONTROL_COLOR)

    def _toggle_advanced(self, _event=None):
        self.show_advanced = not self.show_advanced
        self._build_rows()
        self.win.update_idletasks()
        self.win.geometry(f"{self.WIDTH}x{self.win.winfo_reqheight()}")

    def _position(self):
        """Docks flush below the caption window; above it instead if there
        isn't room below (same virtual-screen-bounds check _dock_chat() uses)."""
        parent = self.overlay.root
        parent.update_idletasks()
        self.win.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        panel_h = self.win.winfo_reqheight()

        user32 = ctypes.windll.user32
        vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
        y = py + ph if py + ph + panel_h <= vy + vh else max(vy, py - panel_h)
        self.win.geometry(f"{self.WIDTH}x{panel_h}+{px}+{y}")

    # --- save -------------------------------------------------------------

    def _save(self):
        updates: dict = {}
        for spec in SETTING_SPECS:
            if spec.key not in self.vars:
                continue  # hidden behind Advanced and never touched
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
        chat_overlay = self.overlay.chat_overlay
        if "enable_chat" in updates and chat_overlay is not None:
            if updates["enable_chat"] != chat_overlay.is_visible():
                chat_overlay.toggle()
        if "chat_channel" in updates and chat_overlay is not None:
            new_channel = updates["chat_channel"]
            chat_overlay.channel_var.set(new_channel or "")
            chat_overlay._on_channel_change()
        if "caption_font_size" in updates:
            self.overlay.scrollback.set_font_size(updates["caption_font_size"])
        if "chat_font_size" in updates and chat_overlay is not None:
            size = chat_overlay.scrollback.set_font_size(updates["chat_font_size"])
            chat_overlay.scrollback.text.tag_configure("name", font=("Segoe UI", size, "bold"))
