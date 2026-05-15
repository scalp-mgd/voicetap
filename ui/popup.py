"""Near-cursor recording popup.

Tkinter borderless always-on-top window. Tracks three states:
  - RECORDING:    red blinking dot + "REC m:ss"
  - TRANSCRIBING: amber dot + "Processing..."
  - DONE / ERROR: brief green / red flash, then hide

Tkinter is not thread-safe; all state changes must be scheduled via
`self.root.after(...)`. The public methods do that automatically so callers
can invoke them from any thread.
"""

from __future__ import annotations

import logging
import sys
import time
import tkinter as tk
from typing import Optional

logger = logging.getLogger(__name__)


# Tk's winfo_screenwidth/height returns only the primary monitor on Windows.
# Clamping popup coords against those bounds drags the popup back onto monitor 1
# whenever the cursor lives on a different display (especially monitors at
# negative virtual coords). Ask Win32 which monitor the cursor is on instead.

_WIN_MONITOR_BOUNDS_FOR_POINT = None

if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        class _POINT(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        class _RECT(ctypes.Structure):
            _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                        ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", _RECT),
                        ("rcWork", _RECT), ("dwFlags", wintypes.DWORD)]

        _u32 = ctypes.windll.user32
        _u32.MonitorFromPoint.argtypes = [_POINT, wintypes.DWORD]
        _u32.MonitorFromPoint.restype = ctypes.c_void_p
        _u32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFO)]
        _u32.GetMonitorInfoW.restype = wintypes.BOOL

        _MONITOR_DEFAULTTONEAREST = 2

        def _win_monitor_bounds_for_point(x, y):
            hmon = _u32.MonitorFromPoint(_POINT(x, y), _MONITOR_DEFAULTTONEAREST)
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if _u32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                return (r.left, r.top, r.right, r.bottom)
            return None

        _WIN_MONITOR_BOUNDS_FOR_POINT = _win_monitor_bounds_for_point
    except Exception as e:
        logger.warning("Win32 monitor helpers unavailable: %s", e)


class NearCursorPopup:
    STATE_HIDDEN = "hidden"
    STATE_RECORDING = "recording"
    STATE_TRANSCRIBING = "transcribing"
    STATE_DONE = "done"
    STATE_ERROR = "error"

    def __init__(
        self,
        root: tk.Tk,
        width: int = 200,
        height: int = 60,
        offset_x: int = 20,
        offset_y: int = 20,
        bg_color: str = "#1a1a1a",
        text_color: str = "#ffffff",
        recording_color: str = "#ff3030",
        transcribing_color: str = "#ffaa00",
        done_color: str = "#30ff60",
    ):
        self.root = root
        self.width = width
        self.height = height
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.bg_color = bg_color
        self.text_color = text_color
        self.recording_color = recording_color
        self.transcribing_color = transcribing_color
        self.done_color = done_color

        self.state = self.STATE_HIDDEN
        self._recording_start: float = 0.0
        self._blink_on: bool = True
        self._blink_job: Optional[str] = None
        self._timer_job: Optional[str] = None

        # Build the toplevel
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        try:
            self.win.attributes("-alpha", 0.92)
        except tk.TclError:
            pass
        self.win.configure(bg=self.bg_color)
        self.win.geometry(f"{width}x{height}+0+0")
        self.win.withdraw()  # start hidden

        # Status dot (canvas circle)
        self.canvas = tk.Canvas(
            self.win, width=20, height=20, bg=self.bg_color,
            highlightthickness=0,
        )
        self.canvas.place(x=12, y=(height - 20) // 2)
        self._dot = self.canvas.create_oval(2, 2, 18, 18, fill=self.recording_color, outline="")

        # Status text
        self.label_status = tk.Label(
            self.win, text="REC",
            bg=self.bg_color, fg=self.text_color,
            font=("Segoe UI", 11, "bold"),
        )
        self.label_status.place(x=40, y=10)

        # Timer text
        self.label_timer = tk.Label(
            self.win, text="0:00",
            bg=self.bg_color, fg="#888888",
            font=("Segoe UI", 9),
        )
        self.label_timer.place(x=40, y=32)

    # ------------------------------------------------------------------ public

    def show_recording(self, cursor_x: int, cursor_y: int):
        self.root.after(0, lambda: self._show_recording(cursor_x, cursor_y))

    def show_transcribing(self):
        self.root.after(0, self._show_transcribing)

    def show_done(self):
        self.root.after(0, self._show_done)

    def show_error(self, message: str = "Error"):
        self.root.after(0, lambda: self._show_error(message))

    def hide(self):
        self.root.after(0, self._hide)

    # ------------------------------------------------------------------ internal

    def _position_near(self, x: int, y: int):
        # Bound to the monitor the cursor is on (multi-monitor support).
        bounds = _WIN_MONITOR_BOUNDS_FOR_POINT(x, y) if _WIN_MONITOR_BOUNDS_FOR_POINT else None
        if bounds is None:
            left, top = 0, 0
            right = self.win.winfo_screenwidth()
            bottom = self.win.winfo_screenheight()
        else:
            left, top, right, bottom = bounds

        px = x + self.offset_x
        py = y + self.offset_y
        # Flip to the opposite side of the cursor if popup would extend past the edge
        if px + self.width > right:
            px = x - self.width - self.offset_x
        if py + self.height > bottom:
            py = y - self.height - self.offset_y
        # Clamp to monitor's top-left
        if px < left:
            px = left
        if py < top:
            py = top
        self.win.geometry(f"{self.width}x{self.height}+{px}+{py}")

    def _cancel_jobs(self):
        if self._blink_job:
            self.root.after_cancel(self._blink_job)
            self._blink_job = None
        if self._timer_job:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None

    def _show_recording(self, x: int, y: int):
        self._cancel_jobs()
        self.state = self.STATE_RECORDING
        self._recording_start = time.time()
        self._position_near(x, y)
        self.canvas.itemconfig(self._dot, fill=self.recording_color)
        self.label_status.configure(text="REC", fg=self.text_color)
        self.label_timer.configure(text="0:00")
        self.win.deiconify()
        self._blink()
        self._tick_timer()

    def _show_transcribing(self):
        self._cancel_jobs()
        self.state = self.STATE_TRANSCRIBING
        self.canvas.itemconfig(self._dot, fill=self.transcribing_color)
        self.label_status.configure(text="Processing...", fg=self.text_color)
        self.label_timer.configure(text="")

    def _show_done(self):
        self._cancel_jobs()
        self.state = self.STATE_DONE
        self.canvas.itemconfig(self._dot, fill=self.done_color)
        self.label_status.configure(text="Pasted", fg=self.text_color)
        self.label_timer.configure(text="")
        # Hide after a short flash
        self.root.after(600, self._hide)

    def _show_error(self, message: str):
        self._cancel_jobs()
        self.state = self.STATE_ERROR
        self.canvas.itemconfig(self._dot, fill="#ff3030")
        self.label_status.configure(text=message[:24], fg="#ff8888")
        self.label_timer.configure(text="")
        self.root.after(1500, self._hide)

    def _hide(self):
        self._cancel_jobs()
        self.state = self.STATE_HIDDEN
        try:
            self.win.withdraw()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ animations

    def _blink(self):
        if self.state != self.STATE_RECORDING:
            return
        self._blink_on = not self._blink_on
        color = self.recording_color if self._blink_on else self.bg_color
        self.canvas.itemconfig(self._dot, fill=color)
        self._blink_job = self.root.after(450, self._blink)

    def _tick_timer(self):
        if self.state != self.STATE_RECORDING:
            return
        elapsed = time.time() - self._recording_start
        m = int(elapsed) // 60
        s = int(elapsed) % 60
        self.label_timer.configure(text=f"{m}:{s:02d}")
        self._timer_job = self.root.after(200, self._tick_timer)
