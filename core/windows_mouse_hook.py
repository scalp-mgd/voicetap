"""Low-level mouse hook for Windows (WH_MOUSE_LL).

The plain pynput Listener can only OBSERVE mouse events — events still
propagate to the active window. So pressing XButton1 to start dictation in
a browser also triggers Back navigation, which is annoying.

This module installs a Windows low-level mouse hook in a dedicated thread.
The hook runs BEFORE the OS dispatches the event to the focused application,
which gives us two options:

  - Return 1 from the hook callback -> Windows discards the event entirely.
    Active application never sees XButton1, so no Back navigation.
  - Return CallNextHookEx(...) -> event proceeds normally.

We selectively suppress only the configured trigger button (and only when
suppress_default=True). All other mouse events flow through untouched.

References:
  - https://learn.microsoft.com/en-us/windows/win32/winmsg/lowlevelmouseproc
  - MSLLHOOKSTRUCT, WH_MOUSE_LL constants from WinUser.h
"""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------- Win32 constants

WH_MOUSE_LL = 14

WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C

XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

# Maps our config strings to a (wm_down, xbutton-or-None) tuple
_BUTTON_MAP = {
    "x1": (WM_XBUTTONDOWN, XBUTTON1),
    "x2": (WM_XBUTTONDOWN, XBUTTON2),
    "middle": (WM_MBUTTONDOWN, None),
    "left": (WM_LBUTTONDOWN, None),
    "right": (WM_RBUTTONDOWN, None),
}

# Corresponding WM_*UP message (to also swallow the release event)
_UP_FOR_DOWN = {
    WM_LBUTTONDOWN: WM_LBUTTONUP,
    WM_RBUTTONDOWN: WM_RBUTTONUP,
    WM_MBUTTONDOWN: WM_MBUTTONUP,
    WM_XBUTTONDOWN: WM_XBUTTONUP,
}


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,                # LRESULT
    ctypes.c_int,                 # nCode
    wintypes.WPARAM,
    wintypes.LPARAM,
)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class WindowsMouseHook:
    """Install a low-level mouse hook in a dedicated thread.

    The hook thread owns a Windows message pump (GetMessage / DispatchMessage)
    which is required for SetWindowsHookEx to deliver callbacks.
    """

    def __init__(
        self,
        button: str = "x1",
        on_press: Optional[Callable[[int, int], None]] = None,
        suppress_default: bool = True,
    ):
        if button not in _BUTTON_MAP:
            raise ValueError(f"Unknown button {button!r}; expected one of {list(_BUTTON_MAP)}")
        wm_down, xbutton = _BUTTON_MAP[button]
        self._wm_down = wm_down
        self._wm_up = _UP_FOR_DOWN[wm_down]
        self._xbutton = xbutton                  # None for non-X buttons
        self._on_press = on_press
        self._suppress = suppress_default
        self._button_name = button

        self._hook_id: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        # Keep a reference to the WINFUNCTYPE wrapper so it isn't GC'd
        self._proc_ref: Optional[HOOKPROC] = None
        self._stop_requested = False

    # ------------------------------------------------------------------ hook proc

    def _make_proc(self) -> HOOKPROC:
        def proc(nCode: int, wParam, lParam):
            if nCode < 0:
                return user32.CallNextHookEx(0, nCode, wParam, lParam)

            wm = wParam & 0xFFFFFFFF

            # Read MSLLHOOKSTRUCT from lParam
            ms = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents

            # XButton check: WM_XBUTTONDOWN puts the button number in the
            # high word of mouseData (1 = XButton1, 2 = XButton2).
            def _is_target_event() -> bool:
                if wm != self._wm_down and wm != self._wm_up:
                    return False
                if self._xbutton is None:
                    return True
                xb = (ms.mouseData >> 16) & 0xFFFF
                return xb == self._xbutton

            if _is_target_event():
                # Only fire on press, not release
                if wm == self._wm_down and self._on_press is not None:
                    try:
                        self._on_press(ms.pt.x, ms.pt.y)
                    except Exception as e:
                        logger.error("on_press handler raised: %s", e)
                if self._suppress:
                    # Eat the event — Windows will not forward to the focused
                    # window. Browser Back / Forward won't fire.
                    return 1

            return user32.CallNextHookEx(0, nCode, wParam, lParam)

        return HOOKPROC(proc)

    # ------------------------------------------------------------------ lifecycle

    def _run(self):
        self._proc_ref = self._make_proc()
        self._thread_id = kernel32.GetCurrentThreadId()
        hmod = kernel32.GetModuleHandleW(None)

        self._hook_id = user32.SetWindowsHookExW(
            WH_MOUSE_LL, self._proc_ref, hmod, 0
        )
        if not self._hook_id:
            err = ctypes.GetLastError()
            logger.error("SetWindowsHookExW failed (err=%d). Falling back to no-suppress mode.", err)
            return

        logger.info(
            "WH_MOUSE_LL installed (button=%s, suppress=%s)",
            self._button_name, self._suppress,
        )

        # Message pump — required for LL hooks to dispatch
        msg = wintypes.MSG()
        while not self._stop_requested:
            r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if r <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        user32.UnhookWindowsHookEx(self._hook_id)
        self._hook_id = None
        logger.info("WH_MOUSE_LL uninstalled")

    def start(self):
        if self._thread is not None:
            return
        self._stop_requested = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="MouseHookThread")
        self._thread.start()

    def stop(self):
        self._stop_requested = True
        # Post a dummy message to wake GetMessage so the loop can exit
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, 0x0012, 0, 0)  # WM_QUIT
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
