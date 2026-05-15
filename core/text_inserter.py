"""Cross-platform clipboard I/O — paste hotkey + selection capture.

Same approach as remote-hand-control: Win32 keybd_event on Windows (works
from background threads, no focus issues), pyautogui on macOS/Linux.
"""

import logging
import platform
import time

import pyperclip

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


def _send_copy():
    if _SYSTEM == "Windows":
        import ctypes
        VK_CONTROL = 0x11
        VK_C = 0x43
        KEYEVENTF_KEYUP = 0x0002
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_C, 0, 0, 0)
        time.sleep(0.02)
        ctypes.windll.user32.keybd_event(VK_C, 0, KEYEVENTF_KEYUP, 0)
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
    elif _SYSTEM == "Darwin":
        import pyautogui
        pyautogui.hotkey("command", "c", _pause=False)
    else:
        import pyautogui
        pyautogui.hotkey("ctrl", "c", _pause=False)


def _paste_windows():
    import ctypes
    VK_CONTROL = 0x11
    VK_V = 0x56
    KEYEVENTF_KEYUP = 0x0002
    ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_V, 0, 0, 0)
    time.sleep(0.02)
    ctypes.windll.user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _paste_mac():
    import pyautogui
    pyautogui.hotkey("command", "v", _pause=False)


def _paste_linux():
    import pyautogui
    pyautogui.hotkey("ctrl", "v", _pause=False)


def insert_text(text: str):
    """Copy text to clipboard and send paste hotkey to the active window."""
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.05)
    if _SYSTEM == "Windows":
        _paste_windows()
    elif _SYSTEM == "Darwin":
        _paste_mac()
    else:
        _paste_linux()
    logger.debug("Pasted (%d chars)", len(text))


def capture_selection(settle_ms: int = 100) -> str:
    """Send Ctrl+C and read the resulting clipboard; "" if nothing selected.

    Uses a unique sentinel to detect "Ctrl+C did nothing because there was
    no selection" — if the clipboard still holds the sentinel after the
    copy, the active app didn't write anything new. Without this check we
    couldn't distinguish "no selection" from "the selection happened to
    equal the previous clipboard content".
    """
    sentinel = f"​​voicetap-sentinel-{time.time_ns()}​​"
    prev_clip = ""
    try:
        prev_clip = pyperclip.paste()
    except Exception:
        pass

    try:
        pyperclip.copy(sentinel)
    except Exception as e:
        logger.warning("Could not seed clipboard sentinel: %s", e)
        return ""

    _send_copy()
    time.sleep(settle_ms / 1000.0)

    try:
        new_clip = pyperclip.paste()
    except Exception:
        new_clip = ""

    # If Ctrl+C left the sentinel untouched, no selection was active.
    if new_clip == sentinel or not new_clip:
        # Restore the user's previous clipboard so we don't leave the sentinel.
        try:
            pyperclip.copy(prev_clip)
        except Exception:
            pass
        return ""
    return new_clip
