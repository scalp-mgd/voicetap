"""Cross-platform text inserter — clipboard + system paste hotkey.

Same approach as remote-hand-control: Win32 keybd_event on Windows (works
from background threads, no focus issues), pyautogui on macOS/Linux.
"""

import logging
import platform
import time

import pyperclip

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


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
