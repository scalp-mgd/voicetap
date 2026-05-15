"""Global mouse listener — fires a callback when the configured side button
is pressed. Implements toggle and hold modes.

Uses pynput.mouse.Listener which works on Windows, macOS, and X11/Linux.
On macOS the user has to grant Accessibility permission to the running
process (System Settings -> Privacy & Security -> Accessibility).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from pynput import mouse

logger = logging.getLogger(__name__)

# Map config strings to pynput Button enum
_BUTTON_MAP = {
    "x1": mouse.Button.x1,        # back / thumb button
    "x2": mouse.Button.x2,        # forward button
    "middle": mouse.Button.middle,
    "left": mouse.Button.left,    # for debugging only
    "right": mouse.Button.right,
}


class MouseHotkeyListener:
    """Listens for the configured mouse button.

    Modes:
      - "toggle": each press fires on_toggle().
      - "hold": press fires on_press(), release fires on_release().
    """

    def __init__(
        self,
        button: str = "x1",
        mode: str = "toggle",
        on_toggle: Optional[Callable[[int, int], None]] = None,
        on_press: Optional[Callable[[int, int], None]] = None,
        on_release: Optional[Callable[[int, int], None]] = None,
    ):
        if button not in _BUTTON_MAP:
            raise ValueError(f"Unknown button {button!r}; expected one of {list(_BUTTON_MAP)}")
        self._target = _BUTTON_MAP[button]
        self._button_name = button
        self._mode = mode
        self._on_toggle = on_toggle
        self._on_press = on_press
        self._on_release = on_release
        self._listener: Optional[mouse.Listener] = None

    def _handler(self, x: int, y: int, button, pressed: bool):
        if button != self._target:
            return
        if self._mode == "toggle":
            if pressed and self._on_toggle:
                logger.debug("toggle fired at (%d, %d)", x, y)
                self._on_toggle(x, y)
        elif self._mode == "hold":
            if pressed and self._on_press:
                self._on_press(x, y)
            elif not pressed and self._on_release:
                self._on_release(x, y)

    def start(self):
        if self._listener is not None:
            return
        self._listener = mouse.Listener(on_click=self._handler)
        self._listener.start()
        logger.info("Mouse listener started (button=%s, mode=%s)", self._button_name, self._mode)

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Mouse listener stopped")
