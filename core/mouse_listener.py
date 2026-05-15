"""Global mouse hotkey listener with optional event suppression.

On Windows, suppression is needed because the browser/OS interprets the side
buttons as Back/Forward navigation by default. We install a low-level mouse
hook (WH_MOUSE_LL) which can EAT the event before the focused application
sees it. See core.windows_mouse_hook for the gory details.

On macOS and Linux we fall back to pynput's observe-only listener (no
suppression). macOS requires Accessibility permission for the running
process; without it, pynput sees nothing.
"""

from __future__ import annotations

import logging
import platform
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()


class MouseHotkeyListener:
    """Listens for the configured mouse button and fires on_toggle (or
    on_press / on_release in hold mode).

    On Windows, set suppress_default=True to prevent the button from also
    doing whatever the OS normally does with it (Back/Forward in browsers,
    etc.).
    """

    def __init__(
        self,
        button: str = "x1",
        mode: str = "toggle",
        on_toggle: Optional[Callable[[int, int], None]] = None,
        on_press: Optional[Callable[[int, int], None]] = None,
        on_release: Optional[Callable[[int, int], None]] = None,
        suppress_default: bool = True,
    ):
        self.button = button
        self.mode = mode
        self.on_toggle = on_toggle
        self.on_press = on_press
        self.on_release = on_release
        self.suppress_default = suppress_default

        self._impl = None  # filled in start()

    # ------------------------------------------------------------------ helpers

    def _toggle_handler(self, x: int, y: int):
        if self.on_toggle:
            self.on_toggle(x, y)

    def _press_handler(self, x: int, y: int):
        if self.on_press:
            self.on_press(x, y)

    def _release_handler(self, x: int, y: int):
        if self.on_release:
            self.on_release(x, y)

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        if self._impl is not None:
            return

        if _SYSTEM == "Windows":
            # Low-level hook supports event suppression
            from core.windows_mouse_hook import WindowsMouseHook
            if self.mode == "toggle":
                on_press_cb = self._toggle_handler
            else:
                # hold mode: WindowsMouseHook only fires on press currently,
                # which is fine for toggle. For hold we still need both edges.
                on_press_cb = self._press_handler
            self._impl = WindowsMouseHook(
                button=self.button,
                on_press=on_press_cb,
                suppress_default=self.suppress_default,
            )
            self._impl.start()
            if self.mode == "hold":
                logger.warning(
                    "hold mode is not fully supported with Windows low-level hook; "
                    "consider switching to toggle mode."
                )
        else:
            # macOS / Linux: pynput listener, no event suppression
            from pynput import mouse as pmouse
            _BUTTON_MAP = {
                "x1": pmouse.Button.x1,
                "x2": pmouse.Button.x2,
                "middle": pmouse.Button.middle,
                "left": pmouse.Button.left,
                "right": pmouse.Button.right,
            }
            target = _BUTTON_MAP[self.button]

            def _on_click(x, y, button, pressed):
                if button != target:
                    return
                if self.mode == "toggle":
                    if pressed:
                        self._toggle_handler(x, y)
                else:  # hold
                    if pressed:
                        self._press_handler(x, y)
                    else:
                        self._release_handler(x, y)

            listener = pmouse.Listener(on_click=_on_click)
            listener.start()
            self._impl = listener
            logger.info(
                "pynput listener started (button=%s, mode=%s) — note: events "
                "are not suppressed on this platform.",
                self.button, self.mode,
            )

    def stop(self):
        if self._impl is None:
            return
        try:
            self._impl.stop()
        except Exception as e:
            logger.warning("Listener stop error: %s", e)
        self._impl = None
