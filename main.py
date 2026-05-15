"""voicetap — push-to-talk dictation via side mouse button.

How it works
------------
1. Daemon listens for side mouse button (XButton1 by default) via pynput.
2. First press: pop up a small "REC" indicator near the cursor, start
   recording from the default microphone.
3. Second press: stop recording, swap popup to "Processing...", send audio
   to Groq Whisper, then to Llama 3.3 for filler-word cleanup.
4. Cleaned text is pasted into the focused window via clipboard + Ctrl+V
   (or Cmd+V on macOS).
5. Popup briefly flashes "Pasted" and disappears.

Threading model
---------------
- Main thread: Tkinter mainloop + popup.
- Mouse listener: pynput's own thread.
- Audio capture: sounddevice callback thread (managed by sounddevice).
- Transcription: spawned worker thread per recording, so the UI never
  blocks while Groq is responding.

All cross-thread UI updates are funneled through `root.after(...)`.
"""

import logging
import platform
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _make_dpi_aware():
    """Tell Windows this process knows about DPI scaling.

    Without this call, mouse coordinates from pynput and window coordinates in
    Tkinter use different units when display scaling is set to anything other
    than 100% — the popup ends up far from the cursor.
    """
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2 (Windows 8.1+, recommended)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (OSError, AttributeError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_make_dpi_aware()

from core.audio_recorder import AudioRecorder
from core.mouse_listener import MouseHotkeyListener
from core.text_inserter import insert_text
from core.transcriber import GroqTranscriber
from ui.popup import NearCursorPopup

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class VoicetapApp:
    def __init__(self, cfg: dict, root: tk.Tk):
        self.cfg = cfg
        self.root = root

        audio_cfg = cfg.get("audio", {})
        self.recorder = AudioRecorder(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            device=audio_cfg.get("device"),
        )

        stt_cfg = cfg.get("stt", {})
        post_cfg = cfg.get("post_edit", {})
        self.transcriber = GroqTranscriber(
            stt_model=stt_cfg.get("model", "whisper-large-v3-turbo"),
            language=stt_cfg.get("language"),
            initial_prompt=stt_cfg.get("initial_prompt"),
            post_edit_enabled=post_cfg.get("enabled", True),
            post_edit_model=post_cfg.get("model", "llama-3.3-70b-versatile"),
            post_edit_temperature=post_cfg.get("temperature", 0.1),
            post_edit_max_tokens=post_cfg.get("max_tokens", 2048),
        )

        popup_cfg = cfg.get("popup", {})
        self.popup = NearCursorPopup(
            root,
            width=popup_cfg.get("width", 200),
            height=popup_cfg.get("height", 60),
            offset_x=popup_cfg.get("offset_x", 20),
            offset_y=popup_cfg.get("offset_y", 20),
            bg_color=popup_cfg.get("bg_color", "#1a1a1a"),
            text_color=popup_cfg.get("text_color", "#ffffff"),
            recording_color=popup_cfg.get("recording_color", "#ff3030"),
            transcribing_color=popup_cfg.get("transcribing_color", "#ffaa00"),
            done_color=popup_cfg.get("done_color", "#30ff60"),
        )

        text_cfg = cfg.get("text_input", {})
        self._trailing_space = text_cfg.get("trailing_space", True)

        hotkey_cfg = cfg.get("hotkey", {})
        self.mouse = MouseHotkeyListener(
            button=hotkey_cfg.get("button", "x1"),
            mode=hotkey_cfg.get("mode", "toggle"),
            on_toggle=self._on_toggle,
        )

        # Prevent double-triggering and overlapping transcriptions
        self._busy_lock = threading.Lock()
        self._transcribing = False

    # ------------------------------------------------------------------ hotkey

    def _on_toggle(self, x: int, y: int):
        """Called from the pynput listener thread on each side-button press."""
        with self._busy_lock:
            if self._transcribing:
                logger.info("Ignored press: previous transcription still in progress")
                return
            if self.recorder.is_recording:
                # Second press: stop and transcribe
                self._stop_and_transcribe()
            else:
                # First press: start
                self._start_recording(x, y)

    def _start_recording(self, x: int, y: int):
        try:
            self.recorder.start()
        except Exception as e:
            logger.error("Failed to start recording: %s", e)
            self.popup.show_error("Mic error")
            return
        self.popup.show_recording(x, y)

    def _stop_and_transcribe(self):
        audio = self.recorder.stop()
        self.popup.show_transcribing()
        self._transcribing = True

        def _worker():
            try:
                text = self.transcriber.transcribe(audio, sample_rate=self.recorder.sample_rate)
                if not text:
                    logger.info("Empty transcription, nothing to paste")
                    self.popup.show_error("(empty)")
                    return
                if self._trailing_space:
                    text = text.rstrip() + " "
                insert_text(text)
                logger.info("Pasted: %s", text[:120])
                self.popup.show_done()
            except Exception as e:
                logger.error("Transcription pipeline failed: %s", e)
                self.popup.show_error("Failed")
            finally:
                with self._busy_lock:
                    self._transcribing = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        self.mouse.start()
        hotkey_cfg = self.cfg.get("hotkey", {})
        logger.info(
            "voicetap ready. Press the mouse %s button (mode=%s) to start/stop dictation.",
            hotkey_cfg.get("button", "x1"),
            hotkey_cfg.get("mode", "toggle"),
        )

    def stop(self):
        if self.recorder.is_recording:
            self.recorder.stop()
        self.mouse.stop()


def main():
    load_dotenv()
    cfg = load_config()

    root = tk.Tk()
    root.withdraw()  # main hidden window — only the popup is visible

    app = VoicetapApp(cfg, root)
    app.start()

    def _on_closing():
        logger.info("Shutting down")
        app.stop()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", _on_closing)

    # Catch Ctrl+C from a console launch
    def _poll_signals():
        root.after(200, _poll_signals)

    root.after(200, _poll_signals)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        _on_closing()


if __name__ == "__main__":
    main()
