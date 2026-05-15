"""Audio buffer recorder — captures mic into memory.

Pure capture, no STT. Transcription is done elsewhere when stop() returns
the buffered audio.
"""

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, device: Optional[int | str] = None):
        self._sample_rate = sample_rate
        self._device = device
        self._stream: Optional[sd.InputStream] = None
        self._recording = threading.Event()
        self._chunks: list = []
        self._chunks_lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def duration_sec(self) -> float:
        """Approximate seconds of audio captured so far."""
        with self._chunks_lock:
            total = sum(c.shape[0] for c in self._chunks)
        return total / self._sample_rate

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.debug("Audio status: %s", status)
        if self._recording.is_set():
            with self._chunks_lock:
                self._chunks.append(indata.copy())

    def start(self):
        if self._recording.is_set():
            logger.warning("start() while already recording")
            return
        with self._chunks_lock:
            self._chunks.clear()
        self._recording.set()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=2048,
            dtype="float32",
            channels=1,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self) -> np.ndarray:
        if not self._recording.is_set():
            return np.array([], dtype=np.float32)
        self._recording.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("Stream close error: %s", e)
            self._stream = None
        with self._chunks_lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).flatten().astype(np.float32)
            self._chunks.clear()
        logger.info("Recording stopped (%.2fs)", len(audio) / self._sample_rate)
        return audio
