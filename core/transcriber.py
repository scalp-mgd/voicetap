"""Transcribe audio via Groq Whisper, then optionally polish via Llama 3.3.

Two-stage pipeline (this is the trick Whispr Flow uses):
  1. Whisper large-v3-turbo: raw STT, ~1s for clips under 30s.
  2. Llama 3.3 70B Versatile: removes filler words ("эээ", "ну", "как бы"),
     fixes repetitions and punctuation, normalizes transliteration mistakes.
     Free tier covers thousands of personal dictations per day.

Whisper auto-detects language per clip when `language` is None, which is the
best setting for users who switch between Russian and English mid-sentence.
"""

import io
import logging
import os
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


_HALLUCINATIONS = {
    "продолжение следует...",
    "продолжение следует.",
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "субтитры подогнал dimatorzok",
    "редактор субтитров а.семкин",
    "субтитры сделал",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching!",
    "♪",
    "...",
}


def _is_hallucination(text: str) -> bool:
    norm = text.strip().lower()
    if norm in _HALLUCINATIONS:
        return True
    for h in _HALLUCINATIONS:
        if norm.startswith(h) and len(norm) < len(h) + 5:
            return True
    return False


def _loud_enough(audio: np.ndarray, threshold: float = 0.0015) -> bool:
    """RMS-based VAD. Threshold tuned low so quiet/distant speech isn't
    silently dropped — Whisper can handle very quiet audio after gain
    normalization, so the only thing we really want to skip here is true
    silence / mic-off cases."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
    return rms >= threshold


def _normalize_audio(audio: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    """Auto-gain: scale audio so its loudest sample sits near target_peak.

    Whisper transcribes much better when input is at a reasonable volume.
    Quiet recordings (peak < 0.2) get a multiplier; already-loud ones are
    left alone. We never amplify what looks like pure silence.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 0.005:
        # Essentially silent — amplifying just multiplies noise.
        return audio
    if peak >= target_peak:
        return audio
    gain = target_peak / peak
    # Cap the gain so room noise doesn't get blasted to clipping levels.
    gain = min(gain, 25.0)
    boosted = np.clip(audio * gain, -1.0, 1.0)
    logger.info("Auto-gain x%.1f applied (peak %.3f -> %.3f)",
                gain, peak, float(np.max(np.abs(boosted))))
    return boosted


_POST_EDIT_SYSTEM = """Ты — редактор сырого голосового ввода. На вход приходит расшифровка от Whisper (русский, английский или смешанная речь).

Твоя задача — вернуть отредактированный текст, готовый к вставке в чат или документ:
- Убери filler words: "эээ", "ммм", "ну", "как бы", "типа", "вот", "значит" (но только когда они действительно паразиты, не когда несут смысл).
- Убери повторы и оговорки ("я хочу — я хотел", "это, это вот").
- Восстанови пунктуацию и заглавные буквы там где их явно не хватает.
- Исправь явные опечатки и неправильную транслитерацию (например "Гроку" → "Groq", "анфрепик" → "Anthropic").

КРИТИЧНО:
- НЕ переводи между языками. Если человек говорил по-русски с английскими терминами — сохрани mixing.
- НЕ добавляй слов или мыслей от себя. Не "улучшай" фразы, не перефразируй стиль.
- НЕ комментируй, не извиняйся, не объясняй. Верни ТОЛЬКО отредактированный текст.
- Если на входе только filler / шум — верни пустую строку.
"""


class GroqTranscriber:
    def __init__(
        self,
        api_key: Optional[str] = None,
        stt_model: str = "whisper-large-v3-turbo",
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        post_edit_enabled: bool = True,
        post_edit_model: str = "llama-3.3-70b-versatile",
        post_edit_temperature: float = 0.1,
        post_edit_max_tokens: int = 2048,
    ):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your "
                "key (get a free one at https://console.groq.com)."
            )
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._stt_model = stt_model
        self._language = language
        self._initial_prompt = initial_prompt
        self._post_edit_enabled = post_edit_enabled
        self._post_edit_model = post_edit_model
        self._post_edit_temp = post_edit_temperature
        self._post_edit_max_tokens = post_edit_max_tokens

    # ------------------------------------------------------------------ STT

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio is None or audio.size == 0:
            return ""

        duration = len(audio) / sample_rate
        if duration < 0.3:
            logger.info("Audio too short (%.2fs), skipping", duration)
            return ""
        if not _loud_enough(audio):
            logger.info("Audio too quiet, skipping")
            return ""

        # Auto-gain so Whisper gets a properly-leveled clip even when the
        # user is far from the mic or speaking softly.
        audio = _normalize_audio(audio)

        wav_bytes = _to_wav_bytes(audio, sample_rate)
        logger.info("Sending %.1fs audio to Whisper (%s)...", duration, self._stt_model)

        kwargs = {
            "file": ("audio.wav", wav_bytes),
            "model": self._stt_model,
            "response_format": "text",
        }
        if self._language:
            kwargs["language"] = self._language
        if self._initial_prompt:
            kwargs["prompt"] = self._initial_prompt

        try:
            result = self._client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            logger.error("Whisper call failed: %s", e)
            return ""

        raw = (result if isinstance(result, str) else getattr(result, "text", "")).strip()
        if _is_hallucination(raw):
            logger.info("Filtered hallucination: %r", raw)
            return ""
        if not raw:
            return ""

        logger.info("Whisper raw: %s", raw[:120])

        if not self._post_edit_enabled:
            return raw

        cleaned = self._post_edit(raw)
        if cleaned and cleaned.strip():
            logger.info("Post-edited:  %s", cleaned[:120])
            return cleaned
        logger.info("Post-edit returned empty, using raw Whisper output")
        return raw

    # ------------------------------------------------------------------ LLM clean

    def _post_edit(self, raw_text: str) -> str:
        """Send raw Whisper output through Llama 3.3 to clean it up."""
        try:
            resp = self._client.chat.completions.create(
                model=self._post_edit_model,
                temperature=self._post_edit_temp,
                max_tokens=self._post_edit_max_tokens,
                messages=[
                    {"role": "system", "content": _POST_EDIT_SYSTEM},
                    {"role": "user", "content": raw_text},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            return content
        except Exception as e:
            logger.error("Post-edit failed: %s", e)
            return raw_text


# ---------------------------------------------------------------------- helpers

def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
