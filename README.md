# voicetap

Push-to-talk dictation via the side button on your mouse. Click the side button — speak — click again — your speech is transcribed, cleaned up by an LLM, and pasted into the focused window.

Built for vibe-engineering: dictate prompts into Claude / Cursor / VS Code without taking hands off the mouse.

## How it works

1. Click the side mouse button (typically the **back / thumb button**). A small recording indicator appears next to your cursor with a pulsing red dot and a timer.
2. Speak naturally. Russian, English, code names, slang, mid-sentence language switches — Whisper large-v3-turbo handles all of it.
3. Click the side button again. Indicator switches to **Processing...**.
4. Within ~1–2 seconds the cleaned text is pasted into whatever window you were in. Indicator flashes **Pasted** and disappears.

## What makes it good

- **Groq Whisper large-v3-turbo** for STT — ~1 second latency, free tier covers thousands of dictations per day, top-tier accuracy on Russian, English, and mixed.
- **LLM post-edit via Llama 3.3 70B** (also free on Groq) — removes filler words ("эээ", "ну", "как бы"), restores punctuation, normalizes transliteration of technical terms. This is the trick Whispr Flow uses.
- **Custom dictionary** — your project names, technical terms, and personal vocabulary go into the Whisper `initial_prompt` config field. Whisper biases toward recognizing them correctly.
- **Auto language detection** — leave `language: null` in config and Whisper guesses per clip. Best setting if you switch between Russian and English mid-sentence.
- **Cross-platform paste** — works in any text field that accepts Ctrl+V / Cmd+V, including web Claude, Slack, VS Code, browsers.

## Install

```bash
pip install -r requirements.txt
```

### Groq API key

1. Get a free key at https://console.groq.com (login with Google works).
2. Settings → API Keys → Create API Key. Copy it (starts with `gsk_...`).
3. Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

`.env` is in `.gitignore`, so the key never leaves your machine.

## Run

```bash
python main.py
```

Or on Windows, double-click `run.bat` (uses `pythonw` so no console window stays open).

The app sits in the background and listens for the side mouse button globally. There is no main window — only the small popup that appears during dictation. To exit, kill the Python process from Task Manager or stop the console.

## Configuration

All knobs live in `config.yaml`:

| Section | Key | Notes |
|---------|-----|-------|
| `hotkey` | `button` | `x1` (back / thumb), `x2` (forward), `middle` |
| `hotkey` | `mode` | `toggle` (click-on / click-off) or `hold` (press-and-hold) |
| `audio` | `sample_rate` | `16000` is what Whisper wants — leave it alone unless you know better |
| `audio` | `device` | `null` = system default. Use a device name string to pick a specific mic |
| `stt` | `model` | `whisper-large-v3-turbo` (fast) or `whisper-large-v3` (slower, slightly more accurate) |
| `stt` | `language` | `null` = auto-detect (recommended). Set to `"ru"` or `"en"` to lock |
| `stt` | `initial_prompt` | Free-form text with project names and technical terms. Whisper biases toward these spellings |
| `post_edit` | `enabled` | `true` to run cleanup pass through Llama. Set `false` for raw Whisper output |
| `post_edit` | `model` | `llama-3.3-70b-versatile` (default, free tier) |

## Platform notes

### Windows

Works out of the box. Side buttons on most gaming mice register as `XButton1` (back) and `XButton2` (forward), which pynput exposes as `x1` and `x2`.

### macOS

Requires **Accessibility permission**: System Settings → Privacy & Security → Accessibility → add Terminal (or whatever runs `python main.py`).

Without it, pynput can't see mouse events and `insert_text` can't trigger Cmd+V.

### Linux

You'll need X11 (Wayland support in pynput is limited as of mid-2026). Install system deps:
```bash
sudo apt install python3-tk python3-dev scrot xdotool
```

## Troubleshooting

**Side button does nothing.** Some Logitech mice and gaming mice with custom drivers (Logi Options+, Razer Synapse) intercept side button events before they reach the OS — they're remapped to internal actions and pynput never sees them. Open your mouse software and set the side buttons to "Generic Button 4 / 5" or "Browser Back / Forward".

**Mic captures nothing.** Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices, then set `audio.device` to the right name in config.

**Text gets pasted in the wrong place.** Some apps move focus while transcription is in progress. Click in the target field BEFORE pressing the side button to start dictation.

**Whisper hallucinates "Продолжение следует..." or "Thanks for watching!" on silence.** Already filtered — but if a new variant slips through, add it to `_HALLUCINATIONS` in `core/transcriber.py`.

**Post-edit drops too much.** Set `post_edit.enabled: false` to disable the LLM pass and use raw Whisper output. Or tune the system prompt in `core/transcriber.py`.

## Architecture

```
+----------------+       +------------------+
|  pynput        |       |  Tkinter Tk()    |  <-- main thread
|  mouse listener|------>|  hidden root +   |
|  (own thread)  |       |  popup Toplevel  |
+----------------+       +------------------+
        |                         ^
        | toggle                  | root.after(...)
        v                         |
+----------------+       +------------------+
|  AudioRecorder | ----> |  Worker thread:  |
|  sounddevice   |       |  Whisper + Llama |
+----------------+       |  + insert_text   |
                         +------------------+
```

Tkinter is not thread-safe; every UI mutation goes through `root.after(0, ...)` from the public popup methods, so callers don't have to care which thread they're on.

## License

MIT.
