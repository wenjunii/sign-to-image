# Sign-to-Visual Pipeline for StreamDiffusionTD

Yes, there is a way to do it: replace the voice transcription stage with a camera-based sign recognition stage, then keep the same OSC prompt bridge into TouchDesigner / StreamDiffusionTD.

This project is a starter implementation of:

```text
webcam -> MediaPipe hand landmarks -> ASL fingerspelling text -> SDXL prompt -> OSC -> StreamDiffusionTD
```

It references the structure of `voice-to-visual-sdtd`, but swaps Whisper/audio capture for webcam hand tracking.

## Features

- Live webcam capture with MediaPipe hand landmarks.
- Rule-based ASL fingerspelling prototype for static handshapes: `A B C D E F I L O U V W Y`.
- Two-hand commands:
  - two open palms: send prompt
  - open palm plus closed fist: insert space
  - two closed fists: clear text
- Same StreamDiffusionTD OSC bridge as the voice project:
  - `/prompt` receives the final visual prompt
  - `/partial_text` receives the current signed text
  - `/gesture` receives the committed sign token
- Same visual identity controls as the voice project: gender, age, and visual representation mode.

## Important Limit

The built-in recognizer is a prototype. It is good for testing the full sign-to-image loop and simple fingerspelling, but full sign language recognition needs a trained temporal model because real signing includes motion, grammar, facial expression, body pose, and context.

The useful architecture is already here: swap `RuleBasedASLRecognizer` in `gesture_rules.py` for a trained ASL or sign-language-recognition model, and keep the OSC / prompt pipeline unchanged.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Usage

1. Open your TouchDesigner StreamDiffusionTD project.
2. Make sure the OSC In DAT is listening on port `7000`.
3. Run the bridge:

```powershell
python sign_to_visual.py
```

If you want manual commit mode, where Space appends the currently detected sign:

```powershell
python sign_to_visual.py --commit-mode manual
```

If your camera is not index `0`:

```powershell
python sign_to_visual.py --camera 1
```

## Controls

| Key | Action |
| --- | --- |
| `Space` | In manual mode, append current sign. In auto mode, insert a space. |
| `Enter` | Send the current signed text as a StreamDiffusion prompt. |
| `Backspace` | Delete one character. |
| `c` | Clear text. |
| `q` or `Esc` | Exit. |
| `m` / `w` / `n` | Man / woman / neutral subject focus. |
| `1` / `2` / `3` | Young / adult / elder age focus. |
| `d` / `b` / `x` | Asian-American / Black and Brown / combined visual mode. |

## Configuration

Edit `.env` after copying from `.env.example`.

Common settings:

```env
OSC_IP=127.0.0.1
OSC_PORT=7000
CAMERA_INDEX=0
SIGN_COMMIT_MODE=auto
SIGN_HOLD_SECONDS=0.75
AUTO_SEND_PROMPT=true
```

Set `AUTO_SEND_PROMPT=false` if you only want prompts sent when you press Enter or use the two-open-palms send command.

## How To Make It Better

For an installation or performance piece, the next upgrade is a trained recognizer:

1. Collect gesture clips from the actual signer and camera setup.
2. Train a temporal model on MediaPipe hand, pose, and face landmarks.
3. Export it to ONNX or TensorFlow Lite.
4. Replace `RuleBasedASLRecognizer` with a model-backed recognizer that returns the same `GestureResult` shape.

That gets you from simple fingerspelling into phrase-level sign recognition without changing the StreamDiffusion side.
