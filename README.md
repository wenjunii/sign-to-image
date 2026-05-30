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
- Trained temporal model path for signer-specific gesture clips.
- Two-hand commands:
  - two open palms: send prompt
  - open palm plus closed fist: insert space
  - two closed fists: clear text
- Same StreamDiffusionTD OSC bridge as the voice project:
  - `/prompt` receives the final visual prompt
  - `/partial_text` receives the current signed text
  - `/gesture` receives the committed sign token
- Same visual identity controls as the voice project: gender, age, and visual representation mode.

## Recognition Modes

The default `rule_based` recognizer is useful for testing the full sign-to-image
loop and simple fingerspelling.

The `temporal_model` recognizer loads a trained signer-specific classifier from
`SIGN_MODEL_PATH`. It uses short MediaPipe hand landmark clips instead of a
single static handshape, so it can learn signs with motion.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

You can also use the Windows helper scripts, which create `.venv` and install
missing dependencies automatically:

```powershell
.\run.ps1
```

## Usage

1. Open your TouchDesigner StreamDiffusionTD project.
2. Make sure the OSC In DAT is listening on port `7001`.
3. Run the bridge:

```powershell
.\run.ps1
```

If you want manual commit mode, where Space appends the currently detected sign:

```powershell
.\run.ps1 --commit-mode manual
```

If your camera is not index `0`:

```powershell
.\run.ps1 --camera 1
```

## Trained Temporal Model Workflow

Collect clips for each sign token you want the model to recognize. Use single
letters like `A`, or word tokens like `WORD:river`.

```powershell
.\collect.ps1 -Label A -Count 30
.\collect.ps1 -Label B -Count 30
.\collect.ps1 -Label WORD:river -Count 30
```

In the collection window, press `r` to record each clip and `q` to exit. Aim for
20-40 clips per label from the actual signer and camera setup. Clips are saved
under `data/clips/`, which is ignored by Git.

Train the temporal classifier:

```powershell
.\train.ps1
```

The trained model is saved to `models/temporal_sign_model.pkl`, which is also
ignored by Git because it is generated from local signer data.

Enable it in `.env`:

```env
SIGN_RECOGNITION_BACKEND=temporal_model
SIGN_MODEL_PATH=models/temporal_sign_model.pkl
```

Then run the bridge:

```powershell
.\run.ps1 --commit-mode manual
```

Manual mode is best for the first pass because it lets you verify predictions
before committing them into the prompt text.

The Python entry points are still available if you want advanced options:

```powershell
.\.venv\Scripts\python.exe collect_gesture_clips.py --help
.\.venv\Scripts\python.exe train_temporal_model.py --help
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
OSC_PORT=7001
CAMERA_INDEX=0
SIGN_RECOGNITION_BACKEND=rule_based
SIGN_MODEL_PATH=models/temporal_sign_model.pkl
SIGN_COMMIT_MODE=auto
SIGN_HOLD_SECONDS=0.75
AUTO_SEND_PROMPT=true
```

Set `AUTO_SEND_PROMPT=false` if you only want prompts sent when you press Enter or use the two-open-palms send command.

## How To Make It Better

For an installation or performance piece, the next upgrade after the local
temporal classifier is a deeper temporal recognizer:

1. Collect gesture clips from the actual signer and camera setup.
2. Train a temporal model on MediaPipe hand, pose, and face landmarks.
3. Export it to ONNX or TensorFlow Lite.
4. Replace the scikit-learn classifier with a model-backed recognizer that returns the same `GestureResult` shape.

That gets you from simple fingerspelling into phrase-level sign recognition without changing the StreamDiffusion side.
