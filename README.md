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
- Optional MediaPipe Holistic pipeline with hands, pose, and face landmarks.
- Experimental GISLR / PopSign-style TensorFlow Lite backend for pretrained isolated-sign models.
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
`SIGN_MODEL_PATH`. It uses short MediaPipe landmark clips instead of a single
static handshape, so it can learn signs with motion.

The `gislr_tflite` recognizer loads a pretrained TensorFlow Lite model trained
on GISLR / PopSign-style isolated signs. It expects MediaPipe Holistic
landmarks in the common `543 x 3` order: face, left hand, pose, right hand. This
mode predicts one isolated vocabulary word at a time, not full ASL sentences.
The current implementation runs MediaPipe Holistic and TFLite inference on CPU.

## Landmark Pipelines

The default `hands` pipeline uses MediaPipe Hands and keeps the existing
two-hand feature format. It is lighter and is the best first choice for a
controlled sign-to-prompt vocabulary.

The optional `holistic` pipeline uses MediaPipe Holistic and adds pose and face
landmarks to the same temporal workflow. This can help with signs where body
position, shoulders, head movement, or facial expression matter, but it is
heavier and needs its own training clips and model.

Hands and Holistic models are not interchangeable. The collection, training,
and live run steps must use the same landmark pipeline.

The GISLR / PopSign TFLite backend always uses the `holistic` pipeline because
those models usually expect face, hand, and pose landmarks in one sequence.
GISLR / PopSign is not a MediaPipe landmark pipeline itself. In this project it
is a recognition backend that reads MediaPipe Holistic landmarks, and it is not
used with the `hands` pipeline.

| Landmark pipeline | Recognition backends |
| --- | --- |
| `hands` | `rule_based`, `temporal_model` |
| `holistic` | `temporal_model`, `gislr_tflite` |

## Signer-Specific Vocabulary

This project is not a general ASL or sign-language interpreter. It is designed
to become a custom sign-to-prompt vocabulary recognizer.

For ASL, an ASL signer or Deaf consultant can help choose appropriate signs and
labels, but the most reliable model should be trained on the actual person who
will use the pipeline. Signing varies by person, region, speed, handedness,
camera angle, body proportions, and expressive style.

For one performer, collect clips from that performer. For multiple users,
collect clips from each user or train separate models per user.

## Recommended Recording Plan

The next practical step is to ask the person who will use the pipeline to help
define and record the vocabulary.

1. Co-design the vocabulary with the signer: decide which letters, words,
   visual prompt tokens, and commands the system should recognize.
2. Confirm the signs and labels. For ASL, involve the signer and, when possible,
   a Deaf/ASL consultant so the vocabulary is appropriate.
3. Record the actual signer performing each vocabulary item in the real setup:
   same camera angle, lighting, distance, framing, and background.
4. Start with 20-40 clips per sign. Add more clips for signs that are subtle,
   fast, two-handed, or easily confused with another sign.
5. Train the model, then test in manual commit mode before using it live with
   StreamDiffusionTD.

The current helper scripts support live webcam collection. If the signer records
regular video files first, add a video-import script later to convert those
videos into the same `data/clips/` landmark format before training.

## Training Data Timing

Training does not need to happen live. The current workflow is:

```text
camera collection -> saved landmark clips -> train later -> run live
```

The helper scripts currently support live webcam collection into `data/clips/`.
Those clips are saved first, and `.\train.ps1` trains the model afterward.
New clips include per-frame timing metadata so training can normalize gestures
to a fixed time window instead of depending only on the number of captured
frames.

Recorded videos can also work in principle:

```text
record videos -> extract MediaPipe landmarks -> save clips -> train later
```

That video-import path is not included in the helper scripts yet. If you use
recorded videos later, keep them close to the real installation setup: same
signer, camera angle, distance, framing, lighting, and background.

## Reusing a Trained Model

The signer should not need to train the model every time. After training, the
model is saved to:

```text
models/temporal_sign_model.pkl
```

Future runs can reuse that model by keeping the file and setting:

```env
SIGN_RECOGNITION_BACKEND=temporal_model
SIGN_MODEL_PATH=models/temporal_sign_model.pkl
```

Retrain or fine-tune only when something important changes: a new signer, new
vocabulary, a very different camera angle, different lighting or framing,
different signing distance, or recurring sign confusions.

The trained model can also be reused in another future pipeline if that
pipeline uses the same input shape:

```text
MediaPipe hand landmarks -> temporal_features.py -> temporal_model.py -> model file
```

For portability, keep a private signer model bundle with:

```text
models/temporal_sign_model.pkl
temporal_features.py
temporal_model.py
requirements.txt
optional: data/clips/ training examples
```

Treat saved clips and trained models as personal data because they reflect the
signer's body movement and signing style. Do not publish or share them unless
the signer explicitly agrees. `data/clips/` and `models/` are ignored by Git by
default.

## Timing And Occlusion Robustness

Temporal recognition is sensitive to frame rate. Collection might run at one
FPS, while live use alongside TouchDesigner / StreamDiffusionTD might run at
another. To reduce that mismatch, training and live inference resample each
gesture into a fixed number of frames over a fixed time window. The default is
`48` frames over `1.4` seconds.

During live inference, the temporal model keeps a time-based rolling buffer
instead of only the last N webcam frames. That helps when system load changes
the camera loop FPS.

MediaPipe can briefly lose a hand during fast motion or hand-over-hand
occlusion. Short missing-hand gaps are forward-filled before feature extraction.
Longer tracking losses still reset the live temporal buffer so stale motion does
not become a false prediction.

You can tune these values when training:

```powershell
.\train.ps1 -Frames 48 -Seconds 1.4 -MaxMissingSeconds 0.18
```

If you trained a model before this timing metadata path was added, collect or
reuse clips and run `.\train.ps1` again so the saved model includes the timing
window and missing-frame settings.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For the optional GISLR / PopSign TFLite backend, install one TensorFlow Lite
runtime in the same venv. On Windows, the recommended lightweight runtime is
LiteRT:

```powershell
pip install ai-edge-litert
```

Or use the GISLR setup helper, which installs LiteRT, downloads the pretrained
model and label map into `models/`, validates the model, and updates local
`.env` for manual GISLR testing:

```powershell
.\setup_gislr.ps1
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

To run a trained Holistic model:

```powershell
.\run.ps1 --recognition-backend temporal_model --landmark-pipeline holistic --model-path models/holistic_temporal_sign_model.pkl --commit-mode manual
```

To run a GISLR / PopSign-style TFLite model:

```powershell
.\run.ps1 --recognition-backend gislr_tflite --model-path models/gislr_model.tflite --label-map models/sign_to_prediction_index_map.json --commit-mode manual
```

## Trained Temporal Model Workflow

Collect clips for each sign token you want the model to recognize. Use single
letters like `A`, or word tokens like `WORD:river`.

```powershell
.\collect.ps1 -Label A -Count 30
.\collect.ps1 -Label B -Count 30
.\collect.ps1 -Label WORD:river -Count 30
```

To collect with MediaPipe Holistic instead of MediaPipe Hands:

```powershell
.\collect.ps1 -Label WORD:river -Pipeline holistic -Count 30
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

To train a Holistic model:

```powershell
.\train.ps1 -Pipeline holistic -Output models/holistic_temporal_sign_model.pkl
```

Enable it in `.env`:

```env
SIGN_RECOGNITION_BACKEND=temporal_model
SIGN_LANDMARK_PIPELINE=hands
SIGN_MODEL_PATH=models/temporal_sign_model.pkl
```

Then run the bridge:

```powershell
.\run.ps1 --commit-mode manual
```

Manual mode is best for the first pass because it lets you verify predictions
before committing them into the prompt text.

## GISLR / PopSign TFLite Workflow

This is the pretrained-model path. It is useful when you have a GISLR /
PopSign-compatible `.tflite` model and a label map, such as Kaggle's
`sign_to_prediction_index_map.json`.

For the helper quick-start path, run:

```powershell
.\setup_gislr.ps1
```

The helper downloads the MIT-licensed Hugging Face
[`sign/kaggle-asl-signs-1st-place`](https://huggingface.co/sign/kaggle-asl-signs-1st-place) model:

```text
models/gislr_model.tflite
models/sign_to_prediction_index_map.json
```

Put local model files under `models/`:

```text
models/gislr_model.tflite
models/sign_to_prediction_index_map.json
```

Those files are ignored by Git by default. Do not commit pretrained weights or
signer data unless the license and consent terms allow it.

Then run:

```powershell
.\run.ps1 --recognition-backend gislr_tflite --model-path models/gislr_model.tflite --label-map models/sign_to_prediction_index_map.json --commit-mode manual
```

If `.\setup_gislr.ps1` updated `.env`, this shorter command is enough:

```powershell
.\run.ps1
```

The backend automatically switches to MediaPipe Holistic, keeps a short
time-based landmark window, resamples it to `SIGN_GISLR_TARGET_FRAMES`, runs the
TFLite model, and converts the winning label into a `WORD:<label>` token for the
prompt buffer.

This backend is CPU-capable by design. The RTX 3080 Ti is still useful for
StreamDiffusionTD image generation, but the current GISLR / PopSign recognition
path does not use CUDA or a GPU delegate. If CPU load is too high, try lowering
camera resolution, shortening the window only if the model supports it, or
tuning camera/preview settings. The LiteRT runtime currently uses one CPU
thread for this model because higher thread counts can be unstable on Windows.

Useful settings:

```env
SIGN_RECOGNITION_BACKEND=gislr_tflite
SIGN_LANDMARK_PIPELINE=holistic
SIGN_MODEL_PATH=models/gislr_model.tflite
SIGN_GISLR_LABEL_MAP=models/sign_to_prediction_index_map.json
SIGN_GISLR_TARGET_FRAMES=64
SIGN_GISLR_WINDOW_SECONDS=1.6
SIGN_GISLR_THREADS=1
```

If the model expects a different frame count or time window, adjust
`SIGN_GISLR_TARGET_FRAMES` and `SIGN_GISLR_WINDOW_SECONDS`.

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
SIGN_LANDMARK_PIPELINE=hands
SIGN_MODEL_PATH=models/temporal_sign_model.pkl
SIGN_GISLR_LABEL_MAP=models/sign_to_prediction_index_map.json
SIGN_GISLR_TARGET_FRAMES=64
SIGN_GISLR_WINDOW_SECONDS=1.6
SIGN_GISLR_THREADS=1
SIGN_COMMIT_MODE=auto
SIGN_HOLD_SECONDS=0.75
AUTO_SEND_PROMPT=true
```

Set `AUTO_SEND_PROMPT=false` if you only want prompts sent when you press Enter or use the two-open-palms send command.

## How To Make It Better

For an installation or performance piece, the next upgrades after these starter
backends are:

1. Test a GISLR / PopSign `.tflite` model against the actual signer and camera
   setup.
2. Compare that pretrained model against the signer-specific `temporal_model`
   workflow.
3. Fine-tune or retrain a deeper temporal model when the pretrained vocabulary
   or signer coverage is not reliable enough.

That gets you from simple fingerspelling into larger isolated-sign vocabularies
without changing the StreamDiffusion side.
