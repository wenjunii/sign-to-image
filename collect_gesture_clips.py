from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from temporal_features import extract_frame_features, safe_label_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled temporal sign clips from the webcam.")
    parser.add_argument("--label", required=True, help="Gesture label to record, for example A or WORD:hello.")
    parser.add_argument("--output", default="data/clips", help="Directory where labeled .npz clips are saved.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--seconds", type=float, default=1.4, help="Seconds to record per clip.")
    parser.add_argument("--count", type=int, default=0, help="Stop after this many saved clips. 0 means unlimited.")
    parser.add_argument("--min-valid-frames", type=int, default=6, help="Minimum frames with at least one detected hand.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the camera image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output) / safe_label_dir(args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Error: could not open camera index {args.camera}.", file=sys.stderr)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    )

    saved = 0
    recording_until = 0.0
    recorded_frames: list[np.ndarray] = []
    started_at = 0.0

    print("\n" + "=" * 58)
    print(f"COLLECTING LABEL: {args.label}")
    print("Press r to record one clip | q/Esc to quit")
    print(f"Saving to: {output_dir}")
    print("=" * 58 + "\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read.", file=sys.stderr)
                return 1

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            hand_landmarks = results.multi_hand_landmarks or []
            handedness = results.multi_handedness or []
            features = extract_frame_features(hand_landmarks, handedness)

            for landmarks in hand_landmarks:
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            now = time.time()
            is_recording = now < recording_until
            if is_recording:
                recorded_frames.append(features)
            elif recorded_frames:
                saved += save_clip(output_dir, args.label, recorded_frames, started_at, args)
                recorded_frames = []
                if args.count and saved >= args.count:
                    break

            draw_overlay(frame, args.label, saved, is_recording, recording_until - now, len(hand_landmarks))
            cv2.imshow("Collect Sign Clips", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), 27}:
                break
            if key == ord("r") and not is_recording:
                started_at = time.time()
                recording_until = started_at + args.seconds
                recorded_frames = []
    finally:
        hands.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} clip(s) for label {args.label}.")
    return 0


def save_clip(output_dir: Path, label: str, frames: list[np.ndarray], started_at: float, args: argparse.Namespace) -> int:
    valid_frames = sum(1 for frame in frames if np.any(frame))
    if valid_frames < args.min_valid_frames:
        print(f"Skipped clip: only {valid_frames} valid hand frame(s).")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{safe_label_dir(label)}_{timestamp}.npz"
    elapsed = max(time.time() - started_at, 1e-6)
    np.savez_compressed(
        path,
        label=label,
        frames=np.asarray(frames, dtype=np.float32),
        valid_frames=np.asarray(valid_frames, dtype=np.int32),
        fps=np.asarray(len(frames) / elapsed, dtype=np.float32),
        seconds=np.asarray(args.seconds, dtype=np.float32),
    )
    print(f"Saved {path} ({len(frames)} frames, {valid_frames} valid).")
    return 1


def draw_overlay(frame, label: str, saved: int, is_recording: bool, seconds_left: float, hands_seen: int) -> None:
    status = f"REC {max(seconds_left, 0.0):.1f}s" if is_recording else "READY"
    color = (0, 0, 255) if is_recording else (255, 255, 255)
    lines = [
        f"Label: {label}",
        f"Status: {status}",
        f"Saved: {saved} | Hands: {hands_seen}",
        "r record | q exit",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 1, cv2.LINE_AA)
        y += 30


if __name__ == "__main__":
    raise SystemExit(main())
