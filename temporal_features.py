from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HAND_LANDMARK_COUNT = 21
POINT_DIMS = 3
HAND_FEATURE_SIZE = 1 + HAND_LANDMARK_COUNT * POINT_DIMS
FRAME_FEATURE_SIZE = HAND_FEATURE_SIZE * 2
DEFAULT_TARGET_FRAMES = 48


def point_to_xyz(point: object) -> tuple[float, float, float]:
    if isinstance(point, tuple) or isinstance(point, list):
        if len(point) == 2:
            return float(point[0]), float(point[1]), 0.0
        return float(point[0]), float(point[1]), float(point[2])
    return float(point.x), float(point.y), float(getattr(point, "z", 0.0))


def extract_frame_features(hand_landmarks: Sequence[object], handedness: Sequence[object]) -> np.ndarray:
    """Return a fixed-width two-hand feature vector for one video frame."""
    slots = np.zeros((2, HAND_FEATURE_SIZE), dtype=np.float32)
    occupied = [False, False]

    for index, landmarks in enumerate(hand_landmarks[:2]):
        label = handedness_label(handedness, index)
        slot = preferred_slot(label)
        if occupied[slot]:
            slot = 1 - slot
        if occupied[slot]:
            continue

        points = landmark_points(landmarks)
        if len(points) != HAND_LANDMARK_COUNT:
            continue

        slots[slot, 0] = 1.0
        slots[slot, 1:] = normalize_hand(points).reshape(-1)
        occupied[slot] = True

    return slots.reshape(-1)


def handedness_label(handedness: Sequence[object], index: int) -> str:
    if index >= len(handedness):
        return "Right"
    item = handedness[index]
    try:
        return str(item.classification[0].label)
    except (AttributeError, IndexError):
        return "Right"


def preferred_slot(label: str) -> int:
    return 0 if label.lower() == "left" else 1


def landmark_points(landmarks: object) -> list[tuple[float, float, float]]:
    raw_points: Iterable[object]
    if hasattr(landmarks, "landmark"):
        raw_points = landmarks.landmark
    else:
        raw_points = landmarks  # type: ignore[assignment]
    return [point_to_xyz(point) for point in raw_points]


def normalize_hand(points: Sequence[tuple[float, float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    wrist = pts[0].copy()
    scale = float(np.linalg.norm(pts[9] - wrist))
    if scale < 1e-6:
        scale = 1.0
    return (pts - wrist) / scale


def resample_sequence(sequence: Sequence[Sequence[float]] | np.ndarray, target_frames: int) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return np.zeros((target_frames, FRAME_FEATURE_SIZE), dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D frame sequence, got shape {frames.shape}.")
    if frames.shape[1] != FRAME_FEATURE_SIZE:
        raise ValueError(f"Expected frame feature size {FRAME_FEATURE_SIZE}, got {frames.shape[1]}.")
    if len(frames) == target_frames:
        return frames.astype(np.float32, copy=False)
    if len(frames) == 1:
        return np.repeat(frames, target_frames, axis=0)

    old_x = np.linspace(0.0, 1.0, num=len(frames), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    resampled = np.empty((target_frames, frames.shape[1]), dtype=np.float32)
    for column in range(frames.shape[1]):
        resampled[:, column] = np.interp(new_x, old_x, frames[:, column])
    return resampled


def temporal_feature_vector(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> np.ndarray:
    frames = resample_sequence(sequence, target_frames)
    velocity = np.vstack([np.zeros((1, frames.shape[1]), dtype=np.float32), np.diff(frames, axis=0)])
    stats = np.concatenate(
        [
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.max(axis=0) - frames.min(axis=0),
        ]
    )
    return np.concatenate([frames.reshape(-1), velocity.reshape(-1), stats]).astype(np.float32)


def safe_label_dir(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("._") or "unlabeled"


def iter_clip_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("**/*.npz"))


def load_clip(path: str | Path) -> tuple[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        label = str(data["label"])
        frames = np.asarray(data["frames"], dtype=np.float32)
    return label, frames
