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
DEFAULT_TARGET_SECONDS = 1.4
DEFAULT_MAX_MISSING_SECONDS = 0.18
DEFAULT_MAX_MISSING_FRAMES = 4


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


def resample_sequence(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    target_frames: int,
    timestamps: Sequence[float] | np.ndarray | None = None,
    window_seconds: float | None = None,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return np.zeros((target_frames, FRAME_FEATURE_SIZE), dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D frame sequence, got shape {frames.shape}.")
    if frames.shape[1] != FRAME_FEATURE_SIZE:
        raise ValueError(f"Expected frame feature size {FRAME_FEATURE_SIZE}, got {frames.shape[1]}.")
    if len(frames) == target_frames and timestamps is None and window_seconds is None:
        return frames.astype(np.float32, copy=False)
    if len(frames) == 1:
        return np.repeat(frames, target_frames, axis=0)

    if timestamps is None:
        old_x = np.linspace(0.0, 1.0, num=len(frames), dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    else:
        old_x = clean_timestamps(timestamps, len(frames))
        if window_seconds is not None and window_seconds > 0:
            end = float(old_x[-1])
            start = end - float(window_seconds)
            new_x = np.linspace(start, end, num=target_frames, dtype=np.float32)
        else:
            new_x = np.linspace(float(old_x[0]), float(old_x[-1]), num=target_frames, dtype=np.float32)

    resampled = np.empty((target_frames, frames.shape[1]), dtype=np.float32)
    for column in range(frames.shape[1]):
        resampled[:, column] = np.interp(new_x, old_x, frames[:, column], left=frames[0, column], right=frames[-1, column])
    return resampled


def temporal_feature_vector(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    timestamps: Sequence[float] | np.ndarray | None = None,
    window_seconds: float | None = None,
    max_missing_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    max_missing_frames: int = DEFAULT_MAX_MISSING_FRAMES,
) -> np.ndarray:
    repaired = repair_missing_hand_frames(sequence, timestamps, max_missing_seconds, max_missing_frames)
    frames = resample_sequence(repaired, target_frames, timestamps, window_seconds)
    velocity = np.vstack([np.zeros((1, frames.shape[1]), dtype=np.float32), np.diff(frames, axis=0)])
    stats = np.concatenate(
        [
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.max(axis=0) - frames.min(axis=0),
        ]
    )
    return np.concatenate([frames.reshape(-1), velocity.reshape(-1), stats]).astype(np.float32)


def repair_missing_hand_frames(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
    max_gap_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    max_gap_frames: int = DEFAULT_MAX_MISSING_FRAMES,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return np.zeros((0, FRAME_FEATURE_SIZE), dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME_FEATURE_SIZE:
        return frames

    repaired = frames.copy()
    times = clean_timestamps(timestamps, len(frames)) if timestamps is not None else None
    slots = repaired.reshape(len(repaired), 2, HAND_FEATURE_SIZE)

    for hand_index in range(2):
        present = slots[:, hand_index, 0] > 0.5
        index = 0
        while index < len(present):
            if present[index]:
                index += 1
                continue

            start = index
            while index < len(present) and not present[index]:
                index += 1
            end = index
            gap_len = end - start

            prev_index = start - 1 if start > 0 and present[start - 1] else None
            next_index = end if end < len(present) and present[end] else None
            if prev_index is None and next_index is None:
                continue

            if gap_len > max_gap_frames:
                continue
            if times is not None and gap_duration_seconds(times, start, end) > max_gap_seconds:
                continue

            source_index = prev_index if prev_index is not None else next_index
            if source_index is not None:
                slots[start:end, hand_index, :] = slots[source_index, hand_index, :]

    return repaired


def clean_timestamps(timestamps: Sequence[float] | np.ndarray, expected_length: int) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) != expected_length:
        raise ValueError(f"Expected {expected_length} timestamps, got {len(times)}.")
    if expected_length == 0:
        return times

    times = times.copy()
    times -= times[0]
    for index in range(1, len(times)):
        if times[index] <= times[index - 1]:
            times[index] = times[index - 1] + 1e-4
    return times.astype(np.float32)


def gap_duration_seconds(times: np.ndarray, start: int, end: int) -> float:
    left = times[start - 1] if start > 0 else times[start]
    right = times[end] if end < len(times) else times[end - 1]
    return float(max(0.0, right - left))


def safe_label_dir(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("._") or "unlabeled"


def iter_clip_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("**/*.npz"))


def load_clip(path: str | Path) -> tuple[str, np.ndarray]:
    label, frames, _, _ = load_clip_with_timing(path)
    return label, frames


def load_clip_with_timing(path: str | Path) -> tuple[str, np.ndarray, np.ndarray | None, float | None]:
    with np.load(path, allow_pickle=False) as data:
        label = str(data["label"])
        frames = np.asarray(data["frames"], dtype=np.float32)
        timestamps = load_clip_timestamps(data, len(frames))
        seconds = float(data["seconds"]) if "seconds" in data else None
    return label, frames, timestamps, seconds


def load_clip_timestamps(data: np.lib.npyio.NpzFile, frame_count: int) -> np.ndarray | None:
    if frame_count <= 0:
        return None
    if "frame_times" in data:
        return clean_timestamps(np.asarray(data["frame_times"], dtype=np.float32), frame_count)
    if "timestamps" in data:
        return clean_timestamps(np.asarray(data["timestamps"], dtype=np.float32), frame_count)
    if "fps" in data:
        fps = float(data["fps"])
        if fps > 1e-6:
            return np.arange(frame_count, dtype=np.float32) / fps
    if "seconds" in data:
        seconds = max(float(data["seconds"]), 1e-6)
        return np.linspace(0.0, seconds, num=frame_count, dtype=np.float32)
    return None
