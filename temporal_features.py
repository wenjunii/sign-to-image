from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HAND_LANDMARK_COUNT = 21
POINT_DIMS = 3
HAND_FEATURE_SIZE = 1 + HAND_LANDMARK_COUNT * POINT_DIMS
FRAME_FEATURE_SIZE = HAND_FEATURE_SIZE * 2
POSE_LANDMARK_COUNT = 33
FACE_LANDMARK_COUNT = 468
POSE_FEATURE_SIZE = 1 + POSE_LANDMARK_COUNT * POINT_DIMS
FACE_FEATURE_SIZE = 1 + FACE_LANDMARK_COUNT * POINT_DIMS
HOLISTIC_FRAME_FEATURE_SIZE = FRAME_FEATURE_SIZE + POSE_FEATURE_SIZE + FACE_FEATURE_SIZE
FEATURE_KIND_HANDS = "mediapipe_two_hand_temporal_v1"
FEATURE_KIND_HOLISTIC = "mediapipe_holistic_temporal_v1"
DEFAULT_TARGET_FRAMES = 48
DEFAULT_TARGET_SECONDS = 1.4
DEFAULT_MAX_MISSING_SECONDS = 0.18
DEFAULT_MAX_MISSING_FRAMES = 4


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    frame_size: int
    kind: str


HANDS_FEATURE_SPEC = FeatureSpec("hands", FRAME_FEATURE_SIZE, FEATURE_KIND_HANDS)
HOLISTIC_FEATURE_SPEC = FeatureSpec("holistic", HOLISTIC_FRAME_FEATURE_SIZE, FEATURE_KIND_HOLISTIC)
FEATURE_SPECS = {
    HANDS_FEATURE_SPEC.name: HANDS_FEATURE_SPEC,
    HOLISTIC_FEATURE_SPEC.name: HOLISTIC_FEATURE_SPEC,
}


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


def extract_holistic_frame_features(results: object) -> np.ndarray:
    hands = extract_holistic_hand_features(results)
    pose = extract_pose_features(getattr(results, "pose_landmarks", None))
    face = extract_face_features(getattr(results, "face_landmarks", None))
    return np.concatenate([hands, pose, face]).astype(np.float32)


def extract_holistic_hand_features(results: object) -> np.ndarray:
    slots = np.zeros((2, HAND_FEATURE_SIZE), dtype=np.float32)
    left = getattr(results, "left_hand_landmarks", None)
    right = getattr(results, "right_hand_landmarks", None)
    if left is not None:
        slots[0, :] = single_hand_features(left)
    if right is not None:
        slots[1, :] = single_hand_features(right)
    return slots.reshape(-1)


def single_hand_features(landmarks: object) -> np.ndarray:
    features = np.zeros(HAND_FEATURE_SIZE, dtype=np.float32)
    points = landmark_points(landmarks)
    if len(points) != HAND_LANDMARK_COUNT:
        return features
    features[0] = 1.0
    features[1:] = normalize_hand(points).reshape(-1)
    return features


def extract_pose_features(landmarks: object | None) -> np.ndarray:
    features = np.zeros(POSE_FEATURE_SIZE, dtype=np.float32)
    if landmarks is None:
        return features
    points = landmark_points(landmarks)
    if len(points) != POSE_LANDMARK_COUNT:
        return features
    features[0] = 1.0
    features[1:] = normalize_body_points(points).reshape(-1)
    return features


def extract_face_features(landmarks: object | None) -> np.ndarray:
    features = np.zeros(FACE_FEATURE_SIZE, dtype=np.float32)
    if landmarks is None:
        return features
    points = landmark_points(landmarks)
    if len(points) != FACE_LANDMARK_COUNT:
        return features
    features[0] = 1.0
    features[1:] = normalize_body_points(points).reshape(-1)
    return features


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


def normalize_body_points(points: Sequence[tuple[float, float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center = pts.mean(axis=0)
    scale = float(np.std(pts[:, :2]))
    if scale < 1e-6:
        scale = 1.0
    return (pts - center) / scale


def get_feature_spec(name: str) -> FeatureSpec:
    key = name.strip().lower()
    if key not in FEATURE_SPECS:
        raise ValueError(f"Unknown landmark pipeline '{name}'. Choose one of: {', '.join(sorted(FEATURE_SPECS))}.")
    return FEATURE_SPECS[key]


def resample_sequence(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    target_frames: int,
    timestamps: Sequence[float] | np.ndarray | None = None,
    window_seconds: float | None = None,
    frame_feature_size: int = FRAME_FEATURE_SIZE,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return np.zeros((target_frames, frame_feature_size), dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected 2D frame sequence, got shape {frames.shape}.")
    if frames.shape[1] != frame_feature_size:
        raise ValueError(f"Expected frame feature size {frame_feature_size}, got {frames.shape[1]}.")
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
    frame_feature_size: int = FRAME_FEATURE_SIZE,
    feature_kind: str = FEATURE_KIND_HANDS,
) -> np.ndarray:
    repaired = repair_missing_frames(sequence, timestamps, max_missing_seconds, max_missing_frames, feature_kind)
    frames = resample_sequence(repaired, target_frames, timestamps, window_seconds, frame_feature_size)
    velocity = np.vstack([np.zeros((1, frames.shape[1]), dtype=np.float32), np.diff(frames, axis=0)])
    stats = np.concatenate(
        [
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.max(axis=0) - frames.min(axis=0),
        ]
    )
    return np.concatenate([frames.reshape(-1), velocity.reshape(-1), stats]).astype(np.float32)


def repair_missing_frames(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
    max_gap_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    max_gap_frames: int = DEFAULT_MAX_MISSING_FRAMES,
    feature_kind: str = FEATURE_KIND_HANDS,
) -> np.ndarray:
    if feature_kind == FEATURE_KIND_HOLISTIC:
        return repair_missing_presence_slots(
            sequence,
            slot_layout=[
                (0, HAND_FEATURE_SIZE),
                (HAND_FEATURE_SIZE, HAND_FEATURE_SIZE),
                (FRAME_FEATURE_SIZE, POSE_FEATURE_SIZE),
                (FRAME_FEATURE_SIZE + POSE_FEATURE_SIZE, FACE_FEATURE_SIZE),
            ],
            timestamps=timestamps,
            max_gap_seconds=max_gap_seconds,
            max_gap_frames=max_gap_frames,
        )
    return repair_missing_hand_frames(sequence, timestamps, max_gap_seconds, max_gap_frames)


def repair_missing_hand_frames(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray | None = None,
    max_gap_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    max_gap_frames: int = DEFAULT_MAX_MISSING_FRAMES,
) -> np.ndarray:
    return repair_missing_presence_slots(
        sequence,
        slot_layout=[(0, HAND_FEATURE_SIZE), (HAND_FEATURE_SIZE, HAND_FEATURE_SIZE)],
        timestamps=timestamps,
        max_gap_seconds=max_gap_seconds,
        max_gap_frames=max_gap_frames,
    )


def repair_missing_presence_slots(
    sequence: Sequence[Sequence[float]] | np.ndarray,
    slot_layout: Sequence[tuple[int, int]],
    timestamps: Sequence[float] | np.ndarray | None = None,
    max_gap_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    max_gap_frames: int = DEFAULT_MAX_MISSING_FRAMES,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return frames
    if frames.ndim != 2:
        return frames

    repaired = frames.copy()
    times = clean_timestamps(timestamps, len(frames)) if timestamps is not None else None
    for start_col, slot_size in slot_layout:
        if start_col + slot_size > frames.shape[1]:
            continue
        repair_slot_gap(repaired[:, start_col : start_col + slot_size], times, max_gap_seconds, max_gap_frames)
    return repaired


def repair_slot_gap(
    slot_view: np.ndarray,
    times: np.ndarray | None,
    max_gap_seconds: float,
    max_gap_frames: int,
) -> None:
    present = slot_view[:, 0] > 0.5
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
            slot_view[start:end, :] = slot_view[source_index, :]


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
