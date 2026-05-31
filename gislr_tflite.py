from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gesture_rules import GestureResult
from temporal_features import DEFAULT_MAX_MISSING_SECONDS, clean_timestamps, landmark_points, resample_sequence
from temporal_model import label_kind


GISLR_FACE_LANDMARK_COUNT = 468
GISLR_HAND_LANDMARK_COUNT = 21
GISLR_POSE_LANDMARK_COUNT = 33
GISLR_LANDMARK_COUNT = (
    GISLR_FACE_LANDMARK_COUNT
    + GISLR_HAND_LANDMARK_COUNT
    + GISLR_POSE_LANDMARK_COUNT
    + GISLR_HAND_LANDMARK_COUNT
)
GISLR_POINT_DIMS = 3
GISLR_FRAME_FEATURE_SIZE = GISLR_LANDMARK_COUNT * GISLR_POINT_DIMS
GISLR_FACE_OFFSET = 0
GISLR_LEFT_HAND_OFFSET = GISLR_FACE_OFFSET + GISLR_FACE_LANDMARK_COUNT
GISLR_POSE_OFFSET = GISLR_LEFT_HAND_OFFSET + GISLR_HAND_LANDMARK_COUNT
GISLR_RIGHT_HAND_OFFSET = GISLR_POSE_OFFSET + GISLR_POSE_LANDMARK_COUNT
DEFAULT_GISLR_TARGET_FRAMES = 64
DEFAULT_GISLR_WINDOW_SECONDS = 1.6
DEFAULT_GISLR_THREADS = 4


@dataclass
class GislrModelInfo:
    path: str
    label_map_path: str
    labels: list[str]
    target_frames: int
    window_seconds: float
    runtime: str
    threads: int


def extract_gislr_frame_landmarks(results: object | None) -> np.ndarray:
    """Return GISLR/PopSign landmark order: face, left hand, pose, right hand."""
    frame = np.zeros((GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
    if results is None:
        return frame

    fill_landmark_block(
        frame,
        GISLR_FACE_OFFSET,
        GISLR_FACE_LANDMARK_COUNT,
        getattr(results, "face_landmarks", None),
    )
    fill_landmark_block(
        frame,
        GISLR_LEFT_HAND_OFFSET,
        GISLR_HAND_LANDMARK_COUNT,
        getattr(results, "left_hand_landmarks", None),
    )
    fill_landmark_block(
        frame,
        GISLR_POSE_OFFSET,
        GISLR_POSE_LANDMARK_COUNT,
        getattr(results, "pose_landmarks", None),
    )
    fill_landmark_block(
        frame,
        GISLR_RIGHT_HAND_OFFSET,
        GISLR_HAND_LANDMARK_COUNT,
        getattr(results, "right_hand_landmarks", None),
    )
    return frame


def fill_landmark_block(frame: np.ndarray, offset: int, count: int, landmarks: object | None) -> None:
    if landmarks is None:
        return
    points = landmark_points(landmarks)
    if len(points) != count:
        return
    frame[offset : offset + count, :] = np.asarray(points, dtype=np.float32)


def gislr_hand_present(frame: np.ndarray) -> bool:
    points = np.asarray(frame, dtype=np.float32)
    if points.shape != (GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS):
        return bool(np.any(points))
    left = points[GISLR_LEFT_HAND_OFFSET : GISLR_LEFT_HAND_OFFSET + GISLR_HAND_LANDMARK_COUNT]
    right = points[GISLR_RIGHT_HAND_OFFSET : GISLR_RIGHT_HAND_OFFSET + GISLR_HAND_LANDMARK_COUNT]
    return bool(np.any(left) or np.any(right))


def resample_gislr_sequence(
    sequence: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    target_frames: int = DEFAULT_GISLR_TARGET_FRAMES,
    timestamps: Sequence[float] | np.ndarray | None = None,
    window_seconds: float | None = DEFAULT_GISLR_WINDOW_SECONDS,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return np.zeros((target_frames, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS):
        raise ValueError(f"Expected GISLR sequence shape (frames, 543, 3), got {frames.shape}.")

    flat = frames.reshape(len(frames), GISLR_FRAME_FEATURE_SIZE)
    resampled = resample_sequence(
        flat,
        target_frames,
        timestamps=timestamps,
        window_seconds=window_seconds,
        frame_feature_size=GISLR_FRAME_FEATURE_SIZE,
    )
    return resampled.reshape(target_frames, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS)


def load_gislr_label_map(path: str) -> list[str]:
    if not path:
        raise RuntimeError("GISLR/PopSign backend needs SIGN_GISLR_LABEL_MAP or --label-map.")
    if not os.path.exists(path):
        raise RuntimeError(f"GISLR/PopSign label map not found at {path}.")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return [str(item) for item in payload]

    if isinstance(payload, dict) and "labels" in payload and isinstance(payload["labels"], list):
        return [str(item) for item in payload["labels"]]

    if isinstance(payload, dict):
        if all(str(key).isdigit() for key in payload):
            ordered = sorted(payload.items(), key=lambda item: int(item[0]))
            return [str(value) for _, value in ordered]
        if all(isinstance(value, int) for value in payload.values()):
            ordered = sorted(payload.items(), key=lambda item: int(item[1]))
            return [str(key) for key, _ in ordered]

    raise RuntimeError(
        "Unsupported GISLR/PopSign label map. Use a list, an index-to-label dict, "
        "or Kaggle's sign_to_prediction_index_map.json format."
    )


def gislr_token_label(label: str) -> str:
    clean = label.strip()
    if not clean:
        return ""
    upper = clean.upper()
    if upper in {"SPACE", "SEND", "CLEAR", "BACKSPACE"}:
        return upper
    if upper.startswith("WORD:"):
        return clean
    if len(clean) == 1 and clean.isalpha():
        return clean.upper()
    return "WORD:" + clean.replace("_", " ")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        return np.zeros_like(values)

    total = float(values.sum())
    if values.min() >= 0.0 and values.max() <= 1.0 and 0.98 <= total <= 1.02:
        return values

    shifted = values - float(values.max())
    exp = np.exp(shifted)
    denom = float(exp.sum())
    if denom <= 1e-12:
        return np.zeros_like(values)
    return (exp / denom).astype(np.float32)


class GislrTfliteModel:
    def __init__(self, model_path: str, num_threads: int = DEFAULT_GISLR_THREADS):
        if not os.path.exists(model_path):
            raise RuntimeError(f"GISLR/PopSign TFLite model not found at {model_path}.")

        interpreter_cls, runtime = load_tflite_interpreter()
        self.num_threads = max(1, int(num_threads))
        self.interpreter = create_tflite_interpreter(interpreter_cls, model_path, self.num_threads)
        self.runtime = runtime
        self.signature_runner = None
        self.signature_input = None
        self.signature_output = None
        self.input_details = []
        self.output_details = []
        self.cached_input_variant: str | None = None

        signatures = self.interpreter.get_signature_list()
        if signatures:
            signature_name = "serving_default" if "serving_default" in signatures else next(iter(signatures))
            signature = signatures[signature_name]
            inputs = signature.get("inputs", [])
            outputs = signature.get("outputs", [])
            if len(inputs) != 1:
                raise RuntimeError("GISLR/PopSign TFLite signatures with more than one input are not supported.")
            if not outputs:
                raise RuntimeError("GISLR/PopSign TFLite signature has no outputs.")
            self.signature_input = inputs[0]
            self.signature_output = outputs[0]
            self.signature_runner = self.interpreter.get_signature_runner(signature_name)
        else:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            if len(self.input_details) != 1:
                raise RuntimeError("GISLR/PopSign TFLite models with more than one input are not supported.")
            if not self.output_details:
                raise RuntimeError("GISLR/PopSign TFLite model has no outputs.")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        if self.signature_runner is not None:
            return self._predict_signature(sequence)
        return self._predict_raw_interpreter(sequence)

    def _predict_signature(self, sequence: np.ndarray) -> np.ndarray:
        errors = []
        for variant, candidate in self._ordered_input_candidates(sequence):
            try:
                outputs = self.signature_runner(**{self.signature_input: candidate})
                self.cached_input_variant = variant
                output = outputs[self.signature_output]
                return normalize_scores(output)
            except Exception as exc:  # pragma: no cover - depends on external model shape.
                errors.append(f"{variant}: {exc}")
        raise RuntimeError("GISLR/PopSign TFLite signature inference failed. " + " | ".join(errors[-3:]))

    def _predict_raw_interpreter(self, sequence: np.ndarray) -> np.ndarray:
        errors = []
        input_index = self.input_details[0]["index"]
        for variant, candidate in self._ordered_input_candidates(sequence):
            try:
                self.interpreter.resize_tensor_input(input_index, candidate.shape, strict=False)
                self.interpreter.allocate_tensors()
                self.interpreter.set_tensor(input_index, candidate)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]["index"])
                self.cached_input_variant = variant
                return normalize_scores(output)
            except Exception as exc:  # pragma: no cover - depends on external model shape.
                errors.append(f"{variant}: {exc}")
        raise RuntimeError("GISLR/PopSign TFLite inference failed. " + " | ".join(errors[-3:]))

    def _ordered_input_candidates(self, sequence: np.ndarray) -> list[tuple[str, np.ndarray]]:
        candidates = input_candidates(sequence)
        if self.cached_input_variant is None:
            return candidates
        return sorted(candidates, key=lambda item: 0 if item[0] == self.cached_input_variant else 1)


def input_candidates(sequence: np.ndarray) -> list[tuple[str, np.ndarray]]:
    frames = np.asarray(sequence, dtype=np.float32)
    flat = frames.reshape(frames.shape[0], GISLR_FRAME_FEATURE_SIZE)
    return [
        ("frames", frames),
        ("batched_frames", frames[np.newaxis, ...]),
        ("flat_frames", flat),
        ("batched_flat_frames", flat[np.newaxis, ...]),
    ]


def load_tflite_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "GISLR/PopSign backend needs a TensorFlow Lite runtime. "
                "Install either tensorflow or tflite-runtime in the project venv."
            ) from exc
        return Interpreter, "tensorflow.lite"
    return Interpreter, "tflite_runtime"


def create_tflite_interpreter(interpreter_cls, model_path: str, num_threads: int):
    try:
        return interpreter_cls(model_path=model_path, num_threads=max(1, int(num_threads)))
    except TypeError:
        return interpreter_cls(model_path=model_path)


class GislrTfliteRecognizer:
    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        frame_extractor,
        target_frames: int = DEFAULT_GISLR_TARGET_FRAMES,
        window_seconds: float = DEFAULT_GISLR_WINDOW_SECONDS,
        num_threads: int = DEFAULT_GISLR_THREADS,
        min_buffer_ratio: float = 0.6,
        max_missing_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    ):
        self.model = GislrTfliteModel(model_path, num_threads=num_threads)
        self.labels = load_gislr_label_map(label_map_path)
        self.frame_extractor = frame_extractor
        self.target_frames = int(target_frames)
        self.window_seconds = float(window_seconds)
        self.max_missing_seconds = float(max_missing_seconds)

        sample = np.asarray(self.frame_extractor(), dtype=np.float32)
        if sample.shape != (GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS):
            raise RuntimeError(f"GISLR/PopSign backend expects frame shape (543, 3), got {sample.shape}.")

        self.buffer: deque[tuple[float, np.ndarray]] = deque()
        self.min_buffer_frames = max(4, int(self.target_frames * min_buffer_ratio))
        self.min_buffer_seconds = self.window_seconds * min_buffer_ratio
        self.max_buffer_frames = self.target_frames * 4
        self.last_hand_at: float | None = None
        self.info = GislrModelInfo(
            path=model_path,
            label_map_path=label_map_path,
            labels=self.labels,
            target_frames=self.target_frames,
            window_seconds=self.window_seconds,
            runtime=self.model.runtime,
            threads=self.model.num_threads,
        )

    def recognize_frame(self) -> GestureResult | None:
        now = time.monotonic()
        frame = np.asarray(self.frame_extractor(), dtype=np.float32)
        if frame.shape != (GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS):
            return None

        if gislr_hand_present(frame):
            self.last_hand_at = now
        elif self.last_hand_at is None or now - self.last_hand_at > self.max_missing_seconds:
            self.buffer.clear()
            return None

        self.buffer.append((now, frame))
        self._prune_buffer(now)
        if len(self.buffer) < self.min_buffer_frames:
            return None

        timestamps = np.asarray([item[0] for item in self.buffer], dtype=np.float64)
        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        if duration < self.min_buffer_seconds:
            return None

        frames = np.asarray([item[1] for item in self.buffer], dtype=np.float32)
        frames = forward_fill_short_hand_gaps(frames, timestamps, self.max_missing_seconds)
        model_input = resample_gislr_sequence(
            frames,
            target_frames=self.target_frames,
            timestamps=timestamps,
            window_seconds=self.window_seconds,
        )
        scores = self.model.predict(model_input)
        if scores.size == 0:
            return None

        best_index = int(np.argmax(scores))
        confidence = float(scores[best_index])
        label = self.labels[best_index] if best_index < len(self.labels) else f"index_{best_index}"
        token = gislr_token_label(label)
        if not token:
            return None

        return GestureResult(
            label=token,
            confidence=confidence,
            kind=label_kind(token),
            pose="GISLR_TFLITE",
            debug=f"gislr window={duration:.2f}s/{self.window_seconds:.2f}s frames={len(self.buffer)}",
        )

    def _prune_buffer(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
        while len(self.buffer) > self.max_buffer_frames:
            self.buffer.popleft()


def forward_fill_short_hand_gaps(
    sequence: np.ndarray,
    timestamps: Sequence[float] | np.ndarray,
    max_gap_seconds: float,
) -> np.ndarray:
    frames = np.asarray(sequence, dtype=np.float32)
    if frames.size == 0:
        return frames

    repaired = frames.copy()
    times = clean_timestamps(timestamps, len(frames))
    hand_ranges = [
        (GISLR_LEFT_HAND_OFFSET, GISLR_LEFT_HAND_OFFSET + GISLR_HAND_LANDMARK_COUNT),
        (GISLR_RIGHT_HAND_OFFSET, GISLR_RIGHT_HAND_OFFSET + GISLR_HAND_LANDMARK_COUNT),
    ]
    for start, end in hand_ranges:
        present = np.any(repaired[:, start:end, :], axis=(1, 2))
        repair_gislr_hand_gap(repaired[:, start:end, :], present, times, max_gap_seconds)
    return repaired


def repair_gislr_hand_gap(
    hand_view: np.ndarray,
    present: np.ndarray,
    times: np.ndarray,
    max_gap_seconds: float,
) -> None:
    index = 0
    while index < len(present):
        if present[index]:
            index += 1
            continue

        start = index
        while index < len(present) and not present[index]:
            index += 1
        end = index

        prev_index = start - 1 if start > 0 and present[start - 1] else None
        next_index = end if end < len(present) and present[end] else None
        if prev_index is None and next_index is None:
            continue

        left = times[start - 1] if start > 0 else times[start]
        right = times[end] if end < len(times) else times[end - 1]
        if float(max(0.0, right - left)) > max_gap_seconds:
            continue

        source_index = prev_index if prev_index is not None else next_index
        if source_index is not None:
            hand_view[start:end, :, :] = hand_view[source_index, :, :]
