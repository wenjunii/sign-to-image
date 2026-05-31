from __future__ import annotations

import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gesture_rules import GestureResult
from temporal_features import (
    DEFAULT_MAX_MISSING_SECONDS,
    DEFAULT_TARGET_FRAMES,
    DEFAULT_TARGET_SECONDS,
    FRAME_FEATURE_SIZE,
    extract_frame_features,
    temporal_feature_vector,
)


@dataclass
class TemporalModelInfo:
    path: str
    labels: list[str]
    target_frames: int
    target_seconds: float


class TemporalModelRecognizer:
    def __init__(
        self,
        model_path: str,
        min_buffer_ratio: float = 0.6,
        max_missing_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
    ):
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Temporal model not found at {model_path}. "
                "Collect clips and run train_temporal_model.py first."
            )

        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.labels = [str(label) for label in payload.get("labels", [])]
        self.target_frames = int(payload.get("target_frames", DEFAULT_TARGET_FRAMES))
        self.target_seconds = float(payload.get("target_seconds", DEFAULT_TARGET_SECONDS))
        self.max_missing_seconds = float(payload.get("max_missing_seconds", max_missing_seconds))
        self.frame_feature_size = int(payload.get("frame_feature_size", FRAME_FEATURE_SIZE))
        if self.frame_feature_size != FRAME_FEATURE_SIZE:
            raise RuntimeError(
                f"Temporal model expects frame size {self.frame_feature_size}, "
                f"but this app produces {FRAME_FEATURE_SIZE}."
            )

        self.buffer: deque[tuple[float, np.ndarray]] = deque()
        self.min_buffer_frames = max(4, int(self.target_frames * min_buffer_ratio))
        self.min_buffer_seconds = self.target_seconds * min_buffer_ratio
        self.max_buffer_frames = self.target_frames * 4
        self.last_hand_at: float | None = None
        self.info = TemporalModelInfo(model_path, self.labels, self.target_frames, self.target_seconds)

    def recognize_frame(self, hand_landmarks: Sequence[object], handedness: Sequence[object]) -> GestureResult | None:
        now = time.monotonic()
        features = extract_frame_features(hand_landmarks, handedness)
        has_hand = bool(np.any(features))

        if has_hand:
            self.last_hand_at = now
        elif self.last_hand_at is None or now - self.last_hand_at > self.max_missing_seconds:
            self.buffer.clear()
            return None

        self.buffer.append((now, features))
        self._prune_buffer(now)
        if len(self.buffer) < self.min_buffer_frames:
            return None

        timestamps = np.asarray([item[0] for item in self.buffer], dtype=np.float64)
        frames = np.asarray([item[1] for item in self.buffer], dtype=np.float32)
        if not np.any(frames):
            return None

        duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        if duration < self.min_buffer_seconds:
            return None

        vector = temporal_feature_vector(
            frames,
            target_frames=self.target_frames,
            timestamps=timestamps,
            window_seconds=self.target_seconds,
            max_missing_seconds=self.max_missing_seconds,
        )
        label, confidence = self._predict(vector)
        if label is None:
            return None

        return GestureResult(
            label=label,
            confidence=confidence,
            kind=label_kind(label),
            pose="TEMPORAL_MODEL",
            debug=f"temporal model window={duration:.2f}s/{self.target_seconds:.2f}s frames={len(self.buffer)}",
        )

    def _prune_buffer(self, now: float) -> None:
        cutoff = now - self.target_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()
        while len(self.buffer) > self.max_buffer_frames:
            self.buffer.popleft()

    def _predict(self, vector: np.ndarray) -> tuple[str | None, float]:
        sample = vector.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(sample)[0]
            best_index = int(np.argmax(probabilities))
            classes = [str(item) for item in self.model.classes_]
            return classes[best_index], float(probabilities[best_index])

        prediction = self.model.predict(sample)
        if len(prediction) == 0:
            return None, 0.0
        return str(prediction[0]), 1.0


def label_kind(label: str) -> str:
    upper = label.upper()
    if len(label) == 1 and label.isalpha():
        return "letter"
    if upper.startswith("WORD:"):
        return "word"
    if upper in {"SPACE", "SEND", "CLEAR", "BACKSPACE"}:
        return "command"
    return "word"
