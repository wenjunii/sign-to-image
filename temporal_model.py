from __future__ import annotations

import os
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gesture_rules import GestureResult
from temporal_features import (
    DEFAULT_TARGET_FRAMES,
    FRAME_FEATURE_SIZE,
    extract_frame_features,
    temporal_feature_vector,
)


@dataclass
class TemporalModelInfo:
    path: str
    labels: list[str]
    target_frames: int


class TemporalModelRecognizer:
    def __init__(self, model_path: str, min_buffer_ratio: float = 0.6):
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
        self.frame_feature_size = int(payload.get("frame_feature_size", FRAME_FEATURE_SIZE))
        if self.frame_feature_size != FRAME_FEATURE_SIZE:
            raise RuntimeError(
                f"Temporal model expects frame size {self.frame_feature_size}, "
                f"but this app produces {FRAME_FEATURE_SIZE}."
            )

        self.buffer: deque[np.ndarray] = deque(maxlen=self.target_frames)
        self.min_buffer_frames = max(4, int(self.target_frames * min_buffer_ratio))
        self.info = TemporalModelInfo(model_path, self.labels, self.target_frames)

    def recognize_frame(self, hand_landmarks: Sequence[object], handedness: Sequence[object]) -> GestureResult | None:
        features = extract_frame_features(hand_landmarks, handedness)
        if not np.any(features):
            self.buffer.clear()
            return None

        self.buffer.append(features)
        if len(self.buffer) < self.min_buffer_frames:
            return None

        vector = temporal_feature_vector(np.asarray(self.buffer, dtype=np.float32), self.target_frames)
        label, confidence = self._predict(vector)
        if label is None:
            return None

        return GestureResult(
            label=label,
            confidence=confidence,
            kind=label_kind(label),
            pose="TEMPORAL_MODEL",
            debug=f"temporal model window={len(self.buffer)}/{self.target_frames}",
        )

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
