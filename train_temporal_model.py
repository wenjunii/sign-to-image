from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from temporal_features import (
    DEFAULT_MAX_MISSING_SECONDS,
    DEFAULT_TARGET_FRAMES,
    DEFAULT_TARGET_SECONDS,
    FRAME_FEATURE_SIZE,
    iter_clip_files,
    load_clip_with_timing,
    temporal_feature_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal sign classifier from collected landmark clips.")
    parser.add_argument("--data", default="data/clips", help="Directory created by collect_gesture_clips.py.")
    parser.add_argument("--output", default="models/temporal_sign_model.pkl", help="Path for the trained model.")
    parser.add_argument("--frames", type=int, default=DEFAULT_TARGET_FRAMES, help="Resampled frame count per clip.")
    parser.add_argument("--seconds", type=float, default=DEFAULT_TARGET_SECONDS, help="Time window represented by each clip.")
    parser.add_argument(
        "--max-missing-seconds",
        type=float,
        default=DEFAULT_MAX_MISSING_SECONDS,
        help="Short missing-hand gaps up to this duration are forward-filled.",
    )
    parser.add_argument("--estimators", type=int, default=400, help="Number of ExtraTrees estimators.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split when each class has enough clips.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips = load_dataset(args.data, args.frames, args.seconds, args.max_missing_seconds)
    if not clips:
        print(f"No clips found in {args.data}. Record clips first.")
        return 1

    labels = [label for label, _ in clips]
    counts = Counter(labels)
    if len(counts) < 2:
        print("Need at least two labels to train a classifier.")
        print_counts(counts)
        return 1

    x = np.vstack([features for _, features in clips])
    y = np.asarray(labels)

    eval_count = max(len(counts), int(np.ceil(len(y) * args.test_size)))
    eval_count = min(eval_count, len(y) - len(counts))
    can_split = min(counts.values()) >= 2 and eval_count >= len(counts)
    if can_split:
        x_train, x_eval, y_train, y_eval = train_test_split(
            x,
            y,
            test_size=eval_count,
            random_state=args.seed,
            stratify=y,
        )
        eval_name = "validation"
    else:
        x_train, y_train = x, y
        x_eval, y_eval = x, y
        eval_name = "training"
        print("Not enough clips per class for a stratified validation split; reporting training metrics.")

    model = ExtraTreesClassifier(
        n_estimators=args.estimators,
        random_state=args.seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_eval)
    accuracy = float(accuracy_score(y_eval, predictions))
    report = classification_report(y_eval, predictions, zero_division=0)

    payload = {
        "model_version": 1,
        "model_type": "sklearn_extra_trees_temporal_clip",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "labels": sorted(counts),
        "label_counts": dict(sorted(counts.items())),
        "target_frames": args.frames,
        "target_seconds": args.seconds,
        "max_missing_seconds": args.max_missing_seconds,
        "frame_feature_size": FRAME_FEATURE_SIZE,
        "feature_kind": "mediapipe_two_hand_temporal_v1",
        "metrics": {
            "eval_name": eval_name,
            "accuracy": accuracy,
            "clip_count": int(len(y)),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print_counts(counts)
    print(f"\n{eval_name.title()} accuracy: {accuracy:.3f}")
    print(report)
    print(f"Saved model to {output_path}")
    print(json.dumps(payload["metrics"], indent=2))
    return 0


def load_dataset(
    data_dir: str,
    target_frames: int,
    target_seconds: float,
    max_missing_seconds: float,
) -> list[tuple[str, np.ndarray]]:
    clips = []
    for path in iter_clip_files(data_dir):
        try:
            label, frames, timestamps, _ = load_clip_with_timing(path)
            clips.append(
                (
                    label,
                    temporal_feature_vector(
                        frames,
                        target_frames=target_frames,
                        timestamps=timestamps,
                        window_seconds=target_seconds,
                        max_missing_seconds=max_missing_seconds,
                    ),
                )
            )
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return clips


def print_counts(counts: Counter[str]) -> None:
    print("\nClip counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
