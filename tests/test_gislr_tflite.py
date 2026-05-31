import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gislr_tflite import (
    GISLR_FACE_OFFSET,
    GISLR_LANDMARK_COUNT,
    GISLR_LEFT_HAND_OFFSET,
    GISLR_POINT_DIMS,
    GISLR_POSE_OFFSET,
    GISLR_RIGHT_HAND_OFFSET,
    extract_gislr_frame_landmarks,
    forward_fill_short_hand_gaps,
    gislr_hand_present,
    gislr_token_label,
    input_candidates,
    load_gislr_label_map,
    normalize_scores,
    resample_gislr_sequence,
)


class FakePoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeLandmarks:
    def __init__(self, count, base):
        self.landmark = [FakePoint(base + index, base + index + 0.1, base + index + 0.2) for index in range(count)]


class FakeHolisticResults:
    def __init__(self):
        self.face_landmarks = FakeLandmarks(468, 100.0)
        self.left_hand_landmarks = FakeLandmarks(21, 200.0)
        self.pose_landmarks = FakeLandmarks(33, 300.0)
        self.right_hand_landmarks = FakeLandmarks(21, 400.0)


class GislrTfliteTests(unittest.TestCase):
    def test_extract_gislr_frame_landmarks_uses_kaggle_order(self):
        frame = extract_gislr_frame_landmarks(FakeHolisticResults())

        self.assertEqual(frame.shape, (GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS))
        self.assertAlmostEqual(float(frame[GISLR_FACE_OFFSET, 0]), 100.0)
        self.assertAlmostEqual(float(frame[GISLR_LEFT_HAND_OFFSET, 0]), 200.0)
        self.assertAlmostEqual(float(frame[GISLR_POSE_OFFSET, 0]), 300.0)
        self.assertAlmostEqual(float(frame[GISLR_RIGHT_HAND_OFFSET, 0]), 400.0)

    def test_gislr_hand_present_ignores_face_and_pose(self):
        frame = np.zeros((GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
        frame[GISLR_FACE_OFFSET, 0] = 1.0
        frame[GISLR_POSE_OFFSET, 0] = 1.0
        self.assertFalse(gislr_hand_present(frame))

        frame[GISLR_LEFT_HAND_OFFSET, 0] = 1.0
        self.assertTrue(gislr_hand_present(frame))

    def test_resample_gislr_sequence_keeps_shape(self):
        sequence = np.zeros((2, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
        sequence[1, GISLR_LEFT_HAND_OFFSET, 0] = 10.0

        resampled = resample_gislr_sequence(sequence, target_frames=5)

        self.assertEqual(resampled.shape, (5, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS))
        self.assertAlmostEqual(float(resampled[-1, GISLR_LEFT_HAND_OFFSET, 0]), 10.0)

    def test_load_gislr_label_map_accepts_kaggle_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps({"hello": 2, "river": 0, "store": 1}), encoding="utf-8")

            self.assertEqual(load_gislr_label_map(str(path)), ["river", "store", "hello"])

    def test_gislr_token_label_converts_words_to_buffer_tokens(self):
        self.assertEqual(gislr_token_label("river"), "WORD:river")
        self.assertEqual(gislr_token_label("ice_cream"), "WORD:ice cream")
        self.assertEqual(gislr_token_label("SPACE"), "SPACE")
        self.assertEqual(gislr_token_label("a"), "A")

    def test_normalize_scores_keeps_probabilities_and_softmaxes_logits(self):
        probs = normalize_scores(np.asarray([0.2, 0.7, 0.1], dtype=np.float32))
        logits = normalize_scores(np.asarray([0.0, 2.0, 1.0], dtype=np.float32))

        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(logits.sum()), 1.0, places=5)
        self.assertEqual(int(np.argmax(logits)), 1)

    def test_input_candidates_include_common_tflite_shapes(self):
        sequence = np.zeros((4, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
        candidates = dict(input_candidates(sequence))

        self.assertEqual(candidates["frames"].shape, (4, 543, 3))
        self.assertEqual(candidates["batched_frames"].shape, (1, 4, 543, 3))
        self.assertEqual(candidates["flat_frames"].shape, (4, 1629))
        self.assertEqual(candidates["batched_flat_frames"].shape, (1, 4, 1629))

    def test_forward_fill_short_hand_gaps_repairs_occlusion(self):
        sequence = np.zeros((3, GISLR_LANDMARK_COUNT, GISLR_POINT_DIMS), dtype=np.float32)
        sequence[0, GISLR_LEFT_HAND_OFFSET, 0] = 5.0
        sequence[2, GISLR_LEFT_HAND_OFFSET, 0] = 9.0

        repaired = forward_fill_short_hand_gaps(sequence, [0.0, 0.05, 0.1], max_gap_seconds=0.2)

        self.assertEqual(float(repaired[1, GISLR_LEFT_HAND_OFFSET, 0]), 5.0)


if __name__ == "__main__":
    unittest.main()
