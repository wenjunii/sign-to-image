import unittest

import numpy as np

from temporal_features import (
    DEFAULT_TARGET_FRAMES,
    FRAME_FEATURE_SIZE,
    HOLISTIC_FRAME_FEATURE_SIZE,
    clean_timestamps,
    extract_frame_features,
    extract_holistic_frame_features,
    get_feature_spec,
    repair_missing_hand_frames,
    resample_sequence,
    safe_label_dir,
    temporal_feature_vector,
)
from temporal_model import label_kind


class FakePoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeLandmarks:
    def __init__(self, count=21):
        self.landmark = [FakePoint(index * 0.01, index * 0.02, index * 0.001) for index in range(count)]


class FakeHolisticResults:
    def __init__(self):
        self.left_hand_landmarks = FakeLandmarks(21)
        self.right_hand_landmarks = FakeLandmarks(21)
        self.pose_landmarks = FakeLandmarks(33)
        self.face_landmarks = FakeLandmarks(468)


class TemporalFeatureTests(unittest.TestCase):
    def test_extract_frame_features_has_fixed_shape(self):
        features = extract_frame_features([FakeLandmarks()], [])

        self.assertEqual(features.shape, (FRAME_FEATURE_SIZE,))
        self.assertTrue(np.any(features))
        self.assertFalse(np.isnan(features).any())

    def test_extract_holistic_features_has_fixed_shape(self):
        features = extract_holistic_frame_features(FakeHolisticResults())

        self.assertEqual(features.shape, (HOLISTIC_FRAME_FEATURE_SIZE,))
        self.assertTrue(np.any(features))
        self.assertFalse(np.isnan(features).any())

    def test_feature_specs_are_named(self):
        self.assertEqual(get_feature_spec("hands").frame_size, FRAME_FEATURE_SIZE)
        self.assertEqual(get_feature_spec("holistic").frame_size, HOLISTIC_FRAME_FEATURE_SIZE)

    def test_resamples_and_flattens_temporal_sequence(self):
        sequence = np.vstack(
            [
                np.zeros(FRAME_FEATURE_SIZE, dtype=np.float32),
                np.ones(FRAME_FEATURE_SIZE, dtype=np.float32),
            ]
        )

        resampled = resample_sequence(sequence, DEFAULT_TARGET_FRAMES)
        vector = temporal_feature_vector(sequence, DEFAULT_TARGET_FRAMES)

        self.assertEqual(resampled.shape, (DEFAULT_TARGET_FRAMES, FRAME_FEATURE_SIZE))
        self.assertEqual(vector.ndim, 1)
        self.assertGreater(vector.shape[0], DEFAULT_TARGET_FRAMES * FRAME_FEATURE_SIZE)

    def test_timestamp_resampling_uses_elapsed_time(self):
        sequence = np.zeros((3, FRAME_FEATURE_SIZE), dtype=np.float32)
        sequence[:, 4] = [0.0, 1.0, 2.0]
        timestamps = np.asarray([0.0, 0.1, 1.0], dtype=np.float32)

        resampled = resample_sequence(sequence, 3, timestamps=timestamps, window_seconds=1.0)

        self.assertAlmostEqual(float(resampled[1, 4]), 1.44, places=2)

    def test_repairs_short_missing_hand_gap(self):
        sequence = np.zeros((3, FRAME_FEATURE_SIZE), dtype=np.float32)
        sequence[0, 0] = 1.0
        sequence[0, 4] = 5.0
        sequence[2, 0] = 1.0
        sequence[2, 4] = 9.0
        timestamps = np.asarray([0.0, 0.05, 0.1], dtype=np.float32)

        repaired = repair_missing_hand_frames(sequence, timestamps=timestamps, max_gap_seconds=0.2)

        self.assertEqual(float(repaired[1, 0]), 1.0)
        self.assertEqual(float(repaired[1, 4]), 5.0)

    def test_cleans_large_monotonic_timestamps_without_losing_delta(self):
        timestamps = clean_timestamps([1_000_000.0, 1_000_000.033], 2)

        self.assertAlmostEqual(float(timestamps[0]), 0.0)
        self.assertAlmostEqual(float(timestamps[1]), 0.033, places=3)

    def test_safe_label_dir_and_label_kind(self):
        self.assertEqual(safe_label_dir("WORD:hello world"), "WORD_hello_world")
        self.assertEqual(label_kind("A"), "letter")
        self.assertEqual(label_kind("WORD:hello"), "word")
        self.assertEqual(label_kind("SPACE"), "command")


if __name__ == "__main__":
    unittest.main()
