import unittest

import numpy as np

from temporal_features import (
    DEFAULT_TARGET_FRAMES,
    FRAME_FEATURE_SIZE,
    extract_frame_features,
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
    def __init__(self):
        self.landmark = [FakePoint(index * 0.01, index * 0.02, index * 0.001) for index in range(21)]


class TemporalFeatureTests(unittest.TestCase):
    def test_extract_frame_features_has_fixed_shape(self):
        features = extract_frame_features([FakeLandmarks()], [])

        self.assertEqual(features.shape, (FRAME_FEATURE_SIZE,))
        self.assertTrue(np.any(features))
        self.assertFalse(np.isnan(features).any())

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

    def test_safe_label_dir_and_label_kind(self):
        self.assertEqual(safe_label_dir("WORD:hello world"), "WORD_hello_world")
        self.assertEqual(label_kind("A"), "letter")
        self.assertEqual(label_kind("WORD:hello"), "word")
        self.assertEqual(label_kind("SPACE"), "command")


if __name__ == "__main__":
    unittest.main()
