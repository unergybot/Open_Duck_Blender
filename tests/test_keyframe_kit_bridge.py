import unittest

import numpy as np

from open_duck_tools.keyframe_kit_bridge import archive_from_keyframe_data
from open_duck_tools.motion import MotionError
from tests.test_motion_import import test_profile


class KeyframeKitBridgeTests(unittest.TestCase):
    def test_converts_dense_keyframe_kit_state_to_native_archive(self):
        profile = test_profile()
        data = {
            "time": np.array([0.0, 0.02, 0.04]),
            "action": np.array([[0.0, 0.0], [0.25, -0.25], [0.5, 0.25]]),
            "body_pos": np.zeros((3, 3, 3)),
            "body_quat": np.tile([1.0, 0.0, 0.0, 0.0], (3, 3, 1)),
            "joint_names": ("hip", "knee"),
            "source_video_sha256": "a" * 64,
            "retarget_config_sha256": "b" * 64,
        }

        archive = archive_from_keyframe_data(data, profile)

        self.assertEqual(archive["fps"].tolist(), [50])
        self.assertEqual(archive["joint_names"].tolist(), ["hip", "knee"])
        self.assertEqual(archive["body_names"].tolist(), ["root", "hip_link", "knee_link"])
        self.assertEqual(archive["joint_pos"].shape, (3, 2))
        self.assertIn("source_video_sha256", str(archive["source_hashes_json"][0]))

    def test_rejects_wrong_joint_order_before_creating_an_action(self):
        profile = test_profile()
        data = {
            "time": np.array([0.0]),
            "action": np.zeros((1, 2)),
            "body_pos": np.zeros((1, 3, 3)),
            "body_quat": np.tile([1.0, 0.0, 0.0, 0.0], (1, 3, 1)),
            "joint_names": ("knee", "hip"),
        }

        with self.assertRaisesRegex(MotionError, "joint_names.*canonical order"):
            archive_from_keyframe_data(data, profile)


if __name__ == "__main__":
    unittest.main()
