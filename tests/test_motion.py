import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_duck_tools.motion import MotionError, build_motion_archive, save_motion_npz


class MotionArchiveTests(unittest.TestCase):
    def fixture(self):
        joint_pos = np.array([[0.0], [0.1], [0.2]], dtype=np.float64)
        body_pos = np.array(
            [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        angles = [0.0, 0.1, 0.2]
        body_quat = np.array(
            [
                [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]]
                for angle in angles
            ],
            dtype=np.float64,
        )
        return joint_pos, body_pos, body_quat

    def test_derives_50hz_linear_and_angular_velocities(self):
        joint_pos, body_pos, body_quat = self.fixture()
        archive = build_motion_archive(
            joint_pos,
            body_pos,
            body_quat,
            fps=50,
            joint_names=("joint",),
            body_names=("body",),
            joint_ranges=((-1.0, 1.0),),
            source_hashes={"mjcf": "abc"},
        )
        np.testing.assert_allclose(archive["joint_vel"][:, 0], [5.0, 5.0, 5.0])
        np.testing.assert_allclose(archive["body_lin_vel_w"][:, 0, 0], [50.0, 50.0, 50.0])
        np.testing.assert_allclose(archive["body_ang_vel_w"][:, 0, 2], [5.0, 5.0, 5.0], atol=1e-10)
        np.testing.assert_allclose(archive["body_quat_w"][:, 0, 0] ** 2 + archive["body_quat_w"][:, 0, 3] ** 2, 1.0)

    def test_makes_quaternion_signs_continuous(self):
        joint_pos, body_pos, body_quat = self.fixture()
        body_quat[1] *= -1
        archive = build_motion_archive(
            joint_pos,
            body_pos,
            body_quat,
            fps=50,
            joint_names=("joint",),
            body_names=("body",),
            joint_ranges=((-1.0, 1.0),),
            source_hashes={},
        )
        dots = np.sum(archive["body_quat_w"][1:] * archive["body_quat_w"][:-1], axis=-1)
        self.assertTrue(np.all(dots >= 0.0))

    def test_rejects_wrong_fps_and_reports_joint_limit_frame(self):
        joint_pos, body_pos, body_quat = self.fixture()
        with self.assertRaisesRegex(MotionError, "50 Hz"):
            build_motion_archive(
                joint_pos,
                body_pos,
                body_quat,
                fps=24,
                joint_names=("joint",),
                body_names=("body",),
                joint_ranges=((-1.0, 1.0),),
                source_hashes={},
            )
        joint_pos[2, 0] = 1.2
        with self.assertRaisesRegex(MotionError, "joint.*frame 2"):
            build_motion_archive(
                joint_pos,
                body_pos,
                body_quat,
                fps=50,
                joint_names=("joint",),
                body_names=("body",),
                joint_ranges=((-1.0, 1.0),),
                source_hashes={},
            )

    def test_saved_npz_has_native_mjlab_keys_and_metadata(self):
        joint_pos, body_pos, body_quat = self.fixture()
        archive = build_motion_archive(
            joint_pos,
            body_pos,
            body_quat,
            fps=50,
            joint_names=("joint",),
            body_names=("body",),
            joint_ranges=((-1.0, 1.0),),
            source_hashes={"mjcf": "abc"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            save_motion_npz(path, archive)
            loaded = np.load(path)
            self.assertEqual(loaded["joint_pos"].shape, (3, 1))
            self.assertEqual(loaded["body_pos_w"].shape, (3, 1, 3))
            self.assertEqual(loaded["fps"].tolist(), [50])
            self.assertEqual(loaded["joint_names"].tolist(), ["joint"])
            self.assertEqual(loaded["schema_version"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
