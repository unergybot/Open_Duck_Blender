import math
import unittest

from mathutils import Matrix, Quaternion, Vector

from open_duck_tools.blender_bridge import body_samples, joint_angles_from_body_matrices
from open_duck_tools.profile import BodySpec, JointSpec, RobotProfile


class BlenderBridgeTests(unittest.TestCase):
    def profile(self):
        return RobotProfile(
            schema_version=1,
            robot_id="fixture",
            joint_names=("hinge",),
            body_names=("root", "child"),
            home_positions=(0.0,),
            joints=(
                JointSpec(
                    name="hinge",
                    parent_body="root",
                    child_body="child",
                    axis=(0.0, 0.0, 1.0),
                    range_rad=(-1.0, 1.0),
                ),
            ),
            bodies=(
                BodySpec("root", None, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
                BodySpec("child", "root", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
            mouth=None,
            source_sha256={},
        )

    def test_extracts_hinge_angle_after_body_rest_transform(self):
        angle = 0.35
        matrices = {
            "root": Matrix.Identity(4),
            "child": Matrix.Translation((1.0, 0.0, 0.0))
            @ Matrix.Rotation(angle, 4, "Z"),
        }
        result = joint_angles_from_body_matrices(matrices, self.profile())
        self.assertAlmostEqual(result[0], angle, places=6)

    def test_exports_positions_and_wxyz_quaternions_in_profile_order(self):
        rotation = Quaternion(Vector((0.0, 0.0, 1.0)), 0.4)
        matrices = {
            "child": Matrix.LocRotScale((1.0, 2.0, 3.0), rotation, (1, 1, 1)),
            "root": Matrix.Identity(4),
        }
        positions, quaternions = body_samples(matrices, self.profile())
        self.assertEqual(positions.tolist(), [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        expected = [math.cos(0.2), 0.0, 0.0, math.sin(0.2)]
        for actual, wanted in zip(quaternions[1], expected):
            self.assertAlmostEqual(actual, wanted, places=6)


if __name__ == "__main__":
    unittest.main()
