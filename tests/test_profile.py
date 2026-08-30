import json
import math
import tempfile
import unittest
from pathlib import Path

from open_duck_tools.profile import (
    ProfileError,
    build_microduck_profile,
    load_mouth_linkage,
)


MJCF = """
<mujoco model="microduck">
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <body name="left_link" pos="0 0.1 0">
        <joint name="left_hip_yaw" axis="0 0 1" range="-0.4 0.5"/>
      </body>
      <body name="head_link" pos="0.02 0 0.03">
        <joint name="head_pitch" axis="0 0 1" range="-0.6 0.7"/>
      </body>
      <body name="right_link" pos="0 -0.1 0">
        <joint name="right_hip_yaw" axis="0 0 1" range="-0.5 0.4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

RUNTIME_MODEL = """
pub const NUM_JOINTS: usize = 4;
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "left_hip_yaw", "head_pitch", "mouth", "right_hip_yaw",
];
pub const DEFAULT_POSITION: [f64; NUM_JOINTS] = [
    0.1, 0.2, 0.0, -0.1,
];
pub const MOUTH_CLOSED: f64 = -5.0 * std::f64::consts::PI / 180.0;
pub const MOUTH_OPEN: f64 = 30.0 * std::f64::consts::PI / 180.0;
"""


def linkage_payload():
    return {
        "schema_version": 1,
        "units": "m",
        "servo": {
            "name": "mouth",
            "closed_rad": math.radians(-5),
            "open_rad": math.radians(30),
        },
        "links": [
            {"name": "lower_beak", "meshes": ["jaw"], "parent": "jaw_soft"}
        ],
        "samples": [
            {
                "servo_rad": math.radians(angle),
                "poses": {
                    "lower_beak": {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }
            for angle in (-5, 12.5, 30)
        ],
        "validation_poses": [
            {
                "servo_rad": math.radians(3.25),
                "poses": {
                    "lower_beak": {
                        "position": [0.001, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }
        ],
    }


class MouthLinkageTests(unittest.TestCase):
    def test_rejects_linkage_without_validation_pose(self):
        payload = linkage_payload()
        payload["validation_poses"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ProfileError, "validation_poses"):
                load_mouth_linkage(path)

    def test_loads_complete_linkage_and_normalizes_quaternions(self):
        payload = linkage_payload()
        payload["samples"][1]["poses"]["lower_beak"]["quaternion_wxyz"] = [2, 0, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            linkage = load_mouth_linkage(path)
        self.assertEqual(linkage.links[0].name, "lower_beak")
        self.assertEqual(linkage.samples[1].poses["lower_beak"].quaternion_wxyz, (1, 0, 0, 0))


class ProfileBuildTests(unittest.TestCase):
    def write_sources(self, root: Path, runtime=RUNTIME_MODEL):
        mjcf = root / "robot_walk.xml"
        model = root / "model.rs"
        mouth = root / "mouth.json"
        mjcf.write_text(MJCF)
        model.write_text(runtime)
        mouth.write_text(json.dumps(linkage_payload()))
        return mjcf, model, mouth

    def test_builds_policy_order_by_removing_only_mouth(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory))
            profile = build_microduck_profile(mjcf, runtime, mouth)
        self.assertEqual(
            profile.joint_names,
            ("left_hip_yaw", "head_pitch", "right_hip_yaw"),
        )
        self.assertEqual(profile.body_names, ("trunk_base", "left_link", "head_link", "right_link"))
        self.assertEqual(profile.home_positions, (0.1, 0.2, -0.1))
        self.assertEqual(profile.joints[0].range_rad, (-0.4, 0.5))

    def test_rejects_runtime_and_mjcf_order_drift(self):
        drifted = RUNTIME_MODEL.replace(
            '"left_hip_yaw", "head_pitch", "mouth", "right_hip_yaw"',
            '"head_pitch", "left_hip_yaw", "mouth", "right_hip_yaw"',
        )
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory), drifted)
            with self.assertRaisesRegex(ProfileError, "joint order"):
                build_microduck_profile(mjcf, runtime, mouth)


if __name__ == "__main__":
    unittest.main()
