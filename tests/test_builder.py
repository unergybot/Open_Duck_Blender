import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np

from open_duck_tools.builder import generate_microduck_scene
from open_duck_tools import addon
from open_duck_tools.profile import ProfileError, build_microduck_profile


MJCF = """
<mujoco model="microduck">
  <compiler meshdir="assets"/>
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <body name="child" pos="0.1 0 0">
        <joint name="hinge" axis="0 0 1" range="-1 1"/>
      </body>
      <body name="jaw_soft" pos="0 0 0.1">
        <geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>
      </body>
    </body>
  </worldbody>
  <asset>
    <mesh name="jaw" file="jaw.stl"/>
    <material name="jaw_material" rgba="1 0.7 0 1"/>
  </asset>
</mujoco>
"""

RUNTIME = """
pub const JOINT_NAMES: [&str; 2] = ["hinge", "mouth"];
pub const DEFAULT_POSITION: [f64; 2] = [0.0, 0.0];
"""


def mouth_payload():
    identity = {
        "position": [0, 0, 0],
        "quaternion_wxyz": [1, 0, 0, 0],
    }
    return {
        "schema_version": 1,
        "units": "m",
        "servo": {
            "name": "mouth",
            "closed_rad": math.radians(-5),
            "open_rad": math.radians(30),
        },
        "links": [{"name": "lower_beak", "meshes": ["jaw"], "parent": "jaw_soft"}],
        "samples": [
            {"servo_rad": math.radians(-5), "poses": {"lower_beak": identity}},
            {
                "servo_rad": math.radians(30),
                "poses": {
                    "lower_beak": {
                        "position": [0.01, 0, 0],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    }
                },
            },
        ],
        "validation_poses": [
            {
                "servo_rad": math.radians(12.5),
                "poses": {
                    "lower_beak": {
                        "position": [0.005, 0, 0],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    }
                },
            }
        ],
    }


def write_binary_stl(path: Path):
    record = struct.pack(
        "<12fH",
        0,
        0,
        1,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0.01,
        0,
        0,
    )
    path.write_bytes(bytes(80) + struct.pack("<I", 1) + record)


class SceneBuilderTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        addon.unregister()

    def test_builds_profiled_rig_visual_and_driven_mouth_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            armature = generate_microduck_scene(profile, mjcf, Path("open_duck_tools"))

        self.assertEqual(armature.get("duck_robot_id"), "microduck-alpha")
        self.assertEqual(bpy.context.scene.render.fps, 50)
        self.assertEqual(
            set(profile.body_names),
            {bone.name for bone in armature.data.bones if not bone.name.startswith("mouth::")},
        )
        self.assertIn("mouth::lower_beak", armature.data.bones)
        self.assertIsNotNone(bpy.data.objects.get("visual::jaw"))
        self.assertTrue(armature.data.get("duck_robot_profile_json"))
        bootstrap = bpy.data.texts.get("open_duck_bootstrap.py")
        self.assertIsNotNone(bootstrap)
        self.assertTrue(bootstrap.use_module)
        self.assertEqual(bootstrap.as_string().rstrip().splitlines()[-1], "register()")
        addon.register()
        armature.duck_mouth_open = 1.0
        bpy.context.view_layer.update()
        self.assertAlmostEqual(
            armature.pose.bones["mouth::lower_beak"].matrix_basis.to_translation().x,
            0.01,
            places=6,
        )

    def test_builds_51_frame_demo_action_from_native_motion_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            motion = root / "demo.npz"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            joint_pos = np.zeros((51, 1), dtype=np.float32)
            joint_pos[25, 0] = 0.4
            np.savez_compressed(
                motion,
                fps=np.array([50], dtype=np.int32),
                joint_names=np.array(["hinge"]),
                joint_pos=joint_pos,
            )
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            armature = generate_microduck_scene(
                profile,
                mjcf,
                Path("open_duck_tools"),
                demo_motion_path=motion,
            )

        self.assertEqual((bpy.context.scene.frame_start, bpy.context.scene.frame_end), (1, 51))
        self.assertIsNotNone(armature.animation_data)
        self.assertEqual(armature.animation_data.action.name, "MicroduckCrouchTest")
        bpy.context.scene.frame_set(26)
        self.assertAlmostEqual(armature.pose.bones["child"].rotation_euler.z, 0.4, places=6)
        self.assertAlmostEqual(armature.duck_mouth_open, 1.0, places=6)

    def test_rejects_malformed_demo_archives_with_profile_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            malformed = (
                {"joint_names": np.array(["hinge"]), "joint_pos": np.array([[0.0]])},
                {
                    "fps": np.array([50]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array(0.0),
                },
                {
                    "fps": np.array([50.9]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array(["bad"]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array([np.nan]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array([np.inf]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
            )
            for index, payload in enumerate(malformed):
                motion = root / f"malformed-{index}.npz"
                np.savez_compressed(motion, **payload)
                with self.subTest(index=index), self.assertRaisesRegex(
                    ProfileError, "demo motion"
                ):
                    generate_microduck_scene(
                        profile,
                        mjcf,
                        Path("open_duck_tools"),
                        demo_motion_path=motion,
                    )


if __name__ == "__main__":
    unittest.main()
