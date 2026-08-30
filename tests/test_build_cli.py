import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import bpy

from open_duck_tools.profile import ProfileError


SCRIPT = Path(__file__).parents[1] / "tools" / "build_microduck_blend.py"
POLICY_SHA256 = "ffa9df070e15a2490b862a16e514fdb76ff8eb5ec1001f0dd3474350dce1aa62"
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
BODY_NAMES = (
    "trunk_base",
    "yaw2roll",
    "hip_l",
    "upper_leg_left",
    "leg",
    "ankle_left",
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "jaw_soft",
    "bearing_roll",
    "hip_l_2",
    "upper_leg_right",
    "leg_2",
    "ankle_right",
)
SPEC = importlib.util.spec_from_file_location("build_microduck_blend", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_binary_stl(path: Path) -> None:
    triangle = struct.pack(
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
    path.write_bytes(bytes(80) + struct.pack("<I", 1) + triangle)


def write_controlled_build_sources(root: Path) -> None:
    robot = root / "rl/src/mjlab_microduck/robot/microduck"
    assets = robot / "assets"
    model = root / "runtime/duck-control/src/model.rs"
    contract = root / "runtime/duck-ipc-proto/src/lib.rs"
    assets.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    contract.parent.mkdir(parents=True)
    bodies = []
    for joint, body in zip(JOINT_NAMES, BODY_NAMES[1:]):
        visuals = (
            '<geom type="mesh" class="visual" mesh="jaw"/>'
            '<geom type="mesh" class="visual" mesh="jaw_soft"/>'
            if body == "jaw_soft"
            else ""
        )
        bodies.append(
            f'<body name="{body}"><joint name="{joint}" axis="0 0 1" '
            f'range="-10 10"/>{visuals}</body>'
        )
    (robot / "robot_walk.xml").write_text(
        '<mujoco model="microduck"><compiler meshdir="assets"/>'
        '<asset><mesh name="jaw" file="jaw.stl"/>'
        '<mesh name="jaw_soft" file="jaw_soft.stl"/></asset>'
        '<worldbody><body name="trunk_base" pos="0 0 0.12">'
        + "".join(bodies)
        + "</body></worldbody></mujoco>"
    )
    write_binary_stl(assets / "jaw.stl")
    write_binary_stl(assets / "jaw_soft.stl")
    model.write_text(
        "pub const DEFAULT_POSITION: [f64; 15] = ["
        + ", ".join("0.0" for _ in range(15))
        + "];"
    )
    contract.write_text(
        "pub const JOINT_NAMES: [&str; 15] = ["
        + ", ".join(f'"{name}"' for name in (*JOINT_NAMES, "mouth"))
        + "];"
    )


class BuildCliTests(unittest.TestCase):
    def test_defaults_to_explicit_approximate_mouth_mode(self):
        args = MODULE._arguments([])
        self.assertIsNone(args.mouth_linkage)

    def test_defaults_to_versioned_crouch_demo_motion(self):
        args = MODULE._arguments([])
        self.assertEqual(
            args.demo_motion,
            SCRIPT.parents[1] / "assets/motions/microduck-crouch-test.npz",
        )

    def test_defaults_to_versioned_policy_motion(self):
        args = MODULE._arguments([])
        self.assertEqual(
            args.policy_motion,
            SCRIPT.parents[1] / "assets/motions/alpha-walking-forward.npz",
        )

    def test_built_blend_reopens_as_walking_milestone_from_controlled_roots(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "microduck-alpha.blend"
            write_controlled_build_sources(root)
            args = MODULE._arguments(
                [
                    "--runtime-root",
                    str(root / "runtime"),
                    "--rl-root",
                    str(root / "rl"),
                    "--output",
                    str(output),
                ]
            )
            try:
                MODULE.build(args)
            except ProfileError as exc:
                self.fail(f"controlled build roots were not portable: {exc}")
            bpy.ops.wm.open_mainfile(filepath=str(output))

            self.assertEqual(
                {action.name for action in bpy.data.actions},
                {"MicroduckCrouchTest", "Policy_alpha_walking_forward"},
            )
            self.assertTrue(bpy.data.actions["MicroduckCrouchTest"].use_fake_user)
            self.assertTrue(
                bpy.data.actions["Policy_alpha_walking_forward"].use_fake_user
            )
            self.assertEqual(
                tuple(bpy.data.actions["MicroduckCrouchTest"].frame_range),
                (1.0, 51.0),
            )
            self.assertEqual(
                tuple(bpy.data.actions["Policy_alpha_walking_forward"].frame_range),
                (1.0, 200.0),
            )
            self.assertEqual(bpy.context.scene.render.fps, 50)
            self.assertEqual(
                (bpy.context.scene.frame_start, bpy.context.scene.frame_end),
                (1, 200),
            )
            self.assertEqual(bpy.context.scene.frame_current, 1)
            self.assertEqual(
                bpy.data.objects["MicroduckRig"].animation_data.action.name,
                "Policy_alpha_walking_forward",
            )
            manifest = json.loads(
                bpy.data.texts["microduck-build-manifest.json"].as_string()
            )
            self.assertEqual(manifest["policy_motion_sha256"], POLICY_SHA256)

    def test_reports_all_missing_canonical_sources_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = MODULE._arguments(
                [
                    "--rl-root", str(root / "rl"),
                    "--runtime-root", str(root / "runtime"),
                ]
            )
            with self.assertRaisesRegex(ProfileError, "required source file"):
                MODULE.build(args)


if __name__ == "__main__":
    unittest.main()
