from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import bpy
import numpy as np

from open_duck_tools.motion import MotionError, build_motion_archive
from open_duck_tools.profile import ProfileError


SCRIPT = Path(__file__).parents[1] / "tools" / "build_microduck_blend.py"
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
        foot_site = (
            '<site name="left_foot" pos="0 -0.0238146 -0.0140852" '
            'quat="0 0 0.707107 0.707107"/>'
            if body == "ankle_left"
            else '<site name="right_foot" pos="0 -0.0238146 -0.0140852" '
            'quat="0.707107 -0.707107 0 0"/>'
            if body == "ankle_right"
            else ""
        )
        visuals = (
            '<geom type="mesh" class="visual" mesh="jaw"/>'
            '<geom type="mesh" class="visual" mesh="jaw_soft"/>'
            if body == "jaw_soft"
            else ""
        )
        bodies.append(
            f'<body name="{body}"><joint name="{joint}" axis="0 0 1" '
            f'range="-10 10"/>{foot_site}{visuals}</body>'
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


def write_controlled_motion(path: Path, frame_count: int, travel: float) -> str:
    root_x = np.linspace(0.0, travel, frame_count)
    body_pos = np.zeros((frame_count, len(BODY_NAMES), 3), dtype=np.float64)
    body_pos[:, :, 0] = root_x[:, None]
    body_pos[:, :, 2] = 0.12
    archive = build_motion_archive(
        np.zeros((frame_count, len(JOINT_NAMES)), dtype=np.float64),
        body_pos,
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]),
            (frame_count, len(BODY_NAMES), 1),
        ),
        fps=50,
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        joint_ranges=tuple((-10.0, 10.0) for _name in JOINT_NAMES),
        source_hashes={"fixture": "controlled"},
    )
    np.savez_compressed(path, **archive)
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            demo_motion = root / "controlled-crouch.npz"
            policy_motion = root / "controlled-policy.npz"
            demo_sha256 = write_controlled_motion(demo_motion, 51, 0.0)
            policy_sha256 = write_controlled_motion(policy_motion, 200, 0.5)
            args = MODULE._arguments(
                [
                    "--runtime-root",
                    str(root / "runtime"),
                    "--rl-root",
                    str(root / "rl"),
                    "--demo-motion",
                    str(demo_motion),
                    "--policy-motion",
                    str(policy_motion),
                    "--output",
                    str(output),
                ]
            )
            try:
                MODULE.build(args)
            except (ProfileError, MotionError) as exc:
                self.fail(f"controlled build roots were not portable: {exc}")
            bpy.ops.wm.open_mainfile(filepath=str(output))

            self.assertEqual(
                {action.name for action in bpy.data.actions},
                {"KinematicCrouchTest", "Policy_alpha_walking_forward"},
            )
            crouch = bpy.data.actions["KinematicCrouchTest"]
            policy = bpy.data.actions["Policy_alpha_walking_forward"]
            self.assertTrue(crouch.use_fake_user)
            self.assertTrue(
                policy.use_fake_user
            )
            self.assertEqual(
                tuple(crouch.frame_range),
                (1.0, 51.0),
            )
            self.assertEqual(
                tuple(policy.frame_range),
                (1.0, 200.0),
            )
            self.assertEqual(crouch["duck_motion_kind"], "kinematic_test")
            self.assertEqual(crouch["duck_source_sha256"], demo_sha256)
            self.assertFalse(crouch["duck_loopable"])
            self.assertFalse(crouch["duck_contact_valid"])
            self.assertEqual(policy["duck_motion_kind"], "policy_rollout")
            self.assertEqual(policy["duck_source_sha256"], policy_sha256)
            self.assertFalse(policy["duck_loopable"])
            self.assertEqual(bpy.context.scene.render.fps, 50)
            self.assertEqual(bpy.context.scene.render.fps_base, 1.0)
            self.assertEqual(
                (bpy.context.scene.frame_start, bpy.context.scene.frame_end),
                (1, 200),
            )
            self.assertEqual(bpy.context.scene.frame_current, 1)
            self.assertEqual(
                bpy.data.objects["MicroduckRig"].animation_data.action.name,
                "Policy_alpha_walking_forward",
            )
            self.assertEqual(
                bpy.data.objects["MicroduckRig"].duck_action_name,
                "Policy_alpha_walking_forward",
            )
            scene_children = {collection.name for collection in bpy.context.scene.collection.children}
            self.assertEqual(scene_children, {"Microduck", "Presentation"})
            microduck = bpy.data.collections["Microduck"]
            self.assertEqual(
                {collection.name for collection in microduck.children},
                {"Rig", "Visuals", "Controls"},
            )
            self.assertEqual(
                {obj.name for obj in bpy.data.collections["Rig"].objects},
                {"MicroduckRig"},
            )
            self.assertEqual(
                {obj.name for obj in bpy.data.collections["Visuals"].objects},
                {"visual::jaw", "visual::jaw_soft"},
            )
            self.assertEqual(
                {obj.name for obj in bpy.data.collections["Presentation"].objects},
                {"Ground", "MicroduckCamera", "KeyLight", "FillLight"},
            )
            ground = bpy.data.objects["Ground"]
            self.assertAlmostEqual(ground.matrix_world.translation.z, 0.0, places=7)
            self.assertGreaterEqual(ground.dimensions.x, 0.65 - 1e-6)
            self.assertGreaterEqual(ground.dimensions.y, 0.30 - 1e-6)
            self.assertIs(bpy.context.scene.camera, bpy.data.objects["MicroduckCamera"])
            armature = bpy.data.objects["MicroduckRig"]
            self.assertEqual(armature.data.display_type, "STICK")
            self.assertTrue(armature.show_in_front)
            self.assertEqual(armature.mode, "OBJECT")
            self.assertIs(bpy.context.view_layer.objects.active, armature)
            self.assertEqual(armature["fk_ik"], 0.0)
            view_regions = [
                area.spaces.active.region_3d
                for screen in bpy.data.screens
                for area in screen.areas
                if area.type == "VIEW_3D"
            ]
            self.assertTrue(view_regions)
            for region in view_regions:
                self.assertLess(region.view_distance, 0.5)
                self.assertAlmostEqual(region.view_location.z, 0.13, places=3)
            manifest = json.loads(
                bpy.data.texts["microduck-build-manifest.json"].as_string()
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["motion_sha256"],
                {
                    "KinematicCrouchTest": demo_sha256,
                    "Policy_alpha_walking_forward": policy_sha256,
                },
            )
            self.assertEqual(manifest["source_sha256"], armature.data["duck_robot_profile_json"] and json.loads(armature.data["duck_robot_profile_json"])["source_sha256"])
            self.assertEqual(manifest["build_blender_version"], bpy.app.version_string)
            self.assertEqual(manifest["mouth_mode"], "image-derived-approximation")

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

    def test_main_reports_motion_error_concisely_without_traceback(self):
        stderr = io.StringIO()
        with mock.patch.object(
            MODULE, "build", side_effect=MotionError("joint_pos must have shape [T,14]")
        ), mock.patch.object(sys, "argv", ["blender", "--"]), redirect_stderr(stderr):
            result = MODULE.main()

        self.assertEqual(result, 2)
        self.assertEqual(
            stderr.getvalue(),
            "Microduck build failed: joint_pos must have shape [T,14]\n",
        )
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
