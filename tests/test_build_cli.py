import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import bpy

from open_duck_tools.profile import ProfileError


SCRIPT = Path(__file__).parents[1] / "tools" / "build_microduck_blend.py"
POLICY_SHA256 = "ffa9df070e15a2490b862a16e514fdb76ff8eb5ec1001f0dd3474350dce1aa62"
SPEC = importlib.util.spec_from_file_location("build_microduck_blend", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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

    def test_built_blend_reopens_as_walking_milestone(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "microduck-alpha.blend"
            args = MODULE._arguments(
                [
                    "--runtime-root",
                    "/home/mcao/MyCode/microduck",
                    "--rl-root",
                    "/home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender",
                    "--output",
                    str(output),
                ]
            )
            MODULE.build(args)
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
