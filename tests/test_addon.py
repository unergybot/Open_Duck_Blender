import unittest
from pathlib import Path
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

from open_duck_tools import addon
from open_duck_tools.profile import profile_to_json
from tests.test_motion_import import action_payload, build_minimal_rig, test_profile


class AddonRegistrationTests(unittest.TestCase):
    def tearDown(self):
        addon.unregister()

    def test_registers_tools_panel_and_is_idempotent(self):
        addon.register()
        addon.register()
        self.assertTrue(hasattr(bpy.types, "DUCK_PT_tools"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_import_motion"))
        self.assertIn(addon.DUCK_OT_import_motion, addon.CLASSES)
        self.assertTrue(hasattr(bpy.types.Object, "duck_colorway"))

    def test_ik_target_uses_ankle_tail_without_changing_orientation(self):
        armature_world = Matrix.Translation((1.0, 2.0, 3.0))
        ankle_matrix = Matrix.Rotation(0.3, 4, "Z")
        ankle_matrix.translation = (0.1, 0.2, 0.3)
        ankle_tail = Vector((0.1, 0.215, 0.3))
        target = addon._ankle_target_matrix(armature_world, ankle_matrix, ankle_tail)
        for actual, expected in zip(target.translation, (1.1, 2.215, 3.3)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertLess(
            target.to_quaternion().rotation_difference(
                (armature_world @ ankle_matrix).to_quaternion()
            ).angle,
            1e-7,
        )


class MotionImportOperatorTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        self.armature.data["duck_robot_profile_json"] = profile_to_json(self.profile)

    def tearDown(self):
        addon.unregister()

    def write_archive(self, directory: Path, name: str, payload: dict) -> Path:
        path = directory / name
        np.savez_compressed(path, **payload)
        return path

    def test_imports_fixture_with_sanitized_default_action_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(
                Path(directory), "Policy alpha walking forward!.npz", action_payload(self.profile)
            )
            result = bpy.ops.duck.import_motion(filepath=str(path), action_name="")

        self.assertEqual(result, {"FINISHED"})
        self.assertEqual(
            self.armature.animation_data.action.name, "Policy_alpha_walking_forward"
        )
        self.assertEqual((bpy.context.scene.frame_start, bpy.context.scene.frame_end), (1, 3))

    def test_cancels_malformed_archive_without_changing_scene(self):
        scene = bpy.context.scene
        prior = bpy.data.actions.new("Prior")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        self.armature.location = (9.0, 8.0, 7.0)
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        self.armature.keyframe_insert(data_path="location", frame=8)
        original_matrix = self.armature.matrix_world.copy()
        original_actions = {action.name for action in bpy.data.actions}
        original_action_range = tuple(prior.frame_range)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(Path(directory), "malformed.npz", {"bad": np.array([1])})
            result = bpy.ops.duck.import_motion(filepath=str(path), action_name="Ignored")

        self.assertEqual(result, {"CANCELLED"})
        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        self.assertEqual(self.armature.matrix_world, original_matrix)
        self.assertEqual({action.name for action in bpy.data.actions}, original_actions)
        self.assertEqual(tuple(prior.frame_range), original_action_range)

if __name__ == "__main__":
    unittest.main()
