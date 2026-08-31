import math
import unittest
from pathlib import Path
import tempfile
from unittest import mock

import bpy
import numpy as np
from mathutils import Matrix

from open_duck_tools import addon
from open_duck_tools.motion import MotionError
from open_duck_tools.motion_import import import_motion_action
from open_duck_tools.profile import profile_to_json
from tests.test_motion_import import (
    action_payload,
    build_minimal_rig,
    test_profile,
)


def play_once_handlers():
    return [
        handler
        for handler in bpy.app.handlers.frame_change_post
        if getattr(handler, "_duck_play_once_handler", False)
    ]


def native_playback_cleanup_handlers():
    return [
        handler
        for handler in bpy.app.handlers.animation_playback_post
        if getattr(handler, "_duck_native_playback_handler", False)
    ]


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

    def test_approximate_mouth_control_is_explicitly_labeled(self):
        addon.register()
        self.assertEqual(
            bpy.types.Object.bl_rna.properties["duck_mouth_open"].name,
            "Mouth (visual approximation)",
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

    def test_failed_import_stops_playback_and_cleans_temporary_handler(self):
        action = bpy.data.actions.new("OneShot")
        action.use_frame_range = True
        action.frame_start = 1
        action.frame_end = 3
        action["duck_loopable"] = False
        self.armature.duck_action_name = action.name
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertTrue(bpy.context.screen.is_animation_playing)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(
                Path(directory), "malformed.npz", {"bad": np.array([1])}
            )
            result = bpy.ops.duck.import_motion(
                filepath=str(path), action_name="Ignored"
            )

        self.assertEqual(result, {"CANCELLED"})
        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(play_once_handlers(), [])


class MotionExportTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export-fixture.npz"
            np.savez_compressed(path, **action_payload(self.profile))
            import_motion_action(
                self.armature, self.profile, path, action_name="ExportFixture"
            )

    def test_rejects_non_50hz_effective_rate_before_changing_frame(self):
        scene = bpy.context.scene
        scene.render.fps = 60
        scene.render.fps_base = 1.0
        scene.frame_set(17)
        changed_frames = []

        def record_frame_change(changed_scene, *_args):
            changed_frames.append(changed_scene.frame_current)

        bpy.app.handlers.frame_change_post.append(record_frame_change)
        try:
            with self.assertRaisesRegex(MotionError, r"effective.*60.*50 Hz"):
                addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        finally:
            bpy.app.handlers.frame_change_post.remove(record_frame_change)

        self.assertEqual(scene.frame_current, 17)
        self.assertEqual(changed_frames, [])

    def test_accepts_100_over_2_and_writes_canonical_50hz_metadata(self):
        scene = bpy.context.scene
        scene.render.fps = 100
        scene.render.fps_base = 2.0
        scene.frame_set(17)

        try:
            archive = addon.collect_armature_motion(
                self.armature, self.profile, 1, 3
            )
        except MotionError as exc:
            self.fail(str(exc))

        self.assertEqual(archive["fps"].tolist(), [50])
        self.assertEqual(scene.frame_current, 17)

    def test_rejects_off_axis_joint_rotation_and_restores_original_frame(self):
        scene = bpy.context.scene
        hip = self.armature.pose.bones["hip_link"]
        for frame, angle in ((1, 0.0), (2, 0.02), (3, 0.0)):
            scene.frame_set(frame)
            hip.rotation_euler.x = angle
            hip.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
        scene.frame_set(17)

        with self.assertRaisesRegex(
            MotionError, r"frame 2.*hip_link.*rotation residual"
        ):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)

        self.assertEqual(scene.frame_current, 17)

    def test_rejects_evaluated_scale_drift_and_identifies_frame_and_body(self):
        scene = bpy.context.scene
        knee = self.armature.pose.bones["knee_link"]
        for frame, scale in ((1, 1.0), (2, 1.002), (3, 1.0)):
            scene.frame_set(frame)
            knee.scale.x = scale
            knee.keyframe_insert(data_path="scale", index=0, frame=frame)
        scene.frame_set(17)

        with self.assertRaisesRegex(
            MotionError, r"frame 2.*knee_link.*affine residual"
        ):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)

        self.assertEqual(scene.frame_current, 17)

    def test_rejects_non_finite_evaluated_matrix_instead_of_canonicalizing_it(self):
        scene = bpy.context.scene
        scene.frame_set(1)
        matrices = addon._evaluated_body_matrices(
            self.armature, self.profile.body_names
        )
        matrices["knee_link"] = matrices["knee_link"].copy()
        matrices["knee_link"][0][0] = math.nan
        scene.frame_set(17)

        with mock.patch.object(
            addon, "_evaluated_body_matrices", return_value=matrices
        ):
            with self.assertRaisesRegex(
                MotionError, r"frame 1.*knee_link.*affine residual"
            ):
                addon.collect_armature_motion(self.armature, self.profile, 1, 1)

        self.assertEqual(scene.frame_current, 17)

    def test_restores_frame_and_subframe_after_success_and_failure(self):
        scene = bpy.context.scene
        scene.frame_set(2, subframe=0.5)
        addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        self.assertEqual(scene.frame_current, 2)
        self.assertAlmostEqual(scene.frame_subframe, 0.5, places=7)

        hip = self.armature.pose.bones["hip_link"]
        for frame, angle in ((1, 0.0), (2, 0.02), (3, 0.0)):
            scene.frame_set(frame)
            hip.rotation_euler.x = angle
            hip.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
        scene.frame_set(2, subframe=0.5)
        with self.assertRaisesRegex(MotionError, r"rotation residual"):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        self.assertEqual(scene.frame_current, 2)
        self.assertAlmostEqual(scene.frame_subframe, 0.5, places=7)


class AnimationControlTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        self.armature.data["duck_robot_profile_json"] = profile_to_json(self.profile)
        self.walk = bpy.data.actions.new("Policy_alpha_walking_forward")
        self.walk.use_frame_range = True
        self.walk.frame_start = 1
        self.walk.frame_end = 200
        self.walk["duck_loopable"] = False
        self.crouch = bpy.data.actions.new("KinematicCrouchTest")
        self.crouch.use_frame_range = True
        self.crouch.frame_start = 1
        self.crouch.frame_end = 51
        self.crouch["duck_loopable"] = False

    def tearDown(self):
        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        addon.unregister()

    def test_registers_beginner_animation_controls(self):
        self.assertTrue(hasattr(bpy.types.Object, "duck_action_name"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_select_action"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_toggle_animation"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_reset_animation"))

    def test_action_search_selection_activates_action_range_and_first_frame(self):
        bpy.context.scene.frame_set(17)

        self.armature.duck_action_name = self.walk.name

        self.assertIs(self.armature.animation_data.action, self.walk)
        self.assertEqual(self.armature.duck_action_name, self.walk.name)
        self.assertEqual(
            (bpy.context.scene.frame_start, bpy.context.scene.frame_end),
            (1, 200),
        )
        self.assertEqual(bpy.context.scene.frame_current, 1)

    def test_action_selection_and_play_force_fk_without_baking(self):
        constraint = self.armature.pose.bones["knee_link"].constraints.new(
            "COPY_LOCATION"
        )
        constraint.name = "DUCK_IK_PLAYBACK"
        constraint.influence = 0.75
        self.armature["fk_ik"] = 1.0

        self.armature.duck_action_name = self.walk.name

        self.assertEqual(constraint.influence, 0.0)
        self.assertEqual(self.armature["fk_ik"], 0.0)
        constraint.influence = 0.5
        self.armature["fk_ik"] = 1.0
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(constraint.influence, 0.0)
        self.assertEqual(self.armature["fk_ik"], 0.0)

    def test_selection_play_and_reset_clear_unkeyed_canonical_pose_contamination(self):
        root = self.armature.pose.bones["root"]
        hip = self.armature.pose.bones["hip_link"]
        knee = self.armature.pose.bones["knee_link"]

        def contaminate():
            root.location.x = 0.05
            hip.rotation_euler.x = 0.2
            hip.location.y = 0.03
            knee.scale.y = 1.2

        def assert_canonical():
            for bone_name in self.profile.body_names:
                np.testing.assert_allclose(
                    self.armature.pose.bones[bone_name].matrix_basis,
                    Matrix.Identity(4),
                    atol=1e-6,
                    err_msg=bone_name,
                )

        contaminate()
        self.armature.duck_action_name = self.walk.name
        assert_canonical()

        contaminate()
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        assert_canonical()
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        contaminate()
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        assert_canonical()

    def test_retained_crouch_action_is_searchable_but_not_a_beginner_preset(self):
        self.assertTrue(hasattr(addon, "BEGINNER_ACTION_PRESETS"))
        self.assertEqual(
            addon.BEGINNER_ACTION_PRESETS,
            (("Policy_alpha_walking_forward", "Walk"),),
        )
        self.armature.duck_action_name = self.walk.name

        result = bpy.ops.duck.select_action(action_name=self.crouch.name)

        self.assertEqual(result, {"FINISHED"})
        self.assertIs(self.armature.animation_data.action, self.crouch)
        self.assertEqual(
            (
                bpy.context.scene.frame_start,
                bpy.context.scene.frame_end,
                bpy.context.scene.frame_current,
            ),
            (1, 51, 1),
        )

    def test_missing_preset_cancels_without_changing_animation(self):
        self.armature.duck_action_name = self.walk.name
        bpy.context.scene.frame_set(25)

        result = bpy.ops.duck.select_action(action_name="MissingAction")

        self.assertEqual(result, {"CANCELLED"})
        self.assertIs(self.armature.animation_data.action, self.walk)
        self.assertEqual(
            (
                bpy.context.scene.frame_start,
                bpy.context.scene.frame_end,
                bpy.context.scene.frame_current,
            ),
            (1, 200, 25),
        )

    def test_play_pause_and_reset_controls_real_screen_playback(self):
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertTrue(bpy.context.screen.is_animation_playing)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertFalse(bpy.context.screen.is_animation_playing)
        bpy.context.scene.frame_set(100)
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(bpy.context.scene.frame_current, 1)

    def test_play_once_restarts_from_action_start_at_end_and_disables_preview(self):
        scene = bpy.context.scene
        self.armature.duck_action_name = self.walk.name
        scene.use_preview_range = True
        scene.frame_preview_start = 10
        scene.frame_preview_end = 20
        scene.frame_set(200)

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        self.assertTrue(bpy.context.screen.is_animation_playing)
        self.assertEqual(scene.frame_current, 1)
        self.assertFalse(scene.use_preview_range)
        self.assertEqual((scene.frame_start, scene.frame_end), (1, 200))

    @unittest.skipUnless(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 5.2+ playback loop API",
    )
    def test_blender_5_2_play_once_uses_stop_end_frame_without_handler(self):
        scene = bpy.context.scene
        scene.playback_loop_mode = "INFINITE"
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        self.assertEqual(scene.playback_loop_mode, "STOP_END_FRAME")
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipUnless(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 5.2+ playback loop API",
    )
    def test_blender_5_2_restores_native_loop_mode_on_every_terminal_path(self):
        scene = bpy.context.scene
        scene.playback_loop_mode = "BOUNCE"
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "STOP_END_FRAME")
        self.assertEqual(len(native_playback_cleanup_handlers()), 1)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        cleanup = native_playback_cleanup_handlers()
        self.assertEqual(len(cleanup), 1)
        cleanup[0](scene, None)
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_blender_4_3_play_once_stops_at_endpoint_and_self_removes_handler(self):
        scene = bpy.context.scene
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        scene.frame_set(200)

        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(scene.frame_current, 200)
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_blender_4_3_handler_cleans_on_pause_reset_selection_and_unregister(self):
        self.armature.duck_action_name = self.walk.name
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(
            bpy.ops.duck.select_action(action_name=self.crouch.name), {"FINISHED"}
        )
        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        addon.unregister()
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_unregister_removes_temporary_handlers_from_every_scene(self):
        other_scene = bpy.data.scenes.new("OtherDuckScene")
        addon._install_play_once_handler(bpy.context.scene)
        addon._install_play_once_handler(other_scene)
        self.assertEqual(len(play_once_handlers()), 2)

        try:
            addon.unregister()
            self.assertEqual(play_once_handlers(), [])
        finally:
            addon._clear_play_once_handlers()


if __name__ == "__main__":
    unittest.main()
