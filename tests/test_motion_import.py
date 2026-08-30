"""Tests for strict, side-effect-free mjlab motion archive loading."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

from open_duck_tools import ImportedMotion
from open_duck_tools.motion import MotionError, build_motion_archive
from open_duck_tools.motion_import import import_motion_action, load_motion
from open_duck_tools.profile import (
    BodySpec,
    JointSpec,
    MouthLinkage,
    RobotProfile,
)


def test_profile() -> RobotProfile:
    """A compact profile whose root has a non-identity MJCF rest transform."""
    return RobotProfile(
        schema_version=1,
        robot_id="test-duck",
        joint_names=("hip", "knee"),
        body_names=("root", "hip_link", "knee_link"),
        home_positions=(0.0, 0.0),
        joints=(
            JointSpec("hip", "root", "hip_link", (0.0, 0.0, 1.0), (-1.0, 1.0)),
            JointSpec("knee", "hip_link", "knee_link", (0.0, 0.0, 1.0), (-0.5, 0.5)),
        ),
        bodies=(
            BodySpec("root", None, (0.0, 0.0, 0.125), (1.0, 0.0, 0.0, 0.0)),
            BodySpec("hip_link", "root", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0)),
            BodySpec("knee_link", "hip_link", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0)),
        ),
        mouth=MouthLinkage(1, 0.0, 1.0, (), (), (), "test"),
        source_sha256={},
    )


def archive_payload(profile: RobotProfile) -> dict[str, np.ndarray]:
    # The negative final quaternion represents the same rotation; the loader
    # must normalize it and choose the interpolation-continuous sign.
    archive = build_motion_archive(
        joint_pos=np.array([[0.0, 0.0], [0.25, -0.25], [0.5, 0.25]]),
        body_pos_w=np.array(
            [
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [1.0, 2.0, 3.2]],
                [[1.5, 2.0, 3.0], [1.5, 2.0, 3.1], [1.5, 2.0, 3.2]],
                [[2.0, 2.0, 3.0], [2.0, 2.0, 3.1], [2.0, 2.0, 3.2]],
            ]
        ),
        body_quat_w=np.array(
            [
                [[2.0, 0.0, 0.0, 0.0]] * 3,
                [[0.0, 0.0, 0.0, 3.0]] * 3,
                [[0.0, 0.0, 0.0, -4.0]] * 3,
            ]
        ),
        fps=50,
        joint_names=profile.joint_names,
        body_names=profile.body_names,
        joint_ranges=tuple(joint.range_rad for joint in profile.joints),
        source_hashes={"fixture": "deadbeef"},
    )
    archive["body_quat_w"][2] *= -1
    return archive


class MotionLoaderTests(unittest.TestCase):
    def write_archive(self, directory: Path, payload: dict[str, np.ndarray]) -> Path:
        path = directory / "motion.npz"
        np.savez_compressed(path, **payload)
        return path

    def test_loads_complete_native_archive_with_normalized_continuous_root_quaternions(self):
        profile = test_profile()
        with tempfile.TemporaryDirectory() as directory:
            motion = load_motion(self.write_archive(Path(directory), archive_payload(profile)), profile)

        self.assertIsInstance(motion, ImportedMotion)
        self.assertEqual(motion.fps, 50)
        self.assertEqual(motion.frames, 3)
        self.assertEqual(motion.joint_pos.shape, (3, 2))
        self.assertEqual(motion.root_pos_w.shape, (3, 3))
        self.assertEqual(motion.root_quat_wxyz.shape, (3, 4))
        self.assertTrue(np.isfinite(motion.joint_pos).all())
        self.assertTrue(np.isfinite(motion.root_pos_w).all())
        self.assertTrue(np.isfinite(motion.root_quat_wxyz).all())
        np.testing.assert_allclose(np.linalg.norm(motion.root_quat_wxyz, axis=1), [1.0, 1.0, 1.0])
        self.assertTrue(np.all(np.sum(motion.root_quat_wxyz[1:] * motion.root_quat_wxyz[:-1], axis=1) >= 0))
        np.testing.assert_allclose(motion.joint_pos[:, 1], [0.0, -0.25, 0.25])

    def test_rejects_each_native_schema_or_sample_contract_violation(self):
        profile = test_profile()
        cases = []
        missing = archive_payload(profile)
        missing.pop("body_ang_vel_w")
        cases.append((missing, r"archive keys.*body_ang_vel_w"))
        extra = archive_payload(profile)
        extra["surprise"] = np.array([1])
        cases.append((extra, r"archive keys.*surprise"))
        names = archive_payload(profile)
        names["joint_names"] = np.array(["wrong", "knee"])
        cases.append((names, r"joint_names.*index 0.*wrong.*hip"))
        body_names = archive_payload(profile)
        body_names["body_names"] = np.array(["root", "wrong", "knee_link"])
        cases.append((body_names, r"body_names.*index 1.*wrong.*hip_link"))
        nonfinite = archive_payload(profile)
        nonfinite["body_pos_w"][1, 2, 0] = np.nan
        cases.append((nonfinite, r"body_pos_w.*frame 1.*index 2"))
        zero = archive_payload(profile)
        zero["body_quat_w"][2, 0] = 0
        cases.append((zero, r"body_quat_w.*frame 2.*index 0"))
        nonunit = archive_payload(profile)
        nonunit["body_quat_w"][1, 2] = (2.0, 0.0, 0.0, 0.0)
        cases.append((nonunit, r"body_quat_w.*frame 1.*index 2.*unit"))
        over_limit = archive_payload(profile)
        over_limit["joint_pos"][1, 1] = 0.75
        cases.append((over_limit, r"joint_pos.*frame 1.*index 1.*knee"))
        scalar_joint_pos = archive_payload(profile)
        scalar_joint_pos["joint_pos"] = np.array(0.0)
        cases.append((scalar_joint_pos, r"joint_pos.*shape"))
        vector_joint_pos = archive_payload(profile)
        vector_joint_pos["joint_pos"] = np.array([0.0, 0.0])
        cases.append((vector_joint_pos, r"joint_pos.*shape"))
        matrix_joint_names = archive_payload(profile)
        matrix_joint_names["joint_names"] = np.array([["hip", "knee"]])
        cases.append((matrix_joint_names, r"joint_names.*one-dimensional"))

        with tempfile.TemporaryDirectory() as directory:
            for index, (payload, message) in enumerate(cases):
                with self.subTest(index=index), self.assertRaisesRegex(MotionError, message):
                    load_motion(self.write_archive(Path(directory), payload), profile)


if __name__ == "__main__":
    unittest.main()


def build_minimal_rig(profile: RobotProfile):
    data = bpy.data.armatures.new("TestRig")
    armature = bpy.data.objects.new("TestRig", data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    world = Matrix.Identity(4)
    for body in profile.bodies:
        world = Matrix.Translation(body.position) if body.parent is None else world @ Matrix.Translation(body.position)
        bone = data.edit_bones.new(body.name)
        bone.matrix = world
        bone.length = 0.01
        if body.parent is not None:
            bone.parent = bones[body.parent]
        bones[body.name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.rotation_mode = "QUATERNION"
    for joint in profile.joints:
        armature.pose.bones[joint.child_body].rotation_mode = "XYZ"
    return armature


def action_payload(profile: RobotProfile) -> dict[str, np.ndarray]:
    payload = archive_payload(profile)
    payload["body_pos_w"][:, 0] = np.array(
        [[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [2.0, 2.0, 3.0]]
    )
    payload["body_quat_w"][:, 0] = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.70710678, 0.0, 0.0, 0.70710678], [0.0, 0.0, 0.0, 1.0]]
    )
    return payload


def assert_vector_close(test: unittest.TestCase, actual, expected):
    np.testing.assert_allclose(tuple(actual), tuple(expected), atol=1e-6)


def assert_quaternion_close(test: unittest.TestCase, actual, expected):
    test.assertAlmostEqual(abs(Quaternion(actual).dot(Quaternion(expected))), 1.0, places=6)


class MotionActionTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)

    def import_action(self, action_name="Walk"):
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            return import_motion_action(self.armature, self.profile, path, action_name=action_name)

    def test_imports_calibrated_root_and_joint_keys_into_fake_user_action(self):
        bpy.context.scene.render.fps_base = 1.25
        action = self.import_action()
        scene = bpy.context.scene
        pose_bone = self.armature.pose.bones["hip_link"]

        self.assertEqual(action.name, "Walk")
        self.assertTrue(action.use_fake_user)
        self.assertEqual((scene.frame_start, scene.frame_end), (1, 3))
        self.assertEqual((scene.render.fps, scene.render.fps_base), (50, 1.0))
        scene.frame_set(3)
        assert_vector_close(self, self.armature.location, (2.0, 2.0, 2.875))
        assert_quaternion_close(self, self.armature.rotation_quaternion, (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(pose_bone.rotation_euler.z, 0.5, places=6)

    def test_preserves_existing_action_when_blender_assigns_collision_suffix(self):
        existing = bpy.data.actions.new("Walk")
        action = self.import_action()
        self.assertEqual(existing.name, "Walk")
        self.assertEqual(action.name, "Walk.001")

    def test_restores_scene_and_removes_partial_action_when_keying_fails(self):
        scene = bpy.context.scene
        prior = bpy.data.actions.new("Prior")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        self.armature.location = (9.0, 8.0, 7.0)
        self.armature.rotation_mode = "QUATERNION"
        self.armature.rotation_quaternion = (0.5, 0.5, 0.5, 0.5)
        pose_bone = self.armature.pose.bones["hip_link"]
        pose_bone.rotation_euler.z = 0.2
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with mock.patch.object(type(pose_bone), "keyframe_insert", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(self.armature, self.profile, path, action_name="Broken")

        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        assert_vector_close(self, self.armature.location, (9.0, 8.0, 7.0))
        assert_quaternion_close(self, self.armature.rotation_quaternion, (0.5, 0.5, 0.5, 0.5))
        self.assertAlmostEqual(pose_bone.rotation_euler.z, 0.2, places=6)
        self.assertIsNone(bpy.data.actions.get("Broken"))

    def test_restores_keyed_action_and_unkeyed_overrides_after_keying_fails(self):
        scene = bpy.context.scene
        created_mouth_property = not hasattr(bpy.types.Object, "duck_mouth_open")
        if created_mouth_property:
            bpy.types.Object.duck_mouth_open = bpy.props.FloatProperty(default=0.0)
            self.addCleanup(delattr, bpy.types.Object, "duck_mouth_open")

        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        mouth_edit_bone = self.armature.data.edit_bones.new("mouth::lower_beak")
        mouth_edit_bone.parent = self.armature.data.edit_bones["root"]
        mouth_edit_bone.matrix = Matrix.Translation((0.0, 0.0, 0.2))
        mouth_edit_bone.length = 0.01
        bpy.ops.object.mode_set(mode="OBJECT")

        prior = bpy.data.actions.new("PriorKeyed")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        hip = self.armature.pose.bones["hip_link"]
        mouth = self.armature.pose.bones["mouth::lower_beak"]

        self.armature.location = (1.0, 2.0, 3.0)
        self.armature.keyframe_insert(data_path="location", frame=7)
        self.armature.location = (5.0, 6.0, 7.0)
        self.armature.keyframe_insert(data_path="location", frame=9)
        hip.rotation_mode = "XYZ"
        hip.rotation_euler.z = 0.1
        hip.keyframe_insert(data_path="rotation_euler", index=2, frame=7)
        hip.rotation_euler.z = 0.9
        hip.keyframe_insert(data_path="rotation_euler", index=2, frame=9)
        mouth.location = (0.01, 0.0, 0.0)
        mouth.keyframe_insert(data_path="location", frame=7)
        mouth.location = (0.03, 0.0, 0.0)
        mouth.keyframe_insert(data_path="location", frame=9)
        self.armature.duck_mouth_open = 0.1
        self.armature.keyframe_insert(data_path="duck_mouth_open", frame=7)
        self.armature.duck_mouth_open = 0.9
        self.armature.keyframe_insert(data_path="duck_mouth_open", frame=9)

        scene.frame_start, scene.frame_end = 7, 9
        scene.render.fps = 23
        scene.render.fps_base = 1.001
        scene.frame_set(8)
        self.armature.rotation_mode = "QUATERNION"
        self.armature.matrix_world = Matrix.Translation((9.0, 8.0, 7.0)) @ Quaternion(
            (0.9238795, 0.0, 0.3826834, 0.0)
        ).to_matrix().to_4x4()
        hip.rotation_mode = "ZYX"
        hip.matrix_basis = Matrix.Translation((0.01, 0.02, 0.03)) @ Quaternion(
            (0.9659258, 0.2588190, 0.0, 0.0)
        ).to_matrix().to_4x4()
        mouth.rotation_mode = "AXIS_ANGLE"
        mouth.matrix_basis = Matrix.Translation((0.04, 0.05, 0.06)) @ Quaternion(
            (0.9848078, 0.0, 0.0, 0.1736482)
        ).to_matrix().to_4x4()
        self.armature.duck_mouth_open = 0.73
        expected_matrix = self.armature.matrix_world.copy()
        expected_hip_basis = hip.matrix_basis.copy()
        expected_mouth_basis = mouth.matrix_basis.copy()

        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(
                Path(directory), action_payload(self.profile)
            )
            with mock.patch.object(
                type(hip), "keyframe_insert", side_effect=RuntimeError("injected")
            ):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(
                        self.armature, self.profile, path, action_name="BrokenKeyed"
                    )

        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual(
            (
                scene.frame_start,
                scene.frame_end,
                scene.frame_current,
                scene.render.fps,
            ),
            (7, 9, 8, 23),
        )
        self.assertAlmostEqual(scene.render.fps_base, 1.001, places=6)
        np.testing.assert_allclose(self.armature.matrix_world, expected_matrix, atol=1e-6)
        self.assertEqual(self.armature.rotation_mode, "QUATERNION")
        np.testing.assert_allclose(hip.matrix_basis, expected_hip_basis, atol=1e-6)
        self.assertEqual(hip.rotation_mode, "ZYX")
        np.testing.assert_allclose(mouth.matrix_basis, expected_mouth_basis, atol=1e-6)
        self.assertEqual(mouth.rotation_mode, "AXIS_ANGLE")
        self.assertAlmostEqual(self.armature.duck_mouth_open, 0.73, places=6)
        self.assertIsNone(bpy.data.actions.get("BrokenKeyed"))

    def test_removes_new_animation_data_when_keying_fails_without_prior_animation(self):
        self.assertIsNone(self.armature.animation_data)
        pose_bone = self.armature.pose.bones["hip_link"]
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with mock.patch.object(type(pose_bone), "keyframe_insert", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(self.armature, self.profile, path, action_name="NoAnimation")

        self.assertIsNone(self.armature.animation_data)
        self.assertIsNone(bpy.data.actions.get("NoAnimation"))

    def test_translates_animation_data_creation_failure_without_mutation(self):
        class FailingAnimationCreateArmature:
            def __init__(self, armature):
                self._armature = armature

            def __getattr__(self, name):
                return getattr(self._armature, name)

            def animation_data_create(self):
                raise RuntimeError("create injected")

        self.assertIsNone(self.armature.animation_data)
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with self.assertRaisesRegex(MotionError, "create injected"):
                import_motion_action(
                    FailingAnimationCreateArmature(self.armature),
                    self.profile,
                    path,
                    action_name="CreateFailure",
                )

        self.assertIsNone(self.armature.animation_data)
        self.assertIsNone(bpy.data.actions.get("CreateFailure"))
