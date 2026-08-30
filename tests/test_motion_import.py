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
        action = self.import_action()
        scene = bpy.context.scene
        pose_bone = self.armature.pose.bones["hip_link"]

        self.assertEqual(action.name, "Walk")
        self.assertTrue(action.use_fake_user)
        self.assertEqual((scene.frame_start, scene.frame_end), (1, 3))
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
